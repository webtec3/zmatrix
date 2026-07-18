#ifndef _GNU_SOURCE
#define _GNU_SOURCE
#endif
// gpu_kernels.cu
#include "../config.h"
#include <cuda_runtime.h>
#include <cuda.h>
#include <cublas_v2.h>
#include <stddef.h>
#include <math.h>
#include <cstdio>
#include <cstdlib>
#include <dlfcn.h>
#include <link.h>
#include <mutex>
#include <cctype>
#include <fstream>
#include <limits>
#include <stdexcept>
#include <string>

// ========== WSL CUDA DRIVER FALLBACK ==========
static void* cuda_driver_handle = nullptr;
static std::string cuda_driver_loaded_path;
static std::string cuda_driver_error;
static std::once_flag cuda_driver_load_once;

static bool running_on_wsl() {
    std::ifstream release("/proc/sys/kernel/osrelease");
    std::string text;
    std::getline(release, text);
    for (char& ch : text) ch = static_cast<char>(std::tolower(static_cast<unsigned char>(ch)));
    return text.find("microsoft") != std::string::npos || text.find("wsl") != std::string::npos;
}

static void release_cuda_driver() {
    if (cuda_driver_handle) {
        dlclose(cuda_driver_handle);
        cuda_driver_handle = nullptr;
    }
    cuda_driver_loaded_path.clear();
    cuda_driver_loaded_path.shrink_to_fit();
    cuda_driver_error.clear();
    cuda_driver_error.shrink_to_fit();
}
// Função para encontrar libcuda.so com fallback para caminhos especiais (WSL)
// Isso resolve o problema onde WSL coloca libcuda.so em /usr/lib/wsl/lib/
static void* load_cuda_driver() {
    // Lista de caminhos a tentar em ordem de prioridade
    const char* cuda_lib_paths[] = {
        // WSL provides the real driver proxy here. These entries must precede
        // loader/default distro paths, which may resolve to a toolkit stub.
        "/usr/lib/wsl/lib/libcuda.so.1",
        "/usr/lib/wsl/lib/libcuda.so",
        "libcuda.so.1",                           // Caminho padrão (via LD_LIBRARY_PATH)
        "/usr/lib/wsl/lib/libcuda.so.1",         // WSL2 específico
        "/usr/lib/x86_64-linux-gnu/libcuda.so.1", // Linux padrão
        "libcuda.so",                             // Fallback sem versão
        "/usr/lib/wsl/lib/libcuda.so",           // WSL2 sem versão
        "/usr/lib/x86_64-linux-gnu/libcuda.so",  // Linux sem versão
        nullptr
    };

    void* handle = nullptr;
    for (int i = 0; cuda_lib_paths[i] != nullptr; i++) {
        handle = dlopen(cuda_lib_paths[i], RTLD_NOW | RTLD_GLOBAL);
        if (handle != nullptr) {
            cuda_driver_handle = handle;
            link_map* map = nullptr;
            if (dlinfo(handle, RTLD_DI_LINKMAP, &map) == 0 && map && map->l_name) {
                cuda_driver_loaded_path = map->l_name;
            } else {
                cuda_driver_loaded_path = cuda_lib_paths[i];
            }
            // On WSL only Microsoft's driver proxy is valid. Do not silently
            // accept a distro/toolkit stub merely because dlopen succeeded.
            if (running_on_wsl() && cuda_driver_loaded_path.find("/usr/lib/wsl/lib/") != 0) {
                cuda_driver_error = "Rejected non-WSL CUDA driver library: " + cuda_driver_loaded_path;
                dlclose(handle);
                handle = nullptr;
                cuda_driver_handle = nullptr;
                cuda_driver_loaded_path.clear();
                continue;
            }
            std::atexit(release_cuda_driver);
            const char *dbg = std::getenv("ZMATRIX_GPU_DEBUG");
            if (dbg && dbg[0] == '1') {
                std::fprintf(stderr, "[zmatrix][gpu] CUDA driver loaded from: %s\n", cuda_driver_loaded_path.c_str());
            }
            return handle;
        }
    }

    // Se chegou aqui, nenhum caminho funcionou
    const char *dbg = std::getenv("ZMATRIX_GPU_DEBUG");
    if (dbg && dbg[0] == '1') {
        std::fprintf(stderr, "[zmatrix][gpu] WARNING: Could not load CUDA driver from any path:\n");
        for (int i = 0; cuda_lib_paths[i] != nullptr; i++) {
            std::fprintf(stderr, "[zmatrix][gpu]   - Tried: %s\n", cuda_lib_paths[i]);
        }
        std::fprintf(stderr, "[zmatrix][gpu] Last dlopen error: %s\n", dlerror());
        std::fprintf(stderr, "[zmatrix][gpu] TROUBLESHOOTING: Try exporting LD_LIBRARY_PATH=/usr/lib/wsl/lib:$LD_LIBRARY_PATH\n");
    }
    return nullptr;
}

// Executar carregamento uma única vez ao inicializar o módulo
static void __attribute__((constructor)) init_cuda_driver() {
    // Tenta carregar libcuda de forma robusta
    // Nota: O construtor é chamado antes de cudaGetDeviceCount
    // Driver loading is intentionally lazy. CPU-only PHP requests must not
    // initialize CUDA or retain driver resources merely by loading ZMatrix.
}

static float *cache_a = nullptr;
static size_t cap_a = 0;
static float *cache_b = nullptr;
static size_t cap_b = 0;
static bool cache_inited = false;
static std::mutex cache_mutex;

static void release_cache() {
    if (cache_a) {
        cudaFree(cache_a);
        cache_a = nullptr;
        cap_a = 0;
    }
    if (cache_b) {
        cudaFree(cache_b);
        cache_b = nullptr;
        cap_b = 0;
    }
}

static inline void init_cache() {
    if (!cache_inited) {
        cache_inited = true;
        std::atexit(release_cache);
    }
}

static inline bool ensure_buffer(size_t n, float *&buf, size_t &cap) {
    if (n <= cap && buf) return true;
    if (buf) {
        cudaError_t free_status = cudaFree(buf);
        buf = nullptr;
        cap = 0;
        if (free_status != cudaSuccess) {
            throw std::runtime_error(std::string("CUDA cache free failed: ") + cudaGetErrorString(free_status));
        }
    }
    cudaError_t err = cudaMalloc(reinterpret_cast<void**>(&buf), n * sizeof(float));
    if (err != cudaSuccess) {
        buf = nullptr;
        cap = 0;
        throw std::runtime_error(std::string("CUDA cache allocation failed: ") + cudaGetErrorString(err));
    }
    cap = n;
    return true;
}

#define CUDA_CHECK_CONTEXT(stmt, context) do { \
    cudaError_t cuda_status_ = (stmt); \
    if (cuda_status_ != cudaSuccess) { \
        throw std::runtime_error(std::string("CUDA error in ") + (context) + \
            " at " + __FILE__ + ":" + std::to_string(__LINE__) + \
            " while evaluating " #stmt ": " + cudaGetErrorString(cuda_status_) + \
            " (code " + std::to_string(static_cast<int>(cuda_status_)) + ")"); \
    } \
} while (0)

#define CUDA_CHECK(stmt) CUDA_CHECK_CONTEXT(stmt, __func__)

#define CUDA_KERNEL_CHECK(context) do { \
    CUDA_CHECK_CONTEXT(cudaPeekAtLastError(), context); \
    CUDA_CHECK_CONTEXT(cudaDeviceSynchronize(), context); \
} while (0)

static const char* cublas_status_name(cublasStatus_t status) {
    switch (status) {
        case CUBLAS_STATUS_SUCCESS: return "CUBLAS_STATUS_SUCCESS";
        case CUBLAS_STATUS_NOT_INITIALIZED: return "CUBLAS_STATUS_NOT_INITIALIZED";
        case CUBLAS_STATUS_ALLOC_FAILED: return "CUBLAS_STATUS_ALLOC_FAILED";
        case CUBLAS_STATUS_INVALID_VALUE: return "CUBLAS_STATUS_INVALID_VALUE";
        case CUBLAS_STATUS_ARCH_MISMATCH: return "CUBLAS_STATUS_ARCH_MISMATCH";
        case CUBLAS_STATUS_MAPPING_ERROR: return "CUBLAS_STATUS_MAPPING_ERROR";
        case CUBLAS_STATUS_EXECUTION_FAILED: return "CUBLAS_STATUS_EXECUTION_FAILED";
        case CUBLAS_STATUS_INTERNAL_ERROR: return "CUBLAS_STATUS_INTERNAL_ERROR";
        default: return "CUBLAS_STATUS_UNKNOWN";
    }
}

#define CUBLAS_CHECK_CONTEXT(stmt, context) do { \
    cublasStatus_t cublas_status_ = (stmt); \
    if (cublas_status_ != CUBLAS_STATUS_SUCCESS) { \
        throw std::runtime_error(std::string("cuBLAS error in ") + (context) + \
            " at " + __FILE__ + ":" + std::to_string(__LINE__) + \
            " while evaluating " #stmt ": " + cublas_status_name(cublas_status_) + \
            " (code " + std::to_string(static_cast<int>(cublas_status_)) + ")"); \
    } \
} while (0)

extern "C" int gpu_available() {
    std::call_once(cuda_driver_load_once, []() { load_cuda_driver(); });
    if (!cuda_driver_handle) {
        if (cuda_driver_error.empty()) cuda_driver_error = "CUDA driver library could not be loaded";
        return 0;
    }

    using CuInit = CUresult (*)(unsigned int);
    using CuDeviceGetCount = CUresult (*)(int*);
    auto cu_init = reinterpret_cast<CuInit>(dlsym(cuda_driver_handle, "cuInit"));
    auto cu_device_get_count = reinterpret_cast<CuDeviceGetCount>(dlsym(cuda_driver_handle, "cuDeviceGetCount"));
    if (!cu_init || !cu_device_get_count) {
        cuda_driver_error = "Loaded CUDA library is not a functional driver: " + cuda_driver_loaded_path;
        return 0;
    }
#ifdef HAVE_WSL
    // This direct reference intentionally keeps libcuda.so.1 as DT_NEEDED.
    // It makes the dynamic loader resolve the WSL proxy through ZMatrix's
    // RUNPATH before libcudart constructors can bind to a distro stub.
    int linked_driver_version = 0;
    CUresult linked_driver_status = cuDriverGetVersion(&linked_driver_version);
    if (linked_driver_status != CUDA_SUCCESS) {
        cuda_driver_error = "Linked WSL CUDA driver validation failed (code " +
            std::to_string(static_cast<int>(linked_driver_status)) + ")";
        return 0;
    }
#endif
    CUresult init_status = cu_init(0);
    int driver_count = 0;
    CUresult count_status = init_status == CUDA_SUCCESS ? cu_device_get_count(&driver_count) : init_status;
    if (init_status != CUDA_SUCCESS || count_status != CUDA_SUCCESS || driver_count < 1) {
        cuda_driver_error = "CUDA Driver API initialization failed for " + cuda_driver_loaded_path +
            " (cuInit=" + std::to_string(static_cast<int>(init_status)) +
            ", cuDeviceGetCount=" + std::to_string(static_cast<int>(count_status)) +
            ", devices=" + std::to_string(driver_count) + ")";
        return 0;
    }

    int count = 0;
    cudaError_t err = cudaGetDeviceCount(&count);
    if (err != cudaSuccess) {
        const char *dbg = std::getenv("ZMATRIX_GPU_DEBUG");
        if (dbg && dbg[0] == '1') {
            std::fprintf(stderr, "[zmatrix][gpu] ERROR: cudaGetDeviceCount failed: %s\n", cudaGetErrorString(err));
            std::fprintf(stderr, "[zmatrix][gpu] TROUBLESHOOTING:\n");
            std::fprintf(stderr, "[zmatrix][gpu]   1. Ensure NVIDIA GPU driver is installed: nvidia-smi\n");
            std::fprintf(stderr, "[zmatrix][gpu]   2. On WSL2, try: export LD_LIBRARY_PATH=/usr/lib/wsl/lib:$LD_LIBRARY_PATH\n");
            std::fprintf(stderr, "[zmatrix][gpu]   3. Or add it permanently to ~/.bashrc\n");
            std::fprintf(stderr, "[zmatrix][gpu]   4. Check if CUDA is properly installed: which nvcc\n");
        }
        cuda_driver_error = std::string("CUDA Runtime initialization failed: ") + cudaGetErrorString(err) +
            " (code " + std::to_string(static_cast<int>(err)) + ")";
        return 0;
    }
    err = cudaFree(nullptr);
    if (err != cudaSuccess) {
        cuda_driver_error = std::string("CUDA context initialization failed: ") + cudaGetErrorString(err) +
            " (code " + std::to_string(static_cast<int>(err)) + ")";
        return 0;
    }
    {
        const char *dbg = std::getenv("ZMATRIX_GPU_DEBUG");
        if (dbg && dbg[0] == '1') {
            std::fprintf(stderr, "[zmatrix][gpu] devices=%d\n", count);
        }
    }
    return (count > 0) ? 1 : 0;
}

extern "C" void gpu_require_available() {
    if (!gpu_available()) {
        throw std::runtime_error(cuda_driver_error.empty() ? "CUDA is unavailable" : cuda_driver_error);
    }
}

extern "C" const char* gpu_driver_path() {
    return cuda_driver_loaded_path.c_str();
}

__global__ void kernel_add(float* a, const float* b, size_t n) {
    size_t i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) {
        a[i] += b[i];
    }
}

__global__ void kernel_sub(float* a, const float* b, size_t n) {
    size_t i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) {
        a[i] -= b[i];
    }
}

__global__ void kernel_mul(float* a, const float* b, size_t n) {
    size_t i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) {
        a[i] *= b[i];
    }
}

__global__ void kernel_relu(float* a, size_t n) {
    size_t i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) {
        float v = a[i];
        a[i] = (v > 0.0f) ? v : 0.0f;
    }
}

__global__ void kernel_leaky_relu(float* a, float alpha, size_t n) {
    size_t i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) {
        float v = a[i];
        a[i] = (v > 0.0f) ? v : alpha * v;
    }
}

__global__ void kernel_leaky_relu_derivative(float* a, float alpha, size_t n) {
    size_t i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) {
        a[i] = (a[i] > 0.0f) ? 1.0f : alpha;
    }
}

__global__ void kernel_sigmoid(float* a, size_t n) {
    size_t i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) {
        float v = a[i];
        a[i] = 1.0f / (1.0f + expf(-v));
    }
}

__global__ void kernel_tanh(float* a, size_t n) {
    size_t i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) {
        a[i] = tanhf(a[i]);
    }
}

__global__ void kernel_exp(float* a, size_t n) {
    size_t i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) {
        a[i] = expf(a[i]);
    }
}

__global__ void kernel_abs(float* a, size_t n) {
    size_t i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) {
        a[i] = fabsf(a[i]);
    }
}

__global__ void kernel_scalar_add(float* a, float value, size_t n) {
    size_t i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) {
        a[i] += value;
    }
}

__global__ void kernel_scalar_sub(float* a, float value, size_t n) {
    size_t i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) {
        a[i] -= value;
    }
}

__global__ void kernel_scalar_mul(float* a, float value, size_t n) {
    size_t i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) {
        a[i] *= value;
    }
}

__global__ void kernel_scalar_div(float* a, float value, size_t n) {
    size_t i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) {
        a[i] /= value;
    }
}

__global__ void kernel_div(float* a, const float* b, size_t n) {
    size_t i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) a[i] /= b[i];
}

__global__ void kernel_pow(float* a, float exponent, size_t n) {
    size_t i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) a[i] = powf(a[i], exponent);
}

__global__ void kernel_log(float* a, size_t n) {
    size_t i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) a[i] = logf(a[i]);
}

__global__ void kernel_fill(float* a, float value, size_t n) {
    size_t i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) a[i] = value;
}

__global__ void kernel_find_zero(const float* values, size_t n, int* found) {
    size_t i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n && values[i] == 0.0f) atomicExch(found, 1);
}

__global__ void kernel_find_nonpositive(const float* values, size_t n, int* found) {
    size_t i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n && values[i] <= 0.0f) atomicExch(found, 1);
}

template <int TILE>
__global__ void kernel_transpose_tiled(const float* input, float* output, size_t rows, size_t cols) {
    __shared__ float tile[TILE][TILE + 1];
    size_t x = blockIdx.x * TILE + threadIdx.x;
    size_t y = blockIdx.y * TILE + threadIdx.y;
    if (x < cols && y < rows) tile[threadIdx.y][threadIdx.x] = input[y * cols + x];
    __syncthreads();
    x = blockIdx.y * TILE + threadIdx.x;
    y = blockIdx.x * TILE + threadIdx.y;
    if (x < rows && y < cols) output[y * rows + x] = tile[threadIdx.x][threadIdx.y];
}

enum ReductionMode : int { REDUCE_SUM = 0, REDUCE_MEAN = 1, REDUCE_MIN = 2, REDUCE_MAX = 3, REDUCE_ARGMIN = 4, REDUCE_ARGMAX = 5 };

__global__ void kernel_reduce_axis(const float* input, float* output, size_t outer, size_t axis_size, size_t inner, int mode) {
    size_t out_index = blockIdx.x * blockDim.x + threadIdx.x;
    size_t out_size = outer * inner;
    if (out_index >= out_size) return;

    size_t outer_index = out_index / inner;
    size_t inner_index = out_index % inner;
    size_t base = outer_index * axis_size * inner + inner_index;
    float value = input[base];
    size_t best_index = 0;

    for (size_t axis_index = 1; axis_index < axis_size; ++axis_index) {
        float candidate = input[base + axis_index * inner];
        if (mode == REDUCE_SUM || mode == REDUCE_MEAN) {
            value += candidate;
        } else if ((mode == REDUCE_MIN || mode == REDUCE_ARGMIN) && candidate < value) {
            value = candidate;
            best_index = axis_index;
        } else if ((mode == REDUCE_MAX || mode == REDUCE_ARGMAX) && candidate > value) {
            value = candidate;
            best_index = axis_index;
        }
    }

    if (mode == REDUCE_MEAN) value /= static_cast<float>(axis_size);
    output[out_index] = (mode == REDUCE_ARGMIN || mode == REDUCE_ARGMAX)
        ? static_cast<float>(best_index)
        : value;
}

__global__ void kernel_arg_value(const float* input, size_t n, size_t* output, int find_max) {
    if (blockIdx.x != 0 || threadIdx.x != 0) return;
    float best = input[0];
    size_t best_index = 0;
    for (size_t i = 1; i < n; ++i) {
        if ((find_max && input[i] > best) || (!find_max && input[i] < best)) {
            best = input[i];
            best_index = i;
        }
    }
    *output = best_index;
}

static inline void launch_1d(size_t n, int &blocks, int &threads) {
    threads = 256;
    blocks = static_cast<int>((n + threads - 1) / threads);
}

extern "C" void gpu_add(float* a, const float* b, size_t n) {
    std::lock_guard<std::mutex> cache_lock(cache_mutex);
    float *d_a = nullptr, *d_b = nullptr;
    int threads = 0, blocks = 0;

    if (n == 0) return;
    init_cache();
    if (!ensure_buffer(n, cache_a, cap_a)) return;
    if (!ensure_buffer(n, cache_b, cap_b)) return;
    d_a = cache_a;
    d_b = cache_b;

    CUDA_CHECK(cudaMemcpy(d_a, a, n * sizeof(float), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_b, b, n * sizeof(float), cudaMemcpyHostToDevice));

    launch_1d(n, blocks, threads);
    kernel_add<<<blocks, threads>>>(d_a, d_b, n);
    CUDA_CHECK(cudaDeviceSynchronize());

    CUDA_CHECK(cudaMemcpy(a, d_a, n * sizeof(float), cudaMemcpyDeviceToHost));

    return;
}

extern "C" void gpu_sub(float* a, const float* b, size_t n) {
    std::lock_guard<std::mutex> cache_lock(cache_mutex);
    float *d_a = nullptr, *d_b = nullptr;
    int threads = 0, blocks = 0;

    if (n == 0) return;
    init_cache();
    if (!ensure_buffer(n, cache_a, cap_a)) return;
    if (!ensure_buffer(n, cache_b, cap_b)) return;
    d_a = cache_a;
    d_b = cache_b;

    CUDA_CHECK(cudaMemcpy(d_a, a, n * sizeof(float), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_b, b, n * sizeof(float), cudaMemcpyHostToDevice));

    launch_1d(n, blocks, threads);
    kernel_sub<<<blocks, threads>>>(d_a, d_b, n);
    CUDA_CHECK(cudaDeviceSynchronize());

    CUDA_CHECK(cudaMemcpy(a, d_a, n * sizeof(float), cudaMemcpyDeviceToHost));

    return;
}

extern "C" void gpu_mul(float* a, const float* b, size_t n) {
    std::lock_guard<std::mutex> cache_lock(cache_mutex);
    float *d_a = nullptr, *d_b = nullptr;
    int threads = 0, blocks = 0;

    if (n == 0) return;
    init_cache();
    if (!ensure_buffer(n, cache_a, cap_a)) return;
    if (!ensure_buffer(n, cache_b, cap_b)) return;
    d_a = cache_a;
    d_b = cache_b;

    CUDA_CHECK(cudaMemcpy(d_a, a, n * sizeof(float), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_b, b, n * sizeof(float), cudaMemcpyHostToDevice));

    launch_1d(n, blocks, threads);
    kernel_mul<<<blocks, threads>>>(d_a, d_b, n);
    CUDA_CHECK(cudaDeviceSynchronize());

    CUDA_CHECK(cudaMemcpy(a, d_a, n * sizeof(float), cudaMemcpyDeviceToHost));

    return;
}

extern "C" void gpu_relu(float* a, size_t n) {
    std::lock_guard<std::mutex> cache_lock(cache_mutex);
    float *d_a = nullptr;
    int threads = 0, blocks = 0;

    if (n == 0) return;
    init_cache();
    if (!ensure_buffer(n, cache_a, cap_a)) return;
    d_a = cache_a;
    CUDA_CHECK(cudaMemcpy(d_a, a, n * sizeof(float), cudaMemcpyHostToDevice));

    launch_1d(n, blocks, threads);
    kernel_relu<<<blocks, threads>>>(d_a, n);
    CUDA_CHECK(cudaDeviceSynchronize());

    CUDA_CHECK(cudaMemcpy(a, d_a, n * sizeof(float), cudaMemcpyDeviceToHost));

    return;
}

extern "C" void gpu_leaky_relu(float* a, float alpha, size_t n) {
    std::lock_guard<std::mutex> cache_lock(cache_mutex);
    float *d_a = nullptr;
    int threads = 0, blocks = 0;

    if (n == 0) return;
    init_cache();
    if (!ensure_buffer(n, cache_a, cap_a)) return;
    d_a = cache_a;
    CUDA_CHECK(cudaMemcpy(d_a, a, n * sizeof(float), cudaMemcpyHostToDevice));

    launch_1d(n, blocks, threads);
    kernel_leaky_relu<<<blocks, threads>>>(d_a, alpha, n);
    CUDA_CHECK(cudaDeviceSynchronize());

    CUDA_CHECK(cudaMemcpy(a, d_a, n * sizeof(float), cudaMemcpyDeviceToHost));

    return;
}

extern "C" void gpu_leaky_relu_derivative(float* a, float alpha, size_t n) {
    std::lock_guard<std::mutex> cache_lock(cache_mutex);
    float *d_a = nullptr;
    int threads = 0, blocks = 0;

    if (n == 0) return;
    init_cache();
    if (!ensure_buffer(n, cache_a, cap_a)) return;
    d_a = cache_a;
    CUDA_CHECK(cudaMemcpy(d_a, a, n * sizeof(float), cudaMemcpyHostToDevice));

    launch_1d(n, blocks, threads);
    kernel_leaky_relu_derivative<<<blocks, threads>>>(d_a, alpha, n);
    CUDA_CHECK(cudaDeviceSynchronize());

    CUDA_CHECK(cudaMemcpy(a, d_a, n * sizeof(float), cudaMemcpyDeviceToHost));

    return;
}

extern "C" void gpu_sigmoid(float* a, size_t n) {
    std::lock_guard<std::mutex> cache_lock(cache_mutex);
    float *d_a = nullptr;
    int threads = 0, blocks = 0;

    if (n == 0) return;
    init_cache();
    if (!ensure_buffer(n, cache_a, cap_a)) return;
    d_a = cache_a;
    CUDA_CHECK(cudaMemcpy(d_a, a, n * sizeof(float), cudaMemcpyHostToDevice));

    launch_1d(n, blocks, threads);
    kernel_sigmoid<<<blocks, threads>>>(d_a, n);
    CUDA_CHECK(cudaDeviceSynchronize());

    CUDA_CHECK(cudaMemcpy(a, d_a, n * sizeof(float), cudaMemcpyDeviceToHost));

    return;
}

extern "C" void gpu_tanh(float* a, size_t n) {
    std::lock_guard<std::mutex> cache_lock(cache_mutex);
    float *d_a = nullptr;
    int threads = 0, blocks = 0;

    if (n == 0) return;
    init_cache();
    if (!ensure_buffer(n, cache_a, cap_a)) return;
    d_a = cache_a;
    CUDA_CHECK(cudaMemcpy(d_a, a, n * sizeof(float), cudaMemcpyHostToDevice));

    launch_1d(n, blocks, threads);
    kernel_tanh<<<blocks, threads>>>(d_a, n);
    CUDA_CHECK(cudaDeviceSynchronize());

    CUDA_CHECK(cudaMemcpy(a, d_a, n * sizeof(float), cudaMemcpyDeviceToHost));

    return;
}

extern "C" void gpu_exp(float* a, size_t n) {
    std::lock_guard<std::mutex> cache_lock(cache_mutex);
    float *d_a = nullptr;
    int threads = 0, blocks = 0;

    if (n == 0) return;
    init_cache();
    if (!ensure_buffer(n, cache_a, cap_a)) return;
    d_a = cache_a;
    CUDA_CHECK(cudaMemcpy(d_a, a, n * sizeof(float), cudaMemcpyHostToDevice));

    launch_1d(n, blocks, threads);
    kernel_exp<<<blocks, threads>>>(d_a, n);
    CUDA_CHECK(cudaDeviceSynchronize());

    CUDA_CHECK(cudaMemcpy(a, d_a, n * sizeof(float), cudaMemcpyDeviceToHost));

    return;
}

extern "C" void gpu_abs(float* a, size_t n) {
    std::lock_guard<std::mutex> cache_lock(cache_mutex);
    float *d_a = nullptr;
    int threads = 0, blocks = 0;

    if (n == 0) return;
    init_cache();
    if (!ensure_buffer(n, cache_a, cap_a)) return;
    d_a = cache_a;
    CUDA_CHECK(cudaMemcpy(d_a, a, n * sizeof(float), cudaMemcpyHostToDevice));

    launch_1d(n, blocks, threads);
    kernel_abs<<<blocks, threads>>>(d_a, n);
    CUDA_CHECK(cudaDeviceSynchronize());

    CUDA_CHECK(cudaMemcpy(a, d_a, n * sizeof(float), cudaMemcpyDeviceToHost));

    return;
}

extern "C" void gpu_scalar_add(float* a, float value, size_t n) {
    std::lock_guard<std::mutex> cache_lock(cache_mutex);
    float *d_a = nullptr;
    int threads = 0, blocks = 0;

    if (n == 0) return;
    init_cache();
    if (!ensure_buffer(n, cache_a, cap_a)) return;
    d_a = cache_a;
    CUDA_CHECK(cudaMemcpy(d_a, a, n * sizeof(float), cudaMemcpyHostToDevice));

    launch_1d(n, blocks, threads);
    kernel_scalar_add<<<blocks, threads>>>(d_a, value, n);
    CUDA_CHECK(cudaDeviceSynchronize());

    CUDA_CHECK(cudaMemcpy(a, d_a, n * sizeof(float), cudaMemcpyDeviceToHost));

    return;
}

extern "C" void gpu_scalar_sub(float* a, float value, size_t n) {
    std::lock_guard<std::mutex> cache_lock(cache_mutex);
    float *d_a = nullptr;
    int threads = 0, blocks = 0;

    if (n == 0) return;
    init_cache();
    if (!ensure_buffer(n, cache_a, cap_a)) return;
    d_a = cache_a;
    CUDA_CHECK(cudaMemcpy(d_a, a, n * sizeof(float), cudaMemcpyHostToDevice));

    launch_1d(n, blocks, threads);
    kernel_scalar_sub<<<blocks, threads>>>(d_a, value, n);
    CUDA_CHECK(cudaDeviceSynchronize());

    CUDA_CHECK(cudaMemcpy(a, d_a, n * sizeof(float), cudaMemcpyDeviceToHost));

    return;
}

extern "C" void gpu_scalar_mul(float* a, float value, size_t n) {
    std::lock_guard<std::mutex> cache_lock(cache_mutex);
    float *d_a = nullptr;
    int threads = 0, blocks = 0;

    if (n == 0) return;
    init_cache();
    if (!ensure_buffer(n, cache_a, cap_a)) return;
    d_a = cache_a;
    CUDA_CHECK(cudaMemcpy(d_a, a, n * sizeof(float), cudaMemcpyHostToDevice));

    launch_1d(n, blocks, threads);
    kernel_scalar_mul<<<blocks, threads>>>(d_a, value, n);
    CUDA_CHECK(cudaDeviceSynchronize());

    CUDA_CHECK(cudaMemcpy(a, d_a, n * sizeof(float), cudaMemcpyDeviceToHost));

    return;
}

extern "C" void gpu_scalar_div(float* a, float value, size_t n) {
    std::lock_guard<std::mutex> cache_lock(cache_mutex);
    float *d_a = nullptr;
    int threads = 0, blocks = 0;

    if (n == 0) return;
    init_cache();
    if (!ensure_buffer(n, cache_a, cap_a)) return;
    d_a = cache_a;
    CUDA_CHECK(cudaMemcpy(d_a, a, n * sizeof(float), cudaMemcpyHostToDevice));

    launch_1d(n, blocks, threads);
    kernel_scalar_div<<<blocks, threads>>>(d_a, value, n);
    CUDA_CHECK(cudaDeviceSynchronize());

    CUDA_CHECK(cudaMemcpy(a, d_a, n * sizeof(float), cudaMemcpyDeviceToHost));

    return;
}

extern "C" void gpu_add_device(float* d_a, const float* d_b, size_t n) {
    int threads = 0, blocks = 0;
    if (n == 0) return;
    launch_1d(n, blocks, threads);
    kernel_add<<<blocks, threads>>>(d_a, d_b, n);
    CUDA_KERNEL_CHECK("add");
}

extern "C" void gpu_sub_device(float* d_a, const float* d_b, size_t n) {
    int threads = 0, blocks = 0;
    if (n == 0) return;
    launch_1d(n, blocks, threads);
    kernel_sub<<<blocks, threads>>>(d_a, d_b, n);
    CUDA_KERNEL_CHECK("sub");
}

extern "C" void gpu_mul_device(float* d_a, const float* d_b, size_t n) {
    int threads = 0, blocks = 0;
    if (n == 0) return;
    launch_1d(n, blocks, threads);
    kernel_mul<<<blocks, threads>>>(d_a, d_b, n);
    CUDA_KERNEL_CHECK("mul");
}

extern "C" void gpu_div_device(float* d_a, const float* d_b, size_t n) {
    int threads = 0, blocks = 0;
    if (n == 0) return;
    launch_1d(n, blocks, threads);

    int* d_found = nullptr;
    CUDA_CHECK_CONTEXT(cudaMalloc(reinterpret_cast<void**>(&d_found), sizeof(int)), "divide validation allocation");
    try {
        CUDA_CHECK_CONTEXT(cudaMemset(d_found, 0, sizeof(int)), "divide validation reset");
        kernel_find_zero<<<blocks, threads>>>(d_b, n, d_found);
        CUDA_KERNEL_CHECK("divide validation");
        int found = 0;
        CUDA_CHECK_CONTEXT(cudaMemcpy(&found, d_found, sizeof(int), cudaMemcpyDeviceToHost), "divide validation result");
        CUDA_CHECK_CONTEXT(cudaFree(d_found), "divide validation free");
        d_found = nullptr;
        if (found) throw std::runtime_error("Divisao por zero detectada");
        kernel_div<<<blocks, threads>>>(d_a, d_b, n);
        CUDA_KERNEL_CHECK("divide");
    } catch (...) {
        if (d_found) cudaFree(d_found);
        throw;
    }
}

extern "C" void gpu_relu_device(float* d_a, size_t n) {
    int threads = 0, blocks = 0;
    if (n == 0) return;
    launch_1d(n, blocks, threads);
    kernel_relu<<<blocks, threads>>>(d_a, n);
    CUDA_KERNEL_CHECK("relu");
}

extern "C" void gpu_leaky_relu_device(float* d_a, float alpha, size_t n) {
    int threads = 0, blocks = 0;
    if (n == 0) return;
    launch_1d(n, blocks, threads);
    kernel_leaky_relu<<<blocks, threads>>>(d_a, alpha, n);
    CUDA_KERNEL_CHECK("leakyRelu");
}

extern "C" void gpu_leaky_relu_derivative_device(float* d_a, float alpha, size_t n) {
    int threads = 0, blocks = 0;
    if (n == 0) return;
    launch_1d(n, blocks, threads);
    kernel_leaky_relu_derivative<<<blocks, threads>>>(d_a, alpha, n);
    CUDA_KERNEL_CHECK("leakyReluDerivative");
}

extern "C" void gpu_sigmoid_device(float* d_a, size_t n) {
    int threads = 0, blocks = 0;
    if (n == 0) return;
    launch_1d(n, blocks, threads);
    kernel_sigmoid<<<blocks, threads>>>(d_a, n);
    CUDA_KERNEL_CHECK("sigmoid");
}

extern "C" void gpu_tanh_device(float* d_a, size_t n) {
    int threads = 0, blocks = 0;
    if (n == 0) return;
    launch_1d(n, blocks, threads);
    kernel_tanh<<<blocks, threads>>>(d_a, n);
    CUDA_KERNEL_CHECK("tanh");
}

extern "C" void gpu_exp_device(float* d_a, size_t n) {
    int threads = 0, blocks = 0;
    if (n == 0) return;
    launch_1d(n, blocks, threads);
    kernel_exp<<<blocks, threads>>>(d_a, n);
    CUDA_KERNEL_CHECK("exp");
}

extern "C" void gpu_abs_device(float* d_a, size_t n) {
    int threads = 0, blocks = 0;
    if (n == 0) return;
    launch_1d(n, blocks, threads);
    kernel_abs<<<blocks, threads>>>(d_a, n);
    CUDA_KERNEL_CHECK("abs");
}

extern "C" void gpu_scalar_add_device(float* d_a, float value, size_t n) {
    int threads = 0, blocks = 0;
    if (n == 0) return;
    launch_1d(n, blocks, threads);
    kernel_scalar_add<<<blocks, threads>>>(d_a, value, n);
    CUDA_KERNEL_CHECK("scalar add");
}

extern "C" void gpu_scalar_sub_device(float* d_a, float value, size_t n) {
    int threads = 0, blocks = 0;
    if (n == 0) return;
    launch_1d(n, blocks, threads);
    kernel_scalar_sub<<<blocks, threads>>>(d_a, value, n);
    CUDA_KERNEL_CHECK("scalar sub");
}

extern "C" void gpu_scalar_mul_device(float* d_a, float value, size_t n) {
    int threads = 0, blocks = 0;
    if (n == 0) return;
    launch_1d(n, blocks, threads);
    kernel_scalar_mul<<<blocks, threads>>>(d_a, value, n);
    CUDA_KERNEL_CHECK("scalar mul");
}

extern "C" void gpu_scalar_div_device(float* d_a, float value, size_t n) {
    int threads = 0, blocks = 0;
    if (n == 0) return;
    launch_1d(n, blocks, threads);
    kernel_scalar_div<<<blocks, threads>>>(d_a, value, n);
    CUDA_KERNEL_CHECK("scalar div");
}

extern "C" void gpu_pow_device(float* d_a, float exponent, size_t n) {
    int threads = 0, blocks = 0;
    if (n == 0) return;
    launch_1d(n, blocks, threads);
    kernel_pow<<<blocks, threads>>>(d_a, exponent, n);
    CUDA_KERNEL_CHECK("pow");
}

extern "C" void gpu_log_device(float* d_a, size_t n) {
    int threads = 0, blocks = 0;
    if (n == 0) return;
    launch_1d(n, blocks, threads);

    int* d_found = nullptr;
    CUDA_CHECK_CONTEXT(cudaMalloc(reinterpret_cast<void**>(&d_found), sizeof(int)), "log validation allocation");
    try {
        CUDA_CHECK_CONTEXT(cudaMemset(d_found, 0, sizeof(int)), "log validation reset");
        kernel_find_nonpositive<<<blocks, threads>>>(d_a, n, d_found);
        CUDA_KERNEL_CHECK("log validation");
        int found = 0;
        CUDA_CHECK_CONTEXT(cudaMemcpy(&found, d_found, sizeof(int), cudaMemcpyDeviceToHost), "log validation result");
        CUDA_CHECK_CONTEXT(cudaFree(d_found), "log validation free");
        d_found = nullptr;
        if (found) throw std::runtime_error("Logaritmo de valor nao positivo.");
        kernel_log<<<blocks, threads>>>(d_a, n);
        CUDA_KERNEL_CHECK("log");
    } catch (...) {
        if (d_found) cudaFree(d_found);
        throw;
    }
}

extern "C" void gpu_fill_device(float* d_a, float value, size_t n) {
    int threads = 0, blocks = 0;
    if (n == 0) return;
    launch_1d(n, blocks, threads);
    kernel_fill<<<blocks, threads>>>(d_a, value, n);
    CUDA_KERNEL_CHECK("fill");
}

extern "C" void gpu_transpose_device(const float* d_input, float* d_output, size_t rows, size_t cols) {
    if (rows == 0 || cols == 0) return;
    constexpr int tile_size = 32;
    dim3 threads(tile_size, tile_size);
    dim3 blocks(static_cast<unsigned int>((cols + tile_size - 1) / tile_size),
                static_cast<unsigned int>((rows + tile_size - 1) / tile_size));
    kernel_transpose_tiled<tile_size><<<blocks, threads>>>(d_input, d_output, rows, cols);
    CUDA_KERNEL_CHECK("transpose");
}

static void launch_reduce_axis(const float* d_input, float* d_output, size_t outer, size_t axis_size, size_t inner, int mode, const char* context) {
    if (!d_input || !d_output) throw std::invalid_argument(std::string(context) + ": null device pointer");
    if (outer == 0 || axis_size == 0 || inner == 0) return;
    size_t output_size = outer * inner;
    int threads = 0, blocks = 0;
    launch_1d(output_size, blocks, threads);
    kernel_reduce_axis<<<blocks, threads>>>(d_input, d_output, outer, axis_size, inner, mode);
    CUDA_KERNEL_CHECK(context);
}

extern "C" void gpu_sum_axis_device(const float* input, float* output, size_t outer, size_t axis_size, size_t inner) {
    launch_reduce_axis(input, output, outer, axis_size, inner, REDUCE_SUM, "sum axis");
}

extern "C" void gpu_mean_axis_device(const float* input, float* output, size_t outer, size_t axis_size, size_t inner) {
    launch_reduce_axis(input, output, outer, axis_size, inner, REDUCE_MEAN, "mean axis");
}

extern "C" void gpu_min_axis_device(const float* input, float* output, size_t outer, size_t axis_size, size_t inner) {
    launch_reduce_axis(input, output, outer, axis_size, inner, REDUCE_MIN, "min axis");
}

extern "C" void gpu_max_axis_device(const float* input, float* output, size_t outer, size_t axis_size, size_t inner) {
    launch_reduce_axis(input, output, outer, axis_size, inner, REDUCE_MAX, "max axis");
}

extern "C" void gpu_arg_axis_device(const float* input, float* output, size_t outer, size_t axis_size, size_t inner, int find_max) {
    launch_reduce_axis(input, output, outer, axis_size, inner, find_max ? REDUCE_ARGMAX : REDUCE_ARGMIN,
        find_max ? "argmax axis" : "argmin axis");
}

static float reduce_value_device(const float* input, size_t n, int mode, const char* context) {
    if (!input || n == 0) throw std::invalid_argument(std::string(context) + ": empty input");
    float* output = nullptr;
    CUDA_CHECK_CONTEXT(cudaMalloc(reinterpret_cast<void**>(&output), sizeof(float)), context);
    try {
        kernel_reduce_axis<<<1, 1>>>(input, output, 1, n, 1, mode);
        CUDA_KERNEL_CHECK(context);
        float result = 0.0f;
        CUDA_CHECK_CONTEXT(cudaMemcpy(&result, output, sizeof(float), cudaMemcpyDeviceToHost), context);
        CUDA_CHECK_CONTEXT(cudaFree(output), context);
        return result;
    } catch (...) {
        if (output) cudaFree(output);
        throw;
    }
}

extern "C" float gpu_sum_value_device(const float* input, size_t n) {
    return reduce_value_device(input, n, REDUCE_SUM, "sum value");
}

extern "C" float gpu_min_value_device(const float* input, size_t n) {
    return reduce_value_device(input, n, REDUCE_MIN, "min value");
}

extern "C" float gpu_max_value_device(const float* input, size_t n) {
    return reduce_value_device(input, n, REDUCE_MAX, "max value");
}

extern "C" size_t gpu_arg_value_device(const float* input, size_t n, int find_max) {
    if (!input || n == 0) throw std::invalid_argument("arg reduction: empty input");
    size_t* output = nullptr;
    CUDA_CHECK_CONTEXT(cudaMalloc(reinterpret_cast<void**>(&output), sizeof(size_t)), "arg reduction allocation");
    try {
        kernel_arg_value<<<1, 1>>>(input, n, output, find_max);
        CUDA_KERNEL_CHECK(find_max ? "argmax value" : "argmin value");
        size_t result = 0;
        CUDA_CHECK_CONTEXT(cudaMemcpy(&result, output, sizeof(size_t), cudaMemcpyDeviceToHost), "arg reduction result");
        CUDA_CHECK_CONTEXT(cudaFree(output), "arg reduction free");
        return result;
    } catch (...) {
        if (output) cudaFree(output);
        throw;
    }
}
// ========== GPU MATMUL FORWARD DECLARATIONS ==========
extern "C" void gpu_matmul_device(const float* d_a, const float* d_b, float* d_c, size_t m, size_t k, size_t n);

static cublasHandle_t sgemm_handle = nullptr;
static std::mutex sgemm_handle_mutex;

static void release_sgemm_handle() {
    std::lock_guard<std::mutex> lock(sgemm_handle_mutex);
    if (sgemm_handle) {
        cublasDestroy(sgemm_handle); // shutdown cleanup must not throw
        sgemm_handle = nullptr;
    }
}

// ========== GPU MATMUL IMPLEMENTATIONS ==========
// gpu_matmul: Host I/O version (copies data to GPU, computes, copies back)
extern "C" void gpu_matmul(const float* a, const float* b, float* c, size_t m, size_t k, size_t n) {
    if (!a || !b || !c) return;
    
    size_t size_a = m * k;
    size_t size_b = k * n;
    size_t size_c = m * n;
    
    float *d_a = nullptr, *d_b = nullptr, *d_c = nullptr;

    try {
        CUDA_CHECK_CONTEXT(cudaMalloc(reinterpret_cast<void**>(&d_a), size_a * sizeof(float)), "host SGEMM A allocation");
        CUDA_CHECK_CONTEXT(cudaMalloc(reinterpret_cast<void**>(&d_b), size_b * sizeof(float)), "host SGEMM B allocation");
        // C is fully overwritten because beta == 0, so no host-to-device copy
        // or initialization is performed for the output buffer.
        CUDA_CHECK_CONTEXT(cudaMalloc(reinterpret_cast<void**>(&d_c), size_c * sizeof(float)), "host SGEMM C allocation");
        CUDA_CHECK_CONTEXT(cudaMemcpy(d_a, a, size_a * sizeof(float), cudaMemcpyHostToDevice), "host SGEMM A upload");
        CUDA_CHECK_CONTEXT(cudaMemcpy(d_b, b, size_b * sizeof(float), cudaMemcpyHostToDevice), "host SGEMM B upload");
        gpu_matmul_device(d_a, d_b, d_c, m, k, n);
        CUDA_CHECK_CONTEXT(cudaMemcpy(c, d_c, size_c * sizeof(float), cudaMemcpyDeviceToHost), "host SGEMM result download");
        CUDA_CHECK_CONTEXT(cudaFree(d_a), "host SGEMM A free"); d_a = nullptr;
        CUDA_CHECK_CONTEXT(cudaFree(d_b), "host SGEMM B free"); d_b = nullptr;
        CUDA_CHECK_CONTEXT(cudaFree(d_c), "host SGEMM C free"); d_c = nullptr;
    } catch (...) {
        if (d_a) cudaFree(d_a);
        if (d_b) cudaFree(d_b);
        if (d_c) cudaFree(d_c);
        throw;
    }
}

// gpu_matmul_device: Device-only version (assumes pointers are already on GPU)
extern "C" void gpu_matmul_device(const float* d_a, const float* d_b, float* d_c, size_t m, size_t k, size_t n) {
    if (!d_a || !d_b || !d_c) return;
    if (m > static_cast<size_t>(std::numeric_limits<int>::max()) ||
        k > static_cast<size_t>(std::numeric_limits<int>::max()) ||
        n > static_cast<size_t>(std::numeric_limits<int>::max())) {
        throw std::overflow_error("gpu_matmul_device dimensions exceed cuBLAS int range");
    }
    
    // A cuBLAS handle contains mutable stream/state, so creation and use are
    // serialized. PHP requests normally execute this path synchronously.
    std::lock_guard<std::mutex> lock(sgemm_handle_mutex);
    if (!sgemm_handle) {
        CUBLAS_CHECK_CONTEXT(cublasCreate(&sgemm_handle), "create SGEMM handle");
        std::atexit(release_sgemm_handle);
    }
    
    // C = A * B where A is m×k, B is k×n
    // ZMatrix stores A[M,K], B[K,N] and C[M,N] contiguously in row-major.
    // The same bytes are seen by cuBLAS as column-major A_col[K,M],
    // B_col[N,K] and C_col[N,M]. Therefore C_row = A_row * B_row is computed
    // as C_col = B_col * A_col, with dimensions N x M x K. No physical
    // transpose or output upload is required.
    
    float alpha = 1.0f, beta = 0.0f;
    CUBLAS_CHECK_CONTEXT(cublasSgemm(
        sgemm_handle,
        CUBLAS_OP_N,
        CUBLAS_OP_N,
        static_cast<int>(n),
        static_cast<int>(m),
        static_cast<int>(k),
        &alpha,
        d_b, static_cast<int>(n),
        d_a, static_cast<int>(k),
        &beta,
        d_c, static_cast<int>(n)
    ), "SGEMM row-major A*B");

    CUDA_CHECK_CONTEXT(cudaDeviceSynchronize(), "SGEMM execution");
}

// ========== GPU MATMUL FORWARD DECLARATIONS ==========
