#ifndef _GNU_SOURCE
#define _GNU_SOURCE
#endif
// gpu_kernels.cu
#include "../config.h"
#include <cuda_runtime.h>
#include <cuda.h>
#include <cublas_v2.h>
#include <cub/cub.cuh>
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

// Reductions are synchronous at the PHP boundary, so a small serialized cache
// safely removes cudaMalloc/cudaFree from repeated global reductions without
// introducing a general-purpose allocator or changing stream semantics.
static void* reduction_temporary = nullptr;
static size_t reduction_temporary_capacity = 0;
static float* reduction_output = nullptr;
static std::mutex reduction_cache_mutex;
static bool reduction_cache_inited = false;
static cublasHandle_t sgemm_handle = nullptr;
static std::mutex sgemm_handle_mutex;
static void release_sgemm_handle();

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

static void release_reduction_cache() {
    if (reduction_temporary) cudaFree(reduction_temporary);
    if (reduction_output) cudaFree(reduction_output);
    reduction_temporary = nullptr;
    reduction_temporary_capacity = 0;
    reduction_output = nullptr;
}

static inline void init_reduction_cache() {
    if (!reduction_cache_inited) {
        reduction_cache_inited = true;
        std::atexit(release_reduction_cache);
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

// Caller holds reduction_cache_mutex. Synchronous reductions and scans reuse
// this single CUB workspace and therefore cannot overlap in the current model.
static void* ensure_cub_temporary_unlocked(size_t required_bytes, const char* context) {
    if (required_bytes <= reduction_temporary_capacity && reduction_temporary) return reduction_temporary;
    void* replacement = nullptr;
    CUDA_CHECK_CONTEXT(cudaMalloc(&replacement, required_bytes), context);
    if (reduction_temporary) {
        const cudaError_t free_status = cudaFree(reduction_temporary);
        if (free_status != cudaSuccess) {
            cudaFree(replacement);
            throw std::runtime_error(std::string(context) + ": CUB cache resize free failed: " + cudaGetErrorString(free_status));
        }
    }
    reduction_temporary = replacement;
    reduction_temporary_capacity = required_bytes;
    return reduction_temporary;
}

template <typename T>
class ScopedCudaBuffer {
public:
    ScopedCudaBuffer(size_t count, const char* context) {
        CUDA_CHECK_CONTEXT(cudaMalloc(reinterpret_cast<void**>(&pointer_), count * sizeof(T)), context);
    }
    ~ScopedCudaBuffer() noexcept {
        if (pointer_) cudaFree(pointer_);
    }
    ScopedCudaBuffer(const ScopedCudaBuffer&) = delete;
    ScopedCudaBuffer& operator=(const ScopedCudaBuffer&) = delete;
    T* get() const noexcept { return pointer_; }
    void release_checked(const char* context) {
        T* pointer = pointer_;
        pointer_ = nullptr;
        CUDA_CHECK_CONTEXT(cudaFree(pointer), context);
    }
private:
    T* pointer_ = nullptr;
};

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

__global__ void kernel_sqrt(float* a, size_t n) {
    size_t i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) a[i] = sqrtf(a[i]);
}

__global__ void kernel_clip(float* a, float min_value, float max_value, size_t n) {
    size_t i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) {
        // Keep the exact std::max(min, std::min(max, value)) CPU ordering,
        // including NaN values in the tensor or in either bound.
        const float inner = (a[i] < max_value) ? a[i] : max_value;
        a[i] = (min_value < inner) ? inner : min_value;
    }
}

__global__ void kernel_softmax_derivative(float* a, size_t n) {
    size_t i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) a[i] = a[i] * (1.0f - a[i]);
}

__global__ void kernel_greater(const float* a, const float* b, float* output,
                               size_t n, size_t broadcast_width, float scalar,
                               int use_scalar) {
    const size_t i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    const float rhs = use_scalar ? scalar : b[broadcast_width ? (i % broadcast_width) : i];
    output[i] = a[i] > rhs ? 1.0f : 0.0f;
}

__global__ void kernel_broadcast_materialize(const float* input, float* output,
                                             const size_t* output_shape,
                                             const size_t* output_strides,
                                             const size_t* input_shape,
                                             const size_t* input_strides,
                                             size_t output_rank, size_t input_rank,
                                             size_t output_size) {
    const size_t linear = blockIdx.x * blockDim.x + threadIdx.x;
    if (linear >= output_size) return;
    const size_t rank_difference = output_rank - input_rank;
    size_t input_offset = 0;
    for (size_t input_axis = 0; input_axis < input_rank; ++input_axis) {
        const size_t output_axis = rank_difference + input_axis;
        const size_t coordinate = (linear / output_strides[output_axis]) % output_shape[output_axis];
        if (input_shape[input_axis] != 1) input_offset += coordinate * input_strides[input_axis];
    }
    output[linear] = input[input_offset];
}

__global__ void kernel_tile(const float* input, float* output,
                            size_t input_size, size_t output_size) {
    const size_t i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < output_size) output[i] = input[i % input_size];
}

__global__ void kernel_cumsum_axis(const float* input, float* output,
                                   size_t rows, size_t cols, int axis) {
    const size_t segment = blockIdx.x * blockDim.x + threadIdx.x;
    if (axis == 1) {
        if (segment >= rows) return;
        double running = 0.0;
        const size_t base = segment * cols;
        for (size_t column = 0; column < cols; ++column) {
            running += input[base + column];
            output[base + column] = static_cast<float>(running);
        }
    } else {
        if (segment >= cols) return;
        double running = 0.0;
        for (size_t row = 0; row < rows; ++row) {
            const size_t index = row * cols + segment;
            running += input[index];
            output[index] = static_cast<float>(running);
        }
    }
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

__global__ void kernel_find_negative(const float* values, size_t n, int* found) {
    size_t i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n && values[i] < 0.0f) atomicExch(found, 1);
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

__global__ void kernel_reduce_axis_serial(const float* input, float* output, size_t outer, size_t axis_size, size_t inner, int mode) {
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

__device__ __forceinline__ bool reduction_candidate_wins(float candidate, size_t candidate_index,
                                                          float current, size_t current_index, int mode) {
    // Preserve the public CPU semantics: a NaN in position zero remains the
    // result, later NaNs are ignored, and ties retain the first index.
    if (candidate_index == SIZE_MAX) return false;
    if (current_index == SIZE_MAX) return true;
    if (isnan(current)) return current_index != 0 && !isnan(candidate);
    if (isnan(candidate)) return false;
    if (mode == REDUCE_MIN || mode == REDUCE_ARGMIN) {
        return candidate < current || (candidate == current && candidate_index < current_index);
    }
    return candidate > current || (candidate == current && candidate_index < current_index);
}

__global__ void kernel_reduce_axis_hierarchical(const float* input, float* output, size_t outer,
                                                 size_t axis_size, size_t inner, int mode) {
    const size_t out_index = blockIdx.x;
    const size_t out_size = outer * inner;
    if (out_index >= out_size) return;

    extern __shared__ unsigned char shared_raw[];
    float* shared_values = reinterpret_cast<float*>(shared_raw);
    size_t* shared_indices = reinterpret_cast<size_t*>(shared_values + blockDim.x);
    const size_t outer_index = out_index / inner;
    const size_t inner_index = out_index % inner;
    const size_t base = outer_index * axis_size * inner + inner_index;

    float local_value = (mode == REDUCE_SUM || mode == REDUCE_MEAN) ? 0.0f : 0.0f;
    size_t local_index = SIZE_MAX;
    for (size_t axis_index = threadIdx.x; axis_index < axis_size; axis_index += blockDim.x) {
        const float candidate = input[base + axis_index * inner];
        if (mode == REDUCE_SUM || mode == REDUCE_MEAN) {
            local_value += candidate;
        } else if (reduction_candidate_wins(candidate, axis_index, local_value, local_index, mode)) {
            local_value = candidate;
            local_index = axis_index;
        }
    }
    shared_values[threadIdx.x] = local_value;
    shared_indices[threadIdx.x] = local_index;
    __syncthreads();

    for (unsigned int offset = blockDim.x / 2; offset > 0; offset >>= 1) {
        if (threadIdx.x < offset) {
            if (mode == REDUCE_SUM || mode == REDUCE_MEAN) {
                shared_values[threadIdx.x] += shared_values[threadIdx.x + offset];
            } else if (reduction_candidate_wins(shared_values[threadIdx.x + offset], shared_indices[threadIdx.x + offset],
                                                shared_values[threadIdx.x], shared_indices[threadIdx.x], mode)) {
                shared_values[threadIdx.x] = shared_values[threadIdx.x + offset];
                shared_indices[threadIdx.x] = shared_indices[threadIdx.x + offset];
            }
        }
        __syncthreads();
    }

    if (threadIdx.x == 0) {
        float value = shared_values[0];
        if (mode == REDUCE_MEAN) value /= static_cast<float>(axis_size);
        output[out_index] = (mode == REDUCE_ARGMIN || mode == REDUCE_ARGMAX)
            ? static_cast<float>(shared_indices[0])
            : value;
    }
}

__global__ void kernel_softmax_rows(float* values, size_t rows, size_t cols, int one_dimensional) {
    const size_t row_index = blockIdx.x;
    if (row_index >= rows) return;
    const size_t base = row_index * cols;
    __shared__ float shared_values[256];
    __shared__ size_t shared_indices[256];
    __shared__ float shared_maximum;

    float local_max = 0.0f;
    size_t local_index = SIZE_MAX;
    for (size_t column = threadIdx.x; column < cols; column += blockDim.x) {
        const float candidate = values[base + column];
        if (reduction_candidate_wins(candidate, column, local_max, local_index, REDUCE_MAX)) {
            local_max = candidate;
            local_index = column;
        }
    }
    shared_values[threadIdx.x] = local_max;
    shared_indices[threadIdx.x] = local_index;
    __syncthreads();

    for (unsigned int offset = blockDim.x / 2; offset > 0; offset >>= 1) {
        if (threadIdx.x < offset &&
            reduction_candidate_wins(shared_values[threadIdx.x + offset], shared_indices[threadIdx.x + offset],
                                     shared_values[threadIdx.x], shared_indices[threadIdx.x], REDUCE_MAX)) {
            shared_values[threadIdx.x] = shared_values[threadIdx.x + offset];
            shared_indices[threadIdx.x] = shared_indices[threadIdx.x + offset];
        }
        __syncthreads();
    }
    if (threadIdx.x == 0) shared_maximum = shared_values[0];
    __syncthreads();
    const float maximum = shared_maximum;

    float local_sum = 0.0f;
    for (size_t column = threadIdx.x; column < cols; column += blockDim.x) {
        const float exponential = expf(values[base + column] - maximum);
        values[base + column] = exponential;
        local_sum += exponential;
    }
    shared_values[threadIdx.x] = local_sum;
    __syncthreads();
    for (unsigned int offset = blockDim.x / 2; offset > 0; offset >>= 1) {
        if (threadIdx.x < offset) shared_values[threadIdx.x] += shared_values[threadIdx.x + offset];
        __syncthreads();
    }

    const float sum = shared_values[0];
    const bool use_uniform = !one_dimensional && (sum == 0.0f || !isfinite(sum));
    const float factor = use_uniform ? (1.0f / static_cast<float>(cols)) : (1.0f / sum);
    if (use_uniform || sum != 0.0f) {
        for (size_t column = threadIdx.x; column < cols; column += blockDim.x) {
            values[base + column] = use_uniform ? factor : values[base + column] * factor;
        }
    }
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
    const size_t block_count = n == 0 ? 0 : ((n - 1) / static_cast<size_t>(threads)) + 1;
    if (block_count > static_cast<size_t>(std::numeric_limits<int>::max())) {
        throw std::overflow_error("CUDA 1D grid exceeds supported int range");
    }
    blocks = static_cast<int>(block_count);
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

    ScopedCudaBuffer<int> found_buffer(1, "divide validation allocation");
    CUDA_CHECK_CONTEXT(cudaMemset(found_buffer.get(), 0, sizeof(int)), "divide validation reset");
    kernel_find_zero<<<blocks, threads>>>(d_b, n, found_buffer.get());
    CUDA_KERNEL_CHECK("divide validation");
    int found = 0;
    CUDA_CHECK_CONTEXT(cudaMemcpy(&found, found_buffer.get(), sizeof(int), cudaMemcpyDeviceToHost), "divide validation result");
    found_buffer.release_checked("divide validation free");
    if (found) throw std::runtime_error("Divisao por zero detectada");
    kernel_div<<<blocks, threads>>>(d_a, d_b, n);
    CUDA_KERNEL_CHECK("divide");
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

    ScopedCudaBuffer<int> found_buffer(1, "log validation allocation");
    CUDA_CHECK_CONTEXT(cudaMemset(found_buffer.get(), 0, sizeof(int)), "log validation reset");
    kernel_find_nonpositive<<<blocks, threads>>>(d_a, n, found_buffer.get());
    CUDA_KERNEL_CHECK("log validation");
    int found = 0;
    CUDA_CHECK_CONTEXT(cudaMemcpy(&found, found_buffer.get(), sizeof(int), cudaMemcpyDeviceToHost), "log validation result");
    found_buffer.release_checked("log validation free");
    if (found) throw std::runtime_error("Logaritmo de valor nao positivo.");
    kernel_log<<<blocks, threads>>>(d_a, n);
    CUDA_KERNEL_CHECK("log");
}

extern "C" void gpu_sqrt_device(float* d_a, size_t n) {
    int threads = 0, blocks = 0;
    if (n == 0) return;
    launch_1d(n, blocks, threads);

    ScopedCudaBuffer<int> found_buffer(1, "sqrt validation allocation");
    CUDA_CHECK_CONTEXT(cudaMemset(found_buffer.get(), 0, sizeof(int)), "sqrt validation reset");
    kernel_find_negative<<<blocks, threads>>>(d_a, n, found_buffer.get());
    CUDA_KERNEL_CHECK("sqrt validation");
    int found = 0;
    CUDA_CHECK_CONTEXT(cudaMemcpy(&found, found_buffer.get(), sizeof(int), cudaMemcpyDeviceToHost), "sqrt validation result");
    found_buffer.release_checked("sqrt validation free");
    if (found) throw std::runtime_error("Raiz quadrada de valor negativo.");
    kernel_sqrt<<<blocks, threads>>>(d_a, n);
    CUDA_KERNEL_CHECK("sqrt");
}

extern "C" void gpu_clip_device(float* d_a, float min_value, float max_value, size_t n) {
    if (std::isnan(min_value) || std::isnan(max_value) || min_value > max_value) {
        throw std::invalid_argument("clip min must be <= max and neither bound may be NaN");
    }
    if (n == 0) return;
    int threads = 0, blocks = 0;
    launch_1d(n, blocks, threads);
    kernel_clip<<<blocks, threads>>>(d_a, min_value, max_value, n);
    CUDA_KERNEL_CHECK("clip");
}

extern "C" void gpu_softmax_device(float* d_a, size_t rows, size_t cols, int one_dimensional) {
    if (!d_a) throw std::invalid_argument("softmax: null device pointer");
    if (rows == 0 || cols == 0) return;
    if (rows > static_cast<size_t>(std::numeric_limits<unsigned int>::max())) {
        throw std::overflow_error("softmax: row grid exceeds CUDA limit");
    }
    constexpr int threads = 256;
    kernel_softmax_rows<<<static_cast<unsigned int>(rows), threads>>>(d_a, rows, cols, one_dimensional);
    CUDA_KERNEL_CHECK("softmax");
}

extern "C" void gpu_softmax_derivative_device(float* d_a, size_t n) {
    if (n == 0) return;
    int threads = 0, blocks = 0;
    launch_1d(n, blocks, threads);
    kernel_softmax_derivative<<<blocks, threads>>>(d_a, n);
    CUDA_KERNEL_CHECK("softmax derivative");
}

extern "C" void gpu_greater_device(const float* d_a, const float* d_b, float* d_output,
                                    size_t n, size_t broadcast_width, float scalar,
                                    int use_scalar) {
    if (n == 0) return;
    if (!d_a || !d_output || (!use_scalar && !d_b)) {
        throw std::invalid_argument("greater: null device pointer");
    }
    int threads = 0, blocks = 0;
    launch_1d(n, blocks, threads);
    kernel_greater<<<blocks, threads>>>(d_a, d_b, d_output, n, broadcast_width, scalar, use_scalar);
    CUDA_KERNEL_CHECK("greater");
}

extern "C" void gpu_broadcast_device(const float* d_input, float* d_output,
                                      const size_t* output_shape, const size_t* output_strides,
                                      const size_t* input_shape, const size_t* input_strides,
                                      size_t output_rank, size_t input_rank, size_t output_size) {
    if (output_size == 0) return;
    if (!d_input || !d_output || !output_shape || !output_strides || !input_shape || !input_strides) {
        throw std::invalid_argument("broadcast: null pointer");
    }
    if (input_rank == 0 || input_rank > output_rank) {
        throw std::invalid_argument("broadcast: invalid rank");
    }
    const size_t metadata_count = 2 * output_rank + 2 * input_rank;
    ScopedCudaBuffer<size_t> metadata(metadata_count, "broadcast metadata allocation");
    size_t* d_output_shape = metadata.get();
    size_t* d_output_strides = d_output_shape + output_rank;
    size_t* d_input_shape = d_output_strides + output_rank;
    size_t* d_input_strides = d_input_shape + input_rank;
    CUDA_CHECK_CONTEXT(cudaMemcpy(d_output_shape, output_shape, output_rank * sizeof(size_t), cudaMemcpyHostToDevice), "broadcast output shape upload");
    CUDA_CHECK_CONTEXT(cudaMemcpy(d_output_strides, output_strides, output_rank * sizeof(size_t), cudaMemcpyHostToDevice), "broadcast output strides upload");
    CUDA_CHECK_CONTEXT(cudaMemcpy(d_input_shape, input_shape, input_rank * sizeof(size_t), cudaMemcpyHostToDevice), "broadcast input shape upload");
    CUDA_CHECK_CONTEXT(cudaMemcpy(d_input_strides, input_strides, input_rank * sizeof(size_t), cudaMemcpyHostToDevice), "broadcast input strides upload");
    int threads = 0, blocks = 0;
    launch_1d(output_size, blocks, threads);
    kernel_broadcast_materialize<<<blocks, threads>>>(d_input, d_output, d_output_shape,
        d_output_strides, d_input_shape, d_input_strides, output_rank, input_rank, output_size);
    CUDA_KERNEL_CHECK("broadcast materialization");
    metadata.release_checked("broadcast metadata free");
}

extern "C" void gpu_tile_device(const float* d_input, float* d_output,
                                 size_t input_size, size_t output_size) {
    if (output_size == 0) return;
    if (!d_input || !d_output || input_size == 0) throw std::invalid_argument("tile: invalid device buffer");
    int threads = 0, blocks = 0;
    launch_1d(output_size, blocks, threads);
    kernel_tile<<<blocks, threads>>>(d_input, d_output, input_size, output_size);
    CUDA_KERNEL_CHECK("tile");
}

extern "C" void gpu_cumsum_device(const float* d_input, float* d_output,
                                   size_t rows, size_t cols, int axis,
                                   int one_dimensional) {
    if (rows == 0 || cols == 0) return;
    if (!d_input || !d_output) throw std::invalid_argument("cumsum: null device pointer");
    if (one_dimensional) {
        const size_t n = rows * cols;
        void* temporary = nullptr;
        size_t temporary_bytes = 0;
        CUDA_CHECK_CONTEXT(cub::DeviceScan::InclusiveSum(temporary, temporary_bytes, d_input, d_output, n), "cumsum CUB size query");
        std::lock_guard<std::mutex> cache_lock(reduction_cache_mutex);
        init_reduction_cache();
        temporary = ensure_cub_temporary_unlocked(temporary_bytes, "cumsum CUB cache");
        CUDA_CHECK_CONTEXT(cub::DeviceScan::InclusiveSum(temporary, temporary_bytes, d_input, d_output, n), "cumsum CUB scan");
        CUDA_KERNEL_CHECK("cumsum CUB execution");
        return;
    }
    if (axis != 0 && axis != 1) throw std::out_of_range("cumsum: invalid CUDA axis");
    const size_t segments = axis == 1 ? rows : cols;
    int threads = 0, blocks = 0;
    launch_1d(segments, blocks, threads);
    kernel_cumsum_axis<<<blocks, threads>>>(d_input, d_output, rows, cols, axis);
    CUDA_KERNEL_CHECK("cumsum axis");
}

extern "C" float gpu_dot_value_device(const float* d_a, const float* d_b, size_t n) {
    if (n == 0) return 0.0f;
    if (!d_a || !d_b) throw std::invalid_argument("dot: null device pointer");
    if (n > static_cast<size_t>(std::numeric_limits<int>::max())) {
        throw std::overflow_error("dot length exceeds cuBLAS int range");
    }
    std::lock_guard<std::mutex> lock(sgemm_handle_mutex);
    if (!sgemm_handle) {
        CUBLAS_CHECK_CONTEXT(cublasCreate(&sgemm_handle), "create cuBLAS handle for dot");
        std::atexit(release_sgemm_handle);
    }
    CUBLAS_CHECK_CONTEXT(cublasSetPointerMode(sgemm_handle, CUBLAS_POINTER_MODE_HOST), "dot host result mode");
    float result = 0.0f;
    CUBLAS_CHECK_CONTEXT(cublasSdot(sgemm_handle, static_cast<int>(n), d_a, 1, d_b, 1, &result), "vector dot");
    CUDA_CHECK_CONTEXT(cudaDeviceSynchronize(), "vector dot execution");
    return result;
}

extern "C" void gpu_matvec_device(const float* d_matrix, const float* d_vector,
                                   float* d_output, size_t rows, size_t cols) {
    if (rows == 0 || cols == 0) return;
    if (!d_matrix || !d_vector || !d_output) throw std::invalid_argument("matvec: null device pointer");
    if (rows > static_cast<size_t>(std::numeric_limits<int>::max()) ||
        cols > static_cast<size_t>(std::numeric_limits<int>::max())) {
        throw std::overflow_error("matvec dimensions exceed cuBLAS int range");
    }
    std::lock_guard<std::mutex> lock(sgemm_handle_mutex);
    if (!sgemm_handle) {
        CUBLAS_CHECK_CONTEXT(cublasCreate(&sgemm_handle), "create cuBLAS handle for matvec");
        std::atexit(release_sgemm_handle);
    }
    CUBLAS_CHECK_CONTEXT(cublasSetPointerMode(sgemm_handle, CUBLAS_POINTER_MODE_HOST), "matvec host scalar mode");
    const float alpha = 1.0f;
    const float beta = 0.0f;
    // A_row[rows,cols] is seen as A_col[cols,rows]. Transposing that
    // column-major view computes y[rows] = A_row * x[cols] without copying A.
    CUBLAS_CHECK_CONTEXT(cublasSgemv(sgemm_handle, CUBLAS_OP_T,
        static_cast<int>(cols), static_cast<int>(rows), &alpha,
        d_matrix, static_cast<int>(cols), d_vector, 1, &beta, d_output, 1),
        "SGEMV row-major A*x");
    CUDA_CHECK_CONTEXT(cudaDeviceSynchronize(), "SGEMV execution");
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
    const char* requested = std::getenv("ZMATRIX_REDUCTION_IMPL");
    const bool serial = requested && std::string(requested) == "serial";
    if (serial) {
        int threads = 0, blocks = 0;
        launch_1d(output_size, blocks, threads);
        kernel_reduce_axis_serial<<<blocks, threads>>>(d_input, d_output, outer, axis_size, inner, mode);
    } else {
        if (output_size > static_cast<size_t>(std::numeric_limits<unsigned int>::max())) {
            throw std::overflow_error(std::string(context) + ": output grid exceeds CUDA limit");
        }
        constexpr int threads = 256;
        const size_t shared_bytes = threads * (sizeof(float) + sizeof(size_t));
        kernel_reduce_axis_hierarchical<<<static_cast<unsigned int>(output_size), threads, shared_bytes>>>(
            d_input, d_output, outer, axis_size, inner, mode);
    }
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
    std::lock_guard<std::mutex> cache_lock(reduction_cache_mutex);
    init_reduction_cache();
    if (!reduction_output) {
        CUDA_CHECK_CONTEXT(cudaMalloc(reinterpret_cast<void**>(&reduction_output), sizeof(float)), context);
    }
    float* output = reduction_output;
    {
        const char* requested = std::getenv("ZMATRIX_REDUCTION_IMPL");
        // Default hybrid: CUB for global sum, the custom hierarchical kernel
        // for operations whose NaN/tie contract CUB does not preserve.
        const bool use_cub = !requested || std::string(requested) == "cub";
        // CUB Sum is numerically compatible within the documented tolerance.
        // DeviceReduce::Min/Max use identity values for NaN inputs and would
        // violate ZTensor's "NaN at index zero wins" CPU contract.
        if (use_cub && mode == REDUCE_SUM) {
            void* temporary = nullptr;
            size_t temporary_bytes = 0;
            CUDA_CHECK_CONTEXT(cub::DeviceReduce::Sum(temporary, temporary_bytes, input, output, n), context);
            temporary = ensure_cub_temporary_unlocked(temporary_bytes, context);
            CUDA_CHECK_CONTEXT(cub::DeviceReduce::Sum(temporary, temporary_bytes, input, output, n), context);
            CUDA_KERNEL_CHECK(context);
        } else {
            launch_reduce_axis(input, output, 1, n, 1, mode, context);
        }
        float result = 0.0f;
        CUDA_CHECK_CONTEXT(cudaMemcpy(&result, output, sizeof(float), cudaMemcpyDeviceToHost), context);
        return result;
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
    // CUB ArgMin/ArgMax are intentionally not used: their NaN handling does
    // not preserve the established CPU semantics or first-index tie rule.
    std::lock_guard<std::mutex> cache_lock(reduction_cache_mutex);
    init_reduction_cache();
    if (!reduction_output) {
        CUDA_CHECK_CONTEXT(cudaMalloc(reinterpret_cast<void**>(&reduction_output), sizeof(float)), "arg reduction allocation");
    }
    launch_reduce_axis(input, reduction_output, 1, n, 1, find_max ? REDUCE_ARGMAX : REDUCE_ARGMIN,
        find_max ? "argmax value" : "argmin value");
    float result = 0.0f;
    CUDA_CHECK_CONTEXT(cudaMemcpy(&result, reduction_output, sizeof(float), cudaMemcpyDeviceToHost), "arg reduction result");
    return static_cast<size_t>(result);
}
// ========== GPU MATMUL FORWARD DECLARATIONS ==========
extern "C" void gpu_matmul_device(const float* d_a, const float* d_b, float* d_c, size_t m, size_t k, size_t n);

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
