// gpu_kernels.cu
#include <cuda_runtime.h>
#include <stddef.h>
#include <math.h>
#include <cstdio>
#include <cstdlib>
#include <dlfcn.h>

// ========== WSL CUDA DRIVER FALLBACK ==========
// Função para encontrar libcuda.so com fallback para caminhos especiais (WSL)
// Isso resolve o problema onde WSL coloca libcuda.so em /usr/lib/wsl/lib/
static void* load_cuda_driver() {
    // Lista de caminhos a tentar em ordem de prioridade
    const char* cuda_lib_paths[] = {
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
            const char *dbg = std::getenv("ZMATRIX_GPU_DEBUG");
            if (dbg && dbg[0] == '1') {
                std::fprintf(stderr, "[zmatrix][gpu] Successfully loaded CUDA driver from: %s\n", cuda_lib_paths[i]);
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
    load_cuda_driver();
}

static float *cache_a = nullptr;
static size_t cap_a = 0;
static float *cache_b = nullptr;
static size_t cap_b = 0;
static bool cache_inited = false;

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
        cudaFree(buf);
        buf = nullptr;
        cap = 0;
    }
    cudaError_t err = cudaMalloc(reinterpret_cast<void**>(&buf), n * sizeof(float));
    if (err != cudaSuccess) {
        buf = nullptr;
        cap = 0;
        return false;
    }
    cap = n;
    return true;
}

#define CUDA_CHECK(stmt) do { \
    cudaError_t err = (stmt); \
    if (err != cudaSuccess) { \
        goto cleanup; \
    } \
} while (0)

extern "C" int gpu_available() {
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

static inline void launch_1d(size_t n, int &blocks, int &threads) {
    threads = 256;
    blocks = static_cast<int>((n + threads - 1) / threads);
}

extern "C" void gpu_add(float* a, const float* b, size_t n) {
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

cleanup:
    return;
}

extern "C" void gpu_sub(float* a, const float* b, size_t n) {
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

cleanup:
    return;
}

extern "C" void gpu_mul(float* a, const float* b, size_t n) {
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

cleanup:
    return;
}

extern "C" void gpu_relu(float* a, size_t n) {
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

cleanup:
    return;
}

extern "C" void gpu_leaky_relu(float* a, float alpha, size_t n) {
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

cleanup:
    return;
}

extern "C" void gpu_leaky_relu_derivative(float* a, float alpha, size_t n) {
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

cleanup:
    return;
}

extern "C" void gpu_sigmoid(float* a, size_t n) {
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

cleanup:
    return;
}

extern "C" void gpu_tanh(float* a, size_t n) {
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

cleanup:
    return;
}

extern "C" void gpu_exp(float* a, size_t n) {
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

cleanup:
    return;
}

extern "C" void gpu_abs(float* a, size_t n) {
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

cleanup:
    return;
}

extern "C" void gpu_scalar_add(float* a, float value, size_t n) {
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

cleanup:
    return;
}

extern "C" void gpu_scalar_sub(float* a, float value, size_t n) {
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

cleanup:
    return;
}

extern "C" void gpu_scalar_mul(float* a, float value, size_t n) {
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

cleanup:
    return;
}

extern "C" void gpu_scalar_div(float* a, float value, size_t n) {
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

cleanup:
    return;
}

extern "C" void gpu_add_device(float* d_a, const float* d_b, size_t n) {
    int threads = 0, blocks = 0;
    if (n == 0) return;
    launch_1d(n, blocks, threads);
    kernel_add<<<blocks, threads>>>(d_a, d_b, n);
    cudaDeviceSynchronize();
}

extern "C" void gpu_sub_device(float* d_a, const float* d_b, size_t n) {
    int threads = 0, blocks = 0;
    if (n == 0) return;
    launch_1d(n, blocks, threads);
    kernel_sub<<<blocks, threads>>>(d_a, d_b, n);
    cudaDeviceSynchronize();
}

extern "C" void gpu_mul_device(float* d_a, const float* d_b, size_t n) {
    int threads = 0, blocks = 0;
    if (n == 0) return;
    launch_1d(n, blocks, threads);
    kernel_mul<<<blocks, threads>>>(d_a, d_b, n);
    cudaDeviceSynchronize();
}

extern "C" void gpu_relu_device(float* d_a, size_t n) {
    int threads = 0, blocks = 0;
    if (n == 0) return;
    launch_1d(n, blocks, threads);
    kernel_relu<<<blocks, threads>>>(d_a, n);
    cudaDeviceSynchronize();
}

extern "C" void gpu_leaky_relu_device(float* d_a, float alpha, size_t n) {
    int threads = 0, blocks = 0;
    if (n == 0) return;
    launch_1d(n, blocks, threads);
    kernel_leaky_relu<<<blocks, threads>>>(d_a, alpha, n);
    cudaDeviceSynchronize();
}

extern "C" void gpu_leaky_relu_derivative_device(float* d_a, float alpha, size_t n) {
    int threads = 0, blocks = 0;
    if (n == 0) return;
    launch_1d(n, blocks, threads);
    kernel_leaky_relu_derivative<<<blocks, threads>>>(d_a, alpha, n);
    cudaDeviceSynchronize();
}

extern "C" void gpu_sigmoid_device(float* d_a, size_t n) {
    int threads = 0, blocks = 0;
    if (n == 0) return;
    launch_1d(n, blocks, threads);
    kernel_sigmoid<<<blocks, threads>>>(d_a, n);
    cudaDeviceSynchronize();
}

extern "C" void gpu_tanh_device(float* d_a, size_t n) {
    int threads = 0, blocks = 0;
    if (n == 0) return;
    launch_1d(n, blocks, threads);
    kernel_tanh<<<blocks, threads>>>(d_a, n);
    cudaDeviceSynchronize();
}

extern "C" void gpu_exp_device(float* d_a, size_t n) {
    int threads = 0, blocks = 0;
    if (n == 0) return;
    launch_1d(n, blocks, threads);
    kernel_exp<<<blocks, threads>>>(d_a, n);
    cudaDeviceSynchronize();
}

extern "C" void gpu_abs_device(float* d_a, size_t n) {
    int threads = 0, blocks = 0;
    if (n == 0) return;
    launch_1d(n, blocks, threads);
    kernel_abs<<<blocks, threads>>>(d_a, n);
    cudaDeviceSynchronize();
}

extern "C" void gpu_scalar_add_device(float* d_a, float value, size_t n) {
    int threads = 0, blocks = 0;
    if (n == 0) return;
    launch_1d(n, blocks, threads);
    kernel_scalar_add<<<blocks, threads>>>(d_a, value, n);
    cudaDeviceSynchronize();
}

extern "C" void gpu_scalar_sub_device(float* d_a, float value, size_t n) {
    int threads = 0, blocks = 0;
    if (n == 0) return;
    launch_1d(n, blocks, threads);
    kernel_scalar_sub<<<blocks, threads>>>(d_a, value, n);
    cudaDeviceSynchronize();
}

extern "C" void gpu_scalar_mul_device(float* d_a, float value, size_t n) {
    int threads = 0, blocks = 0;
    if (n == 0) return;
    launch_1d(n, blocks, threads);
    kernel_scalar_mul<<<blocks, threads>>>(d_a, value, n);
    cudaDeviceSynchronize();
}

extern "C" void gpu_scalar_div_device(float* d_a, float value, size_t n) {
    int threads = 0, blocks = 0;
    if (n == 0) return;
    launch_1d(n, blocks, threads);
    kernel_scalar_div<<<blocks, threads>>>(d_a, value, n);
    cudaDeviceSynchronize();
}
