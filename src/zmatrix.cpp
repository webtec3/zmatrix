#include <memory>
#include <functional>
#include <atomic>
#include <set>  // Para reverse-mode autograd
#include <mutex>  // Para thread-safety em accumulate_grad
#ifdef HAVE_CONFIG_H
#include "config.h"
#endif

#include "php.h"
#include "Zend/zend_exceptions.h"
#include "ext/standard/info.h"
#include <vector>
#include <cmath>        // Para std::fabs, std::expf, std::sqrtf, etc. (inclui math.h)
#include <algorithm>
#include <random>
#include <numeric>
#include <chrono>
#include <cblas.h>      // Para cblas_sgemm agora
#include <thread>
#include <future>
#include <stddef.h>
#include <stdexcept>
#include <limits>
#include <cstdint>
#include <functional>
#include <string>
#include <cstring>
#include <cstdlib>
#include <cstdio>
#include <sstream>
#include <unordered_set>
#ifdef HAVE_CUDA
#include <cuda_runtime.h>
#endif

// #include <any>        // Removido se não usado

// --- OpenMP / SIMD Headers ---
#ifdef _OPENMP
#include <omp.h>
#define HAS_OPENMP 1
#else
#define HAS_OPENMP 0
#endif
#include <immintrin.h> // Para AVX, AVX2, AVX512 intrinsics
#ifdef __AVX2__
#define HAS_AVX2 1
#else
#define HAS_AVX2 0
#endif
#ifdef __AVX512F__
#define HAS_AVX512 1
#else
#define HAS_AVX512 0
#endif
// --- Fim OpenMP / SIMD ---

#include "simd/simd_dispatch.h"
#include "zmatrix_arginfo.h"

#ifndef ZMATRIX_ERRORS_H
#define ZMATRIX_ERRORS_H
#define ZMATRIX_ERR_INVALID_VALUE "Invalid value for the operation"

#endif // ZMATRIX_ERRORS_H

// Error messages (kept)
#define ZMATRIX_ERR_EMPTY_MATRIX "Empty matrix/tensor cannot be processed"

#define ZMATRIX_ERR_INVALID_TYPE "Elements must be numeric (integers or float/double)"
#define ZMATRIX_ERR_SHAPE_MISMATCH "Shape mismatch for the operation"
#define ZMATRIX_ERR_ALLOC_FAILED "Failed to allocate memory for the tensor"
#define ZMATRIX_ERR_NOT_INITIALIZED "ZTensor object not initialized"
#define ZMATRIX_ERR_INCOMPATIBLE_DIMS "Incompatible dimensions for the operation"
#define ZMATRIX_ERR_INVALID_ARG_TYPE "Argument must be an array or ZTensor"
#define ZMATRIX_ERR_INVALID_SHAPE "Invalid shape (dimensions must be positive)"
#define ZMATRIX_ERR_OVERFLOW "Internal computation exceeded the maximum limit (overflow)"
#define ZMATRIX_ERR_UNSUPPORTED_OP "Operation not supported for this tensor type/dimension"

#define ZMATRIX_PARALLEL_THRESHOLD 10000000  // Disable OpenMP - its slower!

static bool zmatrix_cuda_profile_enabled() {
    static const bool enabled = [] {
        const char* value = std::getenv("ZMATRIX_CUDA_PROFILE");
        return value && *value && std::string(value) != "0";
    }();
    return enabled;
}

static double zmatrix_elapsed_ms(std::chrono::steady_clock::time_point start) {
    return std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - start).count();
}

[[maybe_unused]] static void zmatrix_profile_result(const char* operation, double host_output_ms,
                                   double device_allocation_ms, double wrapper_ms,
                                   double operand_transfer_ms = 0.0, double state_update_ms = 0.0) {
    if (!zmatrix_cuda_profile_enabled()) return;
    std::fprintf(stderr,
        "[zmatrix][cuda-wrapper] {\"operation\":\"%s\",\"result_creation_ms\":%.6f,"
        "\"device_allocation_ms\":%.6f,\"operand_transfer_ms\":%.6f,\"cuda_submit_ms\":%.6f,"
        "\"state_update_ms\":%.6f}\n",
        operation, host_output_ms, device_allocation_ms, operand_transfer_ms, wrapper_ms, state_update_ms);
}

#ifdef ZMATRIX_ENABLE_DEBUG_INVARIANTS
#define ZMATRIX_DEBUG_ASSERT(condition, message) do { \
    if (!(condition)) throw std::logic_error(std::string("ZTensor invariant failed: ") + (message)); \
} while (0)
#else
#define ZMATRIX_DEBUG_ASSERT(condition, message) do { (void)sizeof(condition); } while (0)
#endif

#ifdef HAVE_CUDA
#include "gpu_wrapper.h"

static inline bool zmatrix_force_cpu() {
    const char *v = std::getenv("ZMATRIX_FORCE_CPU");
    return (v != nullptr) && (v[0] == '1');
}

static inline bool zmatrix_gpu_debug_enabled() {
    const char *v = std::getenv("ZMATRIX_GPU_DEBUG");
    return (v != nullptr) && (v[0] == '1');
}

static bool zmatrix_async_allocator_requested() {
    static const bool requested = [] {
        const char* value = std::getenv("ZMATRIX_CUDA_ALLOCATOR");
        if (!value || !*value || std::string(value) == "auto") return true;
        return std::string(value) == "async";
    }();
    return requested;
}

static std::atomic<size_t> zmatrix_device_allocations{0};
static std::atomic<size_t> zmatrix_device_frees{0};
static std::atomic<size_t> zmatrix_tensor_destructors{0};

static void zmatrix_profile_lifecycle(const char* event, size_t bytes, const char* mode,
                                      double host_ms, int status = 0) noexcept {
    if (!zmatrix_cuda_profile_enabled()) return;
    std::fprintf(stderr,
        "[zmatrix][cuda-lifecycle] {\"event\":\"%s\",\"bytes\":%zu,\"mode\":\"%s\","
        "\"host_ms\":%.6f,\"status\":%d,\"allocations\":%zu,\"frees\":%zu,\"destructors\":%zu}\n",
        event, bytes, mode, host_ms, status, zmatrix_device_allocations.load(),
        zmatrix_device_frees.load(), zmatrix_tensor_destructors.load());
}

static cudaError_t zmatrix_device_allocate(float** pointer, size_t bytes, bool& used_async) {
    const auto start = std::chrono::steady_clock::now();
    int async_flag = 0;
    const int status = gpu_device_allocate(reinterpret_cast<void**>(pointer), bytes,
        zmatrix_async_allocator_requested() ? 1 : 0, &async_flag);
    used_async = async_flag != 0;
    if (status == static_cast<int>(cudaSuccess)) ++zmatrix_device_allocations;
    zmatrix_profile_lifecycle("allocate", bytes, used_async ? "async" : "legacy",
        zmatrix_elapsed_ms(start), status);
    return static_cast<cudaError_t>(status);
}

static cudaError_t zmatrix_device_free(float* pointer, bool allocation_was_async,
                                       size_t bytes) noexcept {
    const auto start = std::chrono::steady_clock::now();
    const int status = gpu_device_free(pointer, allocation_was_async ? 1 : 0);
    if (status == static_cast<int>(cudaSuccess)) ++zmatrix_device_frees;
    zmatrix_profile_lifecycle("free", bytes, allocation_was_async ? "async" : "legacy",
        zmatrix_elapsed_ms(start), status);
    return static_cast<cudaError_t>(status);
}

static inline void zmatrix_gpu_debug(const char *op, size_t n) {
    if (zmatrix_gpu_debug_enabled()) {
        std::fprintf(stderr, "[zmatrix][gpu] %s n=%zu\n", op, n);
    }
}

#endif

// Gerador aleatório (mantido)
static inline uint64_t xorshift64star(uint64_t& state) {
    state ^= state >> 12; state ^= state << 25; state ^= state >> 27;
    return state * UINT64_C(0x2545F4914F6CDD1D);
}

static std::mt19937& get_global_mt19937() {
    static std::mt19937 gen(std::chrono::high_resolution_clock::now().time_since_epoch().count());
    return gen;
}
// Forward declaration
struct ZTensor;

// --- Autograd Node Structure (reverse-mode, eager-mode) ---
struct AutogradNode {
    // Parents (operandos que geraram este nó)
    std::vector<std::shared_ptr<ZTensor>> parents;

    // Função backward para esta operação
    // Não recebe argumentos - ela captura tudo por closure
    std::function<void()> backward_fn;

    // Identificador da operação (para debug)
    std::string op_name;

    // Mutex para thread-safety na acumulação de gradientes dos pais
    mutable std::mutex backward_lock;

    // Armazena o tensor resultado (fraco para evitar ciclos)
    std::weak_ptr<ZTensor> result_tensor;

    // Ponteiro RAW ao tensor resultado para acesso rápido (CUIDADO: lifetime management)
    ZTensor* result_ptr_raw = nullptr;

    // Ponteiros RAW aos pais para acesso rápido (CUIDADO: lifetime management)
    std::vector<ZTensor*> parents_raw;

    AutogradNode() = default;

    AutogradNode(const std::string& name) : op_name(name) {}
};

// --- Definição COMPLETA de ZTensor PRIMEIRO ---
struct ZTensor {
    // ========== AUTOGRAD STATE ==========

    // Flag: este tensor requer gradiente
    bool requires_grad = false;

    // Gradiente acumulado (inicializa sob demanda)
    std::unique_ptr<ZTensor> grad;

    // Nó do grafo computacional (nullptr para tensores folha)
    std::shared_ptr<AutogradNode> grad_fn = nullptr;

    // Mutex para thread-safety na acumulação de gradientes
    mutable std::mutex grad_mutex;

    // ========== MÉTODOS DE AUTOGRAD ==========

    // Ativa rastreamento de gradientes
    ZTensor& requiresGrad(bool req = true) {
        requires_grad = req;
        if (req && !grad_fn) {
            // Tensor folha: não tem grad_fn
        }
        return *this;
    }

    // Retorna true se este tensor requer gradiente
    bool isRequiresGrad() const {
        return requires_grad;
    }

    // Obtém ou cria o tensor de gradiente
    ZTensor& ensureGrad() {
        if (!grad) {
            grad = std::make_unique<ZTensor>(shape);
            // Inicializa com zeros
            std::fill(grad->data.begin(), grad->data.end(), 0.0f);
        }
        return *grad;
    }

    // Zera gradientes (apenas este tensor)
    void zeroGrad() {
        if (grad) {
#ifdef HAVE_CUDA
            grad->ensure_host();
#endif
            std::fill(grad->data.begin(), grad->data.end(), 0.0f);
#ifdef HAVE_CUDA
            grad->mark_host_modified();
#endif
        }
    }

    // Obtém o gradiente (nullptr se não existir)
    const ZTensor* getGrad() const {
        return grad.get();
    }

    // Acumula gradiente (+=) com thread-safety
    void accumulate_grad(const ZTensor& grad_in) {
        if (grad_in.shape != shape) {
            throw std::invalid_argument("Gradient shape mismatch in accumulate_grad");
        }

#ifdef HAVE_CUDA
        // FIX: garante que o gradiente de entrada esteja sincronizado com o host
        // antes de somá-lo. Antes, esta função lia grad_in.data.data() direto,
        // o que corrompia o resultado se grad_in estivesse residente na GPU.
        grad_in.ensure_host();
#endif

        std::lock_guard<std::mutex> lock(grad_mutex);

        ZTensor& g = ensureGrad();
#ifdef HAVE_CUDA
        g.ensure_host();
#endif
        const size_t N = size();
        if (N == 0) return;

        float* g_data = g.data.data();
        const float* gin_data = grad_in.data.data();

#if HAS_OPENMP
        if (N > ZMATRIX_PARALLEL_THRESHOLD) {
#pragma omp parallel for simd schedule(static)
            for (size_t i = 0; i < N; ++i) {
                g_data[i] += gin_data[i];
            }
        } else {
            for (size_t i = 0; i < N; ++i) {
                g_data[i] += gin_data[i];
            }
        }
#else
        for (size_t i = 0; i < N; ++i) {
            g_data[i] += gin_data[i];
        }
#endif

#ifdef HAVE_CUDA
        g.mark_host_modified();
#endif
    }

    // Backward pass (reverse-mode autodiff)
    // Deve ser chamado apenas em tensores escalares
    void backward() {
        // 1) Deve ser escalar
        if (shape != std::vector<size_t>{1}) {
            throw std::invalid_argument(
                "backward() can only be called on scalar tensors (shape={1})"
            );
        }

        // 2) Gradiente inicial
        ensureGrad();
#ifdef HAVE_CUDA
        grad->ensure_host();
#endif
        grad->data[0] = 1.0f;
#ifdef HAVE_CUDA
        grad->mark_host_modified();
#endif

        if (!grad_fn) return;

        // 3) Ordenação topológica pós-ordem (DFS sem recursão profunda)
        std::vector<std::shared_ptr<AutogradNode>> topo;
        topo.reserve(128);

        std::unordered_set<AutogradNode*> visited;
        visited.reserve(128);
        std::unordered_set<AutogradNode*> expanded;
        expanded.reserve(128);

        std::vector<std::shared_ptr<AutogradNode>> stack;
        stack.push_back(grad_fn);

        while (!stack.empty()) {
            auto node = stack.back();
            if (!node) {
                stack.pop_back();
                continue;
            }

            if (visited.count(node.get())) {
                stack.pop_back();
                continue;
            }

            if (expanded.count(node.get())) {
                visited.insert(node.get());
                topo.push_back(node);
                stack.pop_back();
            } else {
                expanded.insert(node.get());
                // Insere os nós pais na pilha para que sejam processados antes
                for (auto* parent : node->parents_raw) {
                    if (parent && parent->grad_fn) {
                        if (visited.count(parent->grad_fn.get()) == 0) {
                            stack.push_back(parent->grad_fn);
                        }
                    }
                }
            }
        }

        // 4) Backward na ordem reversa da ordenação topológica (pós-ordem reversa)
        for (auto it = topo.rbegin(); it != topo.rend(); ++it) {
            auto& node = *it;
            if (node->backward_fn) {
                node->backward_fn(); // exceção sobe
            }
        }
    }

    std::vector<float> data; // <--- MUDANÇA: double para float
    std::vector<size_t> shape;
    std::vector<size_t> strides;
    size_t offset = 0;
    bool owns_data = true; // ← importante para views
#ifdef HAVE_CUDA
    mutable float *d_data = nullptr;
    mutable size_t d_capacity = 0;
    mutable bool d_async_allocation = false;
    mutable bool device_valid = false;
    mutable bool host_valid = true;
    // True only while an internal result is being fully overwritten on the
    // device. It permits the short construction interval in which neither
    // representation is valid, but must be cleared before the tensor escapes.
    mutable bool device_write_pending = false;
    mutable std::mutex device_mutex; // FIX: protege d_data/device_valid/host_valid contra corrida de dados
#endif


    // Internal device results may defer the host vector until the first D2H.
    // CPU-visible construction keeps the original zero-initialized behavior.
    ZTensor(const std::vector<size_t>& _shape) : ZTensor(_shape, false) {}

    ZTensor(const std::vector<size_t>& _shape, bool defer_host_allocation) : shape(_shape) {
            const size_t total_size = checked_element_count(shape);

            if (total_size > 0) {
                if (!defer_host_allocation) {
                    try {
                        data.resize(total_size, 0.0f);
                    } catch (const std::bad_alloc&) {
                        throw std::runtime_error(ZMATRIX_ERR_ALLOC_FAILED);
                    }
                }
                strides.resize(shape.size());
                size_t stride = 1;
                for (int i = static_cast<int>(shape.size()) - 1; i >= 0; --i) {
                    strides[i] = stride;
                     if (shape[i] > 0 && i > 0 && stride > (std::numeric_limits<size_t>::max() / shape[i])) {
                         throw std::overflow_error(ZMATRIX_ERR_OVERFLOW);
                     }
                     if (shape[i] > 0) {
                        stride *= shape[i];
                     }
                }
            } else {
                data.clear();
                strides.clear();
            }
#ifdef HAVE_CUDA
            host_valid = !defer_host_allocation || total_size == 0;
            device_write_pending = false;
#else
            if (defer_host_allocation && total_size > 0) {
                data.resize(total_size, 0.0f);
            }
#endif
        }

#ifdef HAVE_CUDA
    static ZTensor create_device_result(const std::vector<size_t>& result_shape, bool profile,
                                        double& construction_ms, double& allocation_ms) {
        const auto construction_start = profile ? std::chrono::steady_clock::now()
            : std::chrono::steady_clock::time_point{};
        ZTensor result(result_shape, true);
        construction_ms = profile ? zmatrix_elapsed_ms(construction_start) : 0.0;
        const auto allocation_start = profile ? std::chrono::steady_clock::now()
            : std::chrono::steady_clock::time_point{};
        result.allocate_device_for_write();
        allocation_ms = profile ? zmatrix_elapsed_ms(allocation_start) : 0.0;
        return result;
    }
#endif

    // Construtor Padrão (inalterado)
    ZTensor() = default;

    ZTensor(const ZTensor& other)
        : requires_grad(other.requires_grad),
          grad_fn(other.grad_fn),
          data(other.data),
          shape(other.shape),
          strides(other.strides),
          offset(other.offset),
          owns_data(other.owns_data)
    {
#ifdef HAVE_CUDA
        d_data = nullptr;
        d_capacity = 0;
        d_async_allocation = false;
        device_valid = false;
        host_valid = other.host_valid;
        device_write_pending = false;

        if (other.device_valid) {
            const size_t n = other.size();
            if (n == 0) {
                device_valid = true;
            } else if (other.d_data) {
                try {
                    cuda_check(zmatrix_device_allocate(&d_data, n * sizeof(float), d_async_allocation), "device allocation");
                    cuda_check(cudaMemcpy(d_data, other.d_data, n * sizeof(float), cudaMemcpyDeviceToDevice), "cudaMemcpy D2D");
                    d_capacity = n;
                    device_valid = true;
                } catch (...) {
                    // FIX: evita vazamento de VRAM se cudaMemcpy D2D falhar após cudaMalloc ter sucesso.
                    if (d_data) { zmatrix_device_free(d_data, d_async_allocation, n * sizeof(float)); d_data = nullptr; }
                    d_capacity = 0;
                    d_async_allocation = false;
                    device_valid = false;
                    throw;
                }
            }
        }
#endif
        // Copia gradiente se existir
        if (other.grad) {
            grad = std::make_unique<ZTensor>(*other.grad);
        }
    }

    ZTensor& operator=(const ZTensor& other) {
        if (this == &other) return *this;
#ifdef HAVE_CUDA
        release_device_noexcept();
        // Copia gradiente se existir
        if (other.grad) {
            grad = std::make_unique<ZTensor>(*other.grad);
        }
#endif
        data = other.data;
        shape = other.shape;
        strides = other.strides;
        offset = other.offset;
        owns_data = other.owns_data;
        requires_grad = other.requires_grad;
        grad_fn = other.grad_fn;
#ifdef HAVE_CUDA
        d_data = nullptr;
        d_capacity = 0;
        d_async_allocation = false;
        device_valid = false;
        host_valid = other.host_valid;
        device_write_pending = false;

        if (other.device_valid) {
            const size_t n = other.size();
            if (n == 0) {
                device_valid = true;
            } else if (other.d_data) {
                try {
                    cuda_check(zmatrix_device_allocate(&d_data, n * sizeof(float), d_async_allocation), "device allocation");
                    cuda_check(cudaMemcpy(d_data, other.d_data, n * sizeof(float), cudaMemcpyDeviceToDevice), "cudaMemcpy D2D");
                    d_capacity = n;
                    device_valid = true;
                } catch (...) {
                    // FIX: evita vazamento de VRAM se cudaMemcpy D2D falhar após cudaMalloc ter sucesso.
                    if (d_data) { zmatrix_device_free(d_data, d_async_allocation, n * sizeof(float)); d_data = nullptr; }
                    d_capacity = 0;
                    d_async_allocation = false;
                    device_valid = false;
                    throw;
                }
            }
        }
#endif
        return *this;
    }

    ZTensor(ZTensor&& other) noexcept
        : requires_grad(other.requires_grad),
          grad_fn(std::move(other.grad_fn)),
          data(std::move(other.data)),
          shape(std::move(other.shape)),
          strides(std::move(other.strides)),
          offset(other.offset),
          owns_data(other.owns_data)
    {
        grad = std::move(other.grad);
#ifdef HAVE_CUDA
        d_data = other.d_data;
        d_capacity = other.d_capacity;
        d_async_allocation = other.d_async_allocation;
        device_valid = other.device_valid;
        host_valid = other.host_valid;
        device_write_pending = other.device_write_pending;
        other.d_data = nullptr;
        other.d_capacity = 0;
        other.d_async_allocation = false;
        other.device_valid = false;
        other.host_valid = true;
        other.device_write_pending = false;
#endif
    }

    ZTensor& operator=(ZTensor&& other) noexcept {
        if (this == &other) return *this;
#ifdef HAVE_CUDA
        release_device_noexcept();
#endif
        data = std::move(other.data);
        shape = std::move(other.shape);
        strides = std::move(other.strides);
        offset = other.offset;
        owns_data = other.owns_data;
        // FIX: a move assignment não movia o estado de autograd (requires_grad,
        // grad_fn, grad), causando perda silenciosa do grafo computacional e
        // do gradiente acumulado sempre que "a = std::move(b)" era usado.
        requires_grad = other.requires_grad;
        grad_fn = std::move(other.grad_fn);
        grad = std::move(other.grad);
        other.requires_grad = false;
#ifdef HAVE_CUDA
        d_data = other.d_data;
        d_capacity = other.d_capacity;
        d_async_allocation = other.d_async_allocation;
        device_valid = other.device_valid;
        host_valid = other.host_valid;
        device_write_pending = other.device_write_pending;
        other.d_data = nullptr;
        other.d_capacity = 0;
        other.d_async_allocation = false;
        other.device_valid = false;
        other.host_valid = true;
        other.device_write_pending = false;
#endif
        return *this;
    }

    ~ZTensor() {
#ifdef HAVE_CUDA
        const auto destruction_start = std::chrono::steady_clock::now();
        const size_t released_bytes = d_capacity * sizeof(float);
        const char* allocation_mode = d_async_allocation ? "async" : "legacy";
        release_device_noexcept();
        ++zmatrix_tensor_destructors;
        zmatrix_profile_lifecycle("destructor", released_bytes, allocation_mode,
            zmatrix_elapsed_ms(destruction_start));
#endif
    }

#ifdef HAVE_CUDA
    static inline void cuda_check(cudaError_t err, const char *what) {
        if (err != cudaSuccess) {
            throw std::runtime_error(std::string(what) + ": " + cudaGetErrorString(err));
        }
    }

    void assert_invariants_unlocked() const {
        const size_t n = checked_element_count(shape);
        ZMATRIX_DEBUG_ASSERT(!host_valid || n == 0 || data.size() >= n, "valid host buffer is too small");
        ZMATRIX_DEBUG_ASSERT(!device_valid || n == 0 || (d_data != nullptr && d_capacity >= n), "valid device buffer is missing or too small");
        ZMATRIX_DEBUG_ASSERT(!device_write_pending || (n > 0 && !host_valid && !device_valid && d_data != nullptr && d_capacity >= n),
            "pending device write requires an allocated exclusive device buffer");
        ZMATRIX_DEBUG_ASSERT(n == 0 || host_valid || device_valid || device_write_pending,
            "non-empty tensor has no valid representation or pending device write");
    }

    void mark_synchronized_unlocked() const {
        host_valid = true;
        device_valid = true;
        device_write_pending = false;
        assert_invariants_unlocked();
    }

    void mark_host_modified_unlocked() const {
        host_valid = true;
        device_valid = false;
        device_write_pending = false;
        assert_invariants_unlocked();
    }

    void mark_device_modified_unlocked() const {
        device_valid = true;
        host_valid = false;
        device_write_pending = false;
        assert_invariants_unlocked();
    }

    void invalidate_device_unlocked() const {
        device_valid = false;
        assert_invariants_unlocked();
    }

    void ensure_device() const {
            gpu_require_available();
            std::lock_guard<std::mutex> lock(device_mutex); // FIX: thread-safety
            if (device_valid) return;
            size_t n = size();
            if (n == 0) {
                mark_synchronized_unlocked();
                return;
            }
            if (!host_valid) {
                throw std::runtime_error("Host data is not valid for device upload");
            }
            try {
                if (!d_data || d_capacity < n) {
                    if (d_data) {
                        cudaError_t err = zmatrix_device_free(d_data, d_async_allocation, d_capacity * sizeof(float));
                        d_data = nullptr; // Clear pointer regardless of free result
                        d_async_allocation = false;
                        if (err != cudaSuccess) {
                            throw std::runtime_error(std::string("cudaFree failed: ") + cudaGetErrorString(err));
                        }
                    }
                    cuda_check(zmatrix_device_allocate(&d_data, n * sizeof(float), d_async_allocation), "device allocation");
                    d_capacity = n;
                    if (zmatrix_gpu_debug_enabled()) std::fprintf(stderr, "[zmatrix][gpu] device allocation elements=%zu\n", n);
                }
                // Static zeros()/ones() must remain host tensors until the
                // explicit toGpu() call. At that point generate uniform 0/1
                // contents directly on the device instead of uploading them.
                bool initialized_on_device = false;
                const float first = data[0];
                if (first == 0.0f || first == 1.0f) {
                    initialized_on_device = std::all_of(data.begin() + 1, data.end(),
                        [first](float value) { return value == first; });
                    if (initialized_on_device) gpu_fill_device(d_data, first, n);
                }
                if (!initialized_on_device) {
                    cuda_check(cudaMemcpy(d_data, data.data(), n * sizeof(float), cudaMemcpyHostToDevice), "cudaMemcpy H2D");
                }
                mark_synchronized_unlocked();
                if (zmatrix_gpu_debug_enabled()) {
                    std::fprintf(stderr, initialized_on_device
                        ? "[zmatrix][gpu] device fill elements=%zu host_valid=1 device_valid=1\n"
                        : "[zmatrix][gpu] H2D elements=%zu host_valid=1 device_valid=1\n", n);
                }
            } catch (const std::exception&) {
                // FIX: libera a memória alocada antes de descartar o ponteiro,
                // evitando vazamento de VRAM quando cudaMemcpy falha após cudaMalloc ter sucesso.
                if (d_data) {
                    zmatrix_device_free(d_data, d_async_allocation, d_capacity * sizeof(float));
                    d_data = nullptr;
                }
                d_capacity = 0;
                d_async_allocation = false;
                invalidate_device_unlocked();
                throw;
            }
    }

    // Reserve device storage for an operation that overwrites every element.
    // Unlike ensure_device(), this deliberately performs no H2D transfer. The
    // device buffer remains invalid until the caller confirms successful work
    // with mark_device_modified(). This is required for SGEMM with beta == 0.
    void allocate_device_for_write() const {
        gpu_require_available();
        std::lock_guard<std::mutex> lock(device_mutex);
        const size_t n = size();
        if (n == 0) return;
        if (!d_data || d_capacity < n) {
            if (d_data) {
                cudaError_t free_status = zmatrix_device_free(d_data, d_async_allocation, d_capacity * sizeof(float));
                d_data = nullptr;
                d_capacity = 0;
                d_async_allocation = false;
                cuda_check(free_status, "cudaFree before device output allocation");
            }
            cuda_check(zmatrix_device_allocate(&d_data, n * sizeof(float), d_async_allocation), "device output allocation");
            d_capacity = n;
        }
        host_valid = false;
        device_valid = false;
        device_write_pending = true;
        assert_invariants_unlocked();
        if (zmatrix_gpu_debug_enabled()) {
            std::fprintf(stderr, "[zmatrix][gpu] allocate output n=%zu H2D=0 host_valid=%d device_valid=0\n", n, host_valid ? 1 : 0);
        }
    }

    void ensure_host() const {
        std::lock_guard<std::mutex> lock(device_mutex); // FIX: thread-safety
        if (host_valid) return;
        size_t n = size();
        if (n == 0) {
            mark_synchronized_unlocked();
            return;
        }
        if (!d_data) {
            throw std::runtime_error("Device data is not valid for host download");
        }
        if (data.size() < n) {
            try {
                const_cast<ZTensor*>(this)->data.resize(n);
            } catch (const std::bad_alloc&) {
                throw std::runtime_error(ZMATRIX_ERR_ALLOC_FAILED);
            }
        }
        cuda_check(cudaMemcpy(const_cast<float*>(data.data()), d_data, n * sizeof(float), cudaMemcpyDeviceToHost), "cudaMemcpy D2H");
        mark_synchronized_unlocked();
        if (zmatrix_gpu_debug_enabled()) std::fprintf(stderr, "[zmatrix][gpu] D2H elements=%zu host_valid=1 device_valid=1\n", n);
    }

    void mark_host_modified() {
        std::lock_guard<std::mutex> lock(device_mutex); // FIX: thread-safety
        const bool state_changed = !host_valid || device_valid;
        mark_host_modified_unlocked();
        if (state_changed && zmatrix_gpu_debug_enabled()) std::fprintf(stderr, "[zmatrix][gpu] state host_valid=1 device_valid=0\n");
    }

    void mark_device_modified() const {
        std::lock_guard<std::mutex> lock(device_mutex); // FIX: thread-safety
        const bool state_changed = host_valid || !device_valid;
        mark_device_modified_unlocked();
        if (state_changed && zmatrix_gpu_debug_enabled()) std::fprintf(stderr, "[zmatrix][gpu] state host_valid=0 device_valid=1\n");
    }

    void release_device_noexcept() noexcept {
        std::lock_guard<std::mutex> lock(device_mutex); // FIX: thread-safety
        if (d_data) {
            zmatrix_device_free(d_data, d_async_allocation, d_capacity * sizeof(float));
            d_data = nullptr;
            if (zmatrix_gpu_debug_enabled()) std::fprintf(stderr, "[zmatrix][gpu] device buffer released\n");
        }
        d_capacity = 0;
        d_async_allocation = false;
        device_valid = false;
        device_write_pending = false;
    }

    void free_device() {
        // Public freeDevice() must not discard a newer device-only value.
        // Assignment/destruction use release_device_noexcept() directly.
        ensure_host();
        release_device_noexcept();
    }

    void to_gpu() {
        ensure_device();
    }

    void to_cpu() {
        ensure_host();
        std::lock_guard<std::mutex> lock(device_mutex);
        invalidate_device_unlocked();  // Explicitly leave device residency.
    }

    bool is_on_gpu() const {
        return device_valid;
    }
#endif

    std::string to_string() const {
        std::ostringstream oss;
#ifdef HAVE_CUDA
        ensure_host();
#endif

        if (shape.size() == 2) {
            oss << "[";
            for (size_t i = 0; i < shape[0]; ++i) {
                oss << "[";
                for (size_t j = 0; j < shape[1]; ++j) {
                    oss << at({i, j});
                    if (j < shape[1] - 1) oss << ",";
                }
                oss << "]";
                if (i < shape[0] - 1) oss << ",";
            }
            oss << "]";
        } else {
            oss << "[";
            for (size_t i = 0; i < data.size(); ++i) {
                oss << data[i];
                if (i < data.size() - 1) oss << ",";
            }
            oss << "]";
        }

        return oss.str();
    }

    static size_t checked_element_count(const std::vector<size_t>& shape) {
        if (shape.empty()) return 0;
        size_t total = 1;
        for (size_t dim : shape) {
            if (dim == 0) return 0;
            if (total > std::numeric_limits<size_t>::max() / dim) {
                throw std::overflow_error(ZMATRIX_ERR_OVERFLOW);
            }
            total *= dim;
        }
        return total;
    }

    static size_t compute_total_size(const std::vector<size_t>& shape) {
        return checked_element_count(shape);
    }
    // --- Métodos Utilitários ---
    size_t get_linear_index(const std::vector<size_t>& indices) const {
        if (indices.size() != shape.size()) { throw std::invalid_argument("Number of indices does not match tensor dimensionality"); }
        size_t linear_idx = 0;
        for (size_t i = 0; i < indices.size(); ++i) {
            if (indices[i] >= shape[i]) { throw std::out_of_range("Index out of bounds"); }
            size_t term = indices[i] * strides[i];
            if (linear_idx > (std::numeric_limits<size_t>::max() - term)) { throw std::overflow_error(ZMATRIX_ERR_OVERFLOW); }
            linear_idx += term;
        }
        return linear_idx;
     }
    float& at(const std::vector<size_t>& indices) {
#ifdef HAVE_CUDA
        ensure_host();
        mark_host_modified();
#endif
        if (this->size() == 0) { throw std::out_of_range("Access to empty tensor"); }
        size_t index = get_linear_index(indices);
        if (index >= data.size()) {
            throw std::out_of_range("Calculated index exceeds data size");
        }
        return data[index];
    }
    const float& at(const std::vector<size_t>& indices) const {
#ifdef HAVE_CUDA
         ensure_host();
#endif
         if (this->size() == 0) { throw std::out_of_range("Access to empty tensor"); }
         size_t index = get_linear_index(indices);
         if (index >= data.size()) {
             throw std::out_of_range("Calculated index exceeds data size");
         }
        return data[index];
    }
    bool same_shape(const ZTensor& other) const { return shape == other.shape; }
    size_t size() const {
        return checked_element_count(shape);
     }
    bool empty() const { return this->size() == 0; }

    // --- Métodos de Operações (com float) ---

    // --- Adição (float) - Loop Canônico ---
    void add(const ZTensor& other) {

           // ✋ PROTEÇÃO CONTRA INPLACE EM TENSORES RASTREADOS
           if (this->requires_grad) {
               throw std::logic_error(
                   "In-place operation on tensor with requires_grad=true is not allowed. "
                   "Use add_autograd() for differentiable operations."
               );
           }

           if (!same_shape(other)) {
               throw std::invalid_argument(ZMATRIX_ERR_SHAPE_MISMATCH);
           }
           const size_t N = size();
           if (N == 0) return;

#ifdef HAVE_CUDA
           // FIX: antes só considerava device_valid de 'this'; se 'other' estivesse
           // na GPU e 'this' não, a operação inteira caía para CPU desnecessariamente,
           // forçando um download evitável de 'other'.
           if (device_valid || other.device_valid) {
               ensure_device();
               other.ensure_device();
               gpu_add_device(d_data, other.d_data, N);
               mark_device_modified();
               return;
           }
#endif

#ifdef HAVE_CUDA
           ensure_host();
           other.ensure_host();
#endif
           float * __restrict__ a = data.data();
           const float * __restrict__ b = other.data.data();


        #if HAS_OPENMP
           if (N > ZMATRIX_PARALLEL_THRESHOLD) {
#pragma omp parallel for simd schedule(static)
               for (size_t i = 0; i < N; ++i) {
                   a[i] += b[i];
               }
           } else {
               zmatrix_simd::add_f32(a, b, N);
           }
        #else
           zmatrix_simd::add_f32(a, b, N);
        #endif
#ifdef HAVE_CUDA
           mark_host_modified();
#endif
    }


    void clip(const ZTensor& other) {
        if (!same_shape(other)) throw std::invalid_argument(ZMATRIX_ERR_SHAPE_MISMATCH);

        const size_t N = size();
        if (N == 0) return;
#ifdef HAVE_CUDA
        ensure_host();
        other.ensure_host();
#endif

        float* __restrict__ a = data.data();
        const float* __restrict__ b = other.data.data();

        #if HAS_OPENMP
        if (N > ZMATRIX_PARALLEL_THRESHOLD) {
#pragma omp parallel for simd schedule(static)
            for (size_t i = 0; i < N; ++i) {
                a[i] = std::max(-b[i], std::min(b[i], a[i])); // Exemplo: clip simétrico ±b[i]
            }
        } else {
            for (size_t i = 0; i < N; ++i) {
                a[i] = std::max(-b[i], std::min(b[i], a[i]));
            }
        }
        #else
        for (size_t i = 0; i < N; ++i) {
            a[i] = std::max(-b[i], std::min(b[i], a[i]));
        }
        #endif
#ifdef HAVE_CUDA
        mark_host_modified();
#endif
    }

    void clip_values(float min_value, float max_value) {
        if (std::isnan(min_value) || std::isnan(max_value) || min_value > max_value) {
            throw std::invalid_argument("clip min must be <= max and neither bound may be NaN");
        }
        const size_t N = size();
        if (N == 0) return;
#ifdef HAVE_CUDA
        if (device_valid) {
            gpu_clip_device(d_data, min_value, max_value, N);
            mark_device_modified();
            return;
        }
        ensure_host();
#endif
        float* __restrict__ values = data.data();
#if HAS_OPENMP
        if (N > ZMATRIX_PARALLEL_THRESHOLD) {
#pragma omp parallel for schedule(static)
            for (size_t i = 0; i < N; ++i) {
                values[i] = std::max(min_value, std::min(max_value, values[i]));
            }
        } else
#endif
        {
            for (size_t i = 0; i < N; ++i) {
                values[i] = std::max(min_value, std::min(max_value, values[i]));
            }
        }
#ifdef HAVE_CUDA
        mark_host_modified();
#endif
    }

    ZTensor greater_scalar(float scalar) const {
        const size_t n = size();
        const bool profile = zmatrix_cuda_profile_enabled();
        if (n == 0) {
            ZTensor result(shape);
#ifdef HAVE_CUDA
            if (device_valid) result.ensure_device();
#endif
            return result;
        }
#ifdef HAVE_CUDA
        if (device_valid) {
            double construction_ms = 0.0, allocation_ms = 0.0;
            ZTensor result = create_device_result(shape, profile, construction_ms, allocation_ms);
            const auto wrapper_start = profile ? std::chrono::steady_clock::now() : std::chrono::steady_clock::time_point{};
            gpu_greater_device(d_data, nullptr, result.d_data, n, 0, scalar, 1);
            const double wrapper_ms = profile ? zmatrix_elapsed_ms(wrapper_start) : 0.0;
            const auto state_start = profile ? std::chrono::steady_clock::now() : std::chrono::steady_clock::time_point{};
            result.mark_device_modified();
            const double state_ms = profile ? zmatrix_elapsed_ms(state_start) : 0.0;
            zmatrix_profile_result("greater_scalar", construction_ms, allocation_ms, wrapper_ms, 0.0, state_ms);
            return result;
        }
        ensure_host();
#endif
        ZTensor result(shape);
        for (size_t i = 0; i < n; ++i) result.data[i] = data[i] > scalar ? 1.0f : 0.0f;
        return result;
    }

    ZTensor greater_tensor(const ZTensor& other, size_t broadcast_width = 0) const {
        const size_t n = size();
        if (broadcast_width == 0 && !same_shape(other)) throw std::invalid_argument(ZMATRIX_ERR_SHAPE_MISMATCH);
        if (broadcast_width != 0 && (other.shape != std::vector<size_t>{broadcast_width} ||
            (broadcast_width != 1 && (shape.size() != 2 || shape[1] != broadcast_width)))) {
            throw std::invalid_argument(ZMATRIX_ERR_SHAPE_MISMATCH);
        }
        const bool profile = zmatrix_cuda_profile_enabled();
        if (n == 0) {
            ZTensor result(shape);
#ifdef HAVE_CUDA
            if (device_valid || other.device_valid) result.ensure_device();
#endif
            return result;
        }
#ifdef HAVE_CUDA
        if (device_valid || other.device_valid) {
            const auto transfer_start = profile ? std::chrono::steady_clock::now() : std::chrono::steady_clock::time_point{};
            ensure_device();
            other.ensure_device();
            const double transfer_ms = profile ? zmatrix_elapsed_ms(transfer_start) : 0.0;
            double construction_ms = 0.0, allocation_ms = 0.0;
            ZTensor result = create_device_result(shape, profile, construction_ms, allocation_ms);
            const auto wrapper_start = profile ? std::chrono::steady_clock::now() : std::chrono::steady_clock::time_point{};
            gpu_greater_device(d_data, other.d_data, result.d_data, n, broadcast_width, 0.0f, 0);
            const double wrapper_ms = profile ? zmatrix_elapsed_ms(wrapper_start) : 0.0;
            const auto state_start = profile ? std::chrono::steady_clock::now() : std::chrono::steady_clock::time_point{};
            result.mark_device_modified();
            const double state_ms = profile ? zmatrix_elapsed_ms(state_start) : 0.0;
            zmatrix_profile_result("greater_tensor", construction_ms, allocation_ms, wrapper_ms, transfer_ms, state_ms);
            return result;
        }
        ensure_host();
        other.ensure_host();
#endif
        ZTensor result(shape);
        for (size_t i = 0; i < n; ++i) {
            const size_t rhs = broadcast_width ? i % broadcast_width : i;
            result.data[i] = data[i] > other.data[rhs] ? 1.0f : 0.0f;
        }
        return result;
    }

    ZTensor broadcast_materialized(const ZTensor& source) const {
        const size_t output_rank = shape.size();
        const size_t input_rank = source.shape.size();
        if (input_rank == 0 || input_rank > output_rank) throw std::invalid_argument("Incompatible ranks for broadcast");
        for (size_t i = 0; i < input_rank; ++i) {
            const size_t output_dimension = shape[output_rank - input_rank + i];
            const size_t input_dimension = source.shape[i];
            if (input_dimension != 1 && input_dimension != output_dimension) {
                throw std::invalid_argument("Incompatible for broadcast: dimension " +
                    std::to_string(input_dimension) + " x " + std::to_string(output_dimension));
            }
        }
#ifdef HAVE_CUDA
        ZTensor result(shape, device_valid || source.device_valid);
#else
        ZTensor result(shape);
#endif
        const size_t n = result.size();
        if (n == 0) {
#ifdef HAVE_CUDA
            if (device_valid || source.device_valid) result.ensure_device();
#endif
            return result;
        }
        if (source.size() == 0) throw std::invalid_argument("Cannot broadcast an empty source to a non-empty tensor");
#ifdef HAVE_CUDA
        if (device_valid || source.device_valid) {
            source.ensure_device();
            result.allocate_device_for_write();
            gpu_broadcast_device(source.d_data, result.d_data, shape.data(), strides.data(),
                source.shape.data(), source.strides.data(), output_rank, input_rank, n);
            result.mark_device_modified();
            return result;
        }
        source.ensure_host();
#endif
        for (size_t linear = 0; linear < n; ++linear) {
            size_t input_offset = 0;
            const size_t rank_difference = output_rank - input_rank;
            for (size_t input_axis = 0; input_axis < input_rank; ++input_axis) {
                const size_t output_axis = rank_difference + input_axis;
                const size_t coordinate = (linear / strides[output_axis]) % shape[output_axis];
                if (source.shape[input_axis] != 1) input_offset += coordinate * source.strides[input_axis];
            }
            result.data[linear] = source.data[input_offset];
        }
        return result;
    }

    ZTensor tiled(size_t times) const {
        if (times == 0) throw std::invalid_argument("tile(): parameter times must be >= 1");
        if (shape.empty()) throw std::invalid_argument("tile(): scalar tensor cannot be repeated");
        if (shape[0] > std::numeric_limits<size_t>::max() / times) throw std::overflow_error(ZMATRIX_ERR_OVERFLOW);
        std::vector<size_t> output_shape = shape;
        output_shape[0] *= times;
        const bool profile = zmatrix_cuda_profile_enabled();
        const size_t input_size = size();
        const size_t output_size = checked_element_count(output_shape);
        if (output_size == 0) {
            ZTensor result(output_shape);
#ifdef HAVE_CUDA
            if (device_valid) result.ensure_device();
#endif
            return result;
        }
#ifdef HAVE_CUDA
        if (device_valid) {
            double construction_ms = 0.0, allocation_ms = 0.0;
            ZTensor result = create_device_result(output_shape, profile, construction_ms, allocation_ms);
            const auto wrapper_start = profile ? std::chrono::steady_clock::now() : std::chrono::steady_clock::time_point{};
            gpu_tile_device(d_data, result.d_data, input_size, output_size);
            const double wrapper_ms = profile ? zmatrix_elapsed_ms(wrapper_start) : 0.0;
            const auto state_start = profile ? std::chrono::steady_clock::now() : std::chrono::steady_clock::time_point{};
            result.mark_device_modified();
            const double state_ms = profile ? zmatrix_elapsed_ms(state_start) : 0.0;
            zmatrix_profile_result("tile", construction_ms, allocation_ms, wrapper_ms, 0.0, state_ms);
            return result;
        }
        ensure_host();
#endif
        ZTensor result(output_shape);
        for (size_t repeat = 0; repeat < times; ++repeat) {
            std::copy(data.begin(), data.end(), result.data.begin() + repeat * input_size);
        }
        return result;
    }


    void soma(ZTensor& out, int axis) const {
        if (shape.empty()) throw std::runtime_error(ZMATRIX_ERR_EMPTY_MATRIX);
        if (axis < 0 || static_cast<size_t>(axis) >= shape.size()) throw std::out_of_range("Invalid axis for sum(axis)");

        std::vector<size_t> expected_shape = shape;
        expected_shape.erase(expected_shape.begin() + axis);

        if (out.shape != expected_shape) throw std::invalid_argument(ZMATRIX_ERR_SHAPE_MISMATCH);

        const size_t axis_dim = shape[axis];
        const size_t out_size = out.size();
#ifdef HAVE_CUDA
        if (device_valid) {
            size_t outer = 1;
            size_t inner = 1;
            for (int i = 0; i < axis; ++i) outer *= shape[static_cast<size_t>(i)];
            for (size_t i = static_cast<size_t>(axis) + 1; i < shape.size(); ++i) inner *= shape[i];
            out.allocate_device_for_write();
            gpu_sum_axis_device(d_data, out.d_data, outer, axis_dim, inner);
            out.mark_device_modified();
            return;
        }
        ensure_host();
        out.ensure_host();
#endif

        float* __restrict__ out_data = out.data.data();

        #if HAS_OPENMP
        if (out_size > ZMATRIX_PARALLEL_THRESHOLD) {
            #pragma omp parallel for schedule(static)
            for (size_t i = 0; i < out_size; ++i) {
                std::vector<size_t> idx(shape.size(), 0);
                size_t tmp = i;
                for (int d = (int)shape.size() - 1, j = (int)expected_shape.size() - 1; d >= 0; --d) {
                    if ((size_t)d == (size_t)axis) {
                        idx[d] = 0;
                    } else {
                        idx[d] = tmp % expected_shape[j];
                        tmp /= expected_shape[j];
                        --j;
                    }
                }
                float acc = 0.0f;
                for (size_t k = 0; k < axis_dim; ++k) {
                    idx[axis] = k;
                    acc += at(idx);
                }
                out_data[i] = acc;
            }
        } else
        #endif
        {
            for (size_t i = 0; i < out_size; ++i) {
                std::vector<size_t> idx(shape.size(), 0);
                size_t tmp = i;
                for (int d = (int)shape.size() - 1, j = (int)expected_shape.size() - 1; d >= 0; --d) {
                    if ((size_t)d == (size_t)axis) {
                        idx[d] = 0;
                    } else {
                        idx[d] = tmp % expected_shape[j];
                        tmp /= expected_shape[j];
                        --j;
                    }
                }
                float acc = 0.0f;
                for (size_t k = 0; k < axis_dim; ++k) {
                    idx[axis] = k;
                    acc += at(idx);
                }
                out_data[i] = acc;
            }
        }
#ifdef HAVE_CUDA
        out.mark_host_modified();
#endif
    }

    // --- Subtração (float) - Loop Canônico ---
    void subtract(const ZTensor& other) {
        if (!same_shape(other)) throw std::invalid_argument(ZMATRIX_ERR_SHAPE_MISMATCH);
        const size_t N = size();
        if (N == 0) return;
#ifdef HAVE_CUDA
        // FIX: considerar também other.device_valid (ver add()).
        if (device_valid || other.device_valid) {
            ensure_device();
            other.ensure_device();
            gpu_sub_device(d_data, other.d_data, N);
            mark_device_modified();
            return;
        }
        ensure_host();
        other.ensure_host();
#endif
        float * __restrict__ a = data.data();
        const float * __restrict__ b = other.data.data();


        #if HAS_OPENMP
        if (N > ZMATRIX_PARALLEL_THRESHOLD) {
#pragma omp parallel for simd schedule(static)
            for (size_t i = 0; i < N; ++i) {
                a[i] -= b[i];
            }
        } else {
            zmatrix_simd::sub_f32(a, b, N);
        }
        #else
         zmatrix_simd::sub_f32(a, b, N);
        #endif
#ifdef HAVE_CUDA
        mark_host_modified();
#endif

    }



    // --- Multiplicação Elemento a Elemento (float) - Loop Canônico ---
    void mul(const ZTensor& other) {
        // ✋ PROTEÇÃO CONTRA INPLACE EM TENSORES RASTREADOS
        if (this->requires_grad) {
            throw std::logic_error(
                "In-place operation on tensor with requires_grad=true is not allowed. "
                "Use mul_autograd() for differentiable operations."
            );
        }

        if (!same_shape(other)) throw std::invalid_argument(ZMATRIX_ERR_SHAPE_MISMATCH);
        const size_t N = size();
        if (N == 0) return;
#ifdef HAVE_CUDA
        // FIX: considerar também other.device_valid (ver add()).
        if (device_valid || other.device_valid) {
            ensure_device();
            other.ensure_device();
            gpu_mul_device(d_data, other.d_data, N);
            mark_device_modified();
            return;
        }
        ensure_host();
        other.ensure_host();
#endif
        float * __restrict__ a = data.data();
        const float * __restrict__ b = other.data.data();


        #if HAS_OPENMP
        if (N > ZMATRIX_PARALLEL_THRESHOLD) {
#pragma omp parallel for simd schedule(static)
             for (size_t i = 0; i < N; ++i) {
                a[i] *= b[i];
            }
        } else {
             zmatrix_simd::mul_f32(a, b, N);
        }
        #else
         zmatrix_simd::mul_f32(a, b, N);
        #endif
#ifdef HAVE_CUDA
        mark_host_modified();
#endif
    }



    void scalar_divide(float scalar) {
       const size_t N = size();
           if (N == 0) return;
#ifdef HAVE_CUDA
           if (device_valid) {
               ensure_device();
               gpu_scalar_div_device(d_data, scalar, N);
               mark_device_modified();
               return;
           }
           ensure_host();
#endif
           float * __restrict__ a = data.data();
           #if HAS_OPENMP
           if (N > ZMATRIX_PARALLEL_THRESHOLD) {
#pragma omp parallel for simd schedule(static)
               for (size_t i = 0; i < N; ++i) {
                   a[i] /= scalar;
               }
           } else {
               zmatrix_simd::scalar_div_f32(a, scalar, N);
           }
           #else
           zmatrix_simd::scalar_div_f32(a, scalar, N);
           #endif
#ifdef HAVE_CUDA
           mark_host_modified();
#endif
    }



    // --- Multiplicação por Escalar (float) - Loop Canônico ---
    void multiply_scalar(float scalar) {
        const size_t N = size();
        if (N == 0) return;
#ifdef HAVE_CUDA
        if (device_valid) {
            ensure_device();
            gpu_scalar_mul_device(d_data, scalar, N);
            mark_device_modified();
            return;
        }
        ensure_host();
#endif
        float * __restrict__ a = data.data();
        #if defined(HAVE_CUDA)
        #endif
        #if HAS_OPENMP
        if (N > ZMATRIX_PARALLEL_THRESHOLD) {
#pragma omp parallel for simd schedule(static)
            for (size_t i = 0; i < N; ++i) {
                a[i] *= scalar;
            }
        } else {
            zmatrix_simd::scalar_mul_f32(a, scalar, N);
        }
        #else
        zmatrix_simd::scalar_mul_f32(a, scalar, N);
        #endif
#ifdef HAVE_CUDA
        mark_host_modified();
#endif
    }



    void scalar_add(float value) {
        size_t N = data.size();
#ifdef HAVE_CUDA
        if (device_valid) {
            ensure_device();
            gpu_scalar_add_device(d_data, value, N);
            mark_device_modified();
            return;
        }
        ensure_host();
#endif
        float *ptr = data.data();
        #if defined(HAVE_CUDA)
        #endif

        #if HAS_OPENMP
        if (N > ZMATRIX_PARALLEL_THRESHOLD) {
#pragma omp parallel for simd schedule(static)
            for (size_t i = 0; i < N; ++i) {
                ptr[i] += value;
            }
        } else
        #endif
        {
            zmatrix_simd::scalar_add_f32(ptr, value, N);
        }
#ifdef HAVE_CUDA
        mark_host_modified();
#endif
    }



    void scalar_subtract(float value) {
        size_t N = data.size();
#ifdef HAVE_CUDA
        if (device_valid) {
            ensure_device();
            gpu_scalar_sub_device(d_data, value, N);
            mark_device_modified();
            return;
        }
        ensure_host();
#endif
        float *ptr = data.data();

        #if HAS_OPENMP
        if (N > ZMATRIX_PARALLEL_THRESHOLD) {
#pragma omp parallel for simd schedule(static)
            for (size_t i = 0; i < N; ++i) {
                ptr[i] -= value;
            }
        } else
        #endif
        {
            zmatrix_simd::scalar_sub_f32(ptr, value, N);
        }
#ifdef HAVE_CUDA
        mark_host_modified();
#endif
    }

    void fill(float value) {
            const size_t N = size();
            if (N == 0) return;

    #ifdef HAVE_CUDA
            if (device_valid) {
                gpu_fill_device(d_data, value, N);
                mark_device_modified();
                return;
            }
            ensure_host();
    #endif

            float* __restrict__ p = data.data();

    #if HAS_OPENMP
            if (N > ZMATRIX_PARALLEL_THRESHOLD) {
#pragma omp parallel for simd schedule(static)
                for (size_t i = 0; i < N; ++i) {
                    p[i] = value;
                }
            } else {
                for (size_t i = 0; i < N; ++i) {
                    p[i] = value;
                }
            }
    #else
            for (size_t i = 0; i < N; ++i) {
                p[i] = value;
            }
    #endif

#ifdef HAVE_CUDA
        mark_host_modified();
#endif
    }

    ZTensor reshape(const std::vector<size_t>& new_shape) const {
        // 1. Calcular tamanho total do novo shape
        size_t new_total = 1;
        bool has_zero = false;
        for (size_t dim : new_shape) {
            if (dim == 0) {
                has_zero = true;
                break;
            }
            if (dim > 0 && new_total > (std::numeric_limits<size_t>::max() / dim)) {
                throw std::overflow_error(ZMATRIX_ERR_OVERFLOW);
            }
            new_total *= dim;
        }

        // 2. Verificar compatibilidade do número de elementos
        size_t old_total = this->size();
        if (!has_zero && new_total != old_total) {
            throw std::invalid_argument(ZMATRIX_ERR_SHAPE_MISMATCH);
        }

        // 3. Criar objeto de resultado e reutilizar dados
        ZTensor result;
        result.shape = new_shape;
        if (has_zero) {
            // Shape com dimensão zero: tensor vazio
            result.data.clear();
            result.strides.clear();
            return result;
        }
        #ifdef HAVE_CUDA
        ensure_host();
        #endif
        // NOTA: std::vector::operator= faz cópia profunda (não é uma view
        // compartilhada). O comentário anterior alegando compartilhamento
        // de buffer estava incorreto — esta é uma cópia real dos dados.
        result.data = this->data;

        // 4. Calcular strides para o novo shape
        result.strides.resize(new_shape.size());
        size_t stride = 1;
        for (int i = (int)new_shape.size() - 1; i >= 0; --i) {
            result.strides[i] = stride;
            stride *= new_shape[i];
        }

        return result;
    }

    // --- Matmul (float - sgemm) ---
     ZTensor matmul(const ZTensor& other) const {
         // 1. Verificações
         if (shape.size() != 2 || other.shape.size() != 2) {
             throw std::runtime_error("Matmul (BLAS) is only implemented for 2D tensors");
         }
         if (shape[1] != other.shape[0]) {
             throw std::runtime_error(std::string(ZMATRIX_ERR_INCOMPATIBLE_DIMS) + " (Matmul 2D)");
         }

         const size_t M = shape[0];         // linhas de A
         const size_t K = shape[1];         // colunas de A = linhas de B
         const size_t N = other.shape[1];   // colunas de B

         ZTensor result({M, N});
         if (M == 0 || N == 0 || K == 0) return result;  // Caso degenerado

#ifdef HAVE_CUDA
         const size_t output_elements = M * N;
         // GPU dispatch is explicit: at least one input must have been moved
         // with toGpu(). The other input is uploaded only to satisfy this
         // explicitly selected operation.
         const bool use_gpu = device_valid || other.device_valid;
         if (use_gpu) {
             zmatrix_gpu_debug("matmul", output_elements);
             ensure_device();
             other.ensure_device();
             result.allocate_device_for_write();
             gpu_matmul_device(d_data, other.d_data, result.d_data, M, K, N);
             result.mark_device_modified();
             return result;
         }

         ensure_host();
         other.ensure_host();
#endif
         const float* A_ptr = this->data.data();
         const float* B_ptr = other.data.data();
               float* C_ptr = result.data.data();

         const float alpha = 1.0f;
         const float beta  = 0.0f;

         const CBLAS_INDEX lda = static_cast<CBLAS_INDEX>(K); // leading dimension de A
         const CBLAS_INDEX ldb = static_cast<CBLAS_INDEX>(N); // leading dimension de B
         const CBLAS_INDEX ldc = static_cast<CBLAS_INDEX>(N); // leading dimension de C

         // 2. Chamada otimizada BLAS (float)
         cblas_sgemm(
             CblasRowMajor, CblasNoTrans, CblasNoTrans,
             static_cast<CBLAS_INDEX>(M),
             static_cast<CBLAS_INDEX>(N),
             static_cast<CBLAS_INDEX>(K),
             alpha,
             A_ptr, lda,
             B_ptr, ldb,
             beta,
             C_ptr, ldc
         );

         return result;
     }

    // --- Slice (cópia densa correta) ---
    /**
     * Extrai uma fatia (subarray) do tensor ao longo de um eixo.
     *
     * NOTA DE CORREÇÃO: a versão anterior desta função alegava ser uma
     * "view" zero-copy compartilhando o buffer via `offset`, mas `offset`
     * nunca era de fato usado em get_linear_index()/at() — então o
     * resultado ficava INCORRETO sempre que start > 0 (retornava os
     * primeiros elementos do array completo, não a fatia real a partir
     * de `start`). Esta versão faz uma cópia densa e correta dos dados,
     * com um atalho rápido (memcpy) quando axis == 0.
     *
     * @param axis Eixo ao longo do qual fazer slice
     * @param start Índice inicial (inclusivo)
     * @param end Índice final (exclusivo)
     * @return ZTensor novo e independente contendo apenas a fatia solicitada
     * @throws std::out_of_range Se axis >= ndim
     * @throws std::invalid_argument Se start >= end ou end > shape[axis]
     */
    ZTensor slice(size_t axis, size_t start, size_t end) const {
        // Validação do eixo
        if (axis >= shape.size()) {
            throw std::out_of_range("Eixo inválido para slice");
        }

        // Validação dos índices
        if (start > end) {
            throw std::invalid_argument("slice: start deve ser <= end");
        }

        if (end > shape[axis]) {
            throw std::invalid_argument("slice: end excede dimensão " +
                                       std::to_string(axis));
        }

#ifdef HAVE_CUDA
        ensure_host();
#endif

        std::vector<size_t> new_shape = shape;
        new_shape[axis] = end - start;

        ZTensor result(new_shape);

        // Copia estado de autograd
        result.requires_grad = requires_grad;
        result.grad_fn = grad_fn;

        if (result.size() == 0) {
            return result;
        }

        // Fast path: slice ao longo do eixo mais externo (axis 0) é um
        // bloco contíguo em row-major — pode ser copiado com memcpy.
        if (axis == 0) {
            size_t block = 1;
            for (size_t i = 1; i < shape.size(); ++i) block *= shape[i];
            const float* src = this->data.data() + start * block;
            std::memcpy(result.data.data(), src, (end - start) * block * sizeof(float));
            return result;
        }

        // Caminho genérico: decompõe índice linear e recompõe por eixo.
        std::vector<size_t> dst_idx(shape.size(), 0);
        std::vector<size_t> src_idx(shape.size(), 0);
        const size_t total_out = result.size();

        for (size_t linear = 0; linear < total_out; ++linear) {
            size_t tmp = linear;
            for (int d = static_cast<int>(new_shape.size()) - 1; d >= 0; --d) {
                dst_idx[d] = tmp % new_shape[d];
                tmp /= new_shape[d];
            }
            src_idx = dst_idx;
            src_idx[axis] = dst_idx[axis] + start;

            result.at(dst_idx) = this->at(src_idx);
        }

        return result;
    }

    // --- Transpose (float) ---
    /**
     * Transpõe uma matriz 2D (troca linhas por colunas)
     * @return Novo ZTensor transposto
     * @throws std::runtime_error Se o tensor não for 2D ou shape inválido
     */
    ZTensor transpose() const {
        if (shape.size() != 2) {
            throw std::runtime_error("Transposição requer tensor 2D");
        }

        const size_t rows = shape[0];
        const size_t cols = shape[1];

        if (rows == 0 || cols == 0 || empty()) {
            return ZTensor({cols, rows});
        }

#ifdef HAVE_CUDA
        if (device_valid) {
            ZTensor result({cols, rows});
            result.allocate_device_for_write();
            gpu_transpose_device(d_data, result.d_data, rows, cols);
            result.mark_device_modified();
            return result;
        }
        // FIX: transpose() lia data.data() diretamente sem sincronizar;
        // se o tensor estivesse residente só na GPU, lia memória de host
        // desatualizada. Todas as outras operações já faziam ensure_host()
        // antes de acessar os dados; esta foi a exceção corrigida aqui.
        ensure_host();
#endif

        ZTensor result({cols, rows});
        const float* p_in = this->data.data();
        float* p_out = result.data.data();

        constexpr int TILE_SIZE = 32;

    #if HAS_OPENMP
        const size_t total = rows * cols;
        if (total > 10000) {
#pragma omp parallel for collapse(2) schedule(static)
            for (size_t ii = 0; ii < rows; ii += TILE_SIZE) {
                for (size_t jj = 0; jj < cols; jj += TILE_SIZE) {
                    for (size_t i = ii; i < std::min(ii + TILE_SIZE, rows); ++i) {
                        for (size_t j = jj; j < std::min(jj + TILE_SIZE, cols); ++j) {
                            p_out[j * rows + i] = p_in[i * cols + j];
                        }
                    }
                }
            }
            return result;
        }
    #endif

        // fallback sequencial
        for (size_t i = 0; i < rows; ++i) {
            for (size_t j = 0; j < cols; ++j) {
                p_out[j * rows + i] = p_in[i * cols + j];
            }
        }

        return result;
    }

ZTensor column(size_t col_idx) const {
        if (shape.size() != 2) {
            throw std::runtime_error("column() requer um tensor 2D");
        }
        if (col_idx >= shape[1]) {
            throw std::out_of_range("Índice da coluna fora dos limites");
        }

        // FIX: este método marcava result.requires_grad = requires_grad sem
        // criar um AutogradNode/grad_fn correspondente. Isso fazia o tensor
        // resultante virar um nó-folha: um backward() que passasse por
        // column() nunca devolvia gradiente ao tensor original — perda
        // silenciosa. Até existir uma implementação real de autograd (com
        // "scatter" do gradiente de volta às posições originais), bloqueamos
        // o uso em tensores rastreados.
        if (this->requires_grad) {
            throw std::logic_error(
                "column() ainda não suporta autograd (requires_grad=true). "
                "Use em tensores com requires_grad=false."
            );
        }

#ifdef HAVE_CUDA
        ensure_host();
#endif

        const size_t rows = shape[0];
        const size_t cols = shape[1];

        ZTensor result({rows});
        if (rows == 0) return result;

        const float* p_in = this->data.data();
        float* p_out = result.data.data();

#if HAS_OPENMP
        if (rows > ZMATRIX_PARALLEL_THRESHOLD) {
            #pragma omp parallel for schedule(static)
            for (size_t i = 0; i < rows; ++i) {
                p_out[i] = p_in[i * cols + col_idx];
            }
        } else {
            for (size_t i = 0; i < rows; ++i) {
                p_out[i] = p_in[i * cols + col_idx];
            }
        }
#else
        for (size_t i = 0; i < rows; ++i) {
            p_out[i] = p_in[i * cols + col_idx];
        }
#endif

        return result;
    }

    ZTensor row(size_t row_idx) const {
        if (shape.size() != 2) {
            throw std::runtime_error("row() requer um tensor 2D");
        }
        if (row_idx >= shape[0]) {
            throw std::out_of_range("Índice da linha fora dos limites");
        }

        // FIX: ver nota em column() sobre requires_grad sem grad_fn.
        if (this->requires_grad) {
            throw std::logic_error(
                "row() ainda não suporta autograd (requires_grad=true). "
                "Use em tensores com requires_grad=false."
            );
        }

#ifdef HAVE_CUDA
        ensure_host();
#endif

        const size_t cols = shape[1];

        ZTensor result({cols});
        if (cols == 0) return result;

        const float* p_in = this->data.data();
        float* p_out = result.data.data();

        const float* row_ptr = p_in + (row_idx * cols);

#if HAS_OPENMP
        if (cols > ZMATRIX_PARALLEL_THRESHOLD) {
            #pragma omp parallel for schedule(static)
            for (size_t i = 0; i < cols; ++i) {
                p_out[i] = row_ptr[i];
            }
        } else {
            std::memcpy(p_out, row_ptr, cols * sizeof(float));
        }
#else
        std::memcpy(p_out, row_ptr, cols * sizeof(float));
#endif

        return result;
    }

    ZTensor gather(const std::vector<size_t>& indices) const {
        if (shape.size() != 2) {
            throw std::runtime_error("gather() requer um tensor 2D");
        }

        // FIX: ver nota em column() sobre requires_grad sem grad_fn.
        if (this->requires_grad) {
            throw std::logic_error(
                "gather() ainda não suporta autograd (requires_grad=true). "
                "Use em tensores com requires_grad=false."
            );
        }

#ifdef HAVE_CUDA
        ensure_host();
#endif

        const size_t num_indices = indices.size();
        const size_t cols = shape[1];
        const size_t rows_limit = shape[0];

        // Validação antecipada: fora da região paralela, para não lançar
        // exceção dentro de um #pragma omp parallel for (comportamento
        // indefinido / abort do OpenMP).
        for (size_t idx : indices) {
            if (idx >= rows_limit) {
                throw std::out_of_range("Índice na lista de gather fora dos limites");
            }
        }

        ZTensor result({num_indices, cols});
        if (num_indices == 0 || cols == 0) return result;

        const float* p_in = this->data.data();
        float* p_out = result.data.data();

        // FIX: antes paralelizava sem checar ZMATRIX_PARALLEL_THRESHOLD,
        // diferente da convenção usada no resto do arquivo (OpenMP fica mais
        // lento que sequencial abaixo do threshold, por isso o projeto o
        // desabilita nesses casos).
#if HAS_OPENMP
        if (num_indices > ZMATRIX_PARALLEL_THRESHOLD) {
            #pragma omp parallel for schedule(static)
            for (size_t i = 0; i < num_indices; ++i) {
                size_t row_idx = indices[i];
                std::memcpy(p_out + (i * cols), p_in + (row_idx * cols), cols * sizeof(float));
            }
        } else {
            for (size_t i = 0; i < num_indices; ++i) {
                size_t row_idx = indices[i];
                std::memcpy(p_out + (i * cols), p_in + (row_idx * cols), cols * sizeof(float));
            }
        }
#else
        for (size_t i = 0; i < num_indices; ++i) {
            size_t row_idx = indices[i];
            std::memcpy(p_out + (i * cols), p_in + (row_idx * cols), cols * sizeof(float));
        }
#endif

        return result;
    }

    ZTensor where(size_t feature_index, float threshold) const {
        if (shape.size() != 2) {
            throw std::runtime_error("where() requer um tensor 2D");
        }
        if (feature_index >= shape[1]) {
            throw std::out_of_range("Índice da coluna fora dos limites");
        }

#ifdef HAVE_CUDA
        ensure_host();
#endif

        const size_t rows = shape[0];
        const size_t cols = shape[1];

        ZTensor result({rows});
        float* p_out = result.data.data();
        const float* p_in = this->data.data();

        // FIX: antes paralelizava sem checar ZMATRIX_PARALLEL_THRESHOLD (ver gather()).
#if HAS_OPENMP
        if (rows > ZMATRIX_PARALLEL_THRESHOLD) {
            #pragma omp parallel for schedule(static)
            for (size_t i = 0; i < rows; ++i) {
                float val = p_in[i * cols + feature_index];
                p_out[i] = (val <= threshold) ? 1.0f : 0.0f;
            }
        } else {
            for (size_t i = 0; i < rows; ++i) {
                float val = p_in[i * cols + feature_index];
                p_out[i] = (val <= threshold) ? 1.0f : 0.0f;
            }
        }
#else
        for (size_t i = 0; i < rows; ++i) {
            float val = p_in[i * cols + feature_index];
            p_out[i] = (val <= threshold) ? 1.0f : 0.0f;
        }
#endif

        return result;
    }

    // Retorna um tensor contendo APENAS os índices que atendem à condição (val <= threshold)
        ZTensor find_indices_where(size_t feature_index, float threshold) const {
            if (shape.size() != 2) throw std::runtime_error("requer tensor 2D");
            if (feature_index >= shape[1]) throw std::out_of_range("feature index out of bounds");
#ifdef HAVE_CUDA
            ensure_host();
#endif

            const size_t rows = shape[0];
            const size_t cols = shape[1];
            const float* p_in = this->data.data();

            // 1. Primeiro passamos contando quantos elementos atendem (passada rápida)
            size_t count = 0;
            for (size_t i = 0; i < rows; ++i) {
                if (p_in[i * cols + feature_index] <= threshold) count++;
            }

            // 2. Alocamos o tensor de índices exatamente do tamanho necessário
            ZTensor result({count});
            float* p_out = result.data.data();

            // 3. Preenchemos os índices
            size_t k = 0;
            for (size_t i = 0; i < rows; ++i) {
                if (p_in[i * cols + feature_index] <= threshold) {
                    p_out[k++] = static_cast<float>(i);
                }
            }
            return result;
        }

    float calculate_split_gini(size_t feature_index, float threshold, const ZTensor& y) const {
        if (shape.size() != 2) throw std::runtime_error("X deve ser 2D");
        if (feature_index >= shape[1]) throw std::out_of_range("Coluna fora dos limites");
        if (shape[0] != y.size()) throw std::invalid_argument("X e y devem ter o mesmo número de linhas");

#ifdef HAVE_CUDA
        ensure_host();
        y.ensure_host();
#endif
        const size_t rows = shape[0];
        const size_t cols = shape[1];
        const float* p_X = this->data.data();
        const float* p_y = y.data.data();

        // 1. Acha o maior label para dimensionar o vetor de contagem
        int max_label = 0;
        for(size_t i = 0; i < rows; ++i) {
            if (p_y[i] > max_label) max_label = (int)p_y[i];
        }

        std::vector<int> left_counts(max_label + 1, 0);
        std::vector<int> right_counts(max_label + 1, 0);
        int left_total = 0, right_total = 0;

        // 2. Passada única para contar
        for (size_t i = 0; i < rows; ++i) {
            int label = (int)p_y[i];
            if (p_X[i * cols + feature_index] <= threshold) {
                left_counts[label]++;
                left_total++;
            } else {
                right_counts[label]++;
                right_total++;
            }
        }

        if (left_total == 0 || right_total == 0) return 1.0f;

        // 3. Cálculo do Gini
        auto calc_gini = [&](const std::vector<int>& counts, int total) {
            float sum_sq = 0.0f;
            for (int c : counts) {
                float p = (float)c / (float)total;
                sum_sq += (p * p);
            }
            return 1.0f - sum_sq;
        };

        return ((float)left_total / rows) * calc_gini(left_counts, left_total) +
               ((float)right_total / rows) * calc_gini(right_counts, right_total);
    }

    /**
     * argsort com semântica numpy:
     *  - 1D: retorna vetor {N} com os índices que ordenam o vetor.
     *  - 2D axis=1: cada LINHA é ordenada independentemente (shape preservado).
     *  - 2D axis=0: cada COLUNA é ordenada independentemente (shape preservado).
     *
     * FIX (versão anterior):
     *  1) O ramo "axis=0" criava o resultado com o shape COMPLETO do tensor
     *     original ({rows, cols}) mas só preenchia os primeiros `rows`
     *     elementos do buffer plano — o resto ficava zerado.
     *  2) O comparador usado nesse ramo olhava só a coluna 0
     *     (`p_data[i1 * cols] < p_data[i2 * cols]`), então nem sequer
     *     implementava "ordenar por eixo 0" — ordenava tudo pela primeira
     *     coluna, ignorando as demais.
     *  3) `rows` era calculado com um ternário cujos dois ramos eram
     *     idênticos (dead code).
     */
    ZTensor argsort(size_t axis = 0) const {
        if (shape.size() > 2) throw std::runtime_error("argsort suporta apenas 1D ou 2D");

#ifdef HAVE_CUDA
        ensure_host();
#endif

        // --- Caso 1D: só existe um eixo, axis é ignorado ---
        if (shape.size() == 1) {
            const size_t N = shape[0];
            ZTensor result({N});
            if (N == 0) return result;

            std::vector<size_t> indices(N);
            std::iota(indices.begin(), indices.end(), 0);

            const float* p_data = this->data.data();
            std::sort(indices.begin(), indices.end(), [&](size_t i1, size_t i2) {
                return p_data[i1] < p_data[i2];
            });

            float* p_out = result.data.data();
            for (size_t i = 0; i < N; ++i) {
                p_out[i] = static_cast<float>(indices[i]);
            }
            return result;
        }

        // --- Caso 2D ---
        if (axis != 0 && axis != 1) {
            throw std::out_of_range("argsort: axis deve ser 0 ou 1 para tensor 2D");
        }

        const size_t rows = shape[0];
        const size_t cols = shape[1];

        ZTensor result({rows, cols});
        if (rows == 0 || cols == 0) return result;

        const float* p_in = this->data.data();
        float* p_out = result.data.data();

        if (axis == 1) {
            // Cada LINHA ordenada independentemente (semântica numpy axis=1)
#if HAS_OPENMP
            if (rows > ZMATRIX_PARALLEL_THRESHOLD) {
                #pragma omp parallel for schedule(static)
                for (size_t r = 0; r < rows; ++r) {
                    std::vector<size_t> indices(cols);
                    std::iota(indices.begin(), indices.end(), 0);
                    const float* row_ptr = p_in + (r * cols);
                    std::sort(indices.begin(), indices.end(), [&](size_t i1, size_t i2) {
                        return row_ptr[i1] < row_ptr[i2];
                    });
                    for (size_t c = 0; c < cols; ++c) {
                        p_out[r * cols + c] = static_cast<float>(indices[c]);
                    }
                }
            } else
#endif
            {
                for (size_t r = 0; r < rows; ++r) {
                    std::vector<size_t> indices(cols);
                    std::iota(indices.begin(), indices.end(), 0);
                    const float* row_ptr = p_in + (r * cols);
                    std::sort(indices.begin(), indices.end(), [&](size_t i1, size_t i2) {
                        return row_ptr[i1] < row_ptr[i2];
                    });
                    for (size_t c = 0; c < cols; ++c) {
                        p_out[r * cols + c] = static_cast<float>(indices[c]);
                    }
                }
            }
            return result;
        }

        // axis == 0: cada COLUNA ordenada independentemente (semântica numpy axis=0)
#if HAS_OPENMP
        if (cols > ZMATRIX_PARALLEL_THRESHOLD) {
            #pragma omp parallel for schedule(static)
            for (size_t c = 0; c < cols; ++c) {
                std::vector<size_t> indices(rows);
                std::iota(indices.begin(), indices.end(), 0);
                std::sort(indices.begin(), indices.end(), [&](size_t i1, size_t i2) {
                    return p_in[i1 * cols + c] < p_in[i2 * cols + c];
                });
                for (size_t r = 0; r < rows; ++r) {
                    p_out[r * cols + c] = static_cast<float>(indices[r]);
                }
            }
        } else
#endif
        {
            for (size_t c = 0; c < cols; ++c) {
                std::vector<size_t> indices(rows);
                std::iota(indices.begin(), indices.end(), 0);
                std::sort(indices.begin(), indices.end(), [&](size_t i1, size_t i2) {
                    return p_in[i1 * cols + c] < p_in[i2 * cols + c];
                });
                for (size_t r = 0; r < rows; ++r) {
                    p_out[r * cols + c] = static_cast<float>(indices[r]);
                }
            }
        }

        return result;
    }

    // --- Mode (moda) ---

        // Validação SERIAL de NaN — deve rodar sempre ANTES de qualquer região
        // OpenMP. Exceções C++ não podem atravessar um #pragma omp parallel for
        // (comportamento indefinido / possível std::terminate()), então toda
        // rejeição de NaN é resolvida aqui, fora do paralelismo.
        static void validate_mode_values(const float* values, size_t count)
        {
            for (size_t i = 0; i < count; ++i) {
                if (std::isnan(values[i])) {
                    throw std::runtime_error(
                        "mode: valores NaN não são suportados"
                    );
                }
            }
        }

        // Calcula a moda de 'count' valores a partir de 'start', espaçados por
        // 'stride' elementos (stride=1 para acesso contíguo — ex: linha;
        // stride=cols para acesso não-contíguo — ex: coluna).
        //
        // PRÉ-CONDIÇÃO: o chamador já validou ausência de NaN via
        // validate_mode_values() antes de invocar esta função. Por isso ela NÃO
        // lança exceções — é seguro chamá-la de dentro de uma região OpenMP.
        static float calculate_mode(const float* start, size_t count, size_t stride)
        {
            std::unordered_map<float, size_t> frequencies;
            frequencies.reserve(count);

            for (size_t i = 0; i < count; ++i) {
                ++frequencies[start[i * stride]];
            }

            float best_value = start[0];
            size_t best_count = 0;

            // Desempate determinístico: em frequências iguais, escolhe o menor
            // valor. Não depende da ordem de iteração de unordered_map, pois a
            // condição de atualização compara explicitamente contra o melhor
            // candidato atual a cada iteração.
            for (const auto& [value, frequency] : frequencies) {
                if (
                    frequency > best_count ||
                    (frequency == best_count && value < best_value)
                ) {
                    best_value = value;
                    best_count = frequency;
                }
            }

            return best_value;
        }

        // Moda global (valor mais frequente em todo o tensor).
        // NÃO paralelizado nesta implementação: exigiria um unordered_map
        // compartilhado (risco de corrida) ou merge de mapas por thread
        // (complexidade sem benchmark que justifique o custo até o momento).
        float mode() const
        {
    #ifdef HAVE_CUDA
            // ensure_host() já é const — chamada direta, sem const_cast.
            ensure_host();
    #endif

            const size_t N = size();

            if (N == 0) {
                throw std::runtime_error(
                    "mode: não pode ser aplicado em tensor vazio"
                );
            }

            const float* p = data.data();

            validate_mode_values(p, N);

            return calculate_mode(p, N, 1);
        }

        // Moda ao longo de um eixo. Suporta apenas tensores 1D e 2D.
        ZTensor mode(int axis) const
        {
    #ifdef HAVE_CUDA
            ensure_host();
    #endif

            const size_t N = size();

            if (N == 0) {
                throw std::runtime_error(
                    "mode: não pode ser aplicado em tensor vazio"
                );
            }

            if (shape.size() > 2) {
                throw std::runtime_error(
                    "mode(axis): apenas tensores 1D e 2D são suportados"
                );
            }

            // Normalização de eixo negativo ANTES de qualquer conversão para
            // size_t (evita wraparound silencioso de inteiro sem sinal).
            if (axis < 0) {
                axis += static_cast<int>(shape.size());
            }

            if (axis < 0 || axis >= static_cast<int>(shape.size())) {
                throw std::runtime_error(
                    "axis fora dos limites para mode"
                );
            }

            const float* p = data.data();

            // Validação obrigatoriamente serial, antes de qualquer OpenMP abaixo.
            validate_mode_values(p, N);

            if (shape.size() == 1) {
                ZTensor result({1});
                result.data[0] = calculate_mode(p, N, 1);
                return result;
            }

            const size_t rows = shape[0];
            const size_t cols = shape[1];

            if (axis == 1) {
                // Moda por linha: cada linha é contígua (stride=1).
                ZTensor result({rows});
                float* out = result.data.data();

                // Critério de paralelização considera o TRABALHO TOTAL (N),
                // não apenas o tamanho da saída (rows) — cada iteração processa
                // 'cols' elementos, então o custo real escala com N = rows*cols.
    #if HAS_OPENMP
                if (N > ZMATRIX_PARALLEL_THRESHOLD) {
                    #pragma omp parallel for schedule(static)
                    for (size_t row = 0; row < rows; ++row) {
                        out[row] = calculate_mode(p + row * cols, cols, 1);
                    }
                } else
    #endif
                {
                    for (size_t row = 0; row < rows; ++row) {
                        out[row] = calculate_mode(p + row * cols, cols, 1);
                    }
                }

                return result;
            }

            // axis == 0: moda por coluna — elementos espaçados por 'cols' (stride=cols).
            ZTensor result({cols});
            float* out = result.data.data();

    #if HAS_OPENMP
            if (N > ZMATRIX_PARALLEL_THRESHOLD) {
                #pragma omp parallel for schedule(static)
                for (size_t col = 0; col < cols; ++col) {
                    out[col] = calculate_mode(p + col, rows, cols);
                }
            } else
    #endif
            {
                for (size_t col = 0; col < cols; ++col) {
                    out[col] = calculate_mode(p + col, rows, cols);
                }
            }

            return result;
        }

    // --- Unique ---
        ZTensor unique() const {
    #ifdef HAVE_CUDA
            // Garante que a versão mais recente está na RAM da CPU
            const_cast<ZTensor*>(this)->ensure_host();
    #endif
            if (data.empty()) {
                return ZTensor({0});
            }

            // Cópia profunda rápida
            std::vector<float> temp = data;

            // Ordena e remove duplicatas usando a STL (altamente otimizado no C++)
            std::sort(temp.begin(), temp.end());
            auto last = std::unique(temp.begin(), temp.end());
            temp.erase(last, temp.end());

            // Cria o tensor de resultado
            ZTensor result({temp.size()});
            result.data = std::move(temp); // Move sem copiar novamente

            return result;
        }
    // --- Bincount ---
        ZTensor bincount(const ZTensor* weights = nullptr) const {
                if (shape.size() != 1) throw std::runtime_error("bincount() requer tensor 1D");
        #ifdef HAVE_CUDA
                ensure_host();
        #endif
                const size_t N = size();
                if (weights && weights->size() != N) {
                    throw std::invalid_argument("bincount(): weights deve ter o mesmo tamanho do tensor de índices");
                }
        #ifdef HAVE_CUDA
                if (weights) weights->ensure_host();
        #endif

                long max_val = -1;
                for (size_t i = 0; i < N; ++i) {
                    float v = data[i];
                    if (v < 0.0f || std::floor(v) != v) {
                        throw std::invalid_argument("bincount(): valores devem ser inteiros não negativos");
                    }
                    long iv = static_cast<long>(v);
                    if (iv > max_val) max_val = iv;
                }

                size_t out_size = (max_val < 0) ? 0 : static_cast<size_t>(max_val + 1);
                ZTensor result(std::vector<size_t>{out_size});
                if (out_size == 0) return result;

                float* out_data = result.data.data();
                const float* w = weights ? weights->data.data() : nullptr;

                for (size_t i = 0; i < N; ++i) {
                    size_t bin = static_cast<size_t>(data[i]);
                    out_data[bin] += w ? w[i] : 1.0f;
                }

                return result;
            }

    // --- Argmax ---
        size_t argmax() const {
    #ifdef HAVE_CUDA
    #endif
            const size_t N = size();
            if (N == 0) {
                throw std::runtime_error("argmax: não pode ser aplicado em tensor vazio");
            }

#ifdef HAVE_CUDA
            if (device_valid) return gpu_arg_value_device(d_data, N, 1);
            ensure_host();
#endif
            const float* p = data.data();
            size_t global_max_idx = 0;
            float global_max_val = p[0];

    #if HAS_OPENMP
            if (N > ZMATRIX_PARALLEL_THRESHOLD) {
    #pragma omp parallel
                {
                    // Variáveis locais para cada thread (evita concorrência)
                    size_t local_max_idx = 0;
                    float local_max_val = p[0];

    #pragma omp for nowait
                    for (size_t i = 0; i < N; ++i) {
                        if (p[i] > local_max_val) {
                            local_max_val = p[i];
                            local_max_idx = i;
                        }
                    }

                    // Junta os resultados locais na variável global de forma segura
    #pragma omp critical
                    {
                        if (local_max_val > global_max_val ||
                           (local_max_val == global_max_val && local_max_idx < global_max_idx)) {
                            global_max_val = local_max_val;
                            global_max_idx = local_max_idx;
                        }
                    }
                }
            } else {
                for (size_t i = 1; i < N; ++i) {
                    if (p[i] > global_max_val) {
                        global_max_val = p[i];
                        global_max_idx = i;
                    }
                }
            }
    #else
            for (size_t i = 1; i < N; ++i) {
                if (p[i] > global_max_val) {
                    global_max_val = p[i];
                    global_max_idx = i;
                }
            }
    #endif
            return global_max_idx;
        }
    size_t argmin() const {
    #ifdef HAVE_CUDA
            // NOTA: ensure_host() já é 'const' (opera sobre os campos mutable
            // d_data/device_valid/host_valid), então o const_cast usado em
            // argmax() não é necessário aqui — chamada direta é suficiente.
    #endif
            const size_t N = size();
            if (N == 0) {
                throw std::runtime_error("argmin: não pode ser aplicado em tensor vazio");
            }

#ifdef HAVE_CUDA
            if (device_valid) return gpu_arg_value_device(d_data, N, 0);
            ensure_host();
#endif
            const float* p = data.data();
            size_t global_min_idx = 0;
            float global_min_val = p[0];

    #if HAS_OPENMP
            if (N > ZMATRIX_PARALLEL_THRESHOLD) {
    #pragma omp parallel
                {
                    size_t local_min_idx = 0;
                    float local_min_val = p[0];

    #pragma omp for nowait
                    for (size_t i = 0; i < N; ++i) {
                        if (p[i] < local_min_val) {
                            local_min_val = p[i];
                            local_min_idx = i;
                        }
                    }

    #pragma omp critical
                    {
                        if (local_min_val < global_min_val ||
                           (local_min_val == global_min_val && local_min_idx < global_min_idx)) {
                            global_min_val = local_min_val;
                            global_min_idx = local_min_idx;
                        }
                    }
                }
            } else {
                for (size_t i = 1; i < N; ++i) {
                    if (p[i] < global_min_val) {
                        global_min_val = p[i];
                        global_min_idx = i;
                    }
                }
            }
    #else
            for (size_t i = 1; i < N; ++i) {
                if (p[i] < global_min_val) {
                    global_min_val = p[i];
                    global_min_idx = i;
                }
            }
    #endif
            return global_min_idx;
        }

        // --- sort: 1D ordena o vetor inteiro. 2D: axis=1 ordena cada LINHA,
            // axis=0 ordena cada COLUNA (mesma semântica numpy usada em argsort).
            ZTensor sort(size_t axis = 0) const {
                if (shape.size() > 2) throw std::runtime_error("sort suporta apenas 1D ou 2D");
        #ifdef HAVE_CUDA
                ensure_host();
        #endif
                if (shape.size() == 1) {
                    ZTensor result(shape);
                    std::vector<float> tmp(data.begin(), data.end());
                    std::sort(tmp.begin(), tmp.end());
                    std::copy(tmp.begin(), tmp.end(), result.data.begin());
                    return result;
                }

                if (axis != 0 && axis != 1) {
                    throw std::out_of_range("sort: axis deve ser 0 ou 1 para tensor 2D");
                }

                const size_t rows = shape[0];
                const size_t cols = shape[1];
                ZTensor result(shape);
                if (rows == 0 || cols == 0) return result;

                const float* p_in = data.data();
                float* p_out = result.data.data();

                if (axis == 1) {
        #if HAS_OPENMP
                    if (rows > ZMATRIX_PARALLEL_THRESHOLD) {
                        #pragma omp parallel for schedule(static)
                        for (size_t r = 0; r < rows; ++r) {
                            std::vector<float> row(p_in + r * cols, p_in + r * cols + cols);
                            std::sort(row.begin(), row.end());
                            std::copy(row.begin(), row.end(), p_out + r * cols);
                        }
                    } else
        #endif
                    {
                        for (size_t r = 0; r < rows; ++r) {
                            std::vector<float> row(p_in + r * cols, p_in + r * cols + cols);
                            std::sort(row.begin(), row.end());
                            std::copy(row.begin(), row.end(), p_out + r * cols);
                        }
                    }
                    return result;
                }

                // axis == 0: cada coluna ordenada independentemente
        #if HAS_OPENMP
                if (cols > ZMATRIX_PARALLEL_THRESHOLD) {
                    #pragma omp parallel for schedule(static)
                    for (size_t c = 0; c < cols; ++c) {
                        std::vector<float> col(rows);
                        for (size_t r = 0; r < rows; ++r) col[r] = p_in[r * cols + c];
                        std::sort(col.begin(), col.end());
                        for (size_t r = 0; r < rows; ++r) p_out[r * cols + c] = col[r];
                    }
                } else
        #endif
                {
                    for (size_t c = 0; c < cols; ++c) {
                        std::vector<float> col(rows);
                        for (size_t r = 0; r < rows; ++r) col[r] = p_in[r * cols + c];
                        std::sort(col.begin(), col.end());
                        for (size_t r = 0; r < rows; ++r) p_out[r * cols + c] = col[r];
                    }
                }
                return result;
            }

            // --- isin: máscara 1.0/0.0 indicando se cada elemento está em test_values ---
            ZTensor isin(const std::vector<float>& test_values) const {
        #ifdef HAVE_CUDA
                ensure_host();
        #endif
                const size_t N = size();
                ZTensor result(shape);
                if (N == 0) return result;

                std::unordered_set<float> lookup(test_values.begin(), test_values.end());

                const float* a = data.data();
                float* r = result.data.data();

        #if HAS_OPENMP
                if (N > ZMATRIX_PARALLEL_THRESHOLD) {
                    #pragma omp parallel for schedule(static)
                    for (size_t i = 0; i < N; ++i) {
                        r[i] = (lookup.count(a[i]) > 0) ? 1.0f : 0.0f;
                    }
                } else
        #endif
                {
                    for (size_t i = 0; i < N; ++i) {
                        r[i] = (lookup.count(a[i]) > 0) ? 1.0f : 0.0f;
                    }
                }
                return result;
            }

            // --- cumsum: acumulador em double p/ evitar deriva de precisão em
            // somas longas com float (mesmo padrão de sum()/mean()). axis<0 em 2D
            // usa default axis=1 (cumsum por linha — o mais comum em gradient
            // boosting / probabilidade cumulativa).
            ZTensor cumsum(long axis = -1) const {
                if (shape.size() == 1) {
                    const bool profile = zmatrix_cuda_profile_enabled();
                    const size_t N = size();
                    if (N == 0) {
                        ZTensor result(shape);
#ifdef HAVE_CUDA
                        if (device_valid) result.ensure_device();
#endif
                        return result;
                    }
#ifdef HAVE_CUDA
                    if (device_valid) {
                        double construction_ms = 0.0, allocation_ms = 0.0;
                        ZTensor result = create_device_result(shape, profile, construction_ms, allocation_ms);
                        const auto wrapper_start = profile ? std::chrono::steady_clock::now() : std::chrono::steady_clock::time_point{};
                        gpu_cumsum_device(d_data, result.d_data, 1, N, 0, 1);
                        const double wrapper_ms = profile ? zmatrix_elapsed_ms(wrapper_start) : 0.0;
                        const auto state_start = profile ? std::chrono::steady_clock::now() : std::chrono::steady_clock::time_point{};
                        result.mark_device_modified();
                        const double state_ms = profile ? zmatrix_elapsed_ms(state_start) : 0.0;
                        zmatrix_profile_result("cumsum_1d", construction_ms, allocation_ms, wrapper_ms, 0.0, state_ms);
                        return result;
                    }
                    ensure_host();
#endif
                    ZTensor result(shape);
                    const float* a = data.data();
                    float* r = result.data.data();
                    double running = 0.0;
                    for (size_t i = 0; i < N; ++i) {
                        running += a[i];
                        r[i] = static_cast<float>(running);
                    }
                    return result;
                }

                if (shape.size() != 2) {
                    throw std::runtime_error("cumsum() suporta apenas tensores 1D ou 2D");
                }

                size_t ax = (axis < 0) ? 1 : static_cast<size_t>(axis);
                if (ax != 0 && ax != 1) {
                    throw std::out_of_range("cumsum: axis deve ser 0 ou 1 para tensor 2D");
                }

                const size_t rows = shape[0];
                const size_t cols = shape[1];
                const bool profile = zmatrix_cuda_profile_enabled();
                const auto host_start = profile ? std::chrono::steady_clock::now() : std::chrono::steady_clock::time_point{};
#ifdef HAVE_CUDA
                ZTensor result(shape, device_valid);
#else
                ZTensor result(shape);
#endif
                [[maybe_unused]] const double host_ms = profile ? zmatrix_elapsed_ms(host_start) : 0.0;
                if (rows == 0 || cols == 0) {
#ifdef HAVE_CUDA
                    if (device_valid) result.ensure_device();
#endif
                    return result;
                }

#ifdef HAVE_CUDA
                if (device_valid) {
                    const auto allocation_start = profile ? std::chrono::steady_clock::now() : std::chrono::steady_clock::time_point{};
                    result.allocate_device_for_write();
                    const double allocation_ms = profile ? zmatrix_elapsed_ms(allocation_start) : 0.0;
                    const auto wrapper_start = profile ? std::chrono::steady_clock::now() : std::chrono::steady_clock::time_point{};
                    gpu_cumsum_device(d_data, result.d_data, rows, cols, static_cast<int>(ax), 0);
                    const double wrapper_ms = profile ? zmatrix_elapsed_ms(wrapper_start) : 0.0;
                    result.mark_device_modified();
                    zmatrix_profile_result(ax == 1 ? "cumsum_axis1" : "cumsum_axis0", host_ms, allocation_ms, wrapper_ms);
                    return result;
                }
                ensure_host();
#endif

                const float* p_in = data.data();
                float* p_out = result.data.data();

                if (ax == 1) {
        #if HAS_OPENMP
                    if (rows > ZMATRIX_PARALLEL_THRESHOLD) {
                        #pragma omp parallel for schedule(static)
                        for (size_t r = 0; r < rows; ++r) {
                            double running = 0.0;
                            for (size_t c = 0; c < cols; ++c) {
                                running += p_in[r * cols + c];
                                p_out[r * cols + c] = static_cast<float>(running);
                            }
                        }
                    } else
        #endif
                    {
                        for (size_t r = 0; r < rows; ++r) {
                            double running = 0.0;
                            for (size_t c = 0; c < cols; ++c) {
                                running += p_in[r * cols + c];
                                p_out[r * cols + c] = static_cast<float>(running);
                            }
                        }
                    }
                    return result;
                }

        #if HAS_OPENMP
                if (cols > ZMATRIX_PARALLEL_THRESHOLD) {
                    #pragma omp parallel for schedule(static)
                    for (size_t c = 0; c < cols; ++c) {
                        double running = 0.0;
                        for (size_t r = 0; r < rows; ++r) {
                            running += p_in[r * cols + c];
                            p_out[r * cols + c] = static_cast<float>(running);
                        }
                    }
                } else
        #endif
                {
                    for (size_t c = 0; c < cols; ++c) {
                        double running = 0.0;
                        for (size_t r = 0; r < rows; ++r) {
                            running += p_in[r * cols + c];
                            p_out[r * cols + c] = static_cast<float>(running);
                        }
                    }
                }
                return result;
            }

        void unique_counts(std::vector<float>& out_values, std::vector<long>& out_counts) const {
                if (shape.size() != 1) throw std::runtime_error("uniqueCounts() requer tensor 1D");
        #ifdef HAVE_CUDA
                ensure_host();
        #endif
                std::vector<float> sorted(data.begin(), data.end());
                std::sort(sorted.begin(), sorted.end());

                out_values.clear();
                out_counts.clear();
                for (size_t i = 0; i < sorted.size(); ++i) {
                    if (i == 0 || sorted[i] != sorted[i - 1]) {
                        out_values.push_back(sorted[i]);
                        out_counts.push_back(1);
                    } else {
                        out_counts.back()++;
                    }
                }
            }

            // --- stack: empilha N tensores de mesmo shape em um novo eixo 0 ---
            // Paralelizado: cada thread copia um bloco disjunto de 'dst' (tensors[i]
            // -> dst + i*block), sem overlap entre threads.
            static ZTensor stack(const std::vector<ZTensor*>& tensors) {
                if (tensors.empty()) throw std::runtime_error("stack(): a lista de tensores não pode ser vazia");

                const std::vector<size_t>& first_shape = tensors[0]->shape;
                for (size_t i = 1; i < tensors.size(); ++i) {
                    if (tensors[i]->shape != first_shape) {
                        throw std::invalid_argument("stack(): todos os tensores devem ter o mesmo shape");
                    }
                }

        #ifdef HAVE_CUDA
                for (auto* t : tensors) t->ensure_host();
        #endif

                std::vector<size_t> out_shape;
                out_shape.push_back(tensors.size());
                out_shape.insert(out_shape.end(), first_shape.begin(), first_shape.end());

                ZTensor result(out_shape);
                if (result.size() == 0) return result;

                const size_t block = tensors[0]->size();
                float* dst = result.data.data();

        #if HAS_OPENMP
                if (result.size() > ZMATRIX_PARALLEL_THRESHOLD) {
                    #pragma omp parallel for schedule(static)
                    for (size_t i = 0; i < tensors.size(); ++i) {
                        std::memcpy(dst + i * block, tensors[i]->data.data(), block * sizeof(float));
                    }
                } else
        #endif
                {
                    for (size_t i = 0; i < tensors.size(); ++i) {
                        std::memcpy(dst + i * block, tensors[i]->data.data(), block * sizeof(float));
                    }
                }
                return result;
            }

        // === Variance ===
            // Calcula a variância global. (ddof = 1 para variância amostral, 0 para populacional)
            float variance(int ddof = 0) const {
        #ifdef HAVE_CUDA
                const_cast<ZTensor*>(this)->ensure_host();
        #endif
                const size_t N = size();
                if (N <= static_cast<size_t>(ddof)) {
                    throw std::invalid_argument("variance: tamanho do tensor insuficiente para os graus de liberdade (ddof).");
                }

                float m = mean(); // Aproveita o método otimizado de média
                float sum_sq = 0.0f;
                const float* p = data.data();

        #if HAS_OPENMP
                if (N > ZMATRIX_PARALLEL_THRESHOLD) {
        #pragma omp parallel for reduction(+:sum_sq) schedule(static)
                    for (size_t i = 0; i < N; ++i) {
                        float diff = p[i] - m;
                        sum_sq += diff * diff;
                    }
                } else
        #endif
                {
                    for (size_t i = 0; i < N; ++i) {
                        float diff = p[i] - m;
                        sum_sq += diff * diff;
                    }
                }
                return sum_sq / static_cast<float>(N - ddof);
            }

            // === Median (Mediana) ===
            // Usa std::nth_element (O(N) de complexidade) em vez de ordenação total.
            float median() const {
        #ifdef HAVE_CUDA
                const_cast<ZTensor*>(this)->ensure_host();
        #endif
                const size_t N = size();
                if (N == 0) throw std::runtime_error("median: tensor vazio.");

                std::vector<float> temp = data; // Cópia pois nth_element altera a ordem
                size_t mid = N / 2;

                // Coloca o elemento do meio no lugar certo e particiona o resto
                std::nth_element(temp.begin(), temp.begin() + mid, temp.end());

                if (N % 2 != 0) {
                    return temp[mid]; // Tamanho ímpar, o elemento do meio é a mediana exata
                } else {
                    // Tamanho par: precisamos da média do elemento do meio com o maior da partição à esquerda
                    auto max_left_it = std::max_element(temp.begin(), temp.begin() + mid);
                    return (*max_left_it + temp[mid]) / 2.0f;
                }
            }

            // === Percentile ===
            // q = valor entre 0 e 100. Usa interpolação linear (igual numpy.percentile).
            float percentile(float q) const {
        #ifdef HAVE_CUDA
                const_cast<ZTensor*>(this)->ensure_host();
        #endif
                const size_t N = size();
                if (N == 0) throw std::runtime_error("percentile: tensor vazio.");
                if (q < 0.0f || q > 100.0f) throw std::invalid_argument("percentile: q deve estar entre 0 e 100.");
                if (N == 1) return data[0];

                std::vector<float> temp = data;

                // Calcula a posição (fracionária)
                float pos = static_cast<float>(N - 1) * (q / 100.0f);
                size_t idx = static_cast<size_t>(std::floor(pos));
                float frac = pos - static_cast<float>(idx);

                // Particiona até o índice 'idx' (O(N))
                std::nth_element(temp.begin(), temp.begin() + idx, temp.end());
                float lower = temp[idx];

                if (frac > 0.0f && idx + 1 < N) {
                    // Encontra o exato próximo valor sem ter que ordenar o resto
                    auto next_it = std::min_element(temp.begin() + idx + 1, temp.end());
                    float upper = *next_it;
                    // Interpolação linear entre o valor abaixo e o valor acima
                    return lower + frac * (upper - lower);
                }

                return lower;
            }

            // === Histogram (Para LightGBM e Density) ===
            // Retorna uma tupla com as contagens (counts) e as bordas (bin_edges).
            std::pair<ZTensor, ZTensor> histogram(int bins = 10) const {
        #ifdef HAVE_CUDA
                const_cast<ZTensor*>(this)->ensure_host();
        #endif
                const size_t N = size();
                if (N == 0) throw std::runtime_error("histogram: tensor vazio.");
                if (bins <= 0) throw std::invalid_argument("histogram: o número de bins deve ser > 0.");

                // 1. Achar min e max manualmente para evitar varrer o array duas vezes
                float min_val = data[0];
                float max_val = data[0];
                const float* p = data.data();

                for (size_t i = 1; i < N; ++i) {
                    if (p[i] < min_val) min_val = p[i];
                    if (p[i] > max_val) max_val = p[i];
                }

                if (min_val == max_val) {
                    max_val = min_val + 1.0f; // Evitar divisão por zero caso seja tensor constante
                }

                float bin_width = (max_val - min_val) / static_cast<float>(bins);

                ZTensor t_counts({static_cast<size_t>(bins)}); // Inicia zerado
                ZTensor t_edges({static_cast<size_t>(bins + 1)});

                float* p_counts = t_counts.data.data();
                float* p_edges = t_edges.data.data();

                // 2. Preencher os limites das bordas (bin edges)
                for (int i = 0; i <= bins; ++i) {
                    p_edges[i] = min_val + static_cast<float>(i) * bin_width;
                }

                // 3. Contagem em paralelo atômico
        #if HAS_OPENMP
                if (N > ZMATRIX_PARALLEL_THRESHOLD) {
        #pragma omp parallel for schedule(static)
                    for (size_t i = 0; i < N; ++i) {
                        int b = static_cast<int>((p[i] - min_val) / bin_width);
                        if (b >= bins) b = bins - 1; // Elemento que é exatamente igual ao valor máximo vai no último bin
                        if (b < 0) b = 0;
        #pragma omp atomic
                        p_counts[b] += 1.0f;
                    }
                } else
        #endif
                {
                    for (size_t i = 0; i < N; ++i) {
                        int b = static_cast<int>((p[i] - min_val) / bin_width);
                        if (b >= bins) b = bins - 1;
                        if (b < 0) b = 0;
                        p_counts[b] += 1.0f;
                    }
                }

                return {t_counts, t_edges};
            }

    // --- Concat (Estático) ---
        static ZTensor concat(const std::vector<const ZTensor*>& tensors, int axis = 0) {
            if (tensors.empty()) {
                throw std::invalid_argument("concat: a lista de tensores está vazia");
            }

            size_t ndim = tensors[0]->shape.size();

            // Suporta eixos negativos (ex: -1 é a última dimensão)
            if (axis < 0) axis += static_cast<int>(ndim);
            if (axis < 0 || axis >= static_cast<int>(ndim)) {
                throw std::out_of_range("concat: eixo (axis) fora dos limites");
            }

            std::vector<size_t> out_shape = tensors[0]->shape;
            size_t concat_dim_size = 0;

            // 1. Validação estrutural de todos os tensores
            for (const ZTensor* t : tensors) {
    #ifdef HAVE_CUDA
                const_cast<ZTensor*>(t)->ensure_host();
    #endif
                if (t->shape.size() != ndim) {
                    throw std::invalid_argument("concat: todos os tensores devem ter o mesmo número de dimensões (ndim)");
                }
                for (size_t d = 0; d < ndim; ++d) {
                    if (d != static_cast<size_t>(axis) && t->shape[d] != out_shape[d]) {
                        throw std::invalid_argument("concat: tamanhos incompatíveis nos eixos não-concatenados");
                    }
                }
                concat_dim_size += t->shape[axis];
            }

            out_shape[axis] = concat_dim_size;
            ZTensor result(out_shape);
            float* res_p = result.data.data();

            // 2. Cálculos de tamanho de blocos para memcpy rápido
            size_t outer_size = 1;
            for (size_t d = 0; d < static_cast<size_t>(axis); ++d) outer_size *= out_shape[d];

            size_t inner_size = 1;
            for (size_t d = static_cast<size_t>(axis) + 1; d < ndim; ++d) inner_size *= out_shape[d];

            // 3. Executa a cópia de blocos diretos na memória
    #if HAS_OPENMP
            if (outer_size * concat_dim_size * inner_size > ZMATRIX_PARALLEL_THRESHOLD) {
    #pragma omp parallel for schedule(static)
                for (size_t i = 0; i < outer_size; ++i) {
                    size_t out_offset = i * concat_dim_size * inner_size;
                    for (const ZTensor* t : tensors) {
                        size_t chunk_size = t->shape[axis] * inner_size;
                        size_t in_offset = i * chunk_size;

                        std::memcpy(res_p + out_offset, t->data.data() + in_offset, chunk_size * sizeof(float));
                        out_offset += chunk_size;
                    }
                }
            } else {
    #endif
                for (size_t i = 0; i < outer_size; ++i) {
                    size_t out_offset = i * concat_dim_size * inner_size;
                    for (const ZTensor* t : tensors) {
                        size_t chunk_size = t->shape[axis] * inner_size;
                        size_t in_offset = i * chunk_size;

                        std::memcpy(res_p + out_offset, t->data.data() + in_offset, chunk_size * sizeof(float));
                        out_offset += chunk_size;
                    }
                }
    #if HAS_OPENMP
            }
    #endif

            return result;
        }
    ZTensor arg_reduce_axis(int axis, bool find_max) const {
            if (shape.empty()) throw std::runtime_error(ZMATRIX_ERR_EMPTY_MATRIX);
            if (axis < 0 || static_cast<size_t>(axis) >= shape.size()) {
                throw std::out_of_range("axis fora dos limites para argmax/argmin");
            }

            std::vector<size_t> out_shape = shape;
            out_shape.erase(out_shape.begin() + axis);

            ZTensor result(out_shape);
            const size_t axis_dim = shape[axis];
            const size_t out_size = result.size();
            if (out_size == 0) return result;

#ifdef HAVE_CUDA
            if (device_valid) {
                size_t outer = 1;
                size_t inner = 1;
                for (int i = 0; i < axis; ++i) outer *= shape[static_cast<size_t>(i)];
                for (size_t i = static_cast<size_t>(axis) + 1; i < shape.size(); ++i) inner *= shape[i];
                result.allocate_device_for_write();
                gpu_arg_axis_device(d_data, result.d_data, outer, axis_dim, inner, find_max ? 1 : 0);
                result.mark_device_modified();
                return result;
            }
            ensure_host();
#endif

            float* out_data = result.data.data();

            // FIX: faltava paralelização aqui. Cada iteração já usa seu próprio
            // 'idx' local (declarado dentro do loop), sem estado compartilhado
            // mutável — mesmo padrão de soma(), que já usa este exato threshold.
    #if HAS_OPENMP
            if (out_size > ZMATRIX_PARALLEL_THRESHOLD) {
                #pragma omp parallel for schedule(static)
                for (size_t i = 0; i < out_size; ++i) {
                    std::vector<size_t> idx(shape.size(), 0);
                    size_t tmp = i;
                    for (int d = (int)shape.size() - 1, j = (int)out_shape.size() - 1; d >= 0; --d) {
                        if ((size_t)d == (size_t)axis) {
                            idx[d] = 0;
                        } else {
                            idx[d] = tmp % out_shape[j];
                            tmp /= out_shape[j];
                            --j;
                        }
                    }
                    float best_val = 0.0f;
                    size_t best_idx = 0;
                    for (size_t k = 0; k < axis_dim; ++k) {
                        idx[axis] = k;
                        float v = at(idx);
                        if (k == 0 || (find_max ? (v > best_val) : (v < best_val))) {
                            best_val = v;
                            best_idx = k;
                        }
                    }
                    out_data[i] = static_cast<float>(best_idx);
                }
            } else
    #endif
            {
                for (size_t i = 0; i < out_size; ++i) {
                    std::vector<size_t> idx(shape.size(), 0);
                    size_t tmp = i;
                    for (int d = (int)shape.size() - 1, j = (int)out_shape.size() - 1; d >= 0; --d) {
                        if ((size_t)d == (size_t)axis) {
                            idx[d] = 0;
                        } else {
                            idx[d] = tmp % out_shape[j];
                            tmp /= out_shape[j];
                            --j;
                        }
                    }
                    float best_val = 0.0f;
                    size_t best_idx = 0;
                    for (size_t k = 0; k < axis_dim; ++k) {
                        idx[axis] = k;
                        float v = at(idx);
                        if (k == 0 || (find_max ? (v > best_val) : (v < best_val))) {
                            best_val = v;
                            best_idx = k;
                        }
                    }
                    out_data[i] = static_cast<float>(best_idx);
                }
            }
            return result;
        }

    // --- abs (float) ---
     void abs()  {
         const size_t N = size();
         if (N == 0) return;
#ifdef HAVE_CUDA
         if (device_valid) {
             ensure_device();
             gpu_abs_device(d_data, N);
             mark_device_modified();
             return;
         }
         ensure_host();
#endif
         float * __restrict__ a = data.data();
         #if HAS_OPENMP
         if (N > ZMATRIX_PARALLEL_THRESHOLD) {
#pragma omp parallel for simd schedule(static)
             for(size_t i = 0; i < N; ++i) {
                  a[i] = std::fabs(a[i]);
              }
         } else {
             zmatrix_simd::abs_f32(a, N);
         }
         #else
         zmatrix_simd::abs_f32(a, N);
         #endif
#ifdef HAVE_CUDA
         mark_host_modified();
#endif
     }

    // --- sigmoid (float) ---
     void sigmoid()  {
        const size_t N = size();
        if (N == 0) return;
#ifdef HAVE_CUDA
        if (device_valid) {
            ensure_device();
            gpu_sigmoid_device(d_data, N);
            mark_device_modified();
            return;
        }
        ensure_host();
#endif
        float * __restrict__ a = data.data();
        #if HAS_OPENMP
        if (N > ZMATRIX_PARALLEL_THRESHOLD) {
            #pragma omp parallel for simd schedule(static)
            for(size_t i = 0; i < N; ++i) {
                a[i] = 1.0f / (1.0f + expf(-a[i]));
            }
        } else { // Loop sequencial se pequeno
            for(size_t i = 0; i < N; ++i) {
                a[i] = 1.0f / (1.0f + expf(-a[i]));
            }
        }
        #else // Loop sequencial se não houver OpenMP
        for(size_t i = 0; i < N; ++i) {
            a[i] = 1.0f / (1.0f + expf(-a[i]));
        }
        #endif
#ifdef HAVE_CUDA
        mark_host_modified();
#endif
     }

    void sigmoid_derivative() {
        const size_t N = size();
        if (N == 0) return;

#ifdef HAVE_CUDA
        ensure_host();
#endif
        float* __restrict__ a = data.data();

        #if HAS_OPENMP
        if (N > ZMATRIX_PARALLEL_THRESHOLD) {
#pragma omp parallel for simd schedule(static)
            for (size_t i = 0; i < N; ++i) {
                float sig = a[i];
                a[i] = sig * (1.0f - sig);
            }
        } else
        #endif
        {
            for (size_t i = 0; i < N; ++i) {
                float sig = a[i];
                a[i] = sig * (1.0f - sig);
            }
        }
#ifdef HAVE_CUDA
        mark_host_modified();
#endif
   }

    // --- New Activation Functions ---
    void relu()  {
        const size_t N = size();
        if (N == 0) return;
#ifdef HAVE_CUDA
        if (device_valid) {
            ensure_device();
            gpu_relu_device(d_data, N);
            mark_device_modified();
            return;
        }
        ensure_host();
#endif
        float * __restrict__ a = data.data();


        #if HAS_OPENMP
        if (N > ZMATRIX_PARALLEL_THRESHOLD) {
#pragma omp parallel for simd schedule(static)
            for(size_t i = 0; i < N; ++i) {
                a[i] = std::max(0.0f, a[i]);
            }
        } else {
            zmatrix_simd::relu_f32(a, N);
        }
        #else
        zmatrix_simd::relu_f32(a, N);
        #endif
#ifdef HAVE_CUDA
        mark_host_modified();
#endif

    }


    void relu_derivative() {
        const size_t N = size();
        if (N == 0) return;

#ifdef HAVE_CUDA
        ensure_host();
#endif
        float* __restrict__ a = data.data();

        #if HAS_OPENMP
        if (N > ZMATRIX_PARALLEL_THRESHOLD) {
#pragma omp parallel for simd schedule(static)
            for (size_t i = 0; i < N; ++i) {
                a[i] = (a[i] > 0.0f) ? 1.0f : 0.0f;
            }
        } else {
            zmatrix_simd::relu_derivative_f32(a, N);
        }
        #else
        zmatrix_simd::relu_derivative_f32(a, N);
        #endif
#ifdef HAVE_CUDA
        mark_host_modified();
#endif
    }



    void tanh()  {
        const size_t N = size();
        if (N == 0) return;
#ifdef HAVE_CUDA
        if (device_valid) {
            ensure_device();
            gpu_tanh_device(d_data, N);
            mark_device_modified();
            return;
        }
        ensure_host();
#endif
        float * __restrict__ a = data.data();

        #if HAS_OPENMP
        if (N > ZMATRIX_PARALLEL_THRESHOLD) {
             #pragma omp parallel for simd schedule(static)
             for(size_t i = 0; i < N; ++i) {
                a[i] = std::tanh(a[i]);
            }
        } else { // Loop sequencial se pequeno
             for(size_t i = 0; i < N; ++i) {
                a[i] = std::tanh(a[i]);
            }
        }
        #else // Loop sequencial se não houver OpenMP
         for(size_t i = 0; i < N; ++i) {
            a[i] = std::tanh(a[i]);
        }
        #endif
#ifdef HAVE_CUDA
        mark_host_modified();
#endif
    }

    void tanh_derivative() {
        const size_t N = size();
        if (N == 0) return;

#ifdef HAVE_CUDA
        ensure_host();
#endif
        float* __restrict__ a = data.data();

        #if HAS_OPENMP
        if (N > ZMATRIX_PARALLEL_THRESHOLD) {
#pragma omp parallel for simd schedule(static)
            for (size_t i = 0; i < N; ++i) {
                float t = a[i];
                a[i] = 1.0f - t * t;
            }
        } else
        #endif
        {
            for (size_t i = 0; i < N; ++i) {
                float t = a[i];
                a[i] = 1.0f - t * t;
            }
        }
#ifdef HAVE_CUDA
        mark_host_modified();
#endif
    }

    void leaky_relu(float alpha = 0.01f) {
        const size_t N = size();
        if (N == 0) return;

#ifdef HAVE_CUDA
        if (device_valid) {
            ensure_device();
            gpu_leaky_relu_device(d_data, alpha, N);
            mark_device_modified();
            return;
        }
        ensure_host();
#endif
        float* __restrict__ a = data.data();

        #if HAS_OPENMP
        if (N > ZMATRIX_PARALLEL_THRESHOLD) {
#pragma omp parallel for simd schedule(static)
            for (size_t i = 0; i < N; ++i) {
                a[i] = (a[i] > 0.0f) ? a[i] : alpha * a[i];
            }
        } else {
            zmatrix_simd::leaky_relu_f32(a, alpha, N);
        }
        #else
        zmatrix_simd::leaky_relu_f32(a, alpha, N);
        #endif
#ifdef HAVE_CUDA
        mark_host_modified();
#endif
    }


    void leaky_relu_derivative(float alpha = 0.01f) {
        const size_t N = size();
        if (N == 0) return;

#ifdef HAVE_CUDA
        if (device_valid) {
            ensure_device();
            gpu_leaky_relu_derivative_device(d_data, alpha, N);
            mark_device_modified();
            return;
        }
        ensure_host();
#endif
        float* __restrict__ a = data.data();

        #if HAS_OPENMP
        if (N > ZMATRIX_PARALLEL_THRESHOLD) {
#pragma omp parallel for simd schedule(static)
            for (size_t i = 0; i < N; ++i) {
                a[i] = (a[i] > 0.0f) ? 1.0f : alpha;
            }
        } else {
            zmatrix_simd::leaky_relu_derivative_f32(a, alpha, N);
        }
        #else
        zmatrix_simd::leaky_relu_derivative_f32(a, alpha, N);
        #endif
#ifdef HAVE_CUDA
        mark_host_modified();
#endif
    }


    void softmax() {
        if (shape.empty() || (shape.size() != 1 && shape.size() != 2)) {
            throw std::runtime_error("Softmax requires 1D or 2D tensor");
        }
        if (size() == 0) return;

#ifdef HAVE_CUDA
        if (device_valid) {
            const size_t rows = shape.size() == 1 ? 1 : shape[0];
            const size_t cols = shape.size() == 1 ? shape[0] : shape[1];
            gpu_softmax_device(d_data, rows, cols, shape.size() == 1 ? 1 : 0);
            mark_device_modified();
            return;
        }
        ensure_host();
#endif
        float* __restrict__ a = data.data();

        if (shape.size() == 1) {
            // Versão 1D
            const size_t N = shape[0];
            float maxval = a[0];
            for (size_t i = 1; i < N; ++i)
                maxval = std::max(maxval, a[i]);

            float sum = 0.0f;
            for (size_t i = 0; i < N; ++i) {
                a[i] = expf(a[i] - maxval);
                sum += a[i];
            }

            if (sum != 0.0f) {
                float inv = 1.0f / sum;
                for (size_t i = 0; i < N; ++i)
                    a[i] *= inv;
            }
#ifdef HAVE_CUDA
            mark_host_modified();
#endif
            return;
        }

        // Versão 2D: linha a linha
        size_t N = shape[0], C = shape[1];
        for (size_t i = 0; i < N; ++i) {
            float* row = a + i * C;

            float maxval = row[0];
            for (size_t j = 1; j < C; ++j)
                maxval = std::max(maxval, row[j]);

            float sum = 0.0f;
            for (size_t j = 0; j < C; ++j) {
                row[j] = expf(row[j] - maxval);
                sum += row[j];
            }

            if (sum == 0.0f || !std::isfinite(sum)) {
                float uniform = 1.0f / C;
                for (size_t j = 0; j < C; ++j)
                    row[j] = uniform;
            } else {
                float inv = 1.0f / sum;
                for (size_t j = 0; j < C; ++j)
                    row[j] *= inv;
            }
        }
#ifdef HAVE_CUDA
        mark_host_modified();
#endif
    }


    void softmax_derivative() {
        const size_t N = size();
        if (N == 0) return;

#ifdef HAVE_CUDA
        if (device_valid) {
            gpu_softmax_derivative_device(d_data, N);
            mark_device_modified();
            return;
        }
        ensure_host();
#endif
        float* __restrict__ a = data.data();

        // CUIDADO: isso zera tudo se usado direto — normalmente usamos softmax + cross-entropy juntos
        for (size_t i = 0; i < N; ++i) {
            float si = a[i];
            a[i] = si * (1.0f - si);  // diagonal da jacobiana
        }
#ifdef HAVE_CUDA
        mark_host_modified();
#endif
    }


    // --- New Mathematical Operations ---
    void divide(const ZTensor& other) {
        if (!same_shape(other)) {
           throw std::invalid_argument(ZMATRIX_ERR_SHAPE_MISMATCH);
       }
       const size_t N = size();
       if (N == 0) return;
#ifdef HAVE_CUDA
       if (device_valid || other.device_valid) {
           ensure_device();
           other.ensure_device();
           gpu_div_device(d_data, other.d_data, N);
           mark_device_modified();
           return;
       }
       ensure_host();
       other.ensure_host();
#endif
       float * __restrict__ a = data.data();
       const float * __restrict__ b = other.data.data();

       #if HAS_OPENMP
       if (N > ZMATRIX_PARALLEL_THRESHOLD) {
           std::atomic<bool> error{false};
           #pragma omp parallel for simd schedule(static)
           for (size_t i = 0; i < N; ++i) {
               if (b[i] == 0.0f) {
                   error.store(true, std::memory_order_relaxed);
               } else {
                   a[i] /= b[i];
               }
           }
           if (error.load()) {
               throw std::runtime_error("Divisão por zero detectada");
           }
       } else { // Loop sequencial se pequeno
           for (size_t i = 0; i < N; ++i) {
               if (b[i] == 0.0f) {
                   throw std::runtime_error("Divisão por zero detectada");
               }
               a[i] /= b[i];
           }
       }
       #else // Loop sequencial se não houver OpenMP
       for (size_t i = 0; i < N; ++i) {
           if (b[i] == 0.0f) {
               throw std::runtime_error("Divisão por zero detectada");
           }
           a[i] /= b[i];
       }
       #endif

#ifdef HAVE_CUDA
       mark_host_modified();
#endif
    }

     void pow(float exponent) {
        const size_t N = size();
        if (N == 0) return;
#ifdef HAVE_CUDA
        if (device_valid) {
            gpu_pow_device(d_data, exponent, N);
            mark_device_modified();
            return;
        }
        ensure_host();
#endif
        float * __restrict__ a = data.data();

        #if HAS_OPENMP
        if (N > ZMATRIX_PARALLEL_THRESHOLD) {
            #pragma omp parallel for simd schedule(static)
            for (size_t i = 0; i < N; ++i) {
                a[i] = std::pow(a[i], exponent);
            }
        } else { // Loop sequencial se pequeno
            for (size_t i = 0; i < N; ++i) {
                a[i] = std::pow(a[i], exponent);
            }
        }
        #else // Loop sequencial se não houver OpenMP
        for (size_t i = 0; i < N; ++i) {
            a[i] = std::pow(a[i], exponent);
        }
        #endif
#ifdef HAVE_CUDA
        mark_host_modified();
#endif
    }

    void exp() {
        const size_t N = size();
        if (N == 0) return;
#ifdef HAVE_CUDA
        if (device_valid) {
            ensure_device();
            gpu_exp_device(d_data, N);
            mark_device_modified();
            return;
        }
        ensure_host();
#endif
        float * __restrict__ a = data.data();
        #if HAS_OPENMP
        if (N > ZMATRIX_PARALLEL_THRESHOLD) {
           #pragma omp parallel for simd schedule(static)
            for (size_t i = 0; i < N; ++i) {
                a[i] = expf(a[i]);
            }
        } else { // Loop sequencial se pequeno
            for (size_t i = 0; i < N; ++i) {
                a[i] = expf(a[i]);
            }
        }
        #else // Loop sequencial se não houver OpenMP
        for (size_t i = 0; i < N; ++i) {
            a[i] = expf(a[i]);
        }
        #endif
#ifdef HAVE_CUDA
        mark_host_modified();
#endif
    }

    void log() { // Retorno void, não é const pois modifica o objeto
        const size_t total_size = this->size();
        if (total_size == 0) return;

#ifdef HAVE_CUDA
        if (device_valid) {
            gpu_log_device(d_data, total_size);
            mark_device_modified();
            return;
        }
        ensure_host();
#endif
        float* p_this = this->data.data(); // Ponteiro para os dados DO PRÓPRIO objeto

        // Loop de pré-verificação (serial, o que é bom para lançar exceção antes do paralelo)
        for (size_t i = 0; i < total_size; ++i) {
            if (p_this[i] <= 0.0f) {
                // Use sua macro de erro definida, se tiver uma específica para isso,
                // ou mantenha a mensagem detalhada.
                throw std::runtime_error("Logaritmo de valor não positivo.");
            }
        }

        // Aplicação da lógica de paralelismo com threshold
        #if HAS_OPENMP
        if (total_size > ZMATRIX_PARALLEL_THRESHOLD) { // Verifica o threshold
             // A cláusula simd pode ser benéfica aqui também.
             #pragma omp parallel for simd schedule(static)
             for (size_t i = 0; i < total_size; ++i) {
                 p_this[i] = std::log(p_this[i]); // std::log é sobrecarregado para float
             }
        } else { // Loop sequencial se o tensor for pequeno
             for (size_t i = 0; i < total_size; ++i) {
                p_this[i] = std::log(p_this[i]);
            }
        }
        #else // Loop sequencial se OpenMP não estiver disponível
            for (size_t i = 0; i < total_size; ++i) {
                p_this[i] = std::log(p_this[i]);
            }
        #endif
        // Sem return ZTensor, pois é uma operação in-place
#ifdef HAVE_CUDA
        mark_host_modified();
#endif
    }

    void sqrt() { // Retorno void, n??o ?? const
        const size_t total_size = size();
        if (total_size == 0) return;

#ifdef HAVE_CUDA
        if (device_valid) {
            gpu_sqrt_device(d_data, total_size);
            mark_device_modified();
            return;
        }
        ensure_host();
#endif
        float* p_this = this->data.data();

        for (size_t i = 0; i < total_size; ++i) {
            if (p_this[i] < 0.0f) {
                throw std::runtime_error("Raiz quadrada de valor negativo."); // Ou sua ZMATRIX_ERR_INVALID_VALUE
            }
        }

        #if HAS_OPENMP
        if (total_size > ZMATRIX_PARALLEL_THRESHOLD) {
             #pragma omp parallel for schedule(static)
              for (size_t i = 0; i < total_size; ++i) {
                 p_this[i] = std::sqrt(p_this[i]);
             }
        } else {
             zmatrix_simd::sqrt_f32(p_this, total_size);
        }
        #else
        zmatrix_simd::sqrt_f32(p_this, total_size);
        #endif
#ifdef HAVE_CUDA
        mark_host_modified();
#endif

    }

    // --- Reduções (float input, double accumulator) ---
      double sum() const {
          const size_t N = size();
          if (N == 0) return 0.0;

#ifdef HAVE_CUDA
          if (device_valid) return static_cast<double>(gpu_sum_value_device(d_data, N));
          ensure_host();
#endif
          const float* a = data.data();

          #if HAS_AVX2
          if (N <= ZMATRIX_PARALLEL_THRESHOLD) {
              return zmatrix_simd::sum_f32(a, N);
          }
          #endif

          double total_sum = 0.0;

          #if HAS_OPENMP
          if (N > ZMATRIX_PARALLEL_THRESHOLD) {
              #pragma omp parallel for reduction(+:total_sum) schedule(static)
              for (size_t i = 0; i < N; ++i) {
                  total_sum += a[i];
              }
          } else {
              for (size_t i = 0; i < N; ++i) {
                  total_sum += a[i];
              }
          }
          #else
          for (size_t i = 0; i < N; ++i) {
              total_sum += a[i];
          }
          #endif

          return total_sum;
      }

     double mean() const { // Retorna double
         const size_t N = this->size();
         if (N == 0) return std::numeric_limits<double>::quiet_NaN();
         return sum() / static_cast<double>(N) ;
     }

     float min() const {
         const size_t N = size();
         if (N == 0) return std::numeric_limits<float>::quiet_NaN();
#ifdef HAVE_CUDA
         if (device_valid) return gpu_min_value_device(d_data, N);
         ensure_host();
#endif
         const float* p = data.data();

         #if HAS_AVX2
         if (N <= ZMATRIX_PARALLEL_THRESHOLD) {
             return zmatrix_simd::min_f32(p, N);
         }
         #endif

         float m = p[0];

         #if HAS_OPENMP
          if (N > ZMATRIX_PARALLEL_THRESHOLD) {
               #pragma omp parallel for reduction(min:m) schedule(static)
               for (size_t i = 1; i < N; ++i) {
                   if (p[i] < m) m = p[i];
               }
           } else {
                 for (size_t i = 1; i < N; ++i) {
                     if (p[i] < m) m = p[i];
                 }
           }
          #else
              for (size_t i = 1; i < N; ++i) {
                if (p[i] < m) m = p[i];
              }
          #endif

         return m;
     }

     float max() const {
         const size_t N = size();
         if (N == 0) return std::numeric_limits<float>::quiet_NaN();
#ifdef HAVE_CUDA
         if (device_valid) return gpu_max_value_device(d_data, N);
         ensure_host();
#endif
         const float* p = data.data();

         #if HAS_AVX2
         if (N <= ZMATRIX_PARALLEL_THRESHOLD) {
             return zmatrix_simd::max_f32(p, N);
         }
         #endif

         float M = p[0];

         #if HAS_OPENMP
          if (N > ZMATRIX_PARALLEL_THRESHOLD) {
               #pragma omp parallel for reduction(max:M) schedule(static)
               for (size_t i = 1; i < N; ++i) {
                   if (p[i] > M) M = p[i];
               }
           } else {
                 for (size_t i = 1; i < N; ++i) {
                    if (p[i] > M) M = p[i];
                 }
           }
          #else
              for (size_t i = 1; i < N; ++i) {
                 if (p[i] > M) M = p[i];
              }
          #endif
         return M;
     }

     double std() const {
         const size_t N = size();
         if (N < 2) return std::numeric_limits<double>::quiet_NaN();
         double m = mean();  // já chama sum() otimizado
#ifdef HAVE_CUDA
         ensure_host();
#endif
         const float* p = data.data();
         double sq = 0.0;

          #if HAS_OPENMP
           if (N > ZMATRIX_PARALLEL_THRESHOLD) {
               #pragma omp parallel for reduction(+:sq) schedule(static)
               for (size_t i = 0; i < N; ++i) {
                   double d = static_cast<double>(p[i]) - m;
                   sq += d * d;
               }
           } else {
               for (size_t i = 0; i < N; ++i) {
                    double d = static_cast<double>(p[i]) - m;
                    sq += d * d;
               }
           }
          #else
             for (size_t i = 0; i < N; ++i) {
                double d = static_cast<double>(p[i]) - m;
                sq += d * d;
             }
          #endif

         return std::sqrt(sq / (static_cast<double>(N) - 1.0));
     }


    static inline uint64_t xorshift64star(uint64_t& state) {
        state ^= state >> 12;
        state ^= state << 25;
        state ^= state >> 27;
        return state * UINT64_C(2685821657736338717);
    }

    static ZTensor random(const std::vector<size_t>& shape, float min = 0.0f, float max = 1.0f) {
        if (shape.empty()) {
            throw std::invalid_argument("Shape não pode ser vazio.");
        }

        size_t total_size = 1;
        for (size_t dim : shape) {
            if (dim < 0) {
                throw std::invalid_argument("Dimensões devem ser não negativas.");
            }
            total_size *= dim;
        }

        if (min > max) {
            throw std::invalid_argument("min não pode ser maior que max.");
        }

        ZTensor result;
        result.shape = shape;
        result.strides.resize(shape.size());

        // Row-major strides
        size_t stride = 1;
        for (int i = static_cast<int>(shape.size()) - 1; i >= 0; --i) {
            result.strides[i] = stride;
            stride *= shape[i];
        }

        result.data.resize(total_size);
        if (total_size == 0) {
            return result;
        }

        float* data = result.data.data();
        const float range = max - min;
        const double scale = 1.0 / static_cast<double>(std::numeric_limits<uint64_t>::max());
        std::random_device rd;
        uint64_t base_seed = rd() ^ (static_cast<uint64_t>(
            std::chrono::high_resolution_clock::now().time_since_epoch().count()));

        #if HAS_OPENMP
        if (total_size > ZMATRIX_PARALLEL_THRESHOLD) {
            #pragma omp parallel
            {
                int thread_id = omp_get_thread_num();
                uint64_t local_state = base_seed ^ (thread_id * 0x9E3779B97F4A7C15);

                #pragma omp for schedule(static)
                for (size_t i = 0; i < total_size; ++i) {
                    uint64_t r = xorshift64star(local_state);
                    float r01 = static_cast<float>(r * scale);
                    data[i] = min + r01 * range;
                }
            }
        } else
        #endif
        {
            uint64_t state = base_seed;
            for (size_t i = 0; i < total_size; ++i) {
                uint64_t r = xorshift64star(state);
                float r01 = static_cast<float>(r * scale);
                data[i] = min + r01 * range;
            }
        }

        return result;
    }


     /**
      * Cria um tensor preenchido com zeros
      * @param shape Forma do tensor
      * @return ZTensor preenchido com zeros
      */
     static ZTensor zeros(const std::vector<size_t>& shape) {
         if (shape.empty()) {
             throw std::invalid_argument("Shape não pode ser vazio.");
         }

         size_t total_size = 1;
         for (size_t dim : shape) {
             if (dim == 0) return ZTensor(shape);  // tensor vazio
             total_size *= dim;
         }

         ZTensor result;
         result.shape = shape;
         result.strides.resize(shape.size());

         // Calcula strides (row-major)
         size_t stride = 1;
         for (int i = static_cast<int>(shape.size()) - 1; i >= 0; --i) {
             result.strides[i] = stride;
             stride *= shape[i];
         }

         result.data.resize(total_size, 0.0f); // já zera

         return result;
     }




      /**
       * Cria um tensor preenchido com um valor específico
       * @param shape Forma do tensor
       * @param value Valor para preencher o tensor
       * @return ZTensor preenchido com o valor especificado
       */
      static ZTensor full(const std::vector<size_t>& shape, float value) {
          if (shape.empty()) {
              throw std::invalid_argument("Shape não pode ser vazio.");
          }

          size_t total_size = 1;
          for (size_t dim : shape) {
              if (dim < 0) {
                  throw std::invalid_argument("Dimensões devem ser não negativas.");
              }
              total_size *= dim;
          }

          ZTensor result;
          result.shape = shape;
          result.strides.resize(shape.size());

          // Calcula strides (row-major)
          size_t stride = 1;
          for (int i = static_cast<int>(shape.size()) - 1; i >= 0; --i) {
              result.strides[i] = stride;
              stride *= shape[i];
          }

          result.data.resize(total_size);

          #if HAS_OPENMP
          if (total_size > ZMATRIX_PARALLEL_THRESHOLD) {
              float* data = result.data.data();
              #pragma omp parallel for schedule(static)
              for (size_t i = 0; i < total_size; ++i) {
                  data[i] = value;
              }
          } else
          #endif
          {
              std::fill(result.data.begin(), result.data.end(), value);
          }

          return result;
      }


     /**
       * Cria uma matriz identidade quadrada de tamanho especificado
       * @param s Dimensão da matriz quadrada (renomeado de 'size' para 's')
       * @return Matriz identidade como ZTensor
       */
      static ZTensor identity(size_t s) {
          if (s == 0) {
              return ZTensor({0, 0});
          }

          const size_t total_size = s * s;

          ZTensor result;
          result.shape = {s, s};
          result.strides = {s, 1};
          result.data.resize(total_size, 0.0f); // já zera

          float* data = result.data.data();

          #if HAS_OPENMP
          if (s > ZMATRIX_PARALLEL_THRESHOLD) {
              #pragma omp parallel for schedule(static)
              for (size_t i = 0; i < s; ++i) {
                  data[i * s + i] = 1.0f;
              }
          } else
          #endif
          {
              for (size_t i = 0; i < s; ++i) {
                  data[i * s + i] = 1.0f;
              }
          }

          return result;
      }

    // ========== AUTOGRAD OPERATIONS (OUT-OF-PLACE) ==========

    /**
     * Element-wise addition with autograd support
     * result = a + b
     * Gradients: da = grad_output, db = grad_output
     */
    static ZTensor add_autograd(const ZTensor& a, const ZTensor& b) {
        if (a.shape != b.shape) {
            throw std::invalid_argument(ZMATRIX_ERR_SHAPE_MISMATCH);
        }

#ifdef HAVE_CUDA
        // FIX: as operações de autograd não sincronizavam com a GPU; se 'a'
        // ou 'b' estivessem residentes só no device, o cálculo lia memória
        // de host desatualizada e corrompia o resultado (e o grafo de
        // gradiente) silenciosamente. Aqui garantimos consistência antes
        // de qualquer leitura de a.data/b.data.
        a.ensure_host();
        b.ensure_host();
#endif

        ZTensor result(a.shape);
        const size_t N = a.size();

        if (N > 0) {
            const float* a_data = a.data.data();
            const float* b_data = b.data.data();
            float* r_data = result.data.data();

#if HAS_OPENMP
            if (N > ZMATRIX_PARALLEL_THRESHOLD) {
#pragma omp parallel for simd schedule(static)
                for (size_t i = 0; i < N; ++i) {
                    r_data[i] = a_data[i] + b_data[i];
                }
            } else {
                for (size_t i = 0; i < N; ++i) {
                    r_data[i] = a_data[i] + b_data[i];
                }
            }
#else
            for (size_t i = 0; i < N; ++i) {
                r_data[i] = a_data[i] + b_data[i];
            }
#endif
        }

        // Autograd: cria nó se algum operando requer gradiente
        bool requires_grad = a.requires_grad || b.requires_grad;
        result.requires_grad = requires_grad;

        if (requires_grad) {
            auto node = std::make_shared<AutogradNode>("add");

            // Armazena pointers RAW aos operandos
            ZTensor* a_ptr_raw = const_cast<ZTensor*>(&a);
            ZTensor* b_ptr_raw = const_cast<ZTensor*>(&b);
            node->parents_raw = {a_ptr_raw, b_ptr_raw};

            // Backward function para add
            // dy/da = 1, dy/db = 1
            bool a_req = a.requires_grad;
            bool b_req = b.requires_grad;

            node->backward_fn = [node, a_ptr_raw, b_ptr_raw, a_req, b_req]() {
                // Obtém o gradiente do tensor resultado
                ZTensor* result_raw = node->result_ptr_raw;
                if (!result_raw) {
                    auto result_shared = node->result_tensor.lock();
                    if (!result_shared) return;
                    result_raw = result_shared.get();
                }

                const ZTensor* grad_result = result_raw->getGrad();
                if (!grad_result) return;  // Nada a fazer

                // Para add: ambos os pais recebem o mesmo gradiente
                if (a_req) {
                    a_ptr_raw->accumulate_grad(*grad_result);
                }
                if (b_req) {
                    b_ptr_raw->accumulate_grad(*grad_result);
                }
            };

            result.grad_fn = node;

            // Armazena resultado para que backward possa acessá-lo
            auto result_ptr = std::make_shared<ZTensor>(result);
            node->result_tensor = result_ptr;
        }

        return result;
    }

    /**
     * Element-wise subtraction with autograd support
     * result = a - b
     * Gradients: da = grad_output, db = -grad_output
     */
    static ZTensor sub_autograd(const ZTensor& a, const ZTensor& b) {
        if (a.shape != b.shape) {
            throw std::invalid_argument(ZMATRIX_ERR_SHAPE_MISMATCH);
        }

#ifdef HAVE_CUDA
        // FIX: sincroniza com o host antes de ler os dados (ver add_autograd).
        a.ensure_host();
        b.ensure_host();
#endif

        ZTensor result(a.shape);
        const size_t N = a.size();

        if (N > 0) {
            const float* a_data = a.data.data();
            const float* b_data = b.data.data();
            float* r_data = result.data.data();

#if HAS_OPENMP
            if (N > ZMATRIX_PARALLEL_THRESHOLD) {
#pragma omp parallel for simd schedule(static)
                for (size_t i = 0; i < N; ++i) {
                    r_data[i] = a_data[i] - b_data[i];
                }
            } else {
                for (size_t i = 0; i < N; ++i) {
                    r_data[i] = a_data[i] - b_data[i];
                }
            }
#else
            for (size_t i = 0; i < N; ++i) {
                r_data[i] = a_data[i] - b_data[i];
            }
#endif
        }

        // Autograd
        bool requires_grad = a.requires_grad || b.requires_grad;
        result.requires_grad = requires_grad;

        if (requires_grad) {
            auto node = std::make_shared<AutogradNode>("sub");

            // Armazena pointers RAW aos operandos
            ZTensor* a_ptr_raw = const_cast<ZTensor*>(&a);
            ZTensor* b_ptr_raw = const_cast<ZTensor*>(&b);
            node->parents_raw = {a_ptr_raw, b_ptr_raw};

            bool a_req = a.requires_grad;
            bool b_req = b.requires_grad;

            node->backward_fn = [node, a_ptr_raw, b_ptr_raw, a_req, b_req]() {
                ZTensor* result_raw = node->result_ptr_raw;
                if (!result_raw) {
                    auto result_shared = node->result_tensor.lock();
                    if (!result_shared) return;
                    result_raw = result_shared.get();
                }

                const ZTensor* grad_result = result_raw->getGrad();
                if (!grad_result) return;
#ifdef HAVE_CUDA
                grad_result->ensure_host();
#endif

                if (a_req) {
                    a_ptr_raw->accumulate_grad(*grad_result);
                }
                if (b_req) {
                    // Para sub: gradiente em b é negado
                    ZTensor neg_grad(grad_result->shape);
                    const size_t N = grad_result->size();
                    if (N > 0) {
                        const float* src = grad_result->data.data();
                        float* dst = neg_grad.data.data();
                        for (size_t i = 0; i < N; ++i) {
                            dst[i] = -src[i];
                        }
                    }
                    b_ptr_raw->accumulate_grad(neg_grad);
                }
            };

            result.grad_fn = node;

            // Armazena resultado
            auto result_ptr = std::make_shared<ZTensor>(result);
            node->result_tensor = result_ptr;
        }

        return result;
    }

    /**
     * Element-wise multiplication with autograd support
     * result = a * b
     * Gradients: da = b * grad_output, db = a * grad_output
     */
    static ZTensor mul_autograd(const ZTensor& a, const ZTensor& b) {
        if (a.shape != b.shape) {
            throw std::invalid_argument(ZMATRIX_ERR_SHAPE_MISMATCH);
        }

#ifdef HAVE_CUDA
        // FIX: sincroniza com o host antes de ler os dados (ver add_autograd).
        a.ensure_host();
        b.ensure_host();
#endif

        ZTensor result(a.shape);
        const size_t N = a.size();

        if (N > 0) {
            const float* a_data = a.data.data();
            const float* b_data = b.data.data();
            float* r_data = result.data.data();

#if HAS_OPENMP
            if (N > ZMATRIX_PARALLEL_THRESHOLD) {
#pragma omp parallel for simd schedule(static)
                for (size_t i = 0; i < N; ++i) {
                    r_data[i] = a_data[i] * b_data[i];
                }
            } else {
                for (size_t i = 0; i < N; ++i) {
                    r_data[i] = a_data[i] * b_data[i];
                }
            }
#else
            for (size_t i = 0; i < N; ++i) {
                r_data[i] = a_data[i] * b_data[i];
            }
#endif
        }

        // Autograd
        bool requires_grad = a.requires_grad || b.requires_grad;
        result.requires_grad = requires_grad;

        if (requires_grad) {
            auto node = std::make_shared<AutogradNode>("mul");

            // Armazena pointers RAW aos operandos
            ZTensor* a_ptr_raw = const_cast<ZTensor*>(&a);
            ZTensor* b_ptr_raw = const_cast<ZTensor*>(&b);
            node->parents_raw = {a_ptr_raw, b_ptr_raw};

            bool a_req = a.requires_grad;
            bool b_req = b.requires_grad;

            // Armazena cópias dos valores para usar no backward
            auto a_copy = std::make_shared<ZTensor>(a);
            auto b_copy = std::make_shared<ZTensor>(b);

            node->backward_fn = [node, a_ptr_raw, b_ptr_raw, a_copy, b_copy, a_req, b_req]() {
                ZTensor* result_raw = node->result_ptr_raw;
                if (!result_raw) {
                    auto result_shared = node->result_tensor.lock();
                    if (!result_shared) return;
                    result_raw = result_shared.get();
                }

                const ZTensor* grad_result = result_raw->getGrad();
                if (!grad_result) return;
#ifdef HAVE_CUDA
                grad_result->ensure_host();
                a_copy->ensure_host();
                b_copy->ensure_host();
#endif

                const size_t N = grad_result->size();

                // da = b * grad_output
                if (a_req && N > 0) {
                    ZTensor grad_a(grad_result->shape);
                    const float* b_data = b_copy->data.data();
                    const float* grad_data = grad_result->data.data();
                    float* grad_a_data = grad_a.data.data();

                    for (size_t i = 0; i < N; ++i) {
                        grad_a_data[i] = b_data[i] * grad_data[i];
                    }
                    a_ptr_raw->accumulate_grad(grad_a);
                }

                // db = a * grad_output
                if (b_req && N > 0) {
                    ZTensor grad_b(grad_result->shape);
                    const float* a_data = a_copy->data.data();
                    const float* grad_data = grad_result->data.data();
                    float* grad_b_data = grad_b.data.data();

                    for (size_t i = 0; i < N; ++i) {
                        grad_b_data[i] = a_data[i] * grad_data[i];
                    }
                    b_ptr_raw->accumulate_grad(grad_b);
                }
            };

            result.grad_fn = node;

            // Armazena resultado
            auto result_ptr = std::make_shared<ZTensor>(result);
            node->result_tensor = result_ptr;
        }

        return result;
    }

    /**
     * Sum reduction with autograd support
     * result = sum(tensor)  -> escalar
     * Backward: grad_input[i] = grad_output[0] para cada elemento i
     */
    static ZTensor sum_autograd(const ZTensor& t) {
#ifdef HAVE_CUDA
        // FIX: sincroniza com o host antes de ler os dados (ver add_autograd).
        t.ensure_host();
#endif

        const size_t N = t.size();
        double total = 0.0;

        if (N > 0) {
            const float* data = t.data.data();
#if HAS_OPENMP
            if (N > ZMATRIX_PARALLEL_THRESHOLD) {
#pragma omp parallel for reduction(+:total) schedule(static)
                for (size_t i = 0; i < N; ++i) {
                    total += data[i];
                }
            } else {
                for (size_t i = 0; i < N; ++i) {
                    total += data[i];
                }
            }
#else
            for (size_t i = 0; i < N; ++i) {
                total += data[i];
            }
#endif
        }

        // Resultado é escalar
        ZTensor result({1});
        result.data[0] = static_cast<float>(total);

        // Autograd
        result.requires_grad = t.requires_grad;

        if (t.requires_grad) {
            auto node = std::make_shared<AutogradNode>("sum");

            // Armazena pointer RAW ao operando
            ZTensor* t_ptr_raw = const_cast<ZTensor*>(&t);
            node->parents_raw = {t_ptr_raw};

            // Armazena shape original para broadcast no backward
            auto input_shape = t.shape;
            auto input_size = t.size();

            node->backward_fn = [node, t_ptr_raw, input_shape, input_size]() {
                ZTensor* result_raw = node->result_ptr_raw;
                if (!result_raw) {
                    auto result_shared = node->result_tensor.lock();
                    if (!result_shared) return;
                    result_raw = result_shared.get();
                }

                const ZTensor* grad_result = result_raw->getGrad();
                if (!grad_result) return;
#ifdef HAVE_CUDA
                grad_result->ensure_host();
#endif

                // grad_input[i] = grad_result[0] para cada i
                float grad_val = grad_result->data[0];

                ZTensor grad_input(input_shape);
                if (input_size > 0) {
                    float* grad_data = grad_input.data.data();
                    for (size_t i = 0; i < input_size; ++i) {
                        grad_data[i] = grad_val;
                    }
                }

                t_ptr_raw->accumulate_grad(grad_input);
            };

            result.grad_fn = node;

            // Armazena resultado
            auto result_ptr = std::make_shared<ZTensor>(result);
            node->result_tensor = result_ptr;
        }

        return result;
    }

};
// --- Fim da Definição de ZTensor ---


// Declarações globais das entradas de classe
zend_class_entry *zmatrix_ce;
zend_class_entry *zmatrix_ce_ZTensor;

// Declaração dos manipuladores de objeto
zend_object_handlers zmatrix_ztensor_handlers;

// Estrutura do objeto PHP que encapsula o ZTensor C++
typedef struct _zmatrix_ztensor_object {
    ZTensor *tensor;
    zend_object std;
} zmatrix_ztensor_object;

// --- Definição da Macro Auxiliar ---
static inline zmatrix_ztensor_object *zmatrix_ztensor_from_obj(zend_object *obj) {
    return (zmatrix_ztensor_object*)((char*)(obj) - offsetof(zmatrix_ztensor_object, std));
}
#define Z_MATRIX_ZTENSOR_P(zv) zmatrix_ztensor_from_obj(Z_OBJ_P(zv))
// --- Fim da Macro Auxiliar ---

// Função de criação do objeto PHP ZTensor
zend_object *zmatrix_ztensor_create(zend_class_entry *class_type)
{
    zmatrix_ztensor_object *intern = (zmatrix_ztensor_object *) zend_object_alloc(sizeof(zmatrix_ztensor_object), class_type);
    zend_object_std_init(&intern->std, class_type);
    object_properties_init(&intern->std, class_type);
    intern->tensor = nullptr;
    intern->std.handlers = &zmatrix_ztensor_handlers;
    return &intern->std;
}

// Função para liberar a memória do objeto PHP ZTensor
void zmatrix_ztensor_free(zend_object *object)
{
    zmatrix_ztensor_object *intern = zmatrix_ztensor_from_obj(object);
   if (intern->tensor) {
       if (intern->tensor->owns_data) {
           delete intern->tensor;
       }
       intern->tensor = nullptr;
   }
    zend_object_std_dtor(&intern->std);
}

zend_object *zmatrix_ztensor_clone(zend_object *old_object)
{
    zend_object *new_object = zmatrix_ztensor_create(old_object->ce);
    zmatrix_ztensor_object *old_intern = zmatrix_ztensor_from_obj(old_object);
    zmatrix_ztensor_object *new_intern = zmatrix_ztensor_from_obj(new_object);
    zend_objects_clone_members(new_object, old_object);
    if (!old_intern->tensor) return new_object;
    try {
        new_intern->tensor = new ZTensor(*old_intern->tensor);
    } catch (const std::exception& exception) {
        zend_throw_exception(zend_ce_exception, exception.what(), 0);
    } catch (...) {
        zend_throw_exception(zend_ce_exception, "Unknown error while cloning ZTensor", 0);
    }
    return new_object;
}


// --- Funções Auxiliares de Conversão ---
static void php_array_to_nd_shape(const zval *array, std::vector<size_t>& shape) { if (Z_TYPE_P(array) != IS_ARRAY) return; HashTable *ht = Z_ARRVAL_P(array); size_t count = zend_hash_num_elements(ht); shape.push_back(count); if (count > 0) { zval *first_elem = zend_hash_index_find(ht, 0); if (first_elem && Z_TYPE_P(first_elem) == IS_ARRAY) php_array_to_nd_shape(first_elem, shape); } }
static ZTensor php_array_to_tensor(const zval *array) {
    if (Z_TYPE_P(array) != IS_ARRAY) { throw std::invalid_argument(ZMATRIX_ERR_INVALID_ARG_TYPE); }
    std::vector<size_t> shape;
    php_array_to_nd_shape(array, shape);
    ZTensor tensor(shape); // Cria tensor (float)
    if (tensor.size() == 0) return tensor;

    std::function<void(const zval*, std::vector<size_t>&, size_t)> fill_tensor;
    fill_tensor =
        [&](const zval* current_array_zv, std::vector<size_t>& indices, size_t depth) {
        HashTable *ht = Z_ARRVAL_P(current_array_zv);
        zval *val;
        size_t current_idx = 0;
        size_t expected_count = shape[depth];
        if (zend_hash_num_elements(ht) != expected_count) { throw std::runtime_error("Inconsistência no array"); }

        ZEND_HASH_FOREACH_VAL(ht, val) {
            indices[depth] = current_idx;
            if (depth == shape.size() - 1) {
                if (Z_TYPE_P(val) != IS_LONG && Z_TYPE_P(val) != IS_DOUBLE) { throw std::invalid_argument(ZMATRIX_ERR_INVALID_TYPE); }
                // --- MUDANÇA: Cast para float ao atribuir ---
                tensor.at(indices) = static_cast<float>(zval_get_double(val));
            } else {
                if (Z_TYPE_P(val) != IS_ARRAY) { throw std::runtime_error("Estrutura de array inválida"); }
                fill_tensor(val, indices, depth + 1);
            }
            current_idx++;
        } ZEND_HASH_FOREACH_END();
    };

    std::vector<size_t> current_indices(shape.size());
    try { fill_tensor(array, current_indices, 0); } catch (const std::exception& e) { throw; }
    return tensor;
}
static void tensor_to_php_array(const ZTensor& tensor, zval *return_value) {
    #ifdef HAVE_CUDA
    tensor.ensure_host();
    #endif
    if (tensor.shape.empty()) { array_init(return_value); return; }
    // Tratamento de tensor com dimensão 0 (inalterado)
    if (tensor.empty() && !tensor.shape.empty()) { /* ... */ }

    std::function<void(zval*, std::vector<size_t>&, size_t)> build_array;
    build_array =
        [&](zval* current_php_array, std::vector<size_t>& current_indices, size_t depth) {
        array_init_size(current_php_array, tensor.shape[depth]);
        for (size_t i = 0; i < tensor.shape[depth]; ++i) {
            current_indices[depth] = i;
            if (depth == tensor.shape.size() - 1) {
                try {
                    // --- MUDANÇA: Cast float para double ao retornar para PHP ---
                    add_next_index_double(current_php_array, static_cast<double>(tensor.at(current_indices)));
                } catch (const std::out_of_range& e) { add_next_index_null(current_php_array); }
            } else {
                zval nested_array;
                build_array(&nested_array, current_indices, depth + 1);
                add_next_index_zval(current_php_array, &nested_array);
            }
        }
    };
    std::vector<size_t> initial_indices(tensor.shape.size());
    build_array(return_value, initial_indices, 0);
}

/**
 * Obtém ponteiro para tensor a partir de um zval. Pode ser uma instância de ZTensor ou um array PHP.
 *
 * @param input_zv       Entrada zval (ZTensor ou array)
 * @param output_ptr     Saída: ponteiro para tensor (objeto ou temporário)
 * @param tmp_tensor     Saída temporária (precisa estar no escopo do chamador)
 * @param expected_ce    Classe esperada (ZTensor)
 * @return true se sucesso, false se erro
 */
static bool zmatrix_get_tensor_ptr(zval *input_zv, ZTensor* &output_ptr, ZTensor &tmp_tensor, zend_class_entry *expected_ce)
{
    if (Z_TYPE_P(input_zv) == IS_OBJECT && instanceof_function(Z_OBJCE_P(input_zv), expected_ce)) {
        zmatrix_ztensor_object *obj = Z_MATRIX_ZTENSOR_P(input_zv);
        if (!obj->tensor) {
            zend_throw_exception(zend_ce_exception, ZMATRIX_ERR_NOT_INITIALIZED, 0);
            return false;
        }
        output_ptr = obj->tensor;
        return true;
    } else if (Z_TYPE_P(input_zv) == IS_ARRAY) {
        try {
            tmp_tensor = php_array_to_tensor(input_zv);
            output_ptr = &tmp_tensor;
            return true;
        } catch (const std::exception& e) {
            zend_throw_exception(zend_ce_exception, e.what(), 0);
            return false;
        }
    } else {
        zend_throw_exception(zend_ce_exception, ZMATRIX_ERR_INVALID_ARG_TYPE, 0);
        return false;
    }
}


static void zmatrix_return_tensor_obj(
    const ZTensor& result_tensor,
    zval *return_value,
    zend_class_entry *tensor_ce
) {
    try {
        // Cria um objeto PHP novo; intern->tensor é nullptr aqui
        object_init_ex(return_value, tensor_ce);

        zmatrix_ztensor_object *intern = Z_MATRIX_ZTENSOR_P(return_value);
        if (UNEXPECTED(!intern)) {
            zend_throw_exception(zend_ce_exception, "Failed to initialize ZTensor object", 0);
            ZVAL_NULL(return_value);
            return;
        }

        // Não há tensor antigo para deletar (porque intern foi recém-criado)
        intern->tensor = new ZTensor(result_tensor);
        // Não precisa de owns_data

        // FIX (autograd): addAutograd/subAutograd/mulAutograd/sumAutograd criavam
        // um weak_ptr para uma cópia LOCAL que morria ao fim da função *_autograd,
        // e nunca preenchiam result_ptr_raw. No backward(), node->backward_fn()
        // encontrava a referência morta e retornava sem chamar accumulate_grad() —
        // getGrad() ficava null silenciosamente. Este é o único ponto que conhece
        // o objeto C++ definitivo exposto ao PHP.
        if (intern->tensor->grad_fn) {
            intern->tensor->grad_fn->result_ptr_raw = intern->tensor;
        }
    }
    catch (const std::bad_alloc& e) {
        zend_throw_exception(zend_ce_exception, ZMATRIX_ERR_ALLOC_FAILED, 0);
        ZVAL_NULL(return_value);
    }
    catch (const std::exception& e) {
        zend_throw_exception(zend_ce_exception, e.what(), 0);
        ZVAL_NULL(return_value);
    }
}

// --- Fim Funções Auxiliares ---
#include "zmatrix_methods.h"
// ==========================================================================
// Tabela de Métodos da Classe ZMatrix\ZTensor
// ==========================================================================
static const zend_function_entry zmatrix_ztensor_methods[] = {
    PHP_ME(ZTensor, __construct,      arginfo_ztensor_construct,   ZEND_ACC_PUBLIC | ZEND_ACC_CTOR)
    PHP_ME(ZTensor, __toString,       arginfo_ztensor___tostring, ZEND_ACC_PUBLIC)
    PHP_ME(ZTensor, clip,             arginfo_ztensor_static_clip_range, ZEND_ACC_PUBLIC | ZEND_ACC_STATIC)
    PHP_ME(ZTensor, sum,              arginfo_ztensor_sum_flex,    ZEND_ACC_PUBLIC)
    PHP_ME(ZTensor, add,              arginfo_ztensor_op_other,    ZEND_ACC_PUBLIC)
    PHP_ME(ZTensor, sub,              arginfo_ztensor_op_other,    ZEND_ACC_PUBLIC)
    PHP_ME(ZTensor, mul,              arginfo_ztensor_op_other,    ZEND_ACC_PUBLIC) // Element-wise
    PHP_ME(ZTensor, scalarMultiply,   arginfo_ztensor_op_scalar,   ZEND_ACC_PUBLIC)
    PHP_ME(ZTensor, transpose,        arginfo_ztensor_no_args,     ZEND_ACC_PUBLIC) // TODO: Add axes arg
    PHP_ME(ZTensor, abs,              arginfo_ztensor_no_args,     ZEND_ACC_PUBLIC)
    PHP_ME(ZTensor, sumtotal,         arginfo_ztensor_no_args,     ZEND_ACC_PUBLIC) // TODO: Add axis arg
    PHP_ME(ZTensor, mean,             arginfo_ztensor_no_args,     ZEND_ACC_PUBLIC) // TODO: Add axis arg
    PHP_ME(ZTensor, min,              arginfo_ztensor_no_args,     ZEND_ACC_PUBLIC) // TODO: Add axis arg
    PHP_ME(ZTensor, max,              arginfo_ztensor_no_args,     ZEND_ACC_PUBLIC) // TODO: Add axis arg
    PHP_ME(ZTensor, std,              arginfo_ztensor_no_args,     ZEND_ACC_PUBLIC) // TODO: Add axis arg
    PHP_ME(ZTensor, shape,            arginfo_ztensor_no_args,     ZEND_ACC_PUBLIC)
    PHP_ME(ZTensor, ndim,             arginfo_ztensor_no_args,     ZEND_ACC_PUBLIC)
    PHP_ME(ZTensor, size,             arginfo_ztensor_no_args,     ZEND_ACC_PUBLIC)
    PHP_ME(ZTensor, isEmpty,          arginfo_ztensor_no_args,     ZEND_ACC_PUBLIC)
    PHP_ME(ZTensor, toArray,          arginfo_ztensor_no_args,     ZEND_ACC_PUBLIC)
    // Métodos Estáticos
    PHP_ME(ZTensor, zeros,            arginfo_ztensor_static_shape,       ZEND_ACC_PUBLIC | ZEND_ACC_STATIC) // Corrigido ARG_INFO
    PHP_ME(ZTensor, ones,             arginfo_ztensor_ones,               ZEND_ACC_PUBLIC | ZEND_ACC_STATIC) // Corrigido ARG_INFO
    PHP_ME(ZTensor, full,             arginfo_ztensor_static_shape_value, ZEND_ACC_PUBLIC | ZEND_ACC_STATIC) // Novo método 'full'
    PHP_ME(ZTensor, identity,         arginfo_ztensor_static_identity,    ZEND_ACC_PUBLIC | ZEND_ACC_STATIC)
    PHP_ME(ZTensor, random,           arginfo_ztensor_static_random,      ZEND_ACC_PUBLIC | ZEND_ACC_STATIC)
    PHP_ME(ZTensor, matmul,           arginfo_ztensor_matmul,             ZEND_ACC_PUBLIC)
    PHP_ME(ZTensor, divide,           arginfo_ztensor_divide,             ZEND_ACC_PUBLIC)
    PHP_ME(ZTensor, pow,              arginfo_ztensor_pow,                ZEND_ACC_PUBLIC)
    PHP_ME(ZTensor, sigmoid,          arginfo_ztensor_no_args,     ZEND_ACC_PUBLIC)
    PHP_ME(ZTensor, sigmoidDerivative,  arginfo_ztensor_no_args,     ZEND_ACC_PUBLIC)
    PHP_ME(ZTensor, toGpu,            arginfo_ztensor_no_args,     ZEND_ACC_PUBLIC)
    PHP_ME(ZTensor, toCpu,            arginfo_ztensor_no_args,     ZEND_ACC_PUBLIC)
    PHP_ME(ZTensor, isOnGpu,          arginfo_ztensor_isOnGpu,     ZEND_ACC_PUBLIC)
    PHP_ME(ZTensor, freeDevice,      arginfo_ztensor_no_args,     ZEND_ACC_PUBLIC)
    PHP_ME(ZTensor, softmax,          arginfo_ztensor_softmax,            ZEND_ACC_PUBLIC)
    PHP_ME(ZTensor, softmaxDerivative,   arginfo_ztensor_no_args, ZEND_ACC_PUBLIC)
    PHP_ME(ZTensor, relu,             arginfo_ztensor_relu,               ZEND_ACC_PUBLIC)
    PHP_ME(ZTensor, tanh,             arginfo_ztensor_tanh,               ZEND_ACC_PUBLIC)
    PHP_ME(ZTensor, reluDerivative,      arginfo_ztensor_no_args, ZEND_ACC_PUBLIC)
    PHP_ME(ZTensor, tanhDerivative,      arginfo_ztensor_no_args, ZEND_ACC_PUBLIC)
    PHP_ME(ZTensor, leakyRelu,           arginfo_ztensor_optional_float, ZEND_ACC_PUBLIC)
    PHP_ME(ZTensor, leakyReluDerivative, arginfo_ztensor_optional_float, ZEND_ACC_PUBLIC)
    PHP_ME(ZTensor, exp,              arginfo_ztensor_exp,                ZEND_ACC_PUBLIC)
    PHP_ME(ZTensor, log,              arginfo_ztensor_log,                ZEND_ACC_PUBLIC)
    PHP_ME(ZTensor, sqrt,             arginfo_ztensor_sqrt,               ZEND_ACC_PUBLIC)
    PHP_ME(ZTensor, reshape,          arginfo_ztensor_reshape,            ZEND_ACC_PUBLIC)
    PHP_ME(ZTensor, slice,            arginfo_ztensor_slice,              ZEND_ACC_PUBLIC)
    PHP_ME(ZTensor, column,           arginfo_ztensor_column,             ZEND_ACC_PUBLIC)
    PHP_ME(ZTensor, row,              arginfo_ztensor_row,                ZEND_ACC_PUBLIC)
    PHP_ME(ZTensor, gather,           arginfo_ztensor_gather,             ZEND_ACC_PUBLIC)
    PHP_ME(ZTensor, argsort,          arginfo_ztensor_argsort,            ZEND_ACC_PUBLIC)
    PHP_ME(ZTensor, where,            arginfo_ztensor_where,              ZEND_ACC_PUBLIC)
    PHP_ME(ZTensor, findIndicesWhere, arginfo_ztensor_find_indices_where, ZEND_ACC_PUBLIC)
    PHP_ME(ZTensor, calculateSplitGini, arginfo_ztensor_calculate_split_gini, ZEND_ACC_PUBLIC)
    PHP_ME(ZTensor, arr,              arginfo_ztensor_static_arr,         ZEND_ACC_PUBLIC | ZEND_ACC_STATIC)
    PHP_ME(ZTensor, fill,             arginfo_ztensor_fill,               ZEND_ACC_PUBLIC)
    // Métodos Estáticos de Criação Adicionais
    PHP_ME(ZTensor, randn,            arginfo_ztensor_static_randn,     ZEND_ACC_PUBLIC | ZEND_ACC_STATIC)
    PHP_ME(ZTensor, arange,           arginfo_ztensor_static_arange,    ZEND_ACC_PUBLIC | ZEND_ACC_STATIC)
    PHP_ME(ZTensor, linspace,         arginfo_ztensor_static_linspace,  ZEND_ACC_PUBLIC | ZEND_ACC_STATIC)
    PHP_ME(ZTensor, logspace,         arginfo_ztensor_static_logspace,  ZEND_ACC_PUBLIC | ZEND_ACC_STATIC)
    PHP_ME(ZTensor, eye,              arginfo_ztensor_static_eye,       ZEND_ACC_PUBLIC | ZEND_ACC_STATIC)
    // (zeros, full, identity, random já estão lá)
    PHP_ME(ZTensor, key,                arginfo_ztensor_key,            ZEND_ACC_PUBLIC)
    // Métodos de Instância Adicionais
    PHP_ME(ZTensor, dot,              arginfo_ztensor_dot,              ZEND_ACC_PUBLIC)
    PHP_ME(ZTensor, broadcast,        arginfo_ztensor_broadcast,        ZEND_ACC_PUBLIC)
    PHP_ME(ZTensor, greater,          arginfo_ztensor_greater,          ZEND_ACC_PUBLIC)
    PHP_ME(ZTensor, minimum,          arginfo_ztensor_minimum,          ZEND_ACC_PUBLIC | ZEND_ACC_STATIC)
    PHP_ME(ZTensor, maximum,          arginfo_ztensor_maximum,          ZEND_ACC_PUBLIC | ZEND_ACC_STATIC)
    PHP_ME(ZTensor, mode,             arginfo_ztensor_mode,             ZEND_ACC_PUBLIC)
    PHP_ME(ZTensor, scalarDivide,     arginfo_ztensor_scalarDivide,     ZEND_ACC_PUBLIC)
    PHP_ME(ZTensor, copy,             arginfo_ztensor_copy,             ZEND_ACC_PUBLIC)
    PHP_ME(ZTensor, safe,             arginfo_ztensor_static_safe,      ZEND_ACC_PUBLIC | ZEND_ACC_STATIC)
    PHP_ME(ZTensor, tile,             arginfo_ztensor_static_tile,      ZEND_ACC_PUBLIC | ZEND_ACC_STATIC)

    PHP_ME(ZTensor, unique,           arginfo_ztensor_unique,           ZEND_ACC_PUBLIC)
    PHP_ME(ZTensor, bincount,         arginfo_ztensor_bincount,         ZEND_ACC_PUBLIC)
    PHP_ME(ZTensor, argmax,           arginfo_ztensor_argmax,           ZEND_ACC_PUBLIC)
    PHP_ME(ZTensor, argmin,           arginfo_ztensor_argmin,           ZEND_ACC_PUBLIC)
    PHP_ME(ZTensor, sort,             arginfo_ztensor_sort,             ZEND_ACC_PUBLIC)
    PHP_ME(ZTensor, isin,             arginfo_ztensor_isin,             ZEND_ACC_PUBLIC)
    PHP_ME(ZTensor, cumsum,           arginfo_ztensor_cumsum,           ZEND_ACC_PUBLIC)
    PHP_ME(ZTensor, concat,           arginfo_ztensor_static_concat,    ZEND_ACC_PUBLIC | ZEND_ACC_STATIC)
    PHP_ME(ZTensor, uniqueCounts,     arginfo_ztensor_uniqueCounts,     ZEND_ACC_PUBLIC)
    PHP_ME(ZTensor, stack,            arginfo_ztensor_static_stack,     ZEND_ACC_PUBLIC | ZEND_ACC_STATIC)
    PHP_ME(ZTensor, variance,         arginfo_ztensor_variance,         ZEND_ACC_PUBLIC)
    PHP_ME(ZTensor, median,           arginfo_ztensor_median,           ZEND_ACC_PUBLIC)
    PHP_ME(ZTensor, percentile,       arginfo_ztensor_percentile,       ZEND_ACC_PUBLIC)
    PHP_ME(ZTensor, histogram,        arginfo_ztensor_histogram,        ZEND_ACC_PUBLIC)

    // Autograd methods
    PHP_ME(ZTensor, requiresGrad,     arginfo_class_ZMatrix_ZTensor_requiresGrad,      ZEND_ACC_PUBLIC)
    PHP_ME(ZTensor, isRequiresGrad,    arginfo_class_ZMatrix_ZTensor_isRequiresGrad,  ZEND_ACC_PUBLIC)
    PHP_ME(ZTensor, ensureGrad,       arginfo_class_ZMatrix_ZTensor_ensureGrad,       ZEND_ACC_PUBLIC)
    PHP_ME(ZTensor, zeroGrad,         arginfo_class_ZMatrix_ZTensor_zeroGrad,         ZEND_ACC_PUBLIC)
    PHP_ME(ZTensor, getGrad,          arginfo_class_ZMatrix_ZTensor_getGrad,          ZEND_ACC_PUBLIC)
    PHP_ME(ZTensor, backward,         arginfo_class_ZMatrix_ZTensor_backward,          ZEND_ACC_PUBLIC)

    // Autograd methods
    PHP_ME(ZTensor, addAutograd,     arginfo_add_autograd,               ZEND_ACC_PUBLIC | ZEND_ACC_STATIC)
    PHP_ME(ZTensor, subAutograd,     arginfo_sub_autograd,               ZEND_ACC_PUBLIC | ZEND_ACC_STATIC)
    PHP_ME(ZTensor, mulAutograd,     arginfo_mul_autograd,               ZEND_ACC_PUBLIC | ZEND_ACC_STATIC)
    PHP_ME(ZTensor, sumAutograd,     arginfo_sum_autograd,               ZEND_ACC_PUBLIC | ZEND_ACC_STATIC)

        // TODO: Add PHP_ME for rand, randn, arange, linspace, logspace, eye, etc.
    PHP_FE_END
};


// --- Definição da Função MINFO ---
// Movido para antes de MINIT para garantir que está definida quando MINIT for chamada
// (Embora não seja estritamente necessário pela ordem de chamada do PHP, é mais seguro)
PHP_MINFO_FUNCTION(zmatrix)
{
    php_info_print_table_start();
    php_info_print_table_header(2, "ZMatrix Support", "enabled");
    php_info_print_table_row(2, "Version", "0.4.0-float"); // Indicar versão float
    php_info_print_table_row(2, "Data Type", "float");    // <-- NOVO
    php_info_print_table_row(2, "OpenMP Support (Compile Time)", HAS_OPENMP ? "Yes" : "No");
    php_info_print_table_row(2, "AVX2 Support (Compile Time)", HAS_AVX2 ? "Yes" : "No");
    php_info_print_table_row(2, "AVX512F Support (Compile Time)", HAS_AVX512 ? "Yes" : "No");
    php_info_print_table_row(2, "BLAS Used (Compile Time)", "Yes (cblas - sgemm)"); // Indicar sgemm
    php_info_print_table_end();
}

// --- Definição ÚNICA da Função MINIT ---
PHP_MINIT_FUNCTION(zmatrix)
{
    zend_class_entry ce_zmatrix_ns, ce_ztensor; // Variáveis locais

    // Define a versão (exemplo)
    #ifndef ZMATRIX_VERSION
    #define ZMATRIX_VERSION "0.4.0-float"
    #endif

    // Registra a classe Namespace ZMatrix\ZMatrix
    INIT_NS_CLASS_ENTRY(ce_zmatrix_ns, "ZMatrix", "ZMatrix", NULL);
    // Armazena o class entry do namespace na variável global zmatrix_ce
    zmatrix_ce = zend_register_internal_class(&ce_zmatrix_ns);
    if (zmatrix_ce == NULL) {
        php_error_docref(NULL, E_ERROR, "Failed to register class ZMatrix\\ZMatrix");
        return FAILURE;
    }

    // Registra a classe ZMatrix\ZTensor
    INIT_NS_CLASS_ENTRY(ce_ztensor, "ZMatrix", "ZTensor", zmatrix_ztensor_methods);
    // Armazena o class entry do tensor na variável global zmatrix_ce_ZTensor
    // Opcionalmente, pode herdar do namespace CE: zend_register_internal_class_ex(&ce_ztensor, zmatrix_ce);
    zmatrix_ce_ZTensor = zend_register_internal_class(&ce_ztensor);
    if (zmatrix_ce_ZTensor == NULL) {
         php_error_docref(NULL, E_ERROR, "Failed to register class ZMatrix\\ZTensor");
        return FAILURE;
    }
    zmatrix_ce_ZTensor->create_object = zmatrix_ztensor_create;

    // Configura handlers
    memcpy(&zmatrix_ztensor_handlers, zend_get_std_object_handlers(), sizeof(zend_object_handlers));
    zmatrix_ztensor_handlers.offset = offsetof(zmatrix_ztensor_object, std);
    zmatrix_ztensor_handlers.free_obj = zmatrix_ztensor_free;
    zmatrix_ztensor_handlers.clone_obj = zmatrix_ztensor_clone;

    return SUCCESS;
}

PHP_MSHUTDOWN_FUNCTION(zmatrix)
{
    // Se houver objetos estáticos/globais, libere aqui. Exemplo (não há nenhum declarado atualmente):
    // delete global_tensor;
    // global_tensor = nullptr;

    // Caso esteja usando recursos estáticos persistentes (além dos normais da Zend), limpe aqui.

    return SUCCESS;
}

// ==========================================================================
// Estrutura de Entrada do Módulo (Ordem Corrigida)
// ==========================================================================
zend_module_entry zmatrix_module_entry = {
    STANDARD_MODULE_HEADER,
    "zmatrix",              // Nome da extensão
    NULL,                   // Funções globais (ou NULL)
    PHP_MINIT(zmatrix),     // MINIT
    PHP_MSHUTDOWN(zmatrix), // MSHUTDOWN
    NULL,                   // RINIT
    NULL,                   // RSHUTDOWN
    PHP_MINFO(zmatrix),     // MINFO
    ZMATRIX_VERSION,         // Versão da extensão
    STANDARD_MODULE_PROPERTIES
};

// Macro para tornar o módulo carregável
#ifdef COMPILE_DL_ZMATRIX
ZEND_GET_MODULE(zmatrix)
#endif
