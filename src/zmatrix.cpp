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

#define ZMATRIX_PARALLEL_THRESHOLD 40000

#define ZMATRIX_GPU_THRESHOLD 200000

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

static inline void zmatrix_gpu_debug(const char *op, size_t n) {
    if (zmatrix_gpu_debug_enabled()) {
        std::fprintf(stderr, "[zmatrix][gpu] %s n=%zu\n", op, n);
    }
}

static inline bool zmatrix_should_use_gpu(size_t n) {
    if (zmatrix_force_cpu()) return false;
    return (n >= ZMATRIX_GPU_THRESHOLD) && (gpu_available() != 0);
}
#else
static inline bool zmatrix_should_use_gpu(size_t) { return false; }
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
            std::fill(grad->data.begin(), grad->data.end(), 0.0f);
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
        
        std::lock_guard<std::mutex> lock(grad_mutex);
        
        ZTensor& g = ensureGrad();
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
    }
    
    // Backward pass (reverse-mode autodiff)
    // Deve ser chamado apenas em tensores escalares
    void backward() {
        // Validação: deve ser escalar
        if (shape != std::vector<size_t>{1}) {
            throw std::invalid_argument(
                "backward() can only be called on scalar tensors (shape={1})"
            );
        }
        
        // Inicializa o gradiente deste tensor
        ensureGrad();
        grad->data[0] = 1.0f;
        
        // Percorre o grafo em DFS, visitando cada nó uma única vez
        std::set<std::shared_ptr<AutogradNode>> visited;
        
        std::function<void(std::shared_ptr<AutogradNode>)> backward_recursive = 
            [&](std::shared_ptr<AutogradNode> node) {
                if (!node || visited.count(node)) return;
                visited.insert(node);
                
                // Executa backward_fn deste nó
                if (node->backward_fn) {
                    try {
                        node->backward_fn();
                    } catch (const std::exception& e) {
                        // Log mas continua
                    }
                }
                
                // Recursivamente backward para pais
                for (const auto& parent : node->parents) {
                    if (parent && parent->grad_fn) {
                        backward_recursive(parent->grad_fn);
                    }
                }
            };
        
        // Inicia backward neste nó
        if (grad_fn) {
            backward_recursive(grad_fn);
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
    mutable bool device_valid = false;
    mutable bool host_valid = true;
#endif


    // Construtor Principal (inalterado, exceto 0.0f)
    ZTensor(const std::vector<size_t>& _shape) : shape(_shape) {
            size_t total_size = 1;
            bool has_zero_dim = false;
            for (size_t dim : shape) {
                 if (dim == 0) { has_zero_dim = true; break; }
                 if (dim > 0 && total_size > (std::numeric_limits<size_t>::max() / dim)) {
                     throw std::overflow_error(ZMATRIX_ERR_OVERFLOW);
                 }
                 total_size *= dim;
            }
            if (has_zero_dim) { total_size = 0;}

            if (total_size > 0) {
                try {
                    data.resize(total_size, 0.0f); // Usa 0.0f
                } catch (const std::bad_alloc& e) {
                    throw std::runtime_error(ZMATRIX_ERR_ALLOC_FAILED);
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
        }

    // Construtor Padrão (inalterado)
    ZTensor() = default;

    ZTensor(const ZTensor& other)
        : data(other.data),
          shape(other.shape),
          strides(other.strides),
          offset(other.offset),
          owns_data(other.owns_data)
    {
#ifdef HAVE_CUDA
        other.ensure_host();
        data = other.data;
        d_data = nullptr;
        d_capacity = 0;
        device_valid = false;
        host_valid = true;
#endif
    }

    ZTensor& operator=(const ZTensor& other) {
        if (this == &other) return *this;
#ifdef HAVE_CUDA
        other.ensure_host();
        free_device();
#endif
        data = other.data;
        shape = other.shape;
        strides = other.strides;
        offset = other.offset;
        owns_data = other.owns_data;
#ifdef HAVE_CUDA
        d_data = nullptr;
        d_capacity = 0;
        device_valid = false;
        host_valid = true;
#endif
        return *this;
    }

    ZTensor(ZTensor&& other) noexcept
        : data(std::move(other.data)),
          shape(std::move(other.shape)),
          strides(std::move(other.strides)),
          offset(other.offset),
          owns_data(other.owns_data)
    {
#ifdef HAVE_CUDA
        d_data = other.d_data;
        d_capacity = other.d_capacity;
        device_valid = other.device_valid;
        host_valid = other.host_valid;
        other.d_data = nullptr;
        other.d_capacity = 0;
        other.device_valid = false;
        other.host_valid = true;
#endif
    }

    ZTensor& operator=(ZTensor&& other) noexcept {
        if (this == &other) return *this;
#ifdef HAVE_CUDA
        free_device();
#endif
        data = std::move(other.data);
        shape = std::move(other.shape);
        strides = std::move(other.strides);
        offset = other.offset;
        owns_data = other.owns_data;
#ifdef HAVE_CUDA
        d_data = other.d_data;
        d_capacity = other.d_capacity;
        device_valid = other.device_valid;
        host_valid = other.host_valid;
        other.d_data = nullptr;
        other.d_capacity = 0;
        other.device_valid = false;
        other.host_valid = true;
#endif
        return *this;
    }

    ~ZTensor() {
#ifdef HAVE_CUDA
        free_device();
#endif
    }

#ifdef HAVE_CUDA
    static inline void cuda_check(cudaError_t err, const char *what) {
        if (err != cudaSuccess) {
            throw std::runtime_error(std::string(what) + ": " + cudaGetErrorString(err));
        }
    }

    void ensure_device() const {
        if (device_valid) return;
        size_t n = size();
        if (n == 0) {
            device_valid = true;
            return;
        }
        if (!host_valid) {
            throw std::runtime_error("Host data is not valid for device upload");
        }
        if (!d_data || d_capacity < n) {
            if (d_data) cudaFree(d_data);
            cuda_check(cudaMalloc(reinterpret_cast<void**>(&d_data), n * sizeof(float)), "cudaMalloc");
            d_capacity = n;
        }
        cuda_check(cudaMemcpy(d_data, data.data(), n * sizeof(float), cudaMemcpyHostToDevice), "cudaMemcpy H2D");
        device_valid = true;
    }

    void ensure_host() const {
        if (host_valid) return;
        size_t n = size();
        if (n == 0) {
            host_valid = true;
            return;
        }
        if (!d_data) {
            throw std::runtime_error("Device data is not valid for host download");
        }
        cuda_check(cudaMemcpy(const_cast<float*>(data.data()), d_data, n * sizeof(float), cudaMemcpyDeviceToHost), "cudaMemcpy D2H");
        host_valid = true;
    }

    void mark_host_modified() {
        host_valid = true;
        device_valid = false;
    }

    void mark_device_modified() const {
        device_valid = true;
        host_valid = false;
    }

    void free_device() {
        if (d_data) {
            cudaFree(d_data);
            d_data = nullptr;
        }
        d_capacity = 0;
        device_valid = false;
    }

    void to_gpu() {
        ensure_device();
    }

    void to_cpu() {
        ensure_host();
        device_valid = false;  // Mark as no longer on GPU
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

    static size_t compute_total_size(const std::vector<size_t>& shape) {
        size_t total = 1;
        for (size_t dim : shape) {
            total *= dim;
        }
        return total;
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
        if (shape.empty()) return 0;
        size_t total_size = 1;
        for(size_t dim : shape) {
            if (dim == 0) return 0;
            if (dim > 0 && total_size > (std::numeric_limits<size_t>::max() / dim)) return 0;
            total_size *= dim;
        }
        return total_size;
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
           if (device_valid) {
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

        #if defined(HAVE_CUDA)
           if (zmatrix_should_use_gpu(N)) {
               zmatrix_gpu_debug("add", N);
               gpu_add(a, b, N);
               return;
           }
        #endif

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


    void soma(ZTensor& out, int axis) const {
        if (shape.empty()) throw std::runtime_error(ZMATRIX_ERR_EMPTY_MATRIX);
        if (axis < 0 || static_cast<size_t>(axis) >= shape.size()) throw std::out_of_range("Invalid axis for sum(axis)");

        std::vector<size_t> expected_shape = shape;
        expected_shape.erase(expected_shape.begin() + axis);

        if (out.shape != expected_shape) throw std::invalid_argument(ZMATRIX_ERR_SHAPE_MISMATCH);

        const size_t axis_dim = shape[axis];
        const size_t out_size = out.size();

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
    }

    // --- Subtração (float) - Loop Canônico ---
    void subtract(const ZTensor& other) {
        if (!same_shape(other)) throw std::invalid_argument(ZMATRIX_ERR_SHAPE_MISMATCH);
        const size_t N = size();
        if (N == 0) return;
#ifdef HAVE_CUDA
        if (device_valid) {
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

        #if defined(HAVE_CUDA)
        if (zmatrix_should_use_gpu(N)) {
            zmatrix_gpu_debug("sub", N);
            gpu_sub(a, b, N);
            return;
        }
        #endif

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
        if (device_valid) {
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

        #if defined(HAVE_CUDA)
        if (zmatrix_should_use_gpu(N)) {
            zmatrix_gpu_debug("mul", N);
            gpu_mul(a, b, N);
            return;
        }
        #endif

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
        #if defined(HAVE_CUDA)
           if (zmatrix_should_use_gpu(N)) {
               zmatrix_gpu_debug("scalar_div", N);
               gpu_scalar_div(a, scalar, N);
               return;
           }
        #endif
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
        if (zmatrix_should_use_gpu(N)) {
            zmatrix_gpu_debug("scalar_mul", N);
            gpu_scalar_mul(a, scalar, N);
            return;
        }
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
        if (zmatrix_should_use_gpu(N)) {
            zmatrix_gpu_debug("scalar_add", N);
            gpu_scalar_add(ptr, value, N);
            return;
        }
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
        #if defined(HAVE_CUDA)
        if (zmatrix_should_use_gpu(N)) {
            zmatrix_gpu_debug("scalar_sub", N);
            gpu_scalar_sub(ptr, value, N);
            return;
        }
        #endif

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
        // IMPORTANTE: std::vector copy é rasa (shallow) e compartilha os dados
        // Ambos result e this->data apontam para o mesmo buffer de memória
        // Isto implementa uma "view" eficiente, não uma cópia de dados
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
        #if defined(HAVE_CUDA)
         if (zmatrix_should_use_gpu(N)) {
             zmatrix_gpu_debug("abs", N);
             gpu_abs(a, N);
             return;
         }
        #endif
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
        #if defined(HAVE_CUDA)
        if (zmatrix_should_use_gpu(N)) {
            zmatrix_gpu_debug("sigmoid", N);
            gpu_sigmoid(a, N);
            return;
        }
        #endif
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

        #if defined(HAVE_CUDA)
        if (zmatrix_should_use_gpu(N)) {
            zmatrix_gpu_debug("relu", N);
            gpu_relu(a, N);
            return;
        }
        #endif

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
        #if defined(HAVE_CUDA)
        if (zmatrix_should_use_gpu(N)) {
            zmatrix_gpu_debug("tanh", N);
            gpu_tanh(a, N);
            return;
        }
        #endif

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
        #if defined(HAVE_CUDA)
        if (zmatrix_should_use_gpu(N)) {
            zmatrix_gpu_debug("leaky_relu", N);
            gpu_leaky_relu(a, alpha, N);
            return;
        }
        #endif

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
        #if defined(HAVE_CUDA)
        if (zmatrix_should_use_gpu(N)) {
            zmatrix_gpu_debug("leaky_relu_derivative", N);
            gpu_leaky_relu_derivative(a, alpha, N);
            return;
        }
        #endif

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

#ifdef HAVE_CUDA
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
        #if defined(HAVE_CUDA)
        if (zmatrix_should_use_gpu(N)) {
            zmatrix_gpu_debug("exp", N);
            gpu_exp(a, N);
            return;
        }
        #endif
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
            
            // Captura pais E resultado como shared_ptr para evitar use-after-free
            auto a_ptr = std::make_shared<ZTensor>(a);
            auto b_ptr = std::make_shared<ZTensor>(b);
            auto result_ptr = std::make_shared<ZTensor>(result);
            node->parents = {a_ptr, b_ptr};
            
            // Backward function para add
            // dy/da = 1, dy/db = 1
            bool a_req = a.requires_grad;
            bool b_req = b.requires_grad;
            
            node->backward_fn = [result_ptr, a_ptr, b_ptr, a_req, b_req]() {
                // grad_result é o gradiente no tensor resultado
                const ZTensor* grad_result = result_ptr->getGrad();
                if (!grad_result) return;  // Nada a fazer
                
                // Para add: ambos os pais recebem o mesmo gradiente
                if (a_req) {
                    const_cast<ZTensor*>(a_ptr.get())->accumulate_grad(*grad_result);
                }
                if (b_req) {
                    const_cast<ZTensor*>(b_ptr.get())->accumulate_grad(*grad_result);
                }
            };
            
            result.grad_fn = node;
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
            auto a_ptr = std::make_shared<ZTensor>(a);
            auto b_ptr = std::make_shared<ZTensor>(b);
            auto result_ptr = std::make_shared<ZTensor>(result);
            node->parents = {a_ptr, b_ptr};
            
            bool a_req = a.requires_grad;
            bool b_req = b.requires_grad;
            
            node->backward_fn = [result_ptr, a_ptr, b_ptr, a_req, b_req]() {
                const ZTensor* grad_result = result_ptr->getGrad();
                if (!grad_result) return;
                
                if (a_req) {
                    const_cast<ZTensor*>(a_ptr.get())->accumulate_grad(*grad_result);
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
                    const_cast<ZTensor*>(b_ptr.get())->accumulate_grad(neg_grad);
                }
            };
            
            result.grad_fn = node;
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
            auto a_ptr = std::make_shared<ZTensor>(a);
            auto b_ptr = std::make_shared<ZTensor>(b);
            auto result_ptr = std::make_shared<ZTensor>(result);
            node->parents = {a_ptr, b_ptr};
            
            bool a_req = a.requires_grad;
            bool b_req = b.requires_grad;
            
            // Captura cópias dos operandos para usar no backward
            auto a_copy = std::make_shared<ZTensor>(a);
            auto b_copy = std::make_shared<ZTensor>(b);
            
            node->backward_fn = [result_ptr, a_ptr, b_ptr, a_copy, b_copy, a_req, b_req]() {
                const ZTensor* grad_result = result_ptr->getGrad();
                if (!grad_result) return;
                
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
                    const_cast<ZTensor*>(a_ptr.get())->accumulate_grad(grad_a);
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
                    const_cast<ZTensor*>(b_ptr.get())->accumulate_grad(grad_b);
                }
            };
            
            result.grad_fn = node;
        }
        
        return result;
    }
    
    /**
     * Sum reduction with autograd support
     * result = sum(tensor)  -> escalar
     * Backward: grad_input[i] = grad_output[0] para cada elemento i
     */
    static ZTensor sum_autograd(const ZTensor& t) {
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
            auto t_ptr = std::make_shared<ZTensor>(t);
            auto result_ptr = std::make_shared<ZTensor>(result);
            node->parents = {t_ptr};
            
            // Armazena shape original para broadcast no backward
            auto input_shape = t.shape;
            auto input_size = t.size();
            
            node->backward_fn = [result_ptr, t_ptr, input_shape, input_size]() {
                const ZTensor* grad_result = result_ptr->getGrad();
                if (!grad_result) return;
                
                // grad_input[i] = grad_result[0] para cada i
                float grad_val = grad_result->data[0];
                
                ZTensor grad_input(input_shape);
                if (input_size > 0) {
                    float* grad_data = grad_input.data.data();
                    for (size_t i = 0; i < input_size; ++i) {
                        grad_data[i] = grad_val;
                    }
                }
                
                const_cast<ZTensor*>(t_ptr.get())->accumulate_grad(grad_input);
            };
            
            result.grad_fn = node;
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

        // Não há tensor antigo para deletar (porque intern foi recém-criado)
        intern->tensor = new ZTensor(result_tensor);
        // Não precisa de owns_data
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
    PHP_ME(ZTensor, arr,              arginfo_ztensor_static_arr,         ZEND_ACC_PUBLIC | ZEND_ACC_STATIC)
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
    PHP_ME(ZTensor, scalarDivide,     arginfo_ztensor_scalarDivide,     ZEND_ACC_PUBLIC)
    PHP_ME(ZTensor, copy,             arginfo_ztensor_copy,             ZEND_ACC_PUBLIC)
    PHP_ME(ZTensor, safe,             arginfo_ztensor_static_safe,      ZEND_ACC_PUBLIC | ZEND_ACC_STATIC)
    PHP_ME(ZTensor, tile,             arginfo_ztensor_static_tile,      ZEND_ACC_PUBLIC | ZEND_ACC_STATIC)

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
    "0.3.0",                // Versão da extensão (exemplo)
    STANDARD_MODULE_PROPERTIES
};

// Macro para tornar o módulo carregável
#ifdef COMPILE_DL_ZMATRIX
ZEND_GET_MODULE(zmatrix)
#endif