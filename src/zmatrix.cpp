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
#include <sstream>

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

// Gerador aleatório (mantido)
static inline uint64_t xorshift64star(uint64_t& state) {
    state ^= state >> 12; state ^= state << 25; state ^= state >> 27;
    return state * UINT64_C(0x2545F4914F6CDD1D);
}

static std::mt19937& get_global_mt19937() {
    static std::mt19937 gen(std::chrono::high_resolution_clock::now().time_since_epoch().count());
    return gen;
}
// --- Definição COMPLETA de ZTensor PRIMEIRO ---
struct ZTensor {
    std::vector<float> data; // <--- MUDANÇA: double para float
    std::vector<size_t> shape;
    std::vector<size_t> strides;
    size_t offset = 0;
    bool owns_data = true; // ← importante para views


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
                for (int i = shape.size() - 1; i >= 0; --i) {
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

    std::string to_string() const {
        std::ostringstream oss;

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
        if (this->size() == 0) { throw std::out_of_range("Access to empty tensor"); }
        size_t index = get_linear_index(indices);
        return data[index];
    }
    const float& at(const std::vector<size_t>& indices) const {
         if (this->size() == 0) { throw std::out_of_range("Access to empty tensor"); }
         size_t index = get_linear_index(indices);
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
           if (!same_shape(other)) {
               throw std::runtime_error(ZMATRIX_ERR_SHAPE_MISMATCH);
           }
           const size_t N = size();
           if (N == 0) return;

           float * __restrict__ a = data.data();
           const float * __restrict__ b = other.data.data();

        #if HAS_OPENMP
           if (N > ZMATRIX_PARALLEL_THRESHOLD) {
               #pragma omp parallel for simd schedule(static)
               for (size_t i = 0; i < N; ++i) {
                   a[i] += b[i];
               }
           } else {
               for (size_t i = 0; i < N; ++i) { // Loop sequencial
                   a[i] += b[i];
               }
           }
        #else
           for (size_t i = 0; i < N; ++i) { // Loop sequencial se não houver OpenMP
               a[i] += b[i];
           }
        #endif
    }

    void clip(const ZTensor& other) {
        if (!same_shape(other)) throw std::runtime_error(ZMATRIX_ERR_SHAPE_MISMATCH);

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
    }


    void soma(ZTensor& out, int axis) const {
        if (shape.empty()) throw std::runtime_error(ZMATRIX_ERR_EMPTY_MATRIX);
        if (axis < 0 || static_cast<size_t>(axis) >= shape.size()) throw std::out_of_range("Invalid axis for sum(axis)");

        std::vector<size_t> expected_shape = shape;
        expected_shape.erase(expected_shape.begin() + axis);

        if (out.shape != expected_shape) throw std::runtime_error(ZMATRIX_ERR_SHAPE_MISMATCH);

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
        if (!same_shape(other)) throw std::runtime_error(ZMATRIX_ERR_SHAPE_MISMATCH);
        const size_t N = size();
        if (N == 0) return;
        float * __restrict__ a = data.data();
        const float * __restrict__ b = other.data.data();
        #if HAS_OPENMP
        if (N > ZMATRIX_PARALLEL_THRESHOLD) {
            #pragma omp parallel for simd schedule(static)
            for (size_t i = 0; i < N; ++i) {
                a[i] -= b[i];
            }
        } else { // Loop sequencial se pequeno
            for (size_t i = 0; i < N; ++i) {
                a[i] -= b[i];
            }
        }
        #else // Loop sequencial se não houver OpenMP
        for (size_t i = 0; i < N; ++i) {
            a[i] -= b[i];
        }
        #endif

    }

    // --- Multiplicação Elemento a Elemento (float) - Loop Canônico ---
    void mul(const ZTensor& other) {
        if (!same_shape(other)) throw std::runtime_error(ZMATRIX_ERR_SHAPE_MISMATCH);
        const size_t N = size();
        if (N == 0) return;
        float * __restrict__ a = data.data();
        const float * __restrict__ b = other.data.data();

        #if HAS_OPENMP
        if (N > ZMATRIX_PARALLEL_THRESHOLD) {
             #pragma omp parallel for simd schedule(static)
             for (size_t i = 0; i < N; ++i) {
                a[i] *= b[i];
            }
        } else { // Loop sequencial se pequeno
             for (size_t i = 0; i < N; ++i) {
                a[i] *= b[i];
            }
        }
        #else // Loop sequencial se não houver OpenMP
         for (size_t i = 0; i < N; ++i) {
            a[i] *= b[i];
        }
        #endif
    }

    void scalar_divide(float scalar) {
       const size_t N = size();
           if (N == 0) return;
           float * __restrict__ a = data.data();
           #if HAS_OPENMP
           if (N > ZMATRIX_PARALLEL_THRESHOLD) {
               #pragma omp parallel for simd schedule(static)
               for (size_t i = 0; i < N; ++i) {
                   a[i] /= scalar;
               }
           } else {
               for (size_t i = 0; i < N; ++i) {
                   a[i] /= scalar;
               }
           }
           #else
           for (size_t i = 0; i < N; ++i) {
               a[i] /= scalar;
           }
           #endif
    }

    // --- Multiplicação por Escalar (float) - Loop Canônico ---
    void multiply_scalar(float scalar) {
        const size_t N = size();
        if (N == 0) return;
        float * __restrict__ a = data.data();
        #if HAS_OPENMP
        if (N > ZMATRIX_PARALLEL_THRESHOLD) {
            #pragma omp parallel for simd schedule(static)
            for (size_t i = 0; i < N; ++i) {
                a[i] *= scalar;
            }
        } else { // Loop sequencial se pequeno
            for (size_t i = 0; i < N; ++i) {
                a[i] *= scalar;
            }
        }
        #else // Loop sequencial se não houver OpenMP
        for (size_t i = 0; i < N; ++i) {
            a[i] *= scalar;
        }
        #endif
    }

    void scalar_add(float value) {
        size_t N = data.size();
        float *ptr = data.data();

        #if HAS_OPENMP
        if (N > ZMATRIX_PARALLEL_THRESHOLD) {
            #pragma omp parallel for simd schedule(static)
            for (size_t i = 0; i < N; ++i) {
                ptr[i] += value;
            }
        } else
        #endif
        {
            for (size_t i = 0; i < N; ++i) {
                ptr[i] += value;
            }
        }
    }

    void scalar_subtract(float value) {
        size_t N = data.size();
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
            for (size_t i = 0; i < N; ++i) {
                ptr[i] -= value;
            }
        }
    }

    ZTensor greater_than(const ZTensor& other) const {
        if (!same_shape(other)) {
            throw std::runtime_error(ZMATRIX_ERR_SHAPE_MISMATCH);
        }

        size_t N = size();
        ZTensor result(shape);
        const float* a = data.data();
        const float* b = other.data.data();
        float* r = result.data.data();

    #if HAS_OPENMP
        #pragma omp parallel for simd schedule(static)
    #endif
        for (size_t i = 0; i < N; ++i) {
            r[i] = (a[i] > b[i]) ? 1.0f : 0.0f;
        }

        return result;
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
            throw std::runtime_error(ZMATRIX_ERR_SHAPE_MISMATCH);
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
        result.data = this->data;  // Cópia leve do vetor de dados

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
         // Verificações de dimensão
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

         const float* A_ptr = this->data.data();
         const float* B_ptr = other.data.data();
               float* C_ptr = result.data.data();

         const float alpha = 1.0f;
         const float beta  = 0.0f;

         const CBLAS_INDEX lda = static_cast<CBLAS_INDEX>(K); // leading dimension de A
         const CBLAS_INDEX ldb = static_cast<CBLAS_INDEX>(N); // leading dimension de B
         const CBLAS_INDEX ldc = static_cast<CBLAS_INDEX>(N); // leading dimension de C

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
         float * __restrict__ a = data.data();
         #if HAS_OPENMP
         if (N > ZMATRIX_PARALLEL_THRESHOLD) {
             #pragma omp parallel for simd schedule(static)
             for(size_t i = 0; i < N; ++i) {
                  a[i] = std::fabs(a[i]);
              }
         } else { // Loop sequencial se pequeno
             for(size_t i = 0; i < N; ++i) {
                  a[i] = std::fabs(a[i]);
              }
         }
         #else // Loop sequencial se não houver OpenMP
         for(size_t i = 0; i < N; ++i) {
              a[i] = std::fabs(a[i]);
          }
         #endif
     }

    // --- sigmoid (float) ---
     void sigmoid()  {
        const size_t N = size();
        if (N == 0) return;
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
     }

    void sigmoid_derivative() {
        const size_t N = size();
        if (N == 0) return;

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
   }

    // --- New Activation Functions ---
    void relu()  {
        const size_t N = size();
        if (N == 0) return;
        float * __restrict__ a = data.data();
        #if HAS_OPENMP
        if (N > ZMATRIX_PARALLEL_THRESHOLD) {
            #pragma omp parallel for simd schedule(static)
            for(size_t i = 0; i < N; ++i) {
                a[i] = std::max(0.0f, a[i]);
            }
        } else { // Loop sequencial se pequeno
            for(size_t i = 0; i < N; ++i) {
                a[i] = std::max(0.0f, a[i]);
            }
        }
        #else // Loop sequencial se não houver OpenMP
        for(size_t i = 0; i < N; ++i) {
            a[i] = std::max(0.0f, a[i]);
        }
        #endif

    }

    void relu_derivative() {
        const size_t N = size();
        if (N == 0) return;

        float* __restrict__ a = data.data();

        #if HAS_OPENMP
        if (N > ZMATRIX_PARALLEL_THRESHOLD) {
            #pragma omp parallel for simd schedule(static)
            for (size_t i = 0; i < N; ++i) {
                a[i] = (a[i] > 0.0f) ? 1.0f : 0.0f;
            }
        } else
        #endif
        {
            for (size_t i = 0; i < N; ++i) {
                a[i] = (a[i] > 0.0f) ? 1.0f : 0.0f;
            }
        }
    }


    void tanh()  {
        const size_t N = size();
        if (N == 0) return;
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
    }

    void tanh_derivative() {
        const size_t N = size();
        if (N == 0) return;

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
    }

    void leaky_relu(float alpha = 0.01f) {
        const size_t N = size();
        if (N == 0) return;

        float* __restrict__ a = data.data();

        #if HAS_OPENMP
        if (N > ZMATRIX_PARALLEL_THRESHOLD) {
            #pragma omp parallel for simd schedule(static)
            for (size_t i = 0; i < N; ++i) {
                a[i] = (a[i] > 0.0f) ? a[i] : alpha * a[i];
            }
        } else
        #endif
        {
            for (size_t i = 0; i < N; ++i) {
                a[i] = (a[i] > 0.0f) ? a[i] : alpha * a[i];
            }
        }
    }

    void leaky_relu_derivative(float alpha = 0.01f) {
        const size_t N = size();
        if (N == 0) return;

        float* __restrict__ a = data.data();

        #if HAS_OPENMP
        if (N > ZMATRIX_PARALLEL_THRESHOLD) {
            #pragma omp parallel for simd schedule(static)
            for (size_t i = 0; i < N; ++i) {
                a[i] = (a[i] > 0.0f) ? 1.0f : alpha;
            }
        } else
        #endif
        {
            for (size_t i = 0; i < N; ++i) {
                a[i] = (a[i] > 0.0f) ? 1.0f : alpha;
            }
        }
    }

    void softmax() {
        if (shape.empty() || (shape.size() != 1 && shape.size() != 2)) {
            throw std::runtime_error("Softmax requires 1D or 2D tensor");
        }

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
    }


    void softmax_derivative() {
        const size_t N = size();
        if (N == 0) return;

        float* __restrict__ a = data.data();

        // CUIDADO: isso zera tudo se usado direto — normalmente usamos softmax + cross-entropy juntos
        for (size_t i = 0; i < N; ++i) {
            float si = a[i];
            a[i] = si * (1.0f - si);  // diagonal da jacobiana
        }
    }


    // --- New Mathematical Operations ---
    void divide(const ZTensor& other) {
        if (!same_shape(other)) {
           throw std::runtime_error(ZMATRIX_ERR_SHAPE_MISMATCH);
       }
       const size_t N = size();
       if (N == 0) return;
       float * __restrict__ a = data.data();
       const float * __restrict__ b = other.data.data();

       #if HAS_OPENMP
       if (N > ZMATRIX_PARALLEL_THRESHOLD) {
            #pragma omp parallel for simd schedule(static)
           for (size_t i = 0; i < N; ++i) {
                if (b[i] == 0.0f) {
                    throw std::runtime_error("Divisão por zero detectada");
                }
                a[i] /= b[i];
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

    }

     void pow(float exponent) {
        const size_t N = size();
        if (N == 0) return;
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
    }

    void exp() {
        const size_t N = size();
        if (N == 0) return;
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
    }

    void log() { // Retorno void, não é const pois modifica o objeto
        const size_t total_size = this->size();
        if (total_size == 0) return;

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
    }

    void sqrt() { // Retorno void, não é const
        const size_t total_size = size();
        if (total_size == 0) return;

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
                 p_this[i] = std::sqrt(p_this[i]); // Modifica o dado do próprio objeto
             }
        } else { // Loop sequencial se pequeno
             for (size_t i = 0; i < total_size; ++i) {
                p_this[i] = std::sqrt(p_this[i]); // Modifica o dado do próprio objeto
            }
        }
        #else // Loop sequencial se não houver OpenMP
            for (size_t i = 0; i < total_size; ++i) {
                p_this[i] = std::sqrt(p_this[i]); // Modifica o dado do próprio objeto
            }
        #endif

    }
    // --- Reduções (float input, double accumulator) ---
      double sum() const {
          const size_t N = size();
          if (N == 0) return 0.0;

          const float* a = data.data();
          double total_sum = 0.0;

          // Paralelize apenas se o tensor for grande o suficiente
          #if HAS_OPENMP
          if (N > ZMATRIX_PARALLEL_THRESHOLD) {  // Ajuste este limite conforme necessário
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
         const float* p = data.data();
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
         const float* p = data.data();
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

        if (total_size == 0) {
            return result;
        }

        float* raw_ptr = static_cast<float*>(std::aligned_alloc(32, total_size * sizeof(float)));
        if (!raw_ptr) {
            throw std::bad_alloc();
        }

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
                    raw_ptr[i] = min + r01 * range;
                }
            }
        } else
        #endif
        {
            uint64_t state = base_seed;
            for (size_t i = 0; i < total_size; ++i) {
                uint64_t r = xorshift64star(state);
                float r01 = static_cast<float>(r * scale);
                raw_ptr[i] = min + r01 * range;
            }
        }

        result.data = std::vector<float>(raw_ptr, raw_ptr + total_size);
        std::free(raw_ptr);

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

         // Aloca manualmente e zera (alinhado a 32 bytes para AVX)
         float* raw_ptr = static_cast<float*>(std::aligned_alloc(32, total_size * sizeof(float)));
         if (!raw_ptr) {
             throw std::bad_alloc();
         }

         std::memset(raw_ptr, 0, total_size * sizeof(float));  // 🧠 mais rápido que std::fill

         // Converte para std::vector com ponteiro + tamanho (não copia, apenas encapsula)
         result.data = std::vector<float>(raw_ptr, raw_ptr + total_size);
         std::free(raw_ptr);  // libera o ponteiro base manualmente

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

          if (total_size == 0) {
              return result;
          }

          // Alocação alinhada
          float* raw_ptr = static_cast<float*>(std::aligned_alloc(32, total_size * sizeof(float)));
          if (!raw_ptr) {
              throw std::bad_alloc();
          }

          // Preenchimento
          #if HAS_OPENMP
          if (total_size > ZMATRIX_PARALLEL_THRESHOLD) {
              #pragma omp parallel for schedule(static)
              for (size_t i = 0; i < total_size; ++i) {
                  raw_ptr[i] = value;
              }
          } else
          #endif
          {
              std::fill_n(raw_ptr, total_size, value);
          }

          // Copia para vector compatível
          result.data = std::vector<float>(raw_ptr, raw_ptr + total_size);
          std::free(raw_ptr);

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

          // Alocação crua alinhada para AVX
          float* raw_ptr = static_cast<float*>(std::aligned_alloc(32, total_size * sizeof(float)));
          if (!raw_ptr) {
              throw std::bad_alloc();
          }

          // Zera tudo com memset (muito rápido)
          std::memset(raw_ptr, 0, total_size * sizeof(float));

          // Preenche a diagonal com 1.0f
          #if HAS_OPENMP
          if (s > ZMATRIX_PARALLEL_THRESHOLD) {
              #pragma omp parallel for schedule(static)
              for (size_t i = 0; i < s; ++i) {
                  raw_ptr[i * s + i] = 1.0f;
              }
          } else
          #endif
          {
              for (size_t i = 0; i < s; ++i) {
                  raw_ptr[i * s + i] = 1.0f;
              }
          }

          // Copia para vector (mantendo compatibilidade com estrutura atual)
          result.data = std::vector<float>(raw_ptr, raw_ptr + total_size);
          std::free(raw_ptr);

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
    if (intern->tensor != nullptr) {
        delete intern->tensor;
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


static void zmatrix_return_tensor_obj(const ZTensor& result_tensor, zval *return_value, zend_class_entry *tensor_ce) {
    try {
        object_init_ex(return_value, tensor_ce);

        zmatrix_ztensor_object *result_obj = Z_MATRIX_ZTENSOR_P(return_value);

        if (result_obj->tensor != nullptr) {
            if (result_obj->tensor->owns_data) {
                delete result_obj->tensor;
            }
            result_obj->tensor = nullptr;
        }

        result_obj->tensor = new ZTensor(result_tensor);
        result_obj->tensor->owns_data = true;  // 🔥 ESSENCIAL

    } catch (const std::bad_alloc& e) {
        zend_throw_exception(zend_ce_exception, ZMATRIX_ERR_ALLOC_FAILED, 0);
        ZVAL_NULL(return_value);
    } catch (const std::exception& e) {
        zend_throw_exception(zend_ce_exception, e.what(), 0);
        ZVAL_NULL(return_value);
    }
}


// --- Fim Funções Auxiliares ---

// ==========================================================================
// Métodos da Classe ZMatrix\ZTensor (PHP_METHOD)
// ==========================================================================

/// --- Construtor ---
 ZEND_BEGIN_ARG_INFO_EX(arginfo_ztensor_construct, 0, 0, 0)
     ZEND_ARG_INFO(0, dataOrShape)
 ZEND_END_ARG_INFO()

 PHP_METHOD(ZTensor, __construct)
 {
     zval *data_or_shape_zv = nullptr;
     zmatrix_ztensor_object *self_obj = Z_MATRIX_ZTENSOR_P(ZEND_THIS);

     ZEND_PARSE_PARAMETERS_START(0, 1)
         Z_PARAM_OPTIONAL
         Z_PARAM_ZVAL(data_or_shape_zv)
     ZEND_PARSE_PARAMETERS_END();

     if (self_obj->tensor != nullptr) {
         delete self_obj->tensor;
         self_obj->tensor = nullptr;
     }

     try {
         if (data_or_shape_zv == nullptr) {
             self_obj->tensor = new ZTensor();
         } else if (Z_TYPE_P(data_or_shape_zv) == IS_ARRAY) {
             self_obj->tensor = new ZTensor(php_array_to_tensor(data_or_shape_zv));
         } else if (Z_TYPE_P(data_or_shape_zv) == IS_OBJECT &&
                    instanceof_function(Z_OBJCE_P(data_or_shape_zv), zmatrix_ce_ZTensor)) {
             zmatrix_ztensor_object *other_obj = Z_MATRIX_ZTENSOR_P(data_or_shape_zv);
             if (other_obj->tensor != nullptr) {
                 self_obj->tensor = new ZTensor(*other_obj->tensor); // Cria uma cópia do tensor
             } else {
                 throw std::invalid_argument("Objeto ZTensor fornecido não está inicializado.");
             }
         } else {
             throw std::invalid_argument("Construtor ZTensor aceita apenas ZTensor, array ou nenhum argumento.");
         }
     } catch (const std::exception& e) {
         if (self_obj->tensor != nullptr) {
             delete self_obj->tensor;
             self_obj->tensor = nullptr;
         }
         zend_throw_exception(zend_ce_exception, e.what(), 0);
     }
 }

// --- ARG_INFO para o método estático arr ---
ZEND_BEGIN_ARG_INFO_EX(arginfo_ztensor_static_arr, 0, 0, 1)
    ZEND_ARG_INFO(0, input_data) // Agora é um zval genérico, faremos a verificação de tipo manualmente
ZEND_END_ARG_INFO()

PHP_METHOD(ZTensor, arr)
{
    zval *input_zv;

    // Parseia os parâmetros: esperamos 1 argumento.
    ZEND_PARSE_PARAMETERS_START(1, 1)
        Z_PARAM_ZVAL(input_zv) // Pega o argumento como um zval genérico
    ZEND_PARSE_PARAMETERS_END();

    try {
        if (Z_TYPE_P(input_zv) == IS_ARRAY) {
            // Caso 1: O input é um array PHP
            ZTensor result_tensor = php_array_to_tensor(input_zv); //
            zmatrix_return_tensor_obj(result_tensor, return_value, zmatrix_ce_ZTensor); //
        } else if (Z_TYPE_P(input_zv) == IS_OBJECT &&
                   instanceof_function(Z_OBJCE_P(input_zv), zmatrix_ce_ZTensor)) {
            // Caso 2: O input é um objeto ZTensor
            zmatrix_ztensor_object *other_obj = Z_MATRIX_ZTENSOR_P(input_zv);
            if (other_obj->tensor != nullptr) {
                // Cria uma NOVA instância de ZTensor (C++) como uma cópia do tensor interno do objeto fornecido
                ZTensor new_copied_tensor = ZTensor(*other_obj->tensor); // Chama o construtor de cópia de ZTensor C++
                zmatrix_return_tensor_obj(new_copied_tensor, return_value, zmatrix_ce_ZTensor); //
            } else {
                // O objeto ZTensor fornecido não foi inicializado internamente
                zend_throw_exception(zend_ce_exception, ZMATRIX_ERR_NOT_INITIALIZED, 0);
                RETURN_THROWS();
            }
        } else {
            // Tipo de argumento inválido
            zend_throw_exception_ex(zend_ce_type_error, 0, "ZTensor::arr() expects parameter 1 to be array or ZTensor, %s given", zend_zval_type_name(input_zv));
            RETURN_THROWS();
        }
    } catch (const std::exception& e) {
        // Captura exceções C++ de php_array_to_tensor, construtor de cópia de ZTensor, ou zmatrix_return_tensor_obj
        zend_throw_exception(zend_ce_exception, e.what(), 0);
        RETURN_THROWS();
    }
}
// --- Métodos de Operação Binária (add, sub, mul - elementwise) ---
ZEND_BEGIN_ARG_INFO_EX(arginfo_ztensor_op_other, 0, 0, 1)
    ZEND_ARG_INFO(0, other) // Aceita ZTensor ou array
ZEND_END_ARG_INFO()

PHP_METHOD(ZTensor, add)
{
    zval *other_zv;
    ZEND_PARSE_PARAMETERS_START(1, 1)
        Z_PARAM_ZVAL(other_zv)
    ZEND_PARSE_PARAMETERS_END();

    // 1) self tensor
    zmatrix_ztensor_object *self_obj = Z_MATRIX_ZTENSOR_P(ZEND_THIS);
    if (!self_obj->tensor) {
        zend_throw_exception(zend_ce_exception, ZMATRIX_ERR_NOT_INITIALIZED, 0);
        RETURN_THROWS();
    }
    ZTensor &A = *self_obj->tensor;

    // 2) Escalar (int or float)?
    if (Z_TYPE_P(other_zv) == IS_LONG || Z_TYPE_P(other_zv) == IS_DOUBLE) {
        float scalar = (Z_TYPE_P(other_zv) == IS_LONG
            ? (float)Z_LVAL_P(other_zv)
            : (float)Z_DVAL_P(other_zv));
        A.scalar_add(scalar);
        ZVAL_ZVAL(return_value, ZEND_THIS, 1, 0);
        return;
    }

    // 3) Tensor/array
    ZTensor *other_ptr = nullptr, tmp_other;
    if (!zmatrix_get_tensor_ptr(other_zv, other_ptr, tmp_other, zmatrix_ce_ZTensor)) {
        RETURN_THROWS();
    }
    ZTensor &B = *other_ptr;
    const auto &shapeA = A.shape, &shapeB = B.shape;

    // 4) Vetor‑1D de tamanho 1 → escalar
    if (shapeB.size()==1 && shapeB[0]==1) {
        float scalar = B.data.data()[0];
        A.scalar_add(scalar);
        ZVAL_ZVAL(return_value, ZEND_THIS, 1, 0);
        return;
    }

    // A partir daqui passamos a usar try/catch para que qualquer std::runtime_error
    // seja convertida em exceção PHP
    try {
        // 5) Same shape
        if (shapeA == shapeB) {
            A.add(B);
            ZVAL_ZVAL(return_value, ZEND_THIS, 1, 0);
            return;
        }

        // 6) Broadcast 2D×1D
        if (shapeA.size()==2 && shapeB.size()==1 && shapeB[0]==shapeA[1]) {
            size_t M=shapeA[0], N=shapeA[1];
            ZTensor C(shapeA);
            float *cd=C.data.data(), *bd=(float*)B.data.data();
            for(size_t i=0;i<M;++i){
                memcpy(cd+i*N, bd, N*sizeof(float));
            }
            A.add(C);
            ZVAL_ZVAL(return_value, ZEND_THIS, 1, 0);
            return;
        }

        // 7) Broadcast inverso → exceção PHP
        // 7) Broadcast reverso: B é maior que A mas compatível
        if (shapeB.size() == 2 && shapeA.size() == 1 && shapeA[0] == shapeB[1]) {
            size_t M = shapeB[0], N = shapeB[1];
            ZTensor expandedA(shapeB);
            float *dst = expandedA.data.data();
            const float *src = A.data.data();
            for (size_t i = 0; i < M; ++i) {
                memcpy(dst + i * N, src, N * sizeof(float));
            }
            A = expandedA; // substitui A com broadcast
            A.add(B);      // executa soma
            ZVAL_ZVAL(return_value, ZEND_THIS, 1, 0);
            return;
        }

        zend_throw_exception(zend_ce_exception,
            "For reverse broadcasting, call B->add(A) instead of A->add(B)", 0);
        RETURN_THROWS();
    }
    catch(const std::exception &e) {
        zend_throw_exception(zend_ce_exception, e.what(), 0);
        RETURN_THROWS();
    }
}

// ------------------------------------------------------------
// PHP_METHOD(ZTensor, sub)
PHP_METHOD(ZTensor, sub)
{
    zval *other_zv;
    ZEND_PARSE_PARAMETERS_START(1, 1)
        Z_PARAM_ZVAL(other_zv)
    ZEND_PARSE_PARAMETERS_END();

    // 1) self tensor
    zmatrix_ztensor_object *self_obj = Z_MATRIX_ZTENSOR_P(ZEND_THIS);
    if (!self_obj->tensor) {
        zend_throw_exception(zend_ce_exception, ZMATRIX_ERR_NOT_INITIALIZED, 0);
        RETURN_THROWS();
    }
    ZTensor &A = *self_obj->tensor;

    // 2) Escalar?
    if (Z_TYPE_P(other_zv)==IS_LONG||Z_TYPE_P(other_zv)==IS_DOUBLE) {
        float scalar = (Z_TYPE_P(other_zv)==IS_LONG
            ? (float)Z_LVAL_P(other_zv)
            : (float)Z_DVAL_P(other_zv));
        A.scalar_subtract(scalar);
        ZVAL_ZVAL(return_value, ZEND_THIS, 1, 0);
        return;
    }

    // 3) Tensor/array
    ZTensor *other_ptr=nullptr, tmp_other;
    if(!zmatrix_get_tensor_ptr(other_zv, other_ptr, tmp_other, zmatrix_ce_ZTensor)) {
        RETURN_THROWS();
    }
    ZTensor &B=*other_ptr;
    const auto &shapeA=A.shape, &shapeB=B.shape;

    // 4) Vetor‑1D de tamanho 1 → escalar
    if(shapeB.size()==1 && shapeB[0]==1) {
        float scalar=B.data.data()[0];
        A.scalar_subtract(scalar);
        ZVAL_ZVAL(return_value, ZEND_THIS, 1, 0);
        return;
    }

    try {
        // 5) Same shape
        if (shapeA == shapeB) {
            A.subtract(B);
            ZVAL_ZVAL(return_value, ZEND_THIS, 1, 0);
            return;
        }

        // 6) Broadcast 2D×1D: A [M×N], B [N]
        if (shapeA.size() == 2 && shapeB.size() == 1 && shapeB[0] == shapeA[1]) {
            size_t M = shapeA[0], N = shapeA[1];
            ZTensor C(shapeA);
            float *cd = C.data.data(), *bd = B.data.data();
            for (size_t i = 0; i < M; ++i) {
                memcpy(cd + i * N, bd, N * sizeof(float));
            }
            A.subtract(C);
            ZVAL_ZVAL(return_value, ZEND_THIS, 1, 0);
            return;
        }

        // 7) Broadcast reverso: A [N], B [M×N]
        if (shapeA.size() == 1 && shapeB.size() == 2 && shapeA[0] == shapeB[1]) {
            size_t M = shapeB[0], N = shapeB[1];
            ZTensor C(shapeB);
            float *cd = C.data.data();
            const float *ad = A.data.data();
            for (size_t i = 0; i < M; ++i) {
                memcpy(cd + i * N, ad, N * sizeof(float));
            }
            C.subtract(B); // A (broadcastado) - B
            A = C;
            ZVAL_ZVAL(return_value, ZEND_THIS, 1, 0);
            return;
        }

        // 8) Outros casos incompatíveis
        zend_throw_exception(zend_ce_exception,
            "Shapes incompatíveis para sub() com broadcast", 0);
        RETURN_THROWS();
    }
    catch (const std::exception &e) {
        zend_throw_exception(zend_ce_exception, e.what(), 0);
        RETURN_THROWS();
    }
}

PHP_METHOD(ZTensor, mul)
{
    zval *other_zv;
    ZEND_PARSE_PARAMETERS_START(1, 1)
        Z_PARAM_ZVAL(other_zv)
    ZEND_PARSE_PARAMETERS_END();

    zmatrix_ztensor_object *self_obj = Z_MATRIX_ZTENSOR_P(ZEND_THIS);
    if (!self_obj->tensor) {
        zend_throw_exception(zend_ce_exception, ZMATRIX_ERR_NOT_INITIALIZED, 0);
        RETURN_THROWS();
    }
    ZTensor &A = *self_obj->tensor;

    // 1) Caso escalar (int ou float)
    if (Z_TYPE_P(other_zv) == IS_LONG || Z_TYPE_P(other_zv) == IS_DOUBLE) {
        float scalar = (Z_TYPE_P(other_zv) == IS_LONG)
            ? (float)Z_LVAL_P(other_zv)
            : (float)Z_DVAL_P(other_zv);
        A.multiply_scalar(scalar);
        ZVAL_ZVAL(return_value, ZEND_THIS, 1, 0);
        return;
    }

    // 2) Caso tensor/array
    ZTensor *other_ptr = nullptr, tmp_other;
    if (!zmatrix_get_tensor_ptr(other_zv, other_ptr, tmp_other, zmatrix_ce_ZTensor)) {
        RETURN_THROWS();
    }
    ZTensor &B = *other_ptr;
    const auto &shapeA = A.shape, &shapeB = B.shape;

    // 2a) B é vetor 1D de tamanho 1 → escalar
    if (shapeB.size() == 1 && shapeB[0] == 1) {
        float scalar = B.data.data()[0];
        A.multiply_scalar(scalar);
        ZVAL_ZVAL(return_value, ZEND_THIS, 1, 0);
        return;
    }

    // Bloco try para capturar qualquer std::exception e converter em exceção PHP
    try {
        // 3) Mesmos formatos → operação direta
        if (shapeA == shapeB) {
            A.mul(B);
            ZVAL_ZVAL(return_value, ZEND_THIS, 1, 0);
            return;
        }

        // 4) Broadcast linha: A [M×N], B [N]
        if (shapeA.size() == 2 && shapeB.size() == 1 && shapeB[0] == shapeA[1]) {
            size_t M = shapeA[0], N = shapeA[1];
            ZTensor C(shapeA);
            float *cdat = C.data.data();
            const float *bdat = B.data.data();
            for (size_t i = 0; i < M; ++i) {
                memcpy(cdat + i * N, bdat, N * sizeof(float));
            }
            A.mul(C);
            ZVAL_ZVAL(return_value, ZEND_THIS, 1, 0);
            return;
        }

        // 5) Broadcast reverso: A [N], B [M×N]
        if (shapeA.size() == 1 && shapeB.size() == 2 && shapeA[0] == shapeB[1]) {
            size_t M = shapeB[0], N = shapeB[1];
            ZTensor C(shapeB);
            float *cdat = C.data.data();
            const float *adat = A.data.data();
            for (size_t i = 0; i < M; ++i) {
                memcpy(cdat + i * N, adat, N * sizeof(float));
            }
            C.mul(B);
            A = C; // resultado da operação substitui A
            ZVAL_ZVAL(return_value, ZEND_THIS, 1, 0);
            return;
        }

        // 6) Outros casos não suportados
        zend_throw_exception(zend_ce_exception,
            "Shapes incompatíveis para mul() com broadcast", 0);
        RETURN_THROWS();
    } catch (const std::exception &e) {
        zend_throw_exception(zend_ce_exception, e.what(), 0);
        RETURN_THROWS();
    }
}



// --- Métodos de Operação Escalar ---
ZEND_BEGIN_ARG_INFO_EX(arginfo_ztensor_op_scalar, 0, 0, 1)
    ZEND_ARG_TYPE_INFO(0, scalar, IS_DOUBLE, 0)
ZEND_END_ARG_INFO()

PHP_METHOD(ZTensor, scalarMultiply)
{
    double scalar;
   ZEND_PARSE_PARAMETERS_START(1, 1) Z_PARAM_DOUBLE(scalar) ZEND_PARSE_PARAMETERS_END();
   zmatrix_ztensor_object *self_obj = Z_MATRIX_ZTENSOR_P(ZEND_THIS);
   if (!self_obj->tensor) { zend_throw_exception(zend_ce_exception, ZMATRIX_ERR_NOT_INITIALIZED, 0); RETURN_THROWS(); }
   try { self_obj->tensor->multiply_scalar(static_cast<float>(scalar)); ZVAL_ZVAL(return_value, ZEND_THIS, 1, 0); }
   catch (const std::exception &e) { zend_throw_exception(zend_ce_exception, e.what(), 0); RETURN_THROWS(); }
}

// --- Métodos Unários (transpose, abs, sigmoid) ---
ZEND_BEGIN_ARG_INFO_EX(arginfo_ztensor_no_args, 0, 0, 0)
ZEND_END_ARG_INFO()


ZEND_BEGIN_ARG_INFO_EX(arginfo_ztensor_optional_float, 0, 0, 0)
    ZEND_ARG_TYPE_INFO(0, alpha, IS_DOUBLE, 1)
ZEND_END_ARG_INFO()

PHP_METHOD(ZTensor, transpose)
{
    ZEND_PARSE_PARAMETERS_NONE();
    zmatrix_ztensor_object *self_obj = Z_MATRIX_ZTENSOR_P(ZEND_THIS);
    if (!self_obj->tensor) { zend_throw_exception(zend_ce_exception, ZMATRIX_ERR_NOT_INITIALIZED, 0); RETURN_THROWS(); }
    try { ZTensor result = self_obj->tensor->transpose(); zmatrix_return_tensor_obj(result, return_value, zmatrix_ce_ZTensor); }
    catch (const std::exception& e) { zend_throw_exception(zend_ce_exception, e.what(), 0); RETURN_THROWS(); }
}

PHP_METHOD(ZTensor, abs)
{
   zmatrix_ztensor_object *self_obj = Z_MATRIX_ZTENSOR_P(ZEND_THIS);
   if (!self_obj->tensor) { zend_throw_exception(zend_ce_exception, ZMATRIX_ERR_NOT_INITIALIZED, 0); RETURN_THROWS(); }
   try { self_obj->tensor->abs(); ZVAL_ZVAL(return_value, ZEND_THIS, 1, 0); }
   catch (const std::exception &e) { zend_throw_exception(zend_ce_exception, e.what(), 0); RETURN_THROWS(); }
}

PHP_METHOD(ZTensor, sigmoid)
{
   zmatrix_ztensor_object *self_obj = Z_MATRIX_ZTENSOR_P(ZEND_THIS);
   if (!self_obj->tensor) { zend_throw_exception(zend_ce_exception, ZMATRIX_ERR_NOT_INITIALIZED, 0); RETURN_THROWS(); }
   try { self_obj->tensor->sigmoid(); ZVAL_ZVAL(return_value, ZEND_THIS, 1, 0); }
   catch (const std::exception &e) { zend_throw_exception(zend_ce_exception, e.what(), 0); RETURN_THROWS(); }
}

PHP_METHOD(ZTensor, sigmoidDerivative)
{
    zmatrix_ztensor_object *self_obj = Z_MATRIX_ZTENSOR_P(ZEND_THIS);
    if (!self_obj->tensor) {
        zend_throw_exception(zend_ce_exception, ZMATRIX_ERR_NOT_INITIALIZED, 0);
        RETURN_THROWS();
    }

    try {
        self_obj->tensor->sigmoid_derivative();
        ZVAL_ZVAL(return_value, ZEND_THIS, 1, 0);
    } catch (const std::exception& e) {
        zend_throw_exception(zend_ce_exception, e.what(), 0);
        RETURN_THROWS();
    }
}


// --- Métodos de Redução (sum, mean, min, max, std - global) ---
PHP_METHOD(ZTensor, sumtotal)
{
    ZEND_PARSE_PARAMETERS_NONE();
    zmatrix_ztensor_object *self_obj = Z_MATRIX_ZTENSOR_P(ZEND_THIS);
    if (!self_obj->tensor) { zend_throw_exception(zend_ce_exception, ZMATRIX_ERR_NOT_INITIALIZED, 0); RETURN_THROWS(); }
    try { RETURN_DOUBLE(self_obj->tensor->sum()); }
    catch (const std::exception& e) { zend_throw_exception(zend_ce_exception, e.what(), 0); RETURN_THROWS(); }
}

PHP_METHOD(ZTensor, mean)
{
    ZEND_PARSE_PARAMETERS_NONE();
    zmatrix_ztensor_object *self_obj = Z_MATRIX_ZTENSOR_P(ZEND_THIS);
    if (!self_obj->tensor) { zend_throw_exception(zend_ce_exception, ZMATRIX_ERR_NOT_INITIALIZED, 0); RETURN_THROWS(); }
    try { RETURN_DOUBLE(self_obj->tensor->mean()); }
    catch (const std::exception& e) { zend_throw_exception(zend_ce_exception, e.what(), 0); RETURN_THROWS(); }
}

PHP_METHOD(ZTensor, min)
{
    ZEND_PARSE_PARAMETERS_NONE();
    zmatrix_ztensor_object *self_obj = Z_MATRIX_ZTENSOR_P(ZEND_THIS);
    if (!self_obj->tensor) { zend_throw_exception(zend_ce_exception, ZMATRIX_ERR_NOT_INITIALIZED, 0); RETURN_THROWS(); }
    try { RETURN_DOUBLE(static_cast<double>(self_obj->tensor->min())); } // Cast float->double
    catch (const std::exception& e) { zend_throw_exception(zend_ce_exception, e.what(), 0); RETURN_THROWS(); }
}

PHP_METHOD(ZTensor, max)
{
    ZEND_PARSE_PARAMETERS_NONE();
    zmatrix_ztensor_object *self_obj = Z_MATRIX_ZTENSOR_P(ZEND_THIS);
    if (!self_obj->tensor) { zend_throw_exception(zend_ce_exception, ZMATRIX_ERR_NOT_INITIALIZED, 0); RETURN_THROWS(); }
    try { RETURN_DOUBLE(self_obj->tensor->max()); }
    catch (const std::exception& e) { zend_throw_exception(zend_ce_exception, e.what(), 0); RETURN_THROWS(); }
}

PHP_METHOD(ZTensor, std)
{
    ZEND_PARSE_PARAMETERS_NONE();
    zmatrix_ztensor_object *self_obj = Z_MATRIX_ZTENSOR_P(ZEND_THIS);
    if (!self_obj->tensor) { zend_throw_exception(zend_ce_exception, ZMATRIX_ERR_NOT_INITIALIZED, 0); RETURN_THROWS(); }
    try { RETURN_DOUBLE(static_cast<double>(self_obj->tensor->std())); } // Cast float->double
    catch (const std::exception& e) { zend_throw_exception(zend_ce_exception, e.what(), 0); RETURN_THROWS(); }
}


// --- Métodos de Propriedade/Informação ---
PHP_METHOD(ZTensor, shape)
{
    ZEND_PARSE_PARAMETERS_NONE();
    zmatrix_ztensor_object *self_obj = Z_MATRIX_ZTENSOR_P(ZEND_THIS);
    if (!self_obj->tensor) { zend_throw_exception(zend_ce_exception, ZMATRIX_ERR_NOT_INITIALIZED, 0); RETURN_THROWS(); }
    array_init(return_value);
    for (size_t dim : self_obj->tensor->shape) { add_next_index_long(return_value, dim); }
}

PHP_METHOD(ZTensor, ndim)
{
    ZEND_PARSE_PARAMETERS_NONE();
    zmatrix_ztensor_object *self_obj = Z_MATRIX_ZTENSOR_P(ZEND_THIS);
    if (!self_obj->tensor) { zend_throw_exception(zend_ce_exception, ZMATRIX_ERR_NOT_INITIALIZED, 0); RETURN_THROWS(); }
    RETURN_LONG(self_obj->tensor->shape.size());
}

PHP_METHOD(ZTensor, size)
{
    ZEND_PARSE_PARAMETERS_NONE();
    zmatrix_ztensor_object *self_obj = Z_MATRIX_ZTENSOR_P(ZEND_THIS);
    if (!self_obj->tensor) { zend_throw_exception(zend_ce_exception, ZMATRIX_ERR_NOT_INITIALIZED, 0); RETURN_THROWS(); }
    RETURN_LONG(self_obj->tensor->size());
}

PHP_METHOD(ZTensor, isEmpty)
{
    ZEND_PARSE_PARAMETERS_NONE();
    zmatrix_ztensor_object *self_obj = Z_MATRIX_ZTENSOR_P(ZEND_THIS);
    RETURN_BOOL(self_obj->tensor == nullptr || self_obj->tensor->empty());
}

PHP_METHOD(ZTensor, toArray)
{
     ZEND_PARSE_PARAMETERS_NONE();
    zmatrix_ztensor_object *self_obj = Z_MATRIX_ZTENSOR_P(ZEND_THIS);
    if (!self_obj->tensor) { zend_throw_exception(zend_ce_exception, ZMATRIX_ERR_NOT_INITIALIZED, 0); RETURN_THROWS(); }
    try { tensor_to_php_array(*(self_obj->tensor), return_value); }
    catch (const std::exception& e) { zend_throw_exception(zend_ce_exception, e.what(), 0); RETURN_THROWS(); }
}

// --- Métodos Estáticos (Criação) ---
ZEND_BEGIN_ARG_INFO_EX(arginfo_ztensor_static_shape, 0, 0, 1)
    ZEND_ARG_ARRAY_INFO(0, shape, 0)
ZEND_END_ARG_INFO()

ZEND_BEGIN_ARG_INFO_EX(arginfo_ztensor_static_shape_value, 0, 0, 1)
    ZEND_ARG_ARRAY_INFO(0, shape, 0)
    ZEND_ARG_TYPE_INFO(0, value, IS_DOUBLE, 1) // value agora é opcional
ZEND_END_ARG_INFO()

PHP_METHOD(ZTensor, zeros)
{
    zval *shape_zv;

    ZEND_PARSE_PARAMETERS_START(1, 1)
        Z_PARAM_ARRAY(shape_zv)
    ZEND_PARSE_PARAMETERS_END();

    std::vector<size_t> shape;
    HashTable *ht = Z_ARRVAL_P(shape_zv);
    zval *dim_zv;

    ZEND_HASH_FOREACH_VAL(ht, dim_zv) {
        if (Z_TYPE_P(dim_zv) != IS_LONG || Z_LVAL_P(dim_zv) < 0) {
            zend_throw_exception(zend_ce_exception, ZMATRIX_ERR_INVALID_SHAPE, 0);
            RETURN_THROWS();
        }
        shape.push_back(Z_LVAL_P(dim_zv));
    } ZEND_HASH_FOREACH_END();

    if (shape.empty()) {
        zend_throw_exception(zend_ce_exception, "Shape cannot be empty for zeros", 0);
        RETURN_THROWS();
    }

    try {
        ZTensor result = ZTensor::zeros(shape);  // <-- CORRETO AQUI
        zmatrix_return_tensor_obj(result, return_value, zmatrix_ce_ZTensor);
    } catch (const std::exception& e) {
        zend_throw_exception(zend_ce_exception, e.what(), 0);
        RETURN_THROWS();
    }
}


// Método estático para criar tensor preenchido com valor específico
PHP_METHOD(ZTensor, full)
{
    zval *shape_zv;
    double value;

    ZEND_PARSE_PARAMETERS_START(2, 2)
        Z_PARAM_ARRAY(shape_zv)
        Z_PARAM_DOUBLE(value)
    ZEND_PARSE_PARAMETERS_END();

    std::vector<size_t> shape;
    HashTable *ht = Z_ARRVAL_P(shape_zv);
    zval *dim_zv;

    ZEND_HASH_FOREACH_VAL(ht, dim_zv) {
        if (Z_TYPE_P(dim_zv) != IS_LONG || Z_LVAL_P(dim_zv) < 0) {
            zend_throw_exception(zend_ce_exception, ZMATRIX_ERR_INVALID_SHAPE, 0);
            RETURN_THROWS();
        }
        shape.push_back(Z_LVAL_P(dim_zv));
    } ZEND_HASH_FOREACH_END();

    if (shape.empty()) {
        zend_throw_exception(zend_ce_exception, "Shape cannot be empty for full", 0);
        RETURN_THROWS();
    }

    try {
        ZTensor result = ZTensor::full(shape, static_cast<float>(value)); // ✅ chama a versão otimizada
        zmatrix_return_tensor_obj(result, return_value, zmatrix_ce_ZTensor);
    } catch (const std::exception& e) {
        zend_throw_exception(zend_ce_exception, e.what(), 0);
        RETURN_THROWS();
    }
}



ZEND_BEGIN_ARG_INFO_EX(arginfo_ztensor_static_identity, 0, 0, 1)
    ZEND_ARG_TYPE_INFO(0, size, IS_LONG, 0)
ZEND_END_ARG_INFO()

PHP_METHOD(ZTensor, identity)
{
     zend_long size;
     ZEND_PARSE_PARAMETERS_START(1, 1)
         Z_PARAM_LONG(size)
     ZEND_PARSE_PARAMETERS_END();
     if (size <= 0) {
         zend_throw_exception(zend_ce_exception, "Identity size must be positive", 0);
         RETURN_THROWS();
     }
     ZTensor result = ZTensor::identity(static_cast<size_t>(size));
     zmatrix_return_tensor_obj(result, return_value, zmatrix_ce_ZTensor);
}

ZEND_BEGIN_ARG_INFO_EX(arginfo_ztensor_static_random, 0, 0, 1)
    ZEND_ARG_ARRAY_INFO(0, shape, 0)
    ZEND_ARG_TYPE_INFO(0, min, IS_DOUBLE, 1)
    ZEND_ARG_TYPE_INFO(0, max, IS_DOUBLE, 1)
ZEND_END_ARG_INFO()

// --- Método estático ZTensor::random otimizado ---
PHP_METHOD(ZTensor, random)
{
    zval *shape_zv;
    double min_val = 0.0, max_val = 1.0;

    ZEND_PARSE_PARAMETERS_START(1, 3)
        Z_PARAM_ARRAY(shape_zv)
        Z_PARAM_OPTIONAL
        Z_PARAM_DOUBLE(min_val)
        Z_PARAM_DOUBLE(max_val)
    ZEND_PARSE_PARAMETERS_END();

    std::vector<size_t> shape;
    HashTable *ht = Z_ARRVAL_P(shape_zv);
    zval *dim_zv;

    ZEND_HASH_FOREACH_VAL(ht, dim_zv) {
        if (Z_TYPE_P(dim_zv) != IS_LONG || Z_LVAL_P(dim_zv) < 0) {
            zend_throw_exception(zend_ce_exception, ZMATRIX_ERR_INVALID_SHAPE, 0);
            RETURN_THROWS();
        }
        shape.push_back(Z_LVAL_P(dim_zv));
    } ZEND_HASH_FOREACH_END();

    if (min_val > max_val) {
        zend_throw_exception(zend_ce_exception, "Minimum value cannot be greater than maximum in random", 0);
        RETURN_THROWS();
    }

    try {
        ZTensor result = ZTensor::random(shape, static_cast<float>(min_val), static_cast<float>(max_val));
        zmatrix_return_tensor_obj(result, return_value, zmatrix_ce_ZTensor);
    } catch (const std::exception& e) {
        zend_throw_exception(zend_ce_exception, e.what(), 0);
        RETURN_THROWS();
    }
}


// --- ARG_INFO para matmul ---
// Aceita 1 argumento: outro ZTensor ou array
ZEND_BEGIN_ARG_INFO_EX(arginfo_ztensor_matmul, 0, 0, 1)
    ZEND_ARG_INFO(0, other) // Aceita ZTensor ou array
    // ZEND_ARG_TYPE_INFO(0, use_blas, _IS_BOOL, 1) // Opcional para usar BLAS (não implementado aqui)
ZEND_END_ARG_INFO()

// --- Implementação do Método matmul ---
PHP_METHOD(ZTensor, matmul)
{
    zval *other_zv;
    // zend_bool use_blas = 1; // Opcional BLAS (não usado nesta implementação C++ simples)

    // Parseia o argumento 'other'
    ZEND_PARSE_PARAMETERS_START(1, 1) // Apenas 1 argumento obrigatório por enquanto
        Z_PARAM_ZVAL(other_zv)
        // Z_PARAM_OPTIONAL // Descomentar se adicionar use_blas
        // Z_PARAM_BOOL(use_blas)
    ZEND_PARSE_PARAMETERS_END();

    // Obtém o ponteiro para o tensor interno do objeto atual ($this)
    zmatrix_ztensor_object *self_obj = Z_MATRIX_ZTENSOR_P(ZEND_THIS);
    // Verifica se o tensor interno está inicializado
    if (!self_obj->tensor) {
        zend_throw_exception(zend_ce_exception, ZMATRIX_ERR_NOT_INITIALIZED, 0);
        RETURN_THROWS(); // Retorna indicando que uma exceção foi lançada
    }

    // Obtém o ponteiro para o tensor do argumento 'other', convertendo de array se necessário
    ZTensor *other_ptr = nullptr;
    ZTensor tmp_other; // Armazena tensor temporário se 'other' for array
    if (!zmatrix_get_tensor_ptr(other_zv, other_ptr, tmp_other, zmatrix_ce_ZTensor)) {
        RETURN_THROWS(); // Retorna se a obtenção/conversão falhar
    }

    try {
        // Chama o método matmul da classe C++ ZTensor
        // Este método C++ deve conter a lógica real da multiplicação,
        // incluindo verificações de shape e a computação.
        ZTensor result = self_obj->tensor->matmul(*other_ptr);

        // Cria um novo objeto PHP ZTensor para retornar o resultado
        zmatrix_return_tensor_obj(result, return_value, zmatrix_ce_ZTensor);

    } catch (const std::exception& e) {
        // Captura exceções C++ (ex: shape incompatível, tensor vazio) e as lança como exceções PHP
        zend_throw_exception(zend_ce_exception, e.what(), 0);
        RETURN_THROWS(); // Retorna indicando que uma exceção foi lançada
    }
}

// === Divide ===
ZEND_BEGIN_ARG_INFO_EX(arginfo_ztensor_divide, 0, 0, 1)
    ZEND_ARG_OBJ_INFO(0, other, ZTensor, 0)
ZEND_END_ARG_INFO()

PHP_METHOD(ZTensor, divide)
{
    zval *other_zv;
    ZEND_PARSE_PARAMETERS_START(1, 1)
        Z_PARAM_ZVAL(other_zv)
    ZEND_PARSE_PARAMETERS_END();

    // 1) Pega o objeto interno
    zmatrix_ztensor_object *self_obj = Z_MATRIX_ZTENSOR_P(ZEND_THIS);
    if (!self_obj->tensor) {
        zend_throw_exception(zend_ce_exception, ZMATRIX_ERR_NOT_INITIALIZED, 0);
        RETURN_THROWS();
    }
    ZTensor &A = *self_obj->tensor;

    // 2) Caso ESCALAR (int ou float)
    if (Z_TYPE_P(other_zv) == IS_LONG || Z_TYPE_P(other_zv) == IS_DOUBLE) {
        float scalar = (Z_TYPE_P(other_zv) == IS_LONG)
            ? static_cast<float>(Z_LVAL_P(other_zv))
            : static_cast<float>(Z_DVAL_P(other_zv));
        A.scalar_divide(scalar);
        ZVAL_ZVAL(return_value, ZEND_THIS, 1, 0);
        return;
    }

    // 3) Caso tensor/array
    ZTensor *other_ptr = nullptr, tmp_other;
    if (!zmatrix_get_tensor_ptr(other_zv, other_ptr, tmp_other, zmatrix_ce_ZTensor)) {
        RETURN_THROWS();
    }
    ZTensor &B = *other_ptr;
    const auto &shapeA = A.shape, &shapeB = B.shape;

    // 4) Vetor‑1D de tamanho 1 → trate como escalar
    if (shapeB.size() == 1 && shapeB[0] == 1) {
        float scalar = B.data.data()[0];
        A.scalar_divide(scalar);
        ZVAL_ZVAL(return_value, ZEND_THIS, 1, 0);
        return;
    }

    // 5) Agora o bloco try/catch para os casos de tensor×tensor e broadcast
    try {
        // 5.1) Mesmos formatos → divisão direta
        if (shapeA == shapeB) {
            A.divide(B);
            ZVAL_ZVAL(return_value, ZEND_THIS, 1, 0);
            return;
        }

        // 5.2) Broadcast 2D×1D: A [M×N] ÷ B [N]
        if (shapeA.size() == 2 && shapeB.size() == 1 && shapeB[0] == shapeA[1]) {
            size_t M = shapeA[0], N = shapeA[1];
            ZTensor C(shapeA);
            float *cdat = C.data.data();
            const float *bdat = B.data.data();
            for (size_t i = 0; i < M; ++i) {
                memcpy(cdat + i * N, bdat, N * sizeof(float));
            }
            A.divide(C);
            ZVAL_ZVAL(return_value, ZEND_THIS, 1, 0);
            return;
        }

        // 5.3) Broadcast inverso: A [N], B [M×N]
        if (shapeA.size() == 1 && shapeB.size() == 2 && shapeA[0] == shapeB[1]) {
            size_t M = shapeB[0], N = shapeB[1];
            ZTensor C(shapeB);
            float *cdat = C.data.data();
            const float *adat = A.data.data();
            for (size_t i = 0; i < M; ++i) {
                memcpy(cdat + i * N, adat, N * sizeof(float));
            }
            C.divide(B); // A (broadcastado) ÷ B
            A = C;
            ZVAL_ZVAL(return_value, ZEND_THIS, 1, 0);
            return;
        }

        // 5.4) Outros casos incompatíveis
        zend_throw_exception(zend_ce_exception,
            "Incompatible shapes for divide() with broadcasting", 0);
        RETURN_THROWS();
    }
    catch (const std::exception &e) {
        zend_throw_exception(zend_ce_exception, e.what(), 0);
        RETURN_THROWS();
    }
}


// === Pow ===
ZEND_BEGIN_ARG_INFO_EX(arginfo_ztensor_pow, 0, 0, 1)
    ZEND_ARG_TYPE_INFO(0, exponent, IS_DOUBLE, 0)
ZEND_END_ARG_INFO()

PHP_METHOD(ZTensor, pow)
{
    double exponent;
    ZEND_PARSE_PARAMETERS_START(1,1) Z_PARAM_DOUBLE(exponent) ZEND_PARSE_PARAMETERS_END();
    zmatrix_ztensor_object *self = Z_MATRIX_ZTENSOR_P(ZEND_THIS);
    if (!self->tensor) { zend_throw_exception(zend_ce_exception, ZMATRIX_ERR_NOT_INITIALIZED,0); RETURN_THROWS(); }
    try { self->tensor->pow((float)exponent); ZVAL_ZVAL(return_value,ZEND_THIS,1,0);} catch(const std::exception &e){ zend_throw_exception(zend_ce_exception,e.what(),0); RETURN_THROWS(); }

}

// === ReLU ===
ZEND_BEGIN_ARG_INFO_EX(arginfo_ztensor_relu, 0, 0, 0)
ZEND_END_ARG_INFO()

PHP_METHOD(ZTensor, relu)
{
    zmatrix_ztensor_object *self_obj = Z_MATRIX_ZTENSOR_P(ZEND_THIS);
    if (!self_obj->tensor) { zend_throw_exception(zend_ce_exception, ZMATRIX_ERR_NOT_INITIALIZED, 0); RETURN_THROWS(); }
    try { self_obj->tensor->relu(); ZVAL_ZVAL(return_value, ZEND_THIS, 1, 0); }
    catch (const std::exception &e) { zend_throw_exception(zend_ce_exception, e.what(), 0); RETURN_THROWS(); }
}

// === Tanh ===
ZEND_BEGIN_ARG_INFO_EX(arginfo_ztensor_tanh, 0, 0, 0)
ZEND_END_ARG_INFO()

PHP_METHOD(ZTensor, tanh)
{
   zmatrix_ztensor_object *self_obj = Z_MATRIX_ZTENSOR_P(ZEND_THIS);
   if (!self_obj->tensor) { zend_throw_exception(zend_ce_exception, ZMATRIX_ERR_NOT_INITIALIZED, 0); RETURN_THROWS(); }
   try { self_obj->tensor->tanh(); ZVAL_ZVAL(return_value, ZEND_THIS, 1, 0); }
   catch (const std::exception &e) { zend_throw_exception(zend_ce_exception, e.what(), 0); RETURN_THROWS(); }
}
// === Exp ===
ZEND_BEGIN_ARG_INFO_EX(arginfo_ztensor_exp, 0, 0, 0)
ZEND_END_ARG_INFO()

PHP_METHOD(ZTensor, exp)
{
    // Implementação similar a ReLU, usando .exp()
    ZEND_PARSE_PARAMETERS_NONE();
    zmatrix_ztensor_object *self = Z_MATRIX_ZTENSOR_P(ZEND_THIS);
    if(!self->tensor){ zend_throw_exception(zend_ce_exception,ZMATRIX_ERR_NOT_INITIALIZED,0); RETURN_THROWS(); }
    try{ self->tensor->exp(); ZVAL_ZVAL(return_value,ZEND_THIS,1,0);}catch(const std::exception &e){ zend_throw_exception(zend_ce_exception,e.what(),0); RETURN_THROWS(); }

}
// === Log ===
ZEND_BEGIN_ARG_INFO_EX(arginfo_ztensor_log, 0, 0, 0)
ZEND_END_ARG_INFO()

PHP_METHOD(ZTensor, log)
{
    ZEND_PARSE_PARAMETERS_NONE();
    zmatrix_ztensor_object *self = Z_MATRIX_ZTENSOR_P(ZEND_THIS);
    if(!self->tensor){ zend_throw_exception(zend_ce_exception,ZMATRIX_ERR_NOT_INITIALIZED,0); RETURN_THROWS(); }
    try{
        self->tensor->log(); // Chama o método void ZTensor::log()
        ZVAL_ZVAL(return_value,ZEND_THIS,1,0); // Retorna $this
    }catch(const std::exception &e){ zend_throw_exception(zend_ce_exception,e.what(),0); RETURN_THROWS(); }
}
// === Sqrt ===
ZEND_BEGIN_ARG_INFO_EX(arginfo_ztensor_sqrt, 0, 0, 0)
ZEND_END_ARG_INFO()

PHP_METHOD(ZTensor, sqrt)
{
    ZEND_PARSE_PARAMETERS_NONE();
    zmatrix_ztensor_object *self = Z_MATRIX_ZTENSOR_P(ZEND_THIS);
    if(!self->tensor){ zend_throw_exception(zend_ce_exception,ZMATRIX_ERR_NOT_INITIALIZED,0); RETURN_THROWS(); }
    try{
        self->tensor->sqrt(); // Chama o método void ZTensor::sqrt()
        ZVAL_ZVAL(return_value,ZEND_THIS,1,0); // Retorna $this
    }catch(const std::exception &e){ zend_throw_exception(zend_ce_exception,e.what(),0); RETURN_THROWS(); }
}

PHP_METHOD(ZTensor, reluDerivative)
{
    zmatrix_ztensor_object *self_obj = Z_MATRIX_ZTENSOR_P(ZEND_THIS);
    if (!self_obj->tensor) { zend_throw_exception(zend_ce_exception, ZMATRIX_ERR_NOT_INITIALIZED, 0); RETURN_THROWS(); }
    try { self_obj->tensor->relu_derivative(); ZVAL_ZVAL(return_value, ZEND_THIS, 1, 0); }
    catch (const std::exception &e) { zend_throw_exception(zend_ce_exception, e.what(), 0); RETURN_THROWS(); }
}

// Tanh
PHP_METHOD(ZTensor, tanhDerivative)
{
    zmatrix_ztensor_object *self_obj = Z_MATRIX_ZTENSOR_P(ZEND_THIS);
    if (!self_obj->tensor) { zend_throw_exception(zend_ce_exception, ZMATRIX_ERR_NOT_INITIALIZED, 0); RETURN_THROWS(); }
    try { self_obj->tensor->tanh_derivative(); ZVAL_ZVAL(return_value, ZEND_THIS, 1, 0); }
    catch (const std::exception &e) { zend_throw_exception(zend_ce_exception, e.what(), 0); RETURN_THROWS(); }
}

// Leaky ReLU (com alpha = 0.01 por padrão)
PHP_METHOD(ZTensor, leakyRelu)
{
    zmatrix_ztensor_object *self_obj = Z_MATRIX_ZTENSOR_P(ZEND_THIS);
    if (!self_obj->tensor) { zend_throw_exception(zend_ce_exception, ZMATRIX_ERR_NOT_INITIALIZED, 0); RETURN_THROWS(); }

    double alpha = 0.01;
    ZEND_PARSE_PARAMETERS_START(0, 1)
        Z_PARAM_OPTIONAL
        Z_PARAM_DOUBLE(alpha)
    ZEND_PARSE_PARAMETERS_END();

    try { self_obj->tensor->leaky_relu(static_cast<float>(alpha)); ZVAL_ZVAL(return_value, ZEND_THIS, 1, 0); }
    catch (const std::exception &e) { zend_throw_exception(zend_ce_exception, e.what(), 0); RETURN_THROWS(); }
}

PHP_METHOD(ZTensor, leakyReluDerivative)
{
    zmatrix_ztensor_object *self_obj = Z_MATRIX_ZTENSOR_P(ZEND_THIS);
    if (!self_obj->tensor) { zend_throw_exception(zend_ce_exception, ZMATRIX_ERR_NOT_INITIALIZED, 0); RETURN_THROWS(); }

    double alpha = 0.01;
    ZEND_PARSE_PARAMETERS_START(0, 1)
        Z_PARAM_OPTIONAL
        Z_PARAM_DOUBLE(alpha)
    ZEND_PARSE_PARAMETERS_END();

    try { self_obj->tensor->leaky_relu_derivative(static_cast<float>(alpha)); ZVAL_ZVAL(return_value, ZEND_THIS, 1, 0); }
    catch (const std::exception &e) { zend_throw_exception(zend_ce_exception, e.what(), 0); RETURN_THROWS(); }
}

PHP_METHOD(ZTensor, softmaxDerivative)
{
    zmatrix_ztensor_object *self_obj = Z_MATRIX_ZTENSOR_P(ZEND_THIS);
    if (!self_obj->tensor) { zend_throw_exception(zend_ce_exception, ZMATRIX_ERR_NOT_INITIALIZED, 0); RETURN_THROWS(); }
    try { self_obj->tensor->softmax_derivative(); ZVAL_ZVAL(return_value, ZEND_THIS, 1, 0); }
    catch (const std::exception &e) { zend_throw_exception(zend_ce_exception, e.what(), 0); RETURN_THROWS(); }
}

ZEND_BEGIN_ARG_INFO_EX(arginfo_ztensor_softmax, 0, 0, 0)
ZEND_END_ARG_INFO()

PHP_METHOD(ZTensor, softmax)
{
    ZEND_PARSE_PARAMETERS_NONE();
    zmatrix_ztensor_object *self_obj = Z_MATRIX_ZTENSOR_P(ZEND_THIS);
    if (!self_obj->tensor) {
        zend_throw_exception(zend_ce_exception, ZMATRIX_ERR_NOT_INITIALIZED, 0);
        RETURN_THROWS();
    }

    try {
        self_obj->tensor->softmax();  // agora void
        ZVAL_ZVAL(return_value, ZEND_THIS, 1, 0);  // retorna o próprio objeto
    } catch (const std::exception& e) {
        zend_throw_exception(zend_ce_exception, e.what(), 0);
        RETURN_THROWS();
    }
}


ZEND_BEGIN_ARG_INFO_EX(arginfo_ztensor_reshape, 0, 0, 1)
    ZEND_ARG_ARRAY_INFO(0, shape, 0)
ZEND_END_ARG_INFO()

/* PHP method implementation */
PHP_METHOD(ZTensor, reshape)
{
    zval *zshape;
    HashTable *ht_shape;
    zval *val;
    std::vector<size_t> new_shape;

    ZEND_PARSE_PARAMETERS_START(1, 1)
        Z_PARAM_ARRAY(zshape)
    ZEND_PARSE_PARAMETERS_END();

    zmatrix_ztensor_object *self_obj = Z_MATRIX_ZTENSOR_P(ZEND_THIS);
    if (!self_obj->tensor) {
        zend_throw_exception(zend_ce_exception, ZMATRIX_ERR_NOT_INITIALIZED, 0);
        RETURN_THROWS();
    }

    /* 1. Extrai HashTable do array PHP */
    ht_shape = Z_ARRVAL_P(zshape);

    /* 2. Converte cada elemento em size_t */
    ZEND_HASH_FOREACH_VAL(ht_shape, val) {
        if (Z_TYPE_P(val) != IS_LONG) {
            zend_throw_exception(zend_ce_exception, "All dimensions must be integers", 0);
            RETURN_THROWS();
        }
        zend_long dim = Z_LVAL_P(val);
        if (dim < 0) {
            zend_throw_exception(zend_ce_exception, "Dimensions must be non-negative", 0);
            RETURN_THROWS();
        }
        new_shape.push_back((size_t) dim);
    } ZEND_HASH_FOREACH_END();

    /* 3. Chama o método C++ reshape */
    ZTensor reshaped = self_obj->tensor->reshape(new_shape);

    /* 4. Empacota o resultado em um novo objeto PHP ZTensor */
    object_init_ex(return_value, zmatrix_ce_ZTensor);
    zmatrix_ztensor_object *res_obj = Z_MATRIX_ZTENSOR_P(return_value);
    res_obj->tensor = new ZTensor(std::move(reshaped));
}

// --- ARG_INFO para métodos estáticos de criação adicionais ---

ZEND_BEGIN_ARG_INFO_EX(arginfo_ztensor_static_randn, 0, 0, 1)
    ZEND_ARG_ARRAY_INFO(0, shape, 0)
    ZEND_ARG_TYPE_INFO_WITH_DEFAULT_VALUE(0, mean, IS_DOUBLE, 0, "0.0")
    ZEND_ARG_TYPE_INFO_WITH_DEFAULT_VALUE(0, std_dev, IS_DOUBLE, 0, "1.0")
ZEND_END_ARG_INFO()

PHP_METHOD(ZTensor, randn)
{
    zval *shape_zv;
    double mean = 0.0;
    double std_dev = 1.0;

    ZEND_PARSE_PARAMETERS_START(1, 3)
        Z_PARAM_ARRAY(shape_zv)
        Z_PARAM_OPTIONAL
        Z_PARAM_DOUBLE(mean)
        Z_PARAM_DOUBLE(std_dev)
    ZEND_PARSE_PARAMETERS_END();

    if (std_dev < 0) {
        zend_throw_exception(zend_ce_exception, "Standard deviation (std_dev) cannot be negative for randn", 0);
        RETURN_THROWS();
    }

    std::vector<size_t> shape_vec;
    // Reutilize sua função zmatrix_zval_to_shape ou implemente a lógica aqui
    // Inicio da lógica de zmatrix_zval_to_shape (simplificado, adicione sua validação robusta)
    HashTable *ht_shape = Z_ARRVAL_P(shape_zv);
    zval *dim_val_zv;
    ZEND_HASH_FOREACH_VAL(ht_shape, dim_val_zv) {
        if (Z_TYPE_P(dim_val_zv) != IS_LONG || Z_LVAL_P(dim_val_zv) < 0) {
            zend_throw_exception(zend_ce_exception, ZMATRIX_ERR_INVALID_SHAPE, 0);
            RETURN_THROWS();
        }
        shape_vec.push_back(Z_LVAL_P(dim_val_zv));
    } ZEND_HASH_FOREACH_END();
    if (shape_vec.empty() && zend_hash_num_elements(ht_shape) > 0) { // Caso de array não vazio mas sem longs válidos
         zend_throw_exception(zend_ce_exception, ZMATRIX_ERR_INVALID_SHAPE, 0);
         RETURN_THROWS();
    }
    // Fim da lógica de zmatrix_zval_to_shape

    try {
        ZTensor result(shape_vec);
        size_t total_size = result.size();
        if (total_size > 0) {
            float* data_ptr = result.data.data();
            std::normal_distribution<float> dist(static_cast<float>(mean), static_cast<float>(std_dev));

            #if HAS_OPENMP
            if (total_size > ZMATRIX_PARALLEL_THRESHOLD) {
                // Cada thread precisa de seu próprio estado de gerador para paralelismo seguro de PRNG.
                // Uma abordagem simples para loops é ter um gerador por thread.
                #pragma omp parallel
                {
                    // Seed local para cada thread para evitar que todas produzam a mesma sequência
                    unsigned int seed = static_cast<unsigned int>(std::chrono::high_resolution_clock::now().time_since_epoch().count()) ^ (omp_get_thread_num() << 16);
                    std::mt19937 thread_local_gen(seed);
                    #pragma omp for schedule(static)
                    for (size_t i = 0; i < total_size; ++i) {
                        data_ptr[i] = dist(thread_local_gen);
                    }
                }
            } else {
                std::mt19937& main_gen = get_global_mt19937(); // Para o caso sequencial
                for (size_t i = 0; i < total_size; ++i) {
                    data_ptr[i] = dist(main_gen);
                }
            }
            #else
            std::mt19937& main_gen = get_global_mt19937();
            for (size_t i = 0; i < total_size; ++i) {
                data_ptr[i] = dist(main_gen);
            }
            #endif
        }
        zmatrix_return_tensor_obj(result, return_value, zmatrix_ce_ZTensor);
    } catch (const std::exception& e) {
        zend_throw_exception(zend_ce_exception, e.what(), 0);
        RETURN_THROWS();
    }
}

ZEND_BEGIN_ARG_INFO_EX(arginfo_ztensor_static_arange, 0, 0, 1)
    ZEND_ARG_TYPE_INFO(0, arg1, IS_DOUBLE, 0) // start_or_stop
    ZEND_ARG_TYPE_INFO_WITH_DEFAULT_VALUE(0, arg2, IS_DOUBLE, 1, "null") // stop (pode ser null se não passado explicitamente)
    ZEND_ARG_TYPE_INFO_WITH_DEFAULT_VALUE(0, arg3, IS_DOUBLE, 0, "1.0") // step
ZEND_END_ARG_INFO()

PHP_METHOD(ZTensor, arange)
{
    double arg1_val; // Pode ser start ou stop
    zval *z_arg2 = nullptr; // Pode ser stop ou null
    zval *z_arg3 = nullptr; // Pode ser step ou null

    ZEND_PARSE_PARAMETERS_START(1, 3)
        Z_PARAM_DOUBLE(arg1_val)
        Z_PARAM_OPTIONAL
        Z_PARAM_ZVAL_EX(z_arg2, 1, 0) // allow_null = true
        Z_PARAM_ZVAL_EX(z_arg3, 1, 0) // allow_null = true
    ZEND_PARSE_PARAMETERS_END();

    float start_val, stop_val, step_val;

    int argc = ZEND_NUM_ARGS();

    if (argc == 1) { // arange(stop)
        start_val = 0.0f;
        stop_val = static_cast<float>(arg1_val);
        step_val = 1.0f;
    } else if (argc == 2) { // arange(start, stop) ou arange(stop, step=null) - o segundo caso não é típico.
                            // A API PHP é arange(float $start_or_stop, ?float $stop = null, float $step = 1.0)
                            // Se z_arg2 (stop) é null, então arg1_val é stop, start=0, step=1 (coberto por argc=1 se for assim)
                            // Se z_arg2 (stop) NÃO é null, então arg1_val é start, z_arg2 é stop.
        start_val = static_cast<float>(arg1_val);
        stop_val = static_cast<float>(zval_get_double(z_arg2)); // z_arg2 deve ter sido passado
        step_val = 1.0f;
    } else { // argc == 3, arange(start, stop, step)
        start_val = static_cast<float>(arg1_val);
        stop_val = static_cast<float>(zval_get_double(z_arg2)); // z_arg2 deve ter sido passado
        step_val = static_cast<float>(zval_get_double(z_arg3)); // z_arg3 deve ter sido passado
    }


    if (step_val == 0.0f) {
        zend_throw_exception(zend_ce_exception, "Step cannot be zero for arange", 0);
        RETURN_THROWS();
    }

    std::vector<float> values;
    if (step_val > 0) {
        if (start_val < stop_val) { // Somente adiciona se o intervalo for válido
            for (float current_val = start_val; current_val < stop_val; current_val += step_val) {
                values.push_back(current_val);
            }
        }
    } else { // step_val < 0
        if (start_val > stop_val) { // Somente adiciona se o intervalo for válido
            for (float current_val = start_val; current_val > stop_val; current_val += step_val) {
                values.push_back(current_val);
            }
        }
    }

    size_t count = values.size();

    try {
        ZTensor result_tensor(std::vector<size_t>{count});
        if (count > 0) {
            float* data_ptr = result_tensor.data.data();
            // std::copy é geralmente eficiente para esta tarefa
            // A paralelização de std::copy ou um loop manual aqui para 'arange' pode não ser
            // tão benéfica quanto para operações mais intensas, dado que 'values' já foi construído.
            // Se 'count' for extremamente grande, e a construção de 'values' for o gargalo,
            // a lógica de cálculo de 'count' e preenchimento direto do tensor seria mais complexa
            // mas potencialmente paralelizável.
            std::copy(values.begin(), values.end(), data_ptr);
        }
        zmatrix_return_tensor_obj(result_tensor, return_value, zmatrix_ce_ZTensor);
    } catch (const std::exception& e) {
        zend_throw_exception(zend_ce_exception, e.what(), 0);
        RETURN_THROWS();
    }
}

ZEND_BEGIN_ARG_INFO_EX(arginfo_ztensor_static_linspace, 0, 0, 2)
    ZEND_ARG_TYPE_INFO(0, start, IS_DOUBLE, 0)
    ZEND_ARG_TYPE_INFO(0, stop, IS_DOUBLE, 0)
    ZEND_ARG_TYPE_INFO_WITH_DEFAULT_VALUE(0, num, IS_LONG, 0, "50")
    ZEND_ARG_TYPE_INFO_WITH_DEFAULT_VALUE(0, endpoint, _IS_BOOL, 0, "true")
ZEND_END_ARG_INFO()

PHP_METHOD(ZTensor, linspace)
{
    double start_val, stop_val;
    zend_long num_l = 50; // Renomeado para evitar conflito com a variável num
    zend_bool endpoint_val = 1; // true

    ZEND_PARSE_PARAMETERS_START(2, 4)
        Z_PARAM_DOUBLE(start_val)
        Z_PARAM_DOUBLE(stop_val)
        Z_PARAM_OPTIONAL
        Z_PARAM_LONG(num_l)
        Z_PARAM_BOOL(endpoint_val)
    ZEND_PARSE_PARAMETERS_END();

    if (num_l < 0) {
        zend_throw_exception(zend_ce_exception, "Number of samples (num) cannot be negative for linspace", 0);
        RETURN_THROWS();
    }

    size_t num = static_cast<size_t>(num_l); // Usar size_t para o tamanho

    if (num == 0) {
        ZTensor result_tensor(std::vector<size_t>{0});
        zmatrix_return_tensor_obj(result_tensor, return_value, zmatrix_ce_ZTensor);
        return;
    }

    try {
        ZTensor result_tensor(std::vector<size_t>{num});
        // result.size() já retornaria num se num > 0, ou 0 se num == 0 (coberto acima)
        // Se num > 0, data_ptr será válido.

        if (num > 0) { // Garante que só acessamos data_ptr se houver elementos
            float* data_ptr = result_tensor.data.data();

            if (num == 1) {
                data_ptr[0] = static_cast<float>(start_val);
            } else {
                float step;
                if (endpoint_val) {
                    step = (static_cast<float>(stop_val) - static_cast<float>(start_val)) / (num - 1);
                } else {
                    step = (static_cast<float>(stop_val) - static_cast<float>(start_val)) / num;
                }

                #if HAS_OPENMP
                if (num > ZMATRIX_PARALLEL_THRESHOLD) {
                    #pragma omp parallel for schedule(static) // Removido simd por enquanto
                    for (size_t i = 0; i < num; ++i) {
                        data_ptr[i] = static_cast<float>(start_val) + i * step;
                    }
                } else {
                    for (size_t i = 0; i < num; ++i) {
                        data_ptr[i] = static_cast<float>(start_val) + i * step;
                    }
                }
                #else
                for (size_t i = 0; i < num; ++i) {
                    data_ptr[i] = static_cast<float>(start_val) + i * step;
                }
                #endif
                if (endpoint_val && num > 1) { // Garante que o último ponto seja exatamente stop_val
                     data_ptr[num - 1] = static_cast<float>(stop_val);
                }
            }
        }
        zmatrix_return_tensor_obj(result_tensor, return_value, zmatrix_ce_ZTensor);
    } catch (const std::exception& e) {
        zend_throw_exception(zend_ce_exception, e.what(), 0);
        RETURN_THROWS();
    }
}


ZEND_BEGIN_ARG_INFO_EX(arginfo_ztensor_static_logspace, 0, 0, 2)
    ZEND_ARG_TYPE_INFO(0, start, IS_DOUBLE, 0)
    ZEND_ARG_TYPE_INFO(0, stop, IS_DOUBLE, 0)
    ZEND_ARG_TYPE_INFO_WITH_DEFAULT_VALUE(0, num, IS_LONG, 0, "50")
    ZEND_ARG_TYPE_INFO_WITH_DEFAULT_VALUE(0, endpoint, _IS_BOOL, 0, "true")
    ZEND_ARG_TYPE_INFO_WITH_DEFAULT_VALUE(0, base, IS_DOUBLE, 0, "10.0")
ZEND_END_ARG_INFO()

PHP_METHOD(ZTensor, logspace)
{
    double start_val, stop_val, base_val = 10.0;
    zend_long num_l = 50; // Renomeado
    zend_bool endpoint_val = 1; // true

    ZEND_PARSE_PARAMETERS_START(2, 5)
        Z_PARAM_DOUBLE(start_val)
        Z_PARAM_DOUBLE(stop_val)
        Z_PARAM_OPTIONAL
        Z_PARAM_LONG(num_l)
        Z_PARAM_BOOL(endpoint_val)
        Z_PARAM_DOUBLE(base_val)
    ZEND_PARSE_PARAMETERS_END();

    if (num_l < 0) {
        zend_throw_exception(zend_ce_exception, "Number of samples (num) cannot be negative for logspace", 0);
        RETURN_THROWS();
    }

    size_t num = static_cast<size_t>(num_l); // Usar size_t

    if (num == 0) {
        ZTensor result_tensor(std::vector<size_t>{0});
        zmatrix_return_tensor_obj(result_tensor, return_value, zmatrix_ce_ZTensor);
        return;
    }

    try {
        ZTensor result_tensor(std::vector<size_t>{num});

        if (num > 0) { // Garante que só acessamos data_ptr se houver elementos
            float* data_ptr = result_tensor.data.data();

            if (num == 1) {
                 data_ptr[0] = std::pow(static_cast<float>(base_val), static_cast<float>(start_val));
            } else {
                float step; // Expoente step
                if (endpoint_val) {
                    step = (static_cast<float>(stop_val) - static_cast<float>(start_val)) / (num - 1);
                } else {
                    step = (static_cast<float>(stop_val) - static_cast<float>(start_val)) / num;
                }

                #if HAS_OPENMP
                if (num > ZMATRIX_PARALLEL_THRESHOLD) {
                    #pragma omp parallel for schedule(static) // Removido simd por enquanto
                    for (size_t i = 0; i < num; ++i) {
                        data_ptr[i] = std::pow(static_cast<float>(base_val), static_cast<float>(start_val) + i * step);
                    }
                } else {
                    for (size_t i = 0; i < num; ++i) {
                        data_ptr[i] = std::pow(static_cast<float>(base_val), static_cast<float>(start_val) + i * step);
                    }
                }
                #else
                for (size_t i = 0; i < num; ++i) {
                    data_ptr[i] = std::pow(static_cast<float>(base_val), static_cast<float>(start_val) + i * step);
                }
                #endif
                 if (endpoint_val && num > 1) { // Garante que o último ponto seja exatamente base^stop
                     data_ptr[num - 1] = std::pow(static_cast<float>(base_val), static_cast<float>(stop_val));
                }
            }
        }
        zmatrix_return_tensor_obj(result_tensor, return_value, zmatrix_ce_ZTensor);
    } catch (const std::exception& e) {
        zend_throw_exception(zend_ce_exception, e.what(), 0);
        RETURN_THROWS();
    }
}

ZEND_BEGIN_ARG_INFO_EX(arginfo_ztensor_static_eye, 0, 0, 1)
    ZEND_ARG_TYPE_INFO(0, N, IS_LONG, 0)
    // Para M, como é opcional e pode ser nulo para indicar M=N.
    ZEND_ARG_TYPE_INFO_WITH_DEFAULT_VALUE(0, M, IS_LONG, 1, "null")
    ZEND_ARG_TYPE_INFO_WITH_DEFAULT_VALUE(0, k, IS_LONG, 0, "0")
ZEND_END_ARG_INFO()

PHP_METHOD(ZTensor, eye)
{
    zend_long N_val, M_val_opt = -1, k_val = 0; // M_val_opt = -1 para indicar não fornecido
    zval *M_zv = nullptr;

    ZEND_PARSE_PARAMETERS_START(1, 3)
        Z_PARAM_LONG(N_val)
        Z_PARAM_OPTIONAL
        Z_PARAM_ZVAL_EX(M_zv, 1, 0) // Permite null
        Z_PARAM_LONG(k_val)
    ZEND_PARSE_PARAMETERS_END();

    if (N_val < 0) {
        zend_throw_exception(zend_ce_exception, "Number of rows N cannot be negative for eye", 0);
        RETURN_THROWS();
    }

    size_t rows = static_cast<size_t>(N_val);
    size_t cols;

    if (M_zv == nullptr || Z_TYPE_P(M_zv) == IS_NULL) {
        cols = rows; // Matriz quadrada se M não for fornecido
    } else {
        M_val_opt = zval_get_long(M_zv);
        if (M_val_opt < 0) {
            zend_throw_exception(zend_ce_exception, "Number of columns M cannot be negative for eye", 0);
            RETURN_THROWS();
        }
        cols = static_cast<size_t>(M_val_opt);
    }

    try {
        // Construtor ZTensor já preenche com zeros
        ZTensor result_tensor({rows, cols});
        size_t total_size_eye = result_tensor.size();

        if (rows > 0 && cols > 0 && total_size_eye > 0) { // Só preenche a diagonal se a matriz não for vazia
            float* data_ptr = result_tensor.data.data();
            // O número de elementos a serem setados na diagonal k é no máximo min(rows, cols)
            // A paralelização aqui pode ter mais overhead do que benefício a menos que rows/cols sejam muito grandes.
            // Usaremos um threshold menor ou específico se necessário, ou o ZMATRIX_PARALLEL_THRESHOLD.
            // O loop itera 'rows' vezes, mas a condição interna limita as escritas.
            #if HAS_OPENMP
            if (rows > ZMATRIX_PARALLEL_THRESHOLD / (cols > 0 ? std::min((size_t)100, cols) : 100) ) { // Heurística para paralelizar
                #pragma omp parallel for schedule(static)
                for (size_t i = 0; i < rows; ++i) {
                    zend_long j_long = static_cast<zend_long>(i) + k_val; // k_val pode ser negativo
                    if (j_long >= 0 && static_cast<size_t>(j_long) < cols) {
                        data_ptr[i * cols + static_cast<size_t>(j_long)] = 1.0f;
                    }
                }
            } else {
                for (size_t i = 0; i < rows; ++i) {
                    zend_long j_long = static_cast<zend_long>(i) + k_val;
                    if (j_long >= 0 && static_cast<size_t>(j_long) < cols) {
                        data_ptr[i * cols + static_cast<size_t>(j_long)] = 1.0f;
                    }
                }
            }
            #else
            for (size_t i = 0; i < rows; ++i) {
                zend_long j_long = static_cast<zend_long>(i) + k_val;
                if (j_long >= 0 && static_cast<size_t>(j_long) < cols) {
                    data_ptr[i * cols + static_cast<size_t>(j_long)] = 1.0f;
                }
            }
            #endif
        }
        zmatrix_return_tensor_obj(result_tensor, return_value, zmatrix_ce_ZTensor);
    } catch (const std::exception& e) {
        zend_throw_exception(zend_ce_exception, e.what(), 0);
        RETURN_THROWS();
    }
}


 ZEND_BEGIN_ARG_INFO_EX(arginfo_ztensor_dot, 0, 0, 1)
     ZEND_ARG_INFO(0, other)
 ZEND_END_ARG_INFO()

PHP_METHOD(ZTensor, dot)
{
    zval *other_zv;
    ZEND_PARSE_PARAMETERS_START(1, 1)
        Z_PARAM_ZVAL(other_zv)
    ZEND_PARSE_PARAMETERS_END();

    zmatrix_ztensor_object *self_obj = Z_MATRIX_ZTENSOR_P(ZEND_THIS);
    if (!self_obj->tensor) {
        zend_throw_exception(zend_ce_exception, ZMATRIX_ERR_NOT_INITIALIZED, 0);
        RETURN_THROWS();
    }

    ZTensor *tensor_A = self_obj->tensor;
    ZTensor *tensor_B_ptr = nullptr;
    ZTensor tmp_tensor_B;

    if (!zmatrix_get_tensor_ptr(other_zv, tensor_B_ptr, tmp_tensor_B, zmatrix_ce_ZTensor)) { //
        RETURN_THROWS();
    }
    ZTensor& tensor_B = *tensor_B_ptr;

    try {
        size_t a_ndim = tensor_A->shape.size();
        size_t b_ndim = tensor_B.shape.size();

        // Caso 1: Ambos 1D (produto interno de vetores)
        if (a_ndim == 1 && b_ndim == 1) {
            if (tensor_A->shape[0] == 0 && tensor_B.shape[0] == 0) { // Ambos vazios
                 RETURN_DOUBLE(0.0);
            }
            if (tensor_A->shape[0] != tensor_B.shape[0]) {
                throw std::runtime_error("1D vectors with incompatible shapes for dot product");
            }
            if (tensor_A->shape[0] == 0) { // Ambos tamanho 0 devido à condição anterior
                 RETURN_DOUBLE(0.0);
            }

            float sum_product = 0.0f;
            const float* a_data = tensor_A->data.data();
            const float* b_data = tensor_B.data.data();
            size_t N = tensor_A->shape[0];

            // A redução para float pode ser menos precisa, mas ZTensor usa float.
            // Se precisão dupla for crítica, um acumulador double seria melhor aqui.
            #if HAS_OPENMP
            if (N > ZMATRIX_PARALLEL_THRESHOLD) {
                double omp_sum_product = 0.0; // Use double para redução paralela para precisão
                #pragma omp parallel for reduction(+:omp_sum_product) schedule(static)
                for (size_t i = 0; i < N; ++i) {
                    omp_sum_product += static_cast<double>(a_data[i]) * static_cast<double>(b_data[i]);
                }
                sum_product = static_cast<float>(omp_sum_product);
            } else {
                for (size_t i = 0; i < N; ++i) {
                    sum_product += a_data[i] * b_data[i];
                }
            }
            #else
            for (size_t i = 0; i < N; ++i) {
                sum_product += a_data[i] * b_data[i];
            }
            #endif
            RETURN_DOUBLE(static_cast<double>(sum_product));
        }
        // Caso 2: A é 2D, B é 2D (multiplicação de matrizes)
        else if (a_ndim == 2 && b_ndim == 2) {
            // Delega para o método matmul existente (que já faz validação de shape)
            ZTensor result_tensor = tensor_A->matmul(tensor_B); // matmul já retorna ZTensor
            zmatrix_return_tensor_obj(result_tensor, return_value, zmatrix_ce_ZTensor);
        }
        // Caso 3: A é 2D, B é 1D (produto matriz-vetor)
        else if (a_ndim == 2 && b_ndim == 1) {
            if (tensor_A->shape.empty() || tensor_B.shape.empty()) {
                 throw std::runtime_error("Empty tensor cannot be used in matrix-vector product");
            }
            if (tensor_A->shape[1] != tensor_B.shape[0]) {
                throw std::runtime_error("Incompatible shapes for matrix-vector product (A.cols != B.rows)");
            }
            if (tensor_A->shape[1] == 0) { // Caso onde A é Mx0 e B é 0x1, resultado é Mx1 de zeros
                ZTensor result_tensor({tensor_A->shape[0]}); // Já zerado pelo construtor
                zmatrix_return_tensor_obj(result_tensor, return_value, zmatrix_ce_ZTensor);
                return;
            }


            size_t M = tensor_A->shape[0]; // Linhas de A
            size_t K = tensor_A->shape[1]; // Colunas de A (e tamanho de B)

            ZTensor result_tensor({M}); // Resultado é um vetor de tamanho M (já zerado)
            if (M == 0) { // Se A é 0xK, resultado é vetor de tamanho 0
                 zmatrix_return_tensor_obj(result_tensor, return_value, zmatrix_ce_ZTensor);
                 return;
            }


            const float* a_data = tensor_A->data.data();
            const float* b_data = tensor_B.data.data();
            float* c_data = result_tensor.data.data();

            // C[i] = sum_j (A[i,j] * B[j])
            // Pode usar cblas_sgemv se CBLAS estiver disponível e configurado
            // #ifdef HAVE_CBLAS (ou similar)
            // cblas_sgemv(CblasRowMajor, CblasNoTrans, M, K, 1.0f, a_data, K, b_data, 1, 0.0f, c_data, 1);
            // #else
            // Loop manual se CBLAS não estiver disponível/usado para sgemv
            #if HAS_OPENMP
            if (M * K > ZMATRIX_PARALLEL_THRESHOLD) {
                #pragma omp parallel for schedule(static)
                for (size_t i = 0; i < M; ++i) {
                    float row_sum = 0.0f;
                    for (size_t j = 0; j < K; ++j) {
                        row_sum += a_data[i * K + j] * b_data[j];
                    }
                    c_data[i] = row_sum;
                }
            } else {
                 for (size_t i = 0; i < M; ++i) {
                    float row_sum = 0.0f;
                    for (size_t j = 0; j < K; ++j) {
                        row_sum += a_data[i * K + j] * b_data[j];
                    }
                    c_data[i] = row_sum;
                }
            }
            #else
            for (size_t i = 0; i < M; ++i) {
                float row_sum = 0.0f;
                for (size_t j = 0; j < K; ++j) {
                    row_sum += a_data[i * K + j] * b_data[j];
                }
                c_data[i] = row_sum;
            }
            #endif
            // #endif // Fim do else para HAVE_CBLAS (se usado)
            zmatrix_return_tensor_obj(result_tensor, return_value, zmatrix_ce_ZTensor);
        }
        // TODO: Adicionar outros casos (ex: 1D . 2D) ou N-D se necessário
        else {
            throw std::runtime_error("Unsupported shape combination for dot product");
        }

    } catch (const std::exception& e) {
        zend_throw_exception(zend_ce_exception, e.what(), 0);
        RETURN_THROWS();
    }
}

ZEND_BEGIN_ARG_INFO_EX(arginfo_ztensor_key, 0, 0, 1)
    ZEND_ARG_ARRAY_INFO(0, indices, 0)
ZEND_END_ARG_INFO()

PHP_METHOD(ZTensor, key)
{
    zval *indices_zv;
    ZEND_PARSE_PARAMETERS_START(1, 1)
        Z_PARAM_ARRAY(indices_zv)
    ZEND_PARSE_PARAMETERS_END();

    zmatrix_ztensor_object *self_obj = Z_MATRIX_ZTENSOR_P(ZEND_THIS);
    if (!self_obj->tensor) {
        zend_throw_exception(zend_ce_exception, ZMATRIX_ERR_NOT_INITIALIZED, 0);
        RETURN_THROWS();
    }

    /* Converte o array PHP de índices para std::vector<size_t> */
    HashTable   *ht_idx = Z_ARRVAL_P(indices_zv);
    zval        *idx_zv;
    std::vector<size_t> indices;
    zend_ulong   idx_key;
    zend_string *str_key;

    /* Evita warning “idx_key set but not used” */
    (void) idx_key;

    ZEND_HASH_FOREACH_KEY_VAL(ht_idx, idx_key, str_key, idx_zv) {
        /* Esperamos índices numéricos sequenciais (0,1,2,…) */
        if (str_key != nullptr) {
            zend_throw_exception(zend_ce_exception,
                "ZTensor::key() accepts only numerically indexed arrays", 0);
            RETURN_THROWS();
        }
        /* Cada valor deve ser inteiro >= 0 */
        if (Z_TYPE_P(idx_zv) != IS_LONG) {
            zend_throw_exception(zend_ce_exception,
                "ZTensor::key() expects each index to be an integer", 0);
            RETURN_THROWS();
        }
        zend_long l = Z_LVAL_P(idx_zv);
        if (l < 0) {
            zend_throw_exception(zend_ce_exception,
                "ZTensor::key() indices must be >= 0", 0);
            RETURN_THROWS();
        }
        indices.push_back((size_t) l);
    } ZEND_HASH_FOREACH_END();

    try {
        /* Acessa o elemento; at() lança std::out_of_range se fora de limites */
        float value = self_obj->tensor->at(indices);
        /* Retorna como double (PHP não tem float puro) */
        RETURN_DOUBLE((double) value);
    } catch (const std::out_of_range &e) {
        zend_throw_exception(zend_ce_exception, "Index out of tensor bounds", 0);
        RETURN_THROWS();
    } catch (const std::exception &e) {
        zend_throw_exception(zend_ce_exception, e.what(), 0);
        RETURN_THROWS();
    }
}
ZEND_BEGIN_ARG_INFO_EX(arginfo_ztensor_ones, 0, 0, 1)
    ZEND_ARG_ARRAY_INFO(0, shape, 0)
ZEND_END_ARG_INFO()

PHP_METHOD(ZTensor, ones)
{
    zval *shape_zv;
    ZEND_PARSE_PARAMETERS_START(1, 1)
        Z_PARAM_ARRAY(shape_zv)
    ZEND_PARSE_PARAMETERS_END();

    std::vector<size_t> shape;
    HashTable *ht = Z_ARRVAL_P(shape_zv);
    zval *dim_zv;
    ZEND_HASH_FOREACH_VAL(ht, dim_zv) {
        if (Z_TYPE_P(dim_zv) != IS_LONG || Z_LVAL_P(dim_zv) < 0) {
            zend_throw_exception(zend_ce_exception, ZMATRIX_ERR_INVALID_SHAPE, 0);
            RETURN_THROWS();
        }
        shape.push_back(Z_LVAL_P(dim_zv));
    } ZEND_HASH_FOREACH_END();

    if (shape.empty()) {
        zend_throw_exception(zend_ce_exception, "Shape cannot be empty for ones", 0);
        RETURN_THROWS();
    }

    try {
        ZTensor result(shape);
        std::fill(result.data.begin(), result.data.end(), 1.0f);
        zmatrix_return_tensor_obj(result, return_value, zmatrix_ce_ZTensor);
    } catch (const std::exception& e) {
        zend_throw_exception(zend_ce_exception, e.what(), 0);
        RETURN_THROWS();
    }
}

ZEND_BEGIN_ARG_INFO_EX(arginfo_ztensor_static_clip_range, 0, 0, 3)
    ZEND_ARG_INFO(0, input)
    ZEND_ARG_TYPE_INFO(0, min, IS_DOUBLE, 0)
    ZEND_ARG_TYPE_INFO(0, max, IS_DOUBLE, 0)
ZEND_END_ARG_INFO()

PHP_METHOD(ZTensor, clip)
{
    zval *input_zv;
    double min_val, max_val;

    ZEND_PARSE_PARAMETERS_START(3, 3)
        Z_PARAM_ZVAL(input_zv)
        Z_PARAM_DOUBLE(min_val)
        Z_PARAM_DOUBLE(max_val)
    ZEND_PARSE_PARAMETERS_END();

    ZTensor *input_tensor = nullptr;
    ZTensor tmp_tensor;

    if (!zmatrix_get_tensor_ptr(input_zv, input_tensor, tmp_tensor, zmatrix_ce_ZTensor)) {
        RETURN_THROWS();
    }

    try {
        ZTensor result = *input_tensor;  // cópia do tensor de entrada

        const size_t N = result.size();
        float * __restrict__ a = result.data.data();
        const float fmin = static_cast<float>(min_val);
        const float fmax = static_cast<float>(max_val);

        #if HAS_OPENMP
        if (N > ZMATRIX_PARALLEL_THRESHOLD) {
            #pragma omp parallel for simd schedule(static)
            for (size_t i = 0; i < N; ++i) {
                a[i] = std::max(fmin, std::min(fmax, a[i]));
            }
        } else
        #endif
        {
            for (size_t i = 0; i < N; ++i) {
                a[i] = std::max(fmin, std::min(fmax, a[i]));
            }
        }

        zmatrix_return_tensor_obj(result, return_value, zmatrix_ce_ZTensor);
    } catch (const std::exception& e) {
        zend_throw_exception(zend_ce_exception, e.what(), 0);
        RETURN_THROWS();
    }
}



ZEND_BEGIN_ARG_INFO_EX(arginfo_ztensor_sum_flex, 0, 0, 1)
    ZEND_ARG_INFO(0, other) // Ztensor ou array
    ZEND_ARG_TYPE_INFO_WITH_DEFAULT_VALUE(0, axis, IS_LONG, 1, "null")
ZEND_END_ARG_INFO()


PHP_METHOD(ZTensor, sum)
{
    zval *other_zv;
    zval *axis_zv = nullptr;
    ZEND_PARSE_PARAMETERS_START(1, 2)
        Z_PARAM_ZVAL(other_zv)
        Z_PARAM_OPTIONAL
        Z_PARAM_ZVAL(axis_zv)
    ZEND_PARSE_PARAMETERS_END();

    zmatrix_ztensor_object *self_obj = Z_MATRIX_ZTENSOR_P(ZEND_THIS);
    if (!self_obj->tensor) {
        zend_throw_exception(zend_ce_exception, ZMATRIX_ERR_NOT_INITIALIZED, 0);
        RETURN_THROWS();
    }

    ZTensor *other_ptr = nullptr;
    ZTensor tmp_other;

    if (!zmatrix_get_tensor_ptr(other_zv, other_ptr, tmp_other, zmatrix_ce_ZTensor)) {
        RETURN_THROWS();
    }

    try {
        if (!axis_zv || Z_TYPE_P(axis_zv) == IS_NULL) {
            // axis não fornecido → retorna soma total (escalares somados)
            double total = self_obj->tensor->sum();
            ZTensor scalar_tensor({1});
            scalar_tensor.data[0] = static_cast<float>(total);
            zmatrix_return_tensor_obj(scalar_tensor, return_value, zmatrix_ce_ZTensor);
        } else {
            zend_long axis = Z_LVAL_P(axis_zv);
            self_obj->tensor->soma(*other_ptr, static_cast<int>(axis));
            ZVAL_ZVAL(return_value, ZEND_THIS, 1, 0); // retorno this
        }
    } catch (const std::exception &e) {
        zend_throw_exception(zend_ce_exception, e.what(), 0);
        RETURN_THROWS();
    }
}

ZEND_BEGIN_ARG_INFO_EX(arginfo_ztensor_broadcast, 0, 0, 1)
    ZEND_ARG_OBJ_INFO(0, bias, ZMatrix\\Ztensor, 0)
ZEND_END_ARG_INFO()

PHP_METHOD(Ztensor, broadcast)
{
    zval *bias_zv;
        ZEND_PARSE_PARAMETERS_START(1, 1)
            Z_PARAM_ZVAL(bias_zv)
        ZEND_PARSE_PARAMETERS_END();

        // Obtém o ponteiro para o objeto atual (self) e verifica inicialização
        zmatrix_ztensor_object *self_obj = Z_MATRIX_ZTENSOR_P(ZEND_THIS);
        if (!self_obj->tensor) {
            zend_throw_exception(zend_ce_exception, ZMATRIX_ERR_NOT_INITIALIZED, 0);
            RETURN_THROWS();
        }

        // Obtém o ponteiro para o tensor "bias" (pode ser array ou ZTensor)
        ZTensor *bias_ptr = nullptr;
        ZTensor tmp_bias;
        if (!zmatrix_get_tensor_ptr(bias_zv, bias_ptr, tmp_bias, zmatrix_ce_ZTensor)) {
            RETURN_THROWS();
        }

        try {
            ZTensor &self_tensor = *self_obj->tensor;
            ZTensor &bias_tensor = *bias_ptr;

            const std::vector<size_t> &shapeA = self_tensor.shape;
            const std::vector<size_t> &shapeB = bias_tensor.shape;

            // 1. Verifica compatibilidade de broadcast:
            //    shapes são comparados a partir do fim (direita):
            size_t ndimA = shapeA.size();
            size_t ndimB = shapeB.size();
            for (size_t i = 0; i < ndimB; ++i) {
                size_t dimA = shapeA[ndimA - 1 - i];
                size_t dimB = shapeB[ndimB - 1 - i];
                if (dimB != 1 && dimB != dimA) {
                    throw std::runtime_error("Incompatible for broadcast: dimension " +
                        std::to_string(dimB) + " x " + std::to_string(dimA));
                }
            }

            // 2. Cria o resultado com a forma de self_tensor
            ZTensor result(shapeA);
            size_t total_size = result.size();
            if (total_size == 0) {
                // Se for tensor vazio, basta retornar result (vazio)
                zmatrix_return_tensor_obj(result, return_value, zmatrix_ce_ZTensor);
                return;
            }

            // 3. Pré-computa strides do self e do bias
            const std::vector<size_t> &stridesA = self_tensor.strides;
            const std::vector<size_t> &stridesB = bias_tensor.strides;

            // 4. Para cada elemento em result, calcula o índice correspondente em bias
            //    e copia o valor. O critério:
            //    - Se bias_shape[d] == 1 → sempre usa índice 0 nessa dimensão
            //    - Senão → usa o índice em self naquela dimensão
            const float *dataB = bias_tensor.data.data();
            float *dataR = result.data.data();

            std::vector<size_t> indexA(ndimA), indexB(ndimB);
            for (size_t lin = 0; lin < total_size; ++lin) {
                // 4.1. Reconstrói o índice multidimensional de "lin" em shapeA
                size_t rem = lin;
                for (size_t d = 0; d < ndimA; ++d) {
                    indexA[d] = rem / stridesA[d];
                    rem = rem % stridesA[d];
                }

                // 4.2. Mapeia em indexB (direita-alinhado)
                //      Se bias tem menos dims, as dimensões altas de indexB são consideradas "travadas" em zero
                size_t offsetB = 0;
                if (ndimB == 0) {
                    // bias é escalar: sempre offsetB = 0
                    offsetB = 0;
                } else {
                    // calcula deslocamento do índice multidimensional de bias
                    // alinhando a direita:
                    size_t diff = ndimA - ndimB;
                    for (size_t db = 0; db < ndimB; ++db) {
                        size_t dimB = shapeB[db];
                        size_t dimIndexA = indexA[diff + db];
                        size_t idxB = (dimB == 1 ? 0 : dimIndexA);
                        offsetB += idxB * stridesB[db];
                    }
                }

                // 4.3. Copia do bias
                dataR[lin] = dataB[offsetB];
            }

            // 5. Retorna um novo objeto PHP contendo 'result'
            zmatrix_return_tensor_obj(result, return_value, zmatrix_ce_ZTensor);
        } catch (const std::exception &e) {
            zend_throw_exception(zend_ce_exception, e.what(), 0);
            RETURN_THROWS();
        }
}

// --- ARG_INFO para greater (element-wise “>”) ---
ZEND_BEGIN_ARG_INFO_EX(arginfo_ztensor_greater, 0, 0, 1)
    ZEND_ARG_INFO(0, other) // Aceita ZTensor ou array
ZEND_END_ARG_INFO()

PHP_METHOD(ZTensor, greater)
{
    zval *other_zv;
    ZEND_PARSE_PARAMETERS_START(1, 1)
        Z_PARAM_ZVAL(other_zv)
    ZEND_PARSE_PARAMETERS_END();

    zmatrix_ztensor_object *self_obj = Z_MATRIX_ZTENSOR_P(ZEND_THIS);
    if (!self_obj->tensor) {
        zend_throw_exception(zend_ce_exception, ZMATRIX_ERR_NOT_INITIALIZED, 0);
        RETURN_THROWS();
    }

    ZTensor *other_ptr = nullptr;
    ZTensor tmp_other;

    if (!zmatrix_get_tensor_ptr(other_zv, other_ptr, tmp_other, zmatrix_ce_ZTensor)) {
        RETURN_THROWS();
    }

    try {
        ZTensor result = self_obj->tensor->greater_than(*other_ptr);
        zmatrix_return_tensor_obj(result, return_value, zmatrix_ce_ZTensor);
    } catch (const std::exception& e) {
        zend_throw_exception(zend_ce_exception, e.what(), 0);
        RETURN_THROWS();
    }
}




// --- ARG_INFO para minimum e maximum ---
ZEND_BEGIN_ARG_INFO_EX(arginfo_ztensor_minimum, 0, 0, 2)
    ZEND_ARG_INFO(0, a) // ZTensor ou array
    ZEND_ARG_INFO(0, b) // float
ZEND_END_ARG_INFO()

// --- Método estático minimum ---
PHP_METHOD(ZTensor, minimum)
{
    zval *a_zv;
    double b;
    ZEND_PARSE_PARAMETERS_START(2, 2)
        Z_PARAM_ZVAL(a_zv)
        Z_PARAM_DOUBLE(b)
    ZEND_PARSE_PARAMETERS_END();

    // Converte 'a' para ZTensor*
    ZTensor *A_ptr = nullptr;
    ZTensor tmpA;
    if (!zmatrix_get_tensor_ptr(a_zv, A_ptr, tmpA, zmatrix_ce_ZTensor)) {
        RETURN_THROWS();
    }

    try {
        ZTensor &A = *A_ptr;
        size_t N = A.size();
        ZTensor result(A.shape);
        float *res_data = result.data.data();
        const float *a_data = A.data.data();
        float scalar = static_cast<float>(b);

        #if HAS_OPENMP
        if (N > ZMATRIX_PARALLEL_THRESHOLD) {
            #pragma omp parallel for simd schedule(static)
            for (size_t i = 0; i < N; ++i) {
                res_data[i] = (a_data[i] < scalar ? a_data[i] : scalar);
            }
        } else {
            for (size_t i = 0; i < N; ++i) {
                res_data[i] = (a_data[i] < scalar ? a_data[i] : scalar);
            }
        }
        #else
        for (size_t i = 0; i < N; ++i) {
            res_data[i] = (a_data[i] < scalar ? a_data[i] : scalar);
        }
        #endif

        zmatrix_return_tensor_obj(result, return_value, zmatrix_ce_ZTensor);
    } catch (const std::exception &e) {
        zend_throw_exception(zend_ce_exception, e.what(), 0);
        RETURN_THROWS();
    }
}

ZEND_BEGIN_ARG_INFO_EX(arginfo_ztensor_maximum, 0, 0, 2)
    ZEND_ARG_INFO(0, a) // ZTensor ou array
    ZEND_ARG_INFO(0, b) // float
ZEND_END_ARG_INFO()

// --- Método estático maximum ---
PHP_METHOD(ZTensor, maximum)
{
    zval *a_zv;
    double b;
    ZEND_PARSE_PARAMETERS_START(2, 2)
        Z_PARAM_ZVAL(a_zv)
        Z_PARAM_DOUBLE(b)
    ZEND_PARSE_PARAMETERS_END();

    // Converte 'a' para ZTensor*
    ZTensor *A_ptr = nullptr;
    ZTensor tmpA;
    if (!zmatrix_get_tensor_ptr(a_zv, A_ptr, tmpA, zmatrix_ce_ZTensor)) {
        RETURN_THROWS();
    }

    try {
        ZTensor &A = *A_ptr;
        size_t N = A.size();
        ZTensor result(A.shape);
        float *res_data = result.data.data();
        const float *a_data = A.data.data();
        float scalar = static_cast<float>(b);

        #if HAS_OPENMP
        if (N > ZMATRIX_PARALLEL_THRESHOLD) {
            #pragma omp parallel for simd schedule(static)
            for (size_t i = 0; i < N; ++i) {
                res_data[i] = (a_data[i] > scalar ? a_data[i] : scalar);
            }
        } else {
            for (size_t i = 0; i < N; ++i) {
                res_data[i] = (a_data[i] > scalar ? a_data[i] : scalar);
            }
        }
        #else
        for (size_t i = 0; i < N; ++i) {
            res_data[i] = (a_data[i] > scalar ? a_data[i] : scalar);
        }
        #endif

        zmatrix_return_tensor_obj(result, return_value, zmatrix_ce_ZTensor);
    } catch (const std::exception &e) {
        zend_throw_exception(zend_ce_exception, e.what(), 0);
        RETURN_THROWS();
    }
}

ZEND_BEGIN_ARG_INFO_EX(arginfo_ztensor_scalarDivide, 0, 0, 1)
   ZEND_ARG_INFO(0, scalar) // pode ser float|int|ZTensor|array
ZEND_END_ARG_INFO()

PHP_METHOD(ZTensor, scalarDivide)
{
    zval *other_zv;
    ZEND_PARSE_PARAMETERS_START(1, 1)
        Z_PARAM_ZVAL(other_zv)
    ZEND_PARSE_PARAMETERS_END();

    zmatrix_ztensor_object *self_obj = Z_MATRIX_ZTENSOR_P(ZEND_THIS);
    if (!self_obj->tensor) {
        zend_throw_exception(zend_ce_exception, ZMATRIX_ERR_NOT_INITIALIZED, 0);
        RETURN_THROWS();
    }

    try {
        ZTensor &A = *self_obj->tensor;

        // 1) Se for escalar (int ou float), divide elemento a elemento
        if (Z_TYPE_P(other_zv) == IS_LONG || Z_TYPE_P(other_zv) == IS_DOUBLE) {
            float scalar = (Z_TYPE_P(other_zv) == IS_LONG)
                ? (float)Z_LVAL_P(other_zv)
                : (float)Z_DVAL_P(other_zv);
            A.scalar_divide(scalar);
            ZVAL_ZVAL(return_value, ZEND_THIS, 1, 0);
            return;
        }

        // 2) Caso tensor/array: converter para ZTensor*
        ZTensor *B_ptr = nullptr;
        ZTensor tmpB;
        if (!zmatrix_get_tensor_ptr(other_zv, B_ptr, tmpB, zmatrix_ce_ZTensor)) {
            RETURN_THROWS();
        }
        ZTensor &B = *B_ptr;

        // 3) Mesmos formatos → divisão direta
        if (A.shape == B.shape) {
            A.divide(B);
            ZVAL_ZVAL(return_value, ZEND_THIS, 1, 0);
            return;
        }

        // 4) Broadcast linha: A [M×N], B [N]
        const auto &shapeA = A.shape;
        const auto &shapeB = B.shape;
        if (shapeA.size() == 2 && shapeB.size() == 1 && shapeB[0] == shapeA[1]) {
            size_t M = shapeA[0], N = shapeA[1];
            ZTensor C(shapeA);
            float *cd = C.data.data();
            const float *bd = B.data.data();
            for (size_t i = 0; i < M; ++i) {
                memcpy(cd + i*N, bd, N * sizeof(float));
            }
            A.divide(C);
            ZVAL_ZVAL(return_value, ZEND_THIS, 1, 0);
            return;
        }

        // 5) Broadcast inverso não suportado
        if (shapeA.size() == 1 && shapeB.size() == 2 && shapeA[0] == shapeB[1]) {
            throw std::runtime_error("For reverse broadcasting, call B->scalarDivide(A)");
        }

        throw std::runtime_error(ZMATRIX_ERR_SHAPE_MISMATCH);

    } catch (const std::exception &e) {
        zend_throw_exception(zend_ce_exception, e.what(), 0);
        RETURN_THROWS();
    }
}

ZEND_BEGIN_ARG_INFO_EX(arginfo_ztensor_copy, 0, 0, 0)
ZEND_END_ARG_INFO()

PHP_METHOD(ZTensor, copy)
{
    ZEND_PARSE_PARAMETERS_NONE();

    zmatrix_ztensor_object *self_obj = Z_MATRIX_ZTENSOR_P(ZEND_THIS);
    if (!self_obj->tensor) {
        zend_throw_exception(zend_ce_exception, "Tensor not initialized", 0);
        RETURN_THROWS();
    }

    try {
        ZTensor& A = *self_obj->tensor;
        object_init_ex(return_value, Z_OBJCE_P(ZEND_THIS)); // cria novo objeto do mesmo tipo
        zmatrix_ztensor_object *new_obj = Z_MATRIX_ZTENSOR_P(return_value);
        new_obj->tensor = new ZTensor(A); // copia
    } catch (const std::exception& e) {
        zend_throw_exception(zend_ce_exception, e.what(), 0);
        RETURN_THROWS();
    }
}

ZEND_BEGIN_ARG_INFO_EX(arginfo_ztensor_static_safe, 0, 0, 1)
    ZEND_ARG_INFO(0, input)
    ZEND_ARG_TYPE_INFO_WITH_DEFAULT_VALUE(0, copy, _IS_BOOL, 1, "true")
ZEND_END_ARG_INFO()


PHP_METHOD(ZTensor, safe)
{
    zval *input_zv;
    zend_bool copy = 1;

    ZEND_PARSE_PARAMETERS_START(1, 2)
        Z_PARAM_ZVAL(input_zv)
        Z_PARAM_OPTIONAL
        Z_PARAM_BOOL(copy)
    ZEND_PARSE_PARAMETERS_END();

    try {
        if (Z_TYPE_P(input_zv) == IS_ARRAY) {
            ZTensor tensor = php_array_to_tensor(input_zv);
            zmatrix_return_tensor_obj(tensor, return_value, zmatrix_ce_ZTensor);
            return;
        }

        if (Z_TYPE_P(input_zv) == IS_OBJECT && instanceof_function(Z_OBJCE_P(input_zv), zmatrix_ce_ZTensor)) {
            zmatrix_ztensor_object *obj = Z_MATRIX_ZTENSOR_P(input_zv);
            if (!obj->tensor) {
                zend_throw_exception(zend_ce_exception, ZMATRIX_ERR_NOT_INITIALIZED, 0);
                RETURN_THROWS();
            }

            if (copy) {
                ZTensor tensorCopy = ZTensor(*obj->tensor); // deep copy
                zmatrix_return_tensor_obj(tensorCopy, return_value, zmatrix_ce_ZTensor);
            } else {
                ZVAL_ZVAL(return_value, input_zv, 1, 0); // shallow, reusa
            }
            return;
        }

        zend_throw_exception(zend_ce_exception, "ZTensor::safe() expects an array or ZTensor", 0);
        RETURN_THROWS();

    } catch (const std::exception& e) {
        zend_throw_exception(zend_ce_exception, e.what(), 0);
        RETURN_THROWS();
    }
}

ZEND_BEGIN_ARG_INFO_EX(arginfo_ztensor_static_tile, 0, 0, 2)
    ZEND_ARG_OBJ_INFO(0, tensor, ZMatrix\\ZTensor, 0)
    ZEND_ARG_TYPE_INFO(0, times, IS_LONG, 0)
ZEND_END_ARG_INFO()

PHP_METHOD(ZTensor, tile)
{
    zval *tensor_zv;
    zend_long times;

    ZEND_PARSE_PARAMETERS_START(2, 2)
        Z_PARAM_OBJECT_OF_CLASS(tensor_zv, zmatrix_ce_ZTensor)
        Z_PARAM_LONG(times)
    ZEND_PARSE_PARAMETERS_END();

    if (times <= 0) {
        zend_throw_exception(zend_ce_exception, "tile(): parameter times must be >= 1", 0);
        RETURN_THROWS();
    }

    zmatrix_ztensor_object *obj = Z_MATRIX_ZTENSOR_P(tensor_zv);
    if (!obj->tensor) {
        zend_throw_exception(zend_ce_exception, ZMATRIX_ERR_NOT_INITIALIZED, 0);
        RETURN_THROWS();
    }

    try {
        const ZTensor &input = *obj->tensor;
        const auto &inShape = input.shape;

        if (inShape.empty()) {
            zend_throw_exception(zend_ce_exception, "tile(): scalar tensor cannot be repeated", 0);
            RETURN_THROWS();
        }

        size_t rows = inShape[0];
        size_t cols = (inShape.size() == 2) ? inShape[1] : 1;
        std::vector<size_t> outShape = {rows * (size_t)times};
        if (inShape.size() == 2) outShape.push_back(cols);

        ZTensor result(outShape);
        const float* src = input.data.data();
        float* dst = result.data.data();

        size_t blockSize = rows * cols;
        for (size_t i = 0; i < (size_t)times; ++i) {
            memcpy(dst + i * blockSize, src, blockSize * sizeof(float));
        }

        zmatrix_return_tensor_obj(result, return_value, zmatrix_ce_ZTensor);
    } catch (const std::exception& e) {
        zend_throw_exception(zend_ce_exception, e.what(), 0);
        RETURN_THROWS();
    }
}
ZEND_BEGIN_ARG_WITH_RETURN_TYPE_INFO_EX(arginfo_ztensor___tostring, 0, 0, IS_STRING, 0)
ZEND_END_ARG_INFO()

PHP_METHOD(ZTensor, __toString)
{
    zmatrix_ztensor_object *self_obj = Z_MATRIX_ZTENSOR_P(ZEND_THIS);

    if (!self_obj->tensor) {
        RETURN_STRING("[ZTensor: (not initialized)]");
        return;
    }

    try {
        std::string result = self_obj->tensor->to_string();
        RETURN_STRING(result.c_str());
    } catch (const std::exception& e) {
        zend_throw_exception(zend_ce_exception, e.what(), 0);
        RETURN_STRING("[ZTensor: error]");
    }
}

// TODO: Implementar métodos estáticos rand() e randn() similarmente

// ==========================================================================
// Tabela de Métodos da Classe ZMatrix\ZTensor
// ==========================================================================
static const zend_function_entry zmatrix_ztensor_methods[] = {
    PHP_ME(ZTensor, __construct,      arginfo_ztensor_construct,   ZEND_ACC_PUBLIC | ZEND_ACC_CTOR)
    PHP_ME(ZTensor, __toString, arginfo_ztensor___tostring, ZEND_ACC_PUBLIC)
    PHP_ME(ZTensor, clip,            arginfo_ztensor_static_clip_range, ZEND_ACC_PUBLIC | ZEND_ACC_STATIC)
    PHP_ME(ZTensor, sum,             arginfo_ztensor_sum_flex,    ZEND_ACC_PUBLIC)
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
    PHP_ME(Ztensor, broadcast,        arginfo_ztensor_broadcast,        ZEND_ACC_PUBLIC)
    PHP_ME(ZTensor, greater,          arginfo_ztensor_greater,          ZEND_ACC_PUBLIC)
    PHP_ME(ZTensor, minimum,          arginfo_ztensor_minimum,          ZEND_ACC_PUBLIC | ZEND_ACC_STATIC)
    PHP_ME(ZTensor, maximum,          arginfo_ztensor_maximum,          ZEND_ACC_PUBLIC | ZEND_ACC_STATIC)
    PHP_ME(ZTensor, scalarDivide,     arginfo_ztensor_scalarDivide,     ZEND_ACC_PUBLIC)
    PHP_ME(ZTensor, copy,             arginfo_ztensor_copy,             ZEND_ACC_PUBLIC)
    PHP_ME(ZTensor, safe,             arginfo_ztensor_static_safe,      ZEND_ACC_PUBLIC | ZEND_ACC_STATIC)
    PHP_ME(ZTensor, tile,             arginfo_ztensor_static_tile,      ZEND_ACC_PUBLIC | ZEND_ACC_STATIC)

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

// ==========================================================================
// Lista de Funções Globais (Opcional)
// ==========================================================================
// Se não quiser mais as funções globais, defina a lista como vazia ou NULL.
// static const zend_function_entry zmatrix_functions[] = {
//     PHP_FE_END
// };
// Ou remova completamente e passe NULL para zend_module_entry

// Manter por enquanto para compatibilidade ou teste
static const zend_function_entry zmatrix_functions[] = {
    // PHP_FE(zmatrix_add, ...) // Comente/remova as funções globais se não as quiser mais
    PHP_FE_END
};


// ==========================================================================
// Estrutura de Entrada do Módulo (Ordem Corrigida)
// ==========================================================================
zend_module_entry zmatrix_module_entry = {
    STANDARD_MODULE_HEADER,
    "zmatrix",              // Nome da extensão
    zmatrix_functions,      // Funções globais (ou NULL)
    PHP_MINIT(zmatrix),     // MINIT
    NULL,                   // MSHUTDOWN
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
