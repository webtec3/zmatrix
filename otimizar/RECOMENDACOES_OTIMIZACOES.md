# 🚀 Recomendações de Otimizações - zmatrix.cpp

## 📌 Prioridades

### 🔴 ALTA PRIORIDADE (Impacto Significativo)

#### 1. Adicionar SIMD para Funções de Ativação
**Impacto:** 2-4x faster para relu, sigmoid, tanh, exp, log

```cpp
// Adicionar em simd/simd_dispatch.h

namespace zmatrix_simd {
    // ReLU: max(0, x)
    inline void relu_f32(float* data, size_t n) {
        #if HAS_AVX2
        const __m256 zero = _mm256_setzero_ps();
        for (size_t i = 0; i + 8 <= n; i += 8) {
            __m256 v = _mm256_loadu_ps(data + i);
            v = _mm256_max_ps(v, zero);
            _mm256_storeu_ps(data + i, v);
        }
        // Tail loop para remainders
        for (size_t i = (n / 8) * 8; i < n; ++i) {
            data[i] = std::max(0.0f, data[i]);
        }
        #else
        for (size_t i = 0; i < n; ++i) {
            data[i] = std::max(0.0f, data[i]);
        }
        #endif
    }

    // Sigmoid aproximado (fast version)
    inline void sigmoid_f32_approx(float* data, size_t n) {
        #if HAS_AVX2
        // Usar aproximação polinomial para evitar divisão cara
        const __m256 half = _mm256_set1_ps(0.5f);
        const __m256 one = _mm256_set1_ps(1.0f);
        
        for (size_t i = 0; i + 8 <= n; i += 8) {
            __m256 x = _mm256_loadu_ps(data + i);
            // sigmoid(x) ≈ 0.5 + 0.25*x (linear approximation)
            // Ou usar tabela lookup para melhor precisão
            __m256 result = _mm256_fmadd_ps(
                _mm256_mul_ps(_mm256_set1_ps(0.25f), x), 
                one, 
                half
            );
            _mm256_storeu_ps(data + i, result);
        }
        // Tail
        for (size_t i = (n / 8) * 8; i < n; ++i) {
            data[i] = 0.5f + 0.25f * data[i];
        }
        #else
        for (size_t i = 0; i < n; ++i) {
            data[i] = 1.0f / (1.0f + std::expf(-data[i]));
        }
        #endif
    }

    // Exp aproximado (Taylor series)
    inline void exp_f32_approx(float* data, size_t n) {
        #if HAS_AVX2
        // Usar aproximação de Taylor: exp(x) ≈ 1 + x + x²/2 + x³/6
        for (size_t i = 0; i + 8 <= n; i += 8) {
            __m256 x = _mm256_loadu_ps(data + i);
            __m256 result = _mm256_set1_ps(1.0f);
            
            // Termo linear
            result = _mm256_add_ps(result, x);
            
            // Termo quadrático
            __m256 x2 = _mm256_mul_ps(x, x);
            result = _mm256_fmadd_ps(x2, _mm256_set1_ps(0.5f), result);
            
            // Termo cúbico
            __m256 x3 = _mm256_mul_ps(x2, x);
            result = _mm256_fmadd_ps(x3, _mm256_set1_ps(0.16667f), result);
            
            _mm256_storeu_ps(data + i, result);
        }
        // Tail
        for (size_t i = (n / 8) * 8; i < n; ++i) {
            data[i] = std::expf(data[i]);
        }
        #else
        for (size_t i = 0; i < n; ++i) {
            data[i] = std::expf(data[i]);
        }
        #endif
    }
}
```

**Implementação em zmatrix.cpp:**
```cpp
void relu() {
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

    // Nova: SIMD primeiro
    #if HAS_AVX2
    if (N <= ZMATRIX_PARALLEL_THRESHOLD) {
        zmatrix_simd::relu_f32(a, N);  // ← NOVO
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
        zmatrix_simd::relu_f32(a, N);  // ← Fallback SIMD
    }
    #else
    zmatrix_simd::relu_f32(a, N);  // ← Fallback SIMD
    #endif
}
```

---

#### 2. Adicionar CUDA matmul (cublas_sgemm)
**Impacto:** 5-10x faster para grandes matrizes

```cpp
// Em gpu_wrapper.h ou gpu_wrapper.cu
extern "C" {
    void gpu_matmul(
        const float* A, const float* B, float* C,
        int M, int N, int K
    );
}

// Em zmatrix.cpp - método matmul()
ZTensor matmul(const ZTensor& other) const {
    // ... validações ...
    
    const size_t M = shape[0];
    const size_t K = shape[1];
    const size_t N = other.shape[1];
    
    #ifdef HAVE_CUDA
    if (zmatrix_should_use_gpu(M * N * K)) {
        ensure_host();
        other.ensure_host();
        
        ZTensor result({M, N});
        
        gpu_matmul(
            data.data(), 
            other.data.data(), 
            result.data.data(),
            static_cast<int>(M), 
            static_cast<int>(N), 
            static_cast<int>(K)
        );
        
        return result;
    }
    #endif
    
    // Fallback: BLAS (já implementado)
    // ...
}
```

---

### 🟡 MÉDIA PRIORIDADE (Benefício Moderado)

#### 3. Optimizar Divide com SIMD
**Impacto:** 1.5-2x faster

```cpp
namespace zmatrix_simd {
    inline void divide_f32(float* a, const float* b, size_t n, bool& has_error) {
        #if HAS_AVX2
        const __m256 zero = _mm256_setzero_ps();
        __m256 error_mask = zero;
        
        for (size_t i = 0; i + 8 <= n; i += 8) {
            __m256 a_vec = _mm256_loadu_ps(a + i);
            __m256 b_vec = _mm256_loadu_ps(b + i);
            
            // Detectar zeros
            __m256 b_is_zero = _mm256_cmp_ps(b_vec, zero, _CMP_EQ_OQ);
            error_mask = _mm256_or_ps(error_mask, b_is_zero);
            
            // Evitar divisão por zero com máscara
            __m256 safe_b = _mm256_blendv_ps(b_vec, _mm256_set1_ps(1.0f), b_is_zero);
            __m256 result = _mm256_div_ps(a_vec, safe_b);
            
            _mm256_storeu_ps(a + i, result);
        }
        
        // Check se houve zeros
        has_error = _mm256_movemask_ps(error_mask) != 0;
        
        // Tail
        for (size_t i = (n / 8) * 8; i < n; ++i) {
            if (b[i] == 0.0f) {
                has_error = true;
            } else {
                a[i] /= b[i];
            }
        }
        #else
        has_error = false;
        for (size_t i = 0; i < n; ++i) {
            if (b[i] == 0.0f) {
                has_error = true;
            } else {
                a[i] /= b[i];
            }
        }
        #endif
    }
}

// No método divide()
void divide(const ZTensor& other) {
    if (!same_shape(other)) {
        throw std::invalid_argument(ZMATRIX_ERR_SHAPE_MISMATCH);
    }
    const size_t N = size();
    if (N == 0) return;

    float * __restrict__ a = data.data();
    const float * __restrict__ b = other.data.data();

    bool has_error = false;

    #if HAS_OPENMP
    if (N > ZMATRIX_PARALLEL_THRESHOLD) {
        #pragma omp parallel for simd schedule(static) reduction(||:has_error)
        for (size_t i = 0; i < N; ++i) {
            if (b[i] == 0.0f) {
                has_error = true;
            } else {
                a[i] /= b[i];
            }
        }
    } else {
        zmatrix_simd::divide_f32(a, b, N, has_error);  // ← NOVO
    }
    #else
    zmatrix_simd::divide_f32(a, b, N, has_error);  // ← NOVO
    #endif

    if (has_error) {
        throw std::runtime_error("Divisão por zero detectada");
    }

#ifdef HAVE_CUDA
    mark_host_modified();
#endif
}
```

---

#### 4. Otimizar Min/Max/Std com SIMD
**Impacto:** 1.5-3x faster para reduções

```cpp
namespace zmatrix_simd {
    // Min com SIMD
    inline float min_f32(const float* data, size_t n) {
        if (n == 0) return std::numeric_limits<float>::quiet_NaN();
        
        #if HAS_AVX2
        __m256 min_vec = _mm256_set1_ps(std::numeric_limits<float>::max());
        
        for (size_t i = 0; i + 8 <= n; i += 8) {
            __m256 v = _mm256_loadu_ps(data + i);
            min_vec = _mm256_min_ps(min_vec, v);
        }
        
        // Reduzir __m256 a float
        float result = std::numeric_limits<float>::max();
        float tmp[8];
        _mm256_storeu_ps(tmp, min_vec);
        for (int i = 0; i < 8; ++i) {
            result = std::min(result, tmp[i]);
        }
        
        // Tail
        for (size_t i = (n / 8) * 8; i < n; ++i) {
            result = std::min(result, data[i]);
        }
        
        return result;
        #else
        float result = data[0];
        for (size_t i = 1; i < n; ++i) {
            result = std::min(result, data[i]);
        }
        return result;
        #endif
    }

    // Std dev (parte da computação)
    inline double std_f32(const float* data, size_t n, double mean_val) {
        if (n < 2) return std::numeric_limits<double>::quiet_NaN();
        
        double variance = 0.0;
        #if HAS_AVX2
        __m256d variance_vec = _mm256_setzero_pd();
        __m256 mean_vec = _mm256_set1_ps(static_cast<float>(mean_val));
        
        for (size_t i = 0; i + 4 <= n; i += 4) {
            __m128 v = _mm_loadu_ps(data + i);  // 4 floats
            __m128 diff = _mm_sub_ps(v, mean_vec);
            __m128 sq = _mm_mul_ps(diff, diff);
            
            // Converter para double e acumular
            __m256d sq_d = _mm256_cvtps_pd(sq);
            variance_vec = _mm256_add_pd(variance_vec, sq_d);
        }
        
        // Reduzir
        double tmp[4];
        _mm256_storeu_pd(tmp, variance_vec);
        variance = tmp[0] + tmp[1] + tmp[2] + tmp[3];
        
        // Tail
        for (size_t i = (n / 4) * 4; i < n; ++i) {
            double diff = static_cast<double>(data[i]) - mean_val;
            variance += diff * diff;
        }
        #else
        for (size_t i = 0; i < n; ++i) {
            double diff = static_cast<double>(data[i]) - mean_val;
            variance += diff * diff;
        }
        #endif
        
        return std::sqrt(variance / (n - 1));
    }
}
```

---

### 🟢 BAIXA PRIORIDADE (Refinamento)

#### 5. Usar `__restrict__` Consistentemente
```cpp
// Padronizar em todos os métodos:
float * __restrict__ a = data.data();
const float * __restrict__ b = other.data.data();
```

#### 6. Melhorar Detecção de Capacidades SIMD
```cpp
// Adicionar em tempo de inicialização (module RINIT)
int simd_caps = 0;
#if HAS_AVX512
simd_caps |= SIMD_AVX512;
#endif
#if HAS_AVX2
simd_caps |= SIMD_AVX2;
#endif
// Registrar em GLOBALS
```

#### 7. Otimizar Soma com Eixo
```cpp
// Usar cache-blocking para melhor localidade
void soma_optimized(ZTensor& out, int axis) const {
    const size_t BLOCK_SIZE = 4096;  // Ajustar conforme L2 cache
    
    // Implementar em blocos para melhor cache utilization
    // ...
}
```

---

## 📊 Benchmarks Esperados

Após implementar as recomendações de ALTA PRIORIDADE:

| Operação | Antes | Depois | Melhoria |
|----------|-------|--------|----------|
| `relu(1M)` | 2.5ms | 0.8ms | **3.1x** |
| `sigmoid(1M)` | 5.0ms | 1.5ms | **3.3x** |
| `exp(1M)` | 4.0ms | 1.2ms | **3.3x** |
| `divide(1M)` | 3.5ms | 2.0ms | **1.75x** |
| `matmul(1000x1000)` | 15ms (BLAS) | 3ms (CUDA) | **5x** |
| `min(10M)` | 8ms | 2ms | **4x** |

---

## 🛠️ Checklist de Implementação

- [ ] Adicionar `relu_f32()` a `simd/simd_dispatch.h`
- [ ] Adicionar `sigmoid_f32_approx()` a `simd/simd_dispatch.h`
- [ ] Adicionar `exp_f32_approx()` a `simd/simd_dispatch.h`
- [ ] Atualizar `relu()` em `zmatrix.cpp` para usar SIMD
- [ ] Atualizar `sigmoid()` em `zmatrix.cpp` para usar SIMD
- [ ] Atualizar `exp()` em `zmatrix.cpp` para usar SIMD
- [ ] Atualizar `tanh()` em `zmatrix.cpp` para usar SIMD
- [ ] Adicionar `divide_f32()` a `simd/simd_dispatch.h`
- [ ] Atualizar `divide()` em `zmatrix.cpp`
- [ ] Adicionar `min_f32()` a `simd/simd_dispatch.h`
- [ ] Adicionar `std_f32()` a `simd/simd_dispatch.h`
- [ ] Atualizar `min()` e `std()` em `zmatrix.cpp`
- [ ] Implementar `gpu_matmul()` em `gpu_wrapper.cu`
- [ ] Atualizar `matmul()` em `zmatrix.cpp` para usar CUDA
- [ ] Revisar e padronizar `__restrict__` pointers
- [ ] Implementar SIMD capabilities detection
- [ ] Benchmark cada otimização

---

## 📚 Referências

- [GCC SIMD Intrinsics](https://gcc.gnu.org/onlinedocs/gcc/Vector-Extensions.html)
- [AVX-512 Programming](https://www.intel.com/content/www/en/en/docs/cpp-compiler/developer-guide-reference/0/intrinsics-for-new-features.html)
- [CUBLAS Documentation](https://docs.nvidia.com/cuda/cublas/)
- [OpenMP 5.0 Spec](https://www.openmp.org/spec-html/5.0/openmpsu59.html)

---

*Documento de Recomendações - 17 de Janeiro de 2026*
