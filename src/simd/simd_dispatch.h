#ifndef ZMATRIX_SIMD_DISPATCH_H
#define ZMATRIX_SIMD_DISPATCH_H

#include <cmath>
#include <limits>

#if HAS_AVX2
#include <immintrin.h>
#endif

namespace zmatrix_simd {

inline void abs_f32(float* a, size_t n) {
#if HAS_AVX2
    const __m256 sign_mask = _mm256_set1_ps(-0.0f);
    const size_t vec_size = 8;
    const size_t aligned_n = (n / vec_size) * vec_size;

    for (size_t i = 0; i < aligned_n; i += vec_size) {
        __m256 va = _mm256_loadu_ps(&a[i]);
        __m256 result = _mm256_andnot_ps(sign_mask, va);
        _mm256_storeu_ps(&a[i], result);
    }

    for (size_t i = aligned_n; i < n; ++i) {
        a[i] = std::fabs(a[i]);
    }
#else
    for (size_t i = 0; i < n; ++i) {
        a[i] = std::fabs(a[i]);
    }
#endif
}

inline void sqrt_f32(float* a, size_t n) {
#if HAS_AVX2
    const size_t vec_size = 8;
    const size_t aligned_n = (n / vec_size) * vec_size;

    for (size_t i = 0; i < aligned_n; i += vec_size) {
        __m256 va = _mm256_loadu_ps(&a[i]);
        __m256 result = _mm256_sqrt_ps(va);
        _mm256_storeu_ps(&a[i], result);
    }

    for (size_t i = aligned_n; i < n; ++i) {
        a[i] = std::sqrt(a[i]);
    }
#else
    for (size_t i = 0; i < n; ++i) {
        a[i] = std::sqrt(a[i]);
    }
#endif
}

inline float min_f32(const float* a, size_t n) {
    if (n == 0) return std::numeric_limits<float>::quiet_NaN();
    if (n == 1) return a[0];

    float min_val = a[0];
#if HAS_AVX2
    const size_t vec_size = 8;
    const size_t aligned_n = (n / vec_size) * vec_size;

    __m256 vmin = _mm256_set1_ps(a[0]);
    for (size_t i = 0; i < aligned_n; i += vec_size) {
        __m256 va = _mm256_loadu_ps(&a[i]);
        vmin = _mm256_min_ps(vmin, va);
    }

    __m256 shuf = _mm256_shuffle_ps(vmin, vmin, _MM_SHUFFLE(2, 3, 0, 1));
    __m256 mins = _mm256_min_ps(vmin, shuf);
    shuf = _mm256_shuffle_ps(mins, mins, _MM_SHUFFLE(1, 0, 3, 2));
    mins = _mm256_min_ps(mins, shuf);
    shuf = _mm256_permute2f128_ps(mins, mins, 1);
    mins = _mm256_min_ps(mins, shuf);

    min_val = _mm256_cvtss_f32(mins);

    for (size_t i = aligned_n; i < n; ++i) {
        if (a[i] < min_val) min_val = a[i];
    }
#else
    for (size_t i = 1; i < n; ++i) {
        if (a[i] < min_val) min_val = a[i];
    }
#endif
    return min_val;
}

inline float max_f32(const float* a, size_t n) {
    if (n == 0) return std::numeric_limits<float>::quiet_NaN();
    if (n == 1) return a[0];

    float max_val = a[0];
#if HAS_AVX2
    const size_t vec_size = 8;
    const size_t aligned_n = (n / vec_size) * vec_size;

    __m256 vmax = _mm256_set1_ps(a[0]);
    for (size_t i = 0; i < aligned_n; i += vec_size) {
        __m256 va = _mm256_loadu_ps(&a[i]);
        vmax = _mm256_max_ps(vmax, va);
    }

    __m256 shuf = _mm256_shuffle_ps(vmax, vmax, _MM_SHUFFLE(2, 3, 0, 1));
    __m256 maxs = _mm256_max_ps(vmax, shuf);
    shuf = _mm256_shuffle_ps(maxs, maxs, _MM_SHUFFLE(1, 0, 3, 2));
    maxs = _mm256_max_ps(maxs, shuf);
    shuf = _mm256_permute2f128_ps(maxs, maxs, 1);
    maxs = _mm256_max_ps(maxs, shuf);

    max_val = _mm256_cvtss_f32(maxs);

    for (size_t i = aligned_n; i < n; ++i) {
        if (a[i] > max_val) max_val = a[i];
    }
#else
    for (size_t i = 1; i < n; ++i) {
        if (a[i] > max_val) max_val = a[i];
    }
#endif
    return max_val;
}

inline double sum_f32(const float* a, size_t n) {
    if (n == 0) return 0.0;

    double total = 0.0;
#if HAS_AVX2
    const size_t vec_size = 8;
    const size_t aligned_n = (n / vec_size) * vec_size;

    __m256 vsum = _mm256_setzero_ps();
    for (size_t i = 0; i < aligned_n; i += vec_size) {
        __m256 va = _mm256_loadu_ps(&a[i]);
        vsum = _mm256_add_ps(vsum, va);
    }

    alignas(32) float temp[8];
    _mm256_storeu_ps(temp, vsum);
    for (size_t i = 0; i < 8; ++i) {
        total += static_cast<double>(temp[i]);
    }

    for (size_t i = aligned_n; i < n; ++i) {
        total += static_cast<double>(a[i]);
    }
#else
    for (size_t i = 0; i < n; ++i) {
        total += static_cast<double>(a[i]);
    }
#endif
    return total;
}


inline void add_f32(float* a, const float* b, size_t n) {
#if HAS_AVX2
    const size_t vec_size = 8;
    const size_t aligned_n = (n / vec_size) * vec_size;

    for (size_t i = 0; i < aligned_n; i += vec_size) {
        __m256 va = _mm256_loadu_ps(&a[i]);
        __m256 vb = _mm256_loadu_ps(&b[i]);
        __m256 result = _mm256_add_ps(va, vb);
        _mm256_storeu_ps(&a[i], result);
    }

    for (size_t i = aligned_n; i < n; ++i) {
        a[i] += b[i];
    }
#else
    for (size_t i = 0; i < n; ++i) {
        a[i] += b[i];
    }
#endif
}

inline void sub_f32(float* a, const float* b, size_t n) {
#if HAS_AVX2
    const size_t vec_size = 8;
    const size_t aligned_n = (n / vec_size) * vec_size;

    for (size_t i = 0; i < aligned_n; i += vec_size) {
        __m256 va = _mm256_loadu_ps(&a[i]);
        __m256 vb = _mm256_loadu_ps(&b[i]);
        __m256 result = _mm256_sub_ps(va, vb);
        _mm256_storeu_ps(&a[i], result);
    }

    for (size_t i = aligned_n; i < n; ++i) {
        a[i] -= b[i];
    }
#else
    for (size_t i = 0; i < n; ++i) {
        a[i] -= b[i];
    }
#endif
}

inline void mul_f32(float* a, const float* b, size_t n) {
#if HAS_AVX2
    const size_t vec_size = 8;
    const size_t aligned_n = (n / vec_size) * vec_size;

    for (size_t i = 0; i < aligned_n; i += vec_size) {
        __m256 va = _mm256_loadu_ps(&a[i]);
        __m256 vb = _mm256_loadu_ps(&b[i]);
        __m256 result = _mm256_mul_ps(va, vb);
        _mm256_storeu_ps(&a[i], result);
    }

    for (size_t i = aligned_n; i < n; ++i) {
        a[i] *= b[i];
    }
#else
    for (size_t i = 0; i < n; ++i) {
        a[i] *= b[i];
    }
#endif
}

inline void divide_f32(float* a, const float* b, size_t n) {
#if HAS_AVX2
    const size_t vec_size = 8;
    const size_t aligned_n = (n / vec_size) * vec_size;
    const __m256 zero = _mm256_setzero_ps();

    for (size_t i = 0; i < aligned_n; i += vec_size) {
        __m256 vb = _mm256_loadu_ps(&b[i]);
        __m256 mask = _mm256_cmp_ps(vb, zero, _CMP_EQ_OS);
        
        // Se houver zero, cair para fallback elemento a elemento
        if (_mm256_movemask_ps(mask) != 0) {
            // Há zeros neste bloco - processar elemento a elemento com validação
            for (size_t j = i; j < i + vec_size && j < n; ++j) {
                if (b[j] == 0.0f) {
                    throw std::runtime_error("Divisão por zero detectada");
                }
                a[j] /= b[j];
            }
        } else {
            // Nenhum zero - pode fazer divisão vetorial segura
            __m256 va = _mm256_loadu_ps(&a[i]);
            __m256 result = _mm256_div_ps(va, vb);
            _mm256_storeu_ps(&a[i], result);
        }
    }

    // Resto não-alinhado com verificação de zero
    for (size_t i = aligned_n; i < n; ++i) {
        if (b[i] == 0.0f) {
            throw std::runtime_error("Divisão por zero detectada");
        }
        a[i] /= b[i];
    }
#else
    // Fallback escalar com verificação de zero
    for (size_t i = 0; i < n; ++i) {
        if (b[i] == 0.0f) {
            throw std::runtime_error("Divisão por zero detectada");
        }
        a[i] /= b[i];
    }
#endif
}

inline void relu_f32(float* a, size_t n) {
#if HAS_AVX2
    const size_t vec_size = 8;
    const size_t aligned_n = (n / vec_size) * vec_size;
    const __m256 zero = _mm256_setzero_ps();

    for (size_t i = 0; i < aligned_n; i += vec_size) {
        __m256 va = _mm256_loadu_ps(&a[i]);
        __m256 result = _mm256_max_ps(va, zero);
        _mm256_storeu_ps(&a[i], result);
    }

    for (size_t i = aligned_n; i < n; ++i) {
        a[i] = a[i] > 0.0f ? a[i] : 0.0f;
    }
#else
    for (size_t i = 0; i < n; ++i) {
        a[i] = a[i] > 0.0f ? a[i] : 0.0f;
    }
#endif
}

inline void relu_derivative_f32(float* a, size_t n) {
#if HAS_AVX2
    const size_t vec_size = 8;
    const size_t aligned_n = (n / vec_size) * vec_size;
    const __m256 zero = _mm256_setzero_ps();
    const __m256 ones = _mm256_set1_ps(1.0f);

    for (size_t i = 0; i < aligned_n; i += vec_size) {
        __m256 va = _mm256_loadu_ps(&a[i]);
        __m256 mask = _mm256_cmp_ps(va, zero, _CMP_GT_OS);
        __m256 result = _mm256_and_ps(mask, ones);
        _mm256_storeu_ps(&a[i], result);
    }

    for (size_t i = aligned_n; i < n; ++i) {
        a[i] = (a[i] > 0.0f) ? 1.0f : 0.0f;
    }
#else
    for (size_t i = 0; i < n; ++i) {
        a[i] = (a[i] > 0.0f) ? 1.0f : 0.0f;
    }
#endif
}

inline void leaky_relu_f32(float* a, float alpha, size_t n) {
#if HAS_AVX2
    const size_t vec_size = 8;
    const size_t aligned_n = (n / vec_size) * vec_size;
    const __m256 zero = _mm256_setzero_ps();
    const __m256 alpha_v = _mm256_set1_ps(alpha);

    for (size_t i = 0; i < aligned_n; i += vec_size) {
        __m256 va = _mm256_loadu_ps(&a[i]);
        __m256 mask = _mm256_cmp_ps(va, zero, _CMP_GT_OS);
        __m256 neg = _mm256_mul_ps(va, alpha_v);
        __m256 result = _mm256_blendv_ps(neg, va, mask);
        _mm256_storeu_ps(&a[i], result);
    }

    for (size_t i = aligned_n; i < n; ++i) {
        a[i] = (a[i] > 0.0f) ? a[i] : alpha * a[i];
    }
#else
    for (size_t i = 0; i < n; ++i) {
        a[i] = (a[i] > 0.0f) ? a[i] : alpha * a[i];
    }
#endif
}

inline void leaky_relu_derivative_f32(float* a, float alpha, size_t n) {
#if HAS_AVX2
    const size_t vec_size = 8;
    const size_t aligned_n = (n / vec_size) * vec_size;
    const __m256 zero = _mm256_setzero_ps();
    const __m256 ones = _mm256_set1_ps(1.0f);
    const __m256 alpha_v = _mm256_set1_ps(alpha);

    for (size_t i = 0; i < aligned_n; i += vec_size) {
        __m256 va = _mm256_loadu_ps(&a[i]);
        __m256 mask = _mm256_cmp_ps(va, zero, _CMP_GT_OS);
        __m256 result = _mm256_blendv_ps(alpha_v, ones, mask);
        _mm256_storeu_ps(&a[i], result);
    }

    for (size_t i = aligned_n; i < n; ++i) {
        a[i] = (a[i] > 0.0f) ? 1.0f : alpha;
    }
#else
    for (size_t i = 0; i < n; ++i) {
        a[i] = (a[i] > 0.0f) ? 1.0f : alpha;
    }
#endif
}


inline void scalar_add_f32(float* a, float v, size_t n) {
#if HAS_AVX2
    const size_t vec_size = 8;
    const size_t aligned_n = (n / vec_size) * vec_size;
    const __m256 vv = _mm256_set1_ps(v);

    for (size_t i = 0; i < aligned_n; i += vec_size) {
        __m256 va = _mm256_loadu_ps(&a[i]);
        __m256 result = _mm256_add_ps(va, vv);
        _mm256_storeu_ps(&a[i], result);
    }

    for (size_t i = aligned_n; i < n; ++i) {
        a[i] += v;
    }
#else
    for (size_t i = 0; i < n; ++i) {
        a[i] += v;
    }
#endif
}

inline void scalar_sub_f32(float* a, float v, size_t n) {
#if HAS_AVX2
    const size_t vec_size = 8;
    const size_t aligned_n = (n / vec_size) * vec_size;
    const __m256 vv = _mm256_set1_ps(v);

    for (size_t i = 0; i < aligned_n; i += vec_size) {
        __m256 va = _mm256_loadu_ps(&a[i]);
        __m256 result = _mm256_sub_ps(va, vv);
        _mm256_storeu_ps(&a[i], result);
    }

    for (size_t i = aligned_n; i < n; ++i) {
        a[i] -= v;
    }
#else
    for (size_t i = 0; i < n; ++i) {
        a[i] -= v;
    }
#endif
}

inline void scalar_mul_f32(float* a, float v, size_t n) {
#if HAS_AVX2
    const size_t vec_size = 8;
    const size_t aligned_n = (n / vec_size) * vec_size;
    const __m256 vv = _mm256_set1_ps(v);

    for (size_t i = 0; i < aligned_n; i += vec_size) {
        __m256 va = _mm256_loadu_ps(&a[i]);
        __m256 result = _mm256_mul_ps(va, vv);
        _mm256_storeu_ps(&a[i], result);
    }

    for (size_t i = aligned_n; i < n; ++i) {
        a[i] *= v;
    }
#else
    for (size_t i = 0; i < n; ++i) {
        a[i] *= v;
    }
#endif
}

inline void scalar_div_f32(float* a, float v, size_t n) {
#if HAS_AVX2
    const size_t vec_size = 8;
    const size_t aligned_n = (n / vec_size) * vec_size;
    const __m256 vv = _mm256_set1_ps(v);

    for (size_t i = 0; i < aligned_n; i += vec_size) {
        __m256 va = _mm256_loadu_ps(&a[i]);
        __m256 result = _mm256_div_ps(va, vv);
        _mm256_storeu_ps(&a[i], result);
    }

    for (size_t i = aligned_n; i < n; ++i) {
        a[i] /= v;
    }
#else
    for (size_t i = 0; i < n; ++i) {
        a[i] /= v;
    }
#endif
}

} // namespace zmatrix_simd

#endif // ZMATRIX_SIMD_DISPATCH_H
