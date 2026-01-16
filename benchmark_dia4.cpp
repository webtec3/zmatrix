#include <cstring>
#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <chrono>
#include <immintrin.h>

// ============ ABS ============

// ABS Scalar
static inline void abs_scalar(float* a, size_t n) {
    for (size_t i = 0; i < n; ++i) {
        a[i] = std::fabs(a[i]);
    }
}

// ABS SIMD
static inline void abs_simd(float* a, size_t n) {
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
}

// ============ SQRT ============

// SQRT Scalar
static inline void sqrt_scalar(float* a, size_t n) {
    for (size_t i = 0; i < n; ++i) {
        a[i] = std::sqrt(a[i]);
    }
}

// SQRT SIMD
static inline void sqrt_simd(float* a, size_t n) {
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
}

int main() {
    const size_t size = 2500 * 2500;
    const int iterations = 50;
    
    printf("=== DIA 4: Extended SIMD Benchmark ===\n");
    printf("Size: %zu elements (%zu MB)\n", size, size * 4 / (1024*1024));
    printf("Iterations: %d\n\n", iterations);
    
    float* a = (float*)aligned_alloc(32, size * sizeof(float));
    
    // ==== ABS ====
    printf("[ABS]\n");
    
    // Scalar
    for (size_t i = 0; i < size; ++i) a[i] = (float)i - size/2;
    auto start = std::chrono::high_resolution_clock::now();
    for (int i = 0; i < iterations; ++i) {
        abs_scalar(a, size);
    }
    auto end = std::chrono::high_resolution_clock::now();
    double scalar_time = std::chrono::duration<double, std::milli>(end - start).count();
    double scalar_avg = scalar_time / iterations;
    
    // SIMD
    for (size_t i = 0; i < size; ++i) a[i] = (float)i - size/2;
    start = std::chrono::high_resolution_clock::now();
    for (int i = 0; i < iterations; ++i) {
        abs_simd(a, size);
    }
    end = std::chrono::high_resolution_clock::now();
    double simd_time = std::chrono::duration<double, std::milli>(end - start).count();
    double simd_avg = simd_time / iterations;
    
    printf("  Scalar: %.6f ms | SIMD: %.6f ms | Speedup: %.2fx\n\n", 
           scalar_avg, simd_avg, scalar_avg / simd_avg);
    
    // ==== SQRT ====
    printf("[SQRT]\n");
    
    // Scalar
    for (size_t i = 0; i < size; ++i) a[i] = (float)i + 1;  // Valores positivos
    start = std::chrono::high_resolution_clock::now();
    for (int i = 0; i < iterations; ++i) {
        sqrt_scalar(a, size);
    }
    end = std::chrono::high_resolution_clock::now();
    scalar_time = std::chrono::duration<double, std::milli>(end - start).count();
    scalar_avg = scalar_time / iterations;
    
    // SIMD
    for (size_t i = 0; i < size; ++i) a[i] = (float)i + 1;
    start = std::chrono::high_resolution_clock::now();
    for (int i = 0; i < iterations; ++i) {
        sqrt_simd(a, size);
    }
    end = std::chrono::high_resolution_clock::now();
    simd_time = std::chrono::duration<double, std::milli>(end - start).count();
    simd_avg = simd_time / iterations;
    
    printf("  Scalar: %.6f ms | SIMD: %.6f ms | Speedup: %.2fx\n\n", 
           scalar_avg, simd_avg, scalar_avg / simd_avg);
    
    free(a);
    return 0;
}
