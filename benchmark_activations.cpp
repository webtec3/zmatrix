#include <cstring>
#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <chrono>
#include <immintrin.h>

// ReLU Scalar
static inline void relu_scalar(float* a, size_t n) {
    for (size_t i = 0; i < n; ++i) {
        a[i] = std::max(0.0f, a[i]);
    }
}

// ReLU SIMD
static inline void relu_simd(float* a, size_t n) {
    const __m256 zeros = _mm256_setzero_ps();
    const size_t vec_size = 8;
    const size_t aligned_n = (n / vec_size) * vec_size;
    
    for (size_t i = 0; i < aligned_n; i += vec_size) {
        __m256 va = _mm256_loadu_ps(&a[i]);
        __m256 result = _mm256_max_ps(va, zeros);
        _mm256_storeu_ps(&a[i], result);
    }
    
    for (size_t i = aligned_n; i < n; ++i) {
        a[i] = std::max(0.0f, a[i]);
    }
}

// Sigmoid Scalar
static inline void sigmoid_scalar(float* a, size_t n) {
    for (size_t i = 0; i < n; ++i) {
        a[i] = 1.0f / (1.0f + expf(-a[i]));
    }
}

// Tanh Scalar
static inline void tanh_scalar(float* a, size_t n) {
    for (size_t i = 0; i < n; ++i) {
        a[i] = std::tanh(a[i]);
    }
}

int main() {
    const size_t size = 2500 * 2500;
    const int iterations = 50;
    
    printf("=== Activation Functions: Scalar vs SIMD ===\n");
    printf("Size: %zu elements (%zu MB)\n", size, size * 4 / (1024*1024));
    printf("Iterations: %d\n\n", iterations);
    
    float* a = (float*)aligned_alloc(32, size * sizeof(float));
    
    // ==== ReLU ====
    printf("[ReLU]\n");
    
    // Scalar
    for (size_t i = 0; i < size; ++i) a[i] = (float)i - size/2;  // -size/2 to +size/2
    auto start = std::chrono::high_resolution_clock::now();
    for (int i = 0; i < iterations; ++i) {
        relu_scalar(a, size);
    }
    auto end = std::chrono::high_resolution_clock::now();
    double scalar_time = std::chrono::duration<double, std::milli>(end - start).count();
    double scalar_avg = scalar_time / iterations;
    
    // SIMD
    for (size_t i = 0; i < size; ++i) a[i] = (float)i - size/2;
    start = std::chrono::high_resolution_clock::now();
    for (int i = 0; i < iterations; ++i) {
        relu_simd(a, size);
    }
    end = std::chrono::high_resolution_clock::now();
    double simd_time = std::chrono::duration<double, std::milli>(end - start).count();
    double simd_avg = simd_time / iterations;
    
    printf("  Scalar:  %.6f ms | SIMD: %.6f ms | Speedup: %.2fx\n\n", 
           scalar_avg, simd_avg, scalar_avg / simd_avg);
    
    // ==== Sigmoid ====
    printf("[Sigmoid]\n");
    
    // Scalar
    for (size_t i = 0; i < size; ++i) a[i] = (float)i / size * 4 - 2;  // -2 to +2
    start = std::chrono::high_resolution_clock::now();
    for (int i = 0; i < iterations; ++i) {
        sigmoid_scalar(a, size);
    }
    end = std::chrono::high_resolution_clock::now();
    scalar_time = std::chrono::duration<double, std::milli>(end - start).count();
    scalar_avg = scalar_time / iterations;
    
    // Sigmoid SIMD = same as scalar (transcendental)
    for (size_t i = 0; i < size; ++i) a[i] = (float)i / size * 4 - 2;
    start = std::chrono::high_resolution_clock::now();
    for (int i = 0; i < iterations; ++i) {
        sigmoid_scalar(a, size);  // mesmo código escalar
    }
    end = std::chrono::high_resolution_clock::now();
    simd_time = std::chrono::duration<double, std::milli>(end - start).count();
    simd_avg = simd_time / iterations;
    
    printf("  Scalar:  %.6f ms | Transcendental (sem SIMD ideal)\n\n", scalar_avg);
    
    // ==== Tanh ====
    printf("[Tanh]\n");
    
    // Scalar
    for (size_t i = 0; i < size; ++i) a[i] = (float)i / size * 4 - 2;
    start = std::chrono::high_resolution_clock::now();
    for (int i = 0; i < iterations; ++i) {
        tanh_scalar(a, size);
    }
    end = std::chrono::high_resolution_clock::now();
    scalar_time = std::chrono::duration<double, std::milli>(end - start).count();
    scalar_avg = scalar_time / iterations;
    
    printf("  Scalar:  %.6f ms | Transcendental (sem SIMD ideal)\n\n", scalar_avg);
    
    free(a);
    return 0;
}
