#include <cstring>
#include <cstdio>
#include <cstdlib>
#include <chrono>
#include <immintrin.h>

// SIMD ADD kernel
static inline void add_simd_kernel(float* __restrict__ a, const float* __restrict__ b, size_t n) {
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
}

// Scalar ADD kernel
static inline void add_scalar(float* a, const float* b, size_t n) {
    for (size_t i = 0; i < n; ++i) {
        a[i] += b[i];
    }
}

int main() {
    const size_t size = 2500 * 2500;  // 6.25M elements
    const int iterations = 100;
    
    printf("=== SIMD vs Scalar Benchmark ===\n");
    printf("Size: %zu elements (%zu MB)\n", size, size * 4 / (1024*1024));
    printf("Iterations: %d\n\n", iterations);
    
    // Allocate arrays
    float* a = (float*)aligned_alloc(32, size * sizeof(float));
    float* b = (float*)aligned_alloc(32, size * sizeof(float));
    
    // Initialize
    for (size_t i = 0; i < size; ++i) {
        a[i] = 1.0f;
        b[i] = 2.0f;
    }
    
    // Benchmark SIMD
    auto start = std::chrono::high_resolution_clock::now();
    for (int i = 0; i < iterations; ++i) {
        add_simd_kernel(a, b, size);
    }
    auto end = std::chrono::high_resolution_clock::now();
    double simd_time = std::chrono::duration<double, std::milli>(end - start).count();
    double simd_avg = simd_time / iterations;
    double simd_gflops = (size / (simd_avg / 1000.0)) / 1e9;
    
    printf("[SIMD AVX2]\n");
    printf("  Total: %.4f ms | Per op: %.6f ms | Throughput: %.2f Gflops/s\n\n", 
           simd_time, simd_avg, simd_gflops);
    
    // Reset
    for (size_t i = 0; i < size; ++i) {
        a[i] = 1.0f;
        b[i] = 2.0f;
    }
    
    // Benchmark Scalar
    start = std::chrono::high_resolution_clock::now();
    for (int i = 0; i < iterations; ++i) {
        add_scalar(a, b, size);
    }
    end = std::chrono::high_resolution_clock::now();
    double scalar_time = std::chrono::duration<double, std::milli>(end - start).count();
    double scalar_avg = scalar_time / iterations;
    double scalar_gflops = (size / (scalar_avg / 1000.0)) / 1e9;
    
    printf("[Scalar]\n");
    printf("  Total: %.4f ms | Per op: %.6f ms | Throughput: %.2f Gflops/s\n\n", 
           scalar_time, scalar_avg, scalar_gflops);
    
    printf("[Speedup]\n");
    printf("  SIMD vs Scalar: %.2fx\n", scalar_avg / simd_avg);
    
    free(a);
    free(b);
    
    return 0;
}
