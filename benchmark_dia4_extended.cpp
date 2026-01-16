#include <iostream>
#include <vector>
#include <cmath>
#include <chrono>
#include <immintrin.h>

const size_t ARRAY_SIZE = 6250000;
const int ITERATIONS = 50;

void min_scalar(const float* arr, size_t n) {
    volatile float result = arr[0];
    for (size_t i = 1; i < n; ++i) {
        if (arr[i] < result) result = arr[i];
    }
}

void max_scalar(const float* arr, size_t n) {
    volatile float result = arr[0];
    for (size_t i = 1; i < n; ++i) {
        if (arr[i] > result) result = arr[i];
    }
}

double sum_scalar(const float* arr, size_t n) {
    double total = 0.0;
    for (size_t i = 0; i < n; ++i) {
        total += arr[i];
    }
    return total;
}

void min_simd(const float* arr, size_t n) {
    if (n == 0) return;
    const size_t vec_size = 8;
    const size_t aligned_n = (n / vec_size) * vec_size;
    __m256 vmin = _mm256_set1_ps(arr[0]);
    for (size_t i = 0; i < aligned_n; i += vec_size) {
        __m256 va = _mm256_loadu_ps(&arr[i]);
        vmin = _mm256_min_ps(vmin, va);
    }
    __m256 shuf = _mm256_shuffle_ps(vmin, vmin, _MM_SHUFFLE(2, 3, 0, 1));
    __m256 mins = _mm256_min_ps(vmin, shuf);
    shuf = _mm256_shuffle_ps(mins, mins, _MM_SHUFFLE(1, 0, 3, 2));
    mins = _mm256_min_ps(mins, shuf);
    shuf = _mm256_permute2f128_ps(mins, mins, 1);
    mins = _mm256_min_ps(mins, shuf);
    volatile float result = _mm256_cvtss_f32(mins);
    for (size_t i = aligned_n; i < n; ++i) {
        if (arr[i] < result) result = arr[i];
    }
}

void max_simd(const float* arr, size_t n) {
    if (n == 0) return;
    const size_t vec_size = 8;
    const size_t aligned_n = (n / vec_size) * vec_size;
    __m256 vmax = _mm256_set1_ps(arr[0]);
    for (size_t i = 0; i < aligned_n; i += vec_size) {
        __m256 va = _mm256_loadu_ps(&arr[i]);
        vmax = _mm256_max_ps(vmax, va);
    }
    __m256 shuf = _mm256_shuffle_ps(vmax, vmax, _MM_SHUFFLE(2, 3, 0, 1));
    __m256 maxs = _mm256_max_ps(vmax, shuf);
    shuf = _mm256_shuffle_ps(maxs, maxs, _MM_SHUFFLE(1, 0, 3, 2));
    maxs = _mm256_max_ps(maxs, shuf);
    shuf = _mm256_permute2f128_ps(maxs, maxs, 1);
    maxs = _mm256_max_ps(maxs, shuf);
    volatile float result = _mm256_cvtss_f32(maxs);
    for (size_t i = aligned_n; i < n; ++i) {
        if (arr[i] > result) result = arr[i];
    }
}

double sum_simd(const float* arr, size_t n) {
    if (n == 0) return 0.0;
    const size_t vec_size = 8;
    const size_t aligned_n = (n / vec_size) * vec_size;
    __m256 vsum = _mm256_setzero_ps();
    for (size_t i = 0; i < aligned_n; i += vec_size) {
        __m256 va = _mm256_loadu_ps(&arr[i]);
        vsum = _mm256_add_ps(vsum, va);
    }
    __m256 hadd1 = _mm256_shuffle_ps(vsum, vsum, _MM_SHUFFLE(2, 3, 0, 1));
    __m256 hadd2 = _mm256_add_ps(vsum, hadd1);
    __m256 hadd3 = _mm256_shuffle_ps(hadd2, hadd2, _MM_SHUFFLE(1, 0, 3, 2));
    __m256 hadd4 = _mm256_add_ps(hadd2, hadd3);
    float *result_arr = (float*)&hadd4;
    double total = static_cast<double>(result_arr[0]);
    for (size_t i = aligned_n; i < n; ++i) {
        total += static_cast<double>(arr[i]);
    }
    return total;
}

void benchmark(const std::string& name, void (*func)(const float*, size_t), const float* arr, size_t n) {
    auto start = std::chrono::high_resolution_clock::now();
    for (int i = 0; i < ITERATIONS; ++i) {
        func(arr, n);
    }
    auto end = std::chrono::high_resolution_clock::now();
    double elapsed_ms = std::chrono::duration<double, std::milli>(end - start).count();
    double avg_ms = elapsed_ms / ITERATIONS;
    std::cout << name << ": " << avg_ms << " ms\n";
}

double benchmark_sum(const std::string& name, double (*func)(const float*, size_t), const float* arr, size_t n) {
    auto start = std::chrono::high_resolution_clock::now();
    volatile double result = 0.0;
    for (int i = 0; i < ITERATIONS; ++i) {
        result = func(arr, n);
    }
    auto end = std::chrono::high_resolution_clock::now();
    double elapsed_ms = std::chrono::duration<double, std::milli>(end - start).count();
    double avg_ms = elapsed_ms / ITERATIONS;
    std::cout << name << ": " << avg_ms << " ms\n";
    return avg_ms;
}

int main() {
    float* arr = new float[ARRAY_SIZE];
    for (size_t i = 0; i < ARRAY_SIZE; ++i) {
        arr[i] = 1.0f + (i % 100);
    }
    
    std::cout << "Array size: " << ARRAY_SIZE << " floats\n";
    std::cout << "Iterations: " << ITERATIONS << "\n\n";
    
    std::cout << "[MIN]\n";
    benchmark("Scalar  ", min_scalar, arr, ARRAY_SIZE);
    benchmark("SIMD    ", min_simd, arr, ARRAY_SIZE);
    std::cout << "\n";
    
    std::cout << "[MAX]\n";
    benchmark("Scalar  ", max_scalar, arr, ARRAY_SIZE);
    benchmark("SIMD    ", max_simd, arr, ARRAY_SIZE);
    std::cout << "\n";
    
    std::cout << "[SUM]\n";
    benchmark_sum("Scalar  ", sum_scalar, arr, ARRAY_SIZE);
    benchmark_sum("SIMD    ", sum_simd, arr, ARRAY_SIZE);
    std::cout << "\n";
    
    delete[] arr;
    return 0;
}
