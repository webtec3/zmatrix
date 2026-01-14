#ifndef GPU_KERNELS_H
#define GPU_KERNELS_H

#include <stddef.h>

extern "C" {
    int gpu_available();
    void gpu_add(float* a, const float* b, size_t n);
    void gpu_sub(float* a, const float* b, size_t n);
    void gpu_mul(float* a, const float* b, size_t n);
    void gpu_relu(float* a, size_t n);
    void gpu_leaky_relu(float* a, float alpha, size_t n);
    void gpu_leaky_relu_derivative(float* a, float alpha, size_t n);
    void gpu_sigmoid(float* a, size_t n);
    void gpu_tanh(float* a, size_t n);
    void gpu_exp(float* a, size_t n);
    void gpu_abs(float* a, size_t n);
    void gpu_scalar_add(float* a, float value, size_t n);
    void gpu_scalar_sub(float* a, float value, size_t n);
    void gpu_scalar_mul(float* a, float value, size_t n);
    void gpu_scalar_div(float* a, float value, size_t n);
    void gpu_add_device(float* d_a, const float* d_b, size_t n);
    void gpu_sub_device(float* d_a, const float* d_b, size_t n);
    void gpu_mul_device(float* d_a, const float* d_b, size_t n);
    void gpu_relu_device(float* d_a, size_t n);
    void gpu_leaky_relu_device(float* d_a, float alpha, size_t n);
    void gpu_leaky_relu_derivative_device(float* d_a, float alpha, size_t n);
    void gpu_sigmoid_device(float* d_a, size_t n);
    void gpu_tanh_device(float* d_a, size_t n);
    void gpu_exp_device(float* d_a, size_t n);
    void gpu_abs_device(float* d_a, size_t n);
    void gpu_scalar_add_device(float* d_a, float value, size_t n);
    void gpu_scalar_sub_device(float* d_a, float value, size_t n);
    void gpu_scalar_mul_device(float* d_a, float value, size_t n);
    void gpu_scalar_div_device(float* d_a, float value, size_t n);
    void gpu_softmax(const float* A, float* B, const int* shape, int dims, int axis);
    void gpu_sin(const float* A, float* B, int N);
    void gpu_cos(const float* A, float* B, int N);
    void gpu_tan(const float* A, float* B, int N);
    void gpu_round(const float* A, float* B, int N);
    void gpu_floor(const float* A, float* B, int N);
    void gpu_ceil(const float* A, float* B, int N);
    void gpu_trunc(const float* A, float* B, int N);
    void gpu_negate(const float* A, float* B, int N);
    void gpu_sign(const float* A, float* B, int N);
    void gpu_reciprocal(const float* A, float* B, int N);
    void gpu_abs_diff(const float* A, const float* B, float* C, int N);
    void gpu_max(const float* A, const float* B, float* C, int N);
    void gpu_min(const float* A, const float* B, float* C, int N);
    void gpu_transpose(const float* A, float* B, int rows, int cols);
    void gpu_multiply_scalar(const float* A, float scalar, float* B, int N);
    void gpu_fill_random_uniform(void* curand_generator, float* A, int N, float min_val, float max_val);
    void gpu_fill_random_normal(void* curand_generator, float* A, int N, float mean, float stddev);
    void gpu_sum_all(const float* A, float* result, int N);
    void gpu_variance_all(const float* A, float* result, int N, float mean);
    void gpu_min_all(const float* A, float* result, int N);
    void gpu_max_all(const float* A, float* result, int N);
}

#endif // GPU_KERNELS_H
