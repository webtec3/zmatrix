#ifndef GPU_KERNELS_H
#define GPU_KERNELS_H

extern "C" {
    void gpu_add(const float* A, const float* B, float* C, int N);
    void gpu_mul(const float* A, const float* B, float* C, int N);
    void gpu_relu(const float* A, float* B, int N);
    void gpu_leaky_relu(const float* A, float alpha, float* B, int N);
    void gpu_sigmoid(const float* A, float* B, int N);
    void gpu_tanh(const float* A, float* B, int N);
    void gpu_softmax(const float* A, float* B, const int* shape, int dims, int axis);
    void gpu_abs(const float* A, float* B, int N);
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