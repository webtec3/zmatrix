#ifndef GPU_KERNELS_H
#define GPU_KERNELS_H

#include <stddef.h>

extern "C" {
    int gpu_available();
    void gpu_require_available();
    const char* gpu_driver_path();
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
    void gpu_div_device(float* d_a, const float* d_b, size_t n);
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
    void gpu_adam_update_device(
        float* d_parameter,
        const float* d_gradient,
        float* d_first_moment,
        float* d_second_moment,
        float learning_rate,
        float beta1,
        float beta2,
        float epsilon,
        float one_minus_beta1,
        float one_minus_beta2,
        float bias_correction1,
        float bias_correction2,
        size_t n
    );
    void gpu_pow_device(float* d_a, float exponent, size_t n);
    void gpu_log_device(float* d_a, size_t n);
    void gpu_sqrt_device(float* d_a, size_t n);
    void gpu_clip_device(float* d_a, float min_value, float max_value, size_t n);
    void gpu_softmax_device(float* d_a, size_t rows, size_t cols, int one_dimensional);
void gpu_softmax_derivative_device(float* d_a, size_t n);
void gpu_greater_device(const float* d_a, const float* d_b, float* d_output, size_t n, size_t broadcast_width, float scalar, int use_scalar);
void gpu_broadcast_device(const float* d_input, float* d_output, const size_t* output_shape, const size_t* output_strides, const size_t* input_shape, const size_t* input_strides, size_t output_rank, size_t input_rank, size_t output_size);
void gpu_tile_device(const float* d_input, float* d_output, size_t input_size, size_t output_size);
void gpu_cumsum_device(const float* d_input, float* d_output, size_t rows, size_t cols, int axis, int one_dimensional);
float gpu_dot_value_device(const float* d_a, const float* d_b, size_t n);
void gpu_matvec_device(const float* d_matrix, const float* d_vector, float* d_output, size_t rows, size_t cols);
void gpu_fill_device(float* d_a, float value, size_t n);
    void gpu_transpose_device(const float* d_input, float* d_output, size_t rows, size_t cols);
    void gpu_sum_axis_device(const float* d_input, float* d_output, size_t outer, size_t axis_size, size_t inner);
    void gpu_mean_axis_device(const float* d_input, float* d_output, size_t outer, size_t axis_size, size_t inner);
    void gpu_min_axis_device(const float* d_input, float* d_output, size_t outer, size_t axis_size, size_t inner);
    void gpu_max_axis_device(const float* d_input, float* d_output, size_t outer, size_t axis_size, size_t inner);
    void gpu_arg_axis_device(const float* d_input, float* d_output, size_t outer, size_t axis_size, size_t inner, int find_max);
    float gpu_sum_value_device(const float* d_input, size_t n);
    float gpu_min_value_device(const float* d_input, size_t n);
    float gpu_max_value_device(const float* d_input, size_t n);
    size_t gpu_arg_value_device(const float* d_input, size_t n, int find_max);
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
