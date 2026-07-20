#pragma once

#include <stddef.h>

#ifdef HAVE_CUDA
extern "C" int gpu_available();
extern "C" void gpu_require_available();
extern "C" const char* gpu_driver_path();
extern "C" int gpu_memory_pools_supported();
extern "C" int gpu_device_allocate(void** pointer, size_t bytes, int request_async, int* used_async);
extern "C" int gpu_device_free(void* pointer, int allocation_was_async);
extern "C" int gpu_device_memory_info(size_t* free_bytes, size_t* total_bytes);
extern "C" void gpu_add(float* a, const float* b, size_t n);
extern "C" void gpu_sub(float* a, const float* b, size_t n);
extern "C" void gpu_mul(float* a, const float* b, size_t n);
extern "C" void gpu_relu(float* a, size_t n);
extern "C" void gpu_leaky_relu(float* a, float alpha, size_t n);
extern "C" void gpu_leaky_relu_derivative(float* a, float alpha, size_t n);
extern "C" void gpu_sigmoid(float* a, size_t n);
extern "C" void gpu_tanh(float* a, size_t n);
extern "C" void gpu_exp(float* a, size_t n);
extern "C" void gpu_abs(float* a, size_t n);
extern "C" void gpu_scalar_add(float* a, float value, size_t n);
extern "C" void gpu_scalar_sub(float* a, float value, size_t n);
extern "C" void gpu_scalar_mul(float* a, float value, size_t n);
extern "C" void gpu_scalar_div(float* a, float value, size_t n);
extern "C" void gpu_matmul(const float* a, const float* b, float* c, size_t m, size_t k, size_t n);
extern "C" void gpu_matmul_device(const float* d_a, const float* d_b, float* d_c, size_t m, size_t k, size_t n);

extern "C" void gpu_add_device(float* d_a, const float* d_b, size_t n);
extern "C" void gpu_sub_device(float* d_a, const float* d_b, size_t n);
extern "C" void gpu_mul_device(float* d_a, const float* d_b, size_t n);
extern "C" void gpu_div_device(float* d_a, const float* d_b, size_t n);
extern "C" void gpu_relu_device(float* d_a, size_t n);
extern "C" void gpu_leaky_relu_device(float* d_a, float alpha, size_t n);
extern "C" void gpu_leaky_relu_derivative_device(float* d_a, float alpha, size_t n);
extern "C" void gpu_sigmoid_device(float* d_a, size_t n);
extern "C" void gpu_tanh_device(float* d_a, size_t n);
extern "C" void gpu_exp_device(float* d_a, size_t n);
extern "C" void gpu_abs_device(float* d_a, size_t n);
extern "C" void gpu_scalar_add_device(float* d_a, float value, size_t n);
extern "C" void gpu_scalar_sub_device(float* d_a, float value, size_t n);
extern "C" void gpu_scalar_mul_device(float* d_a, float value, size_t n);
extern "C" void gpu_scalar_div_device(float* d_a, float value, size_t n);
extern "C" void gpu_adam_update_device(
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
extern "C" void gpu_pow_device(float* d_a, float exponent, size_t n);
extern "C" void gpu_log_device(float* d_a, size_t n);
extern "C" void gpu_sqrt_device(float* d_a, size_t n);
extern "C" void gpu_clip_device(float* d_a, float min_value, float max_value, size_t n);
extern "C" void gpu_softmax_device(float* d_a, size_t rows, size_t cols, int one_dimensional);
extern "C" void gpu_softmax_derivative_device(float* d_a, size_t n);
extern "C" void gpu_greater_device(const float* d_a, const float* d_b, float* d_output, size_t n, size_t broadcast_width, float scalar, int use_scalar);
extern "C" void gpu_broadcast_device(const float* d_input, float* d_output, const size_t* output_shape, const size_t* output_strides, const size_t* input_shape, const size_t* input_strides, size_t output_rank, size_t input_rank, size_t output_size);
extern "C" void gpu_tile_device(const float* d_input, float* d_output, size_t input_size, size_t output_size);
extern "C" void gpu_cumsum_device(const float* d_input, float* d_output, size_t rows, size_t cols, int axis, int one_dimensional);
extern "C" float gpu_dot_value_device(const float* d_a, const float* d_b, size_t n);
extern "C" void gpu_matvec_device(const float* d_matrix, const float* d_vector, float* d_output, size_t rows, size_t cols);
extern "C" void gpu_fill_device(float* d_a, float value, size_t n);
extern "C" void gpu_transpose_device(const float* d_input, float* d_output, size_t rows, size_t cols);
extern "C" void gpu_sum_axis_device(const float* d_input, float* d_output, size_t outer, size_t axis_size, size_t inner);
extern "C" void gpu_mean_axis_device(const float* d_input, float* d_output, size_t outer, size_t axis_size, size_t inner);
extern "C" void gpu_min_axis_device(const float* d_input, float* d_output, size_t outer, size_t axis_size, size_t inner);
extern "C" void gpu_max_axis_device(const float* d_input, float* d_output, size_t outer, size_t axis_size, size_t inner);
extern "C" void gpu_arg_axis_device(const float* d_input, float* d_output, size_t outer, size_t axis_size, size_t inner, int find_max);
extern "C" float gpu_sum_value_device(const float* d_input, size_t n);
extern "C" float gpu_min_value_device(const float* d_input, size_t n);
extern "C" float gpu_max_value_device(const float* d_input, size_t n);
extern "C" size_t gpu_arg_value_device(const float* d_input, size_t n, int find_max);
#endif
