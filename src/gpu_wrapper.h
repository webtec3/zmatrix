#pragma once

#include <stddef.h>

#ifdef HAVE_CUDA
extern "C" int gpu_available();
extern "C" void gpu_require_available();
extern "C" const char* gpu_driver_path();
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
extern "C" void gpu_pow_device(float* d_a, float exponent, size_t n);
extern "C" void gpu_log_device(float* d_a, size_t n);
extern "C" void gpu_sqrt_device(float* d_a, size_t n);
extern "C" void gpu_clip_device(float* d_a, float min_value, float max_value, size_t n);
extern "C" void gpu_softmax_device(float* d_a, size_t rows, size_t cols, int one_dimensional);
extern "C" void gpu_softmax_derivative_device(float* d_a, size_t n);
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
