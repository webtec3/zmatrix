#pragma once

#include <stddef.h>

#ifdef HAVE_CUDA
extern "C" int gpu_available();
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
#endif