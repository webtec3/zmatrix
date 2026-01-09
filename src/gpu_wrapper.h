#pragma once

#ifdef HAVE_CUDA
extern "C" void gpu_add(float* a, const float* b, size_t n);
#endif