// gpu_kernels.cu
#include <cuda_runtime.h>

extern "C" void gpu_add(float* a, const float* b, size_t n);

__global__ void kernel_add(float* a, const float* b, size_t n) {
    size_t i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) {
        a[i] += b[i];
    }
}

extern "C" void gpu_add(float* a, const float* b, size_t n) {
    float *d_a = nullptr, *d_b = nullptr;

    cudaMalloc((void**)&d_a, n * sizeof(float));
    cudaMalloc((void**)&d_b, n * sizeof(float));

    cudaMemcpy(d_a, a, n * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_b, b, n * sizeof(float), cudaMemcpyHostToDevice);

    int threads = 256;
    int blocks = (n + threads - 1) / threads;
    kernel_add<<<blocks, threads>>>(d_a, d_b, n);

    cudaMemcpy(a, d_a, n * sizeof(float), cudaMemcpyDeviceToHost);

    cudaFree(d_a);
    cudaFree(d_b);
}
