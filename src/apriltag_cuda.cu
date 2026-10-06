#include <cuda_runtime.h>

__global__ void k_bgr_to_gray(const unsigned char* bgr, unsigned char* gray, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) {
        return;
    }
    const unsigned char* p = bgr + i * 3;
    gray[i] = static_cast<unsigned char>((p[0] * 29 + p[1] * 150 + p[2] * 77) >> 8);
}

extern "C" void calico_cuda_bgr_to_gray(const unsigned char* bgr, unsigned char* gray,
        int width, int height) {
    const int n = width * height;
    unsigned char* d_bgr = nullptr;
    unsigned char* d_gray = nullptr;
    cudaMalloc(&d_bgr, static_cast<size_t>(n) * 3);
    cudaMalloc(&d_gray, static_cast<size_t>(n));
    cudaMemcpy(d_bgr, bgr, static_cast<size_t>(n) * 3, cudaMemcpyHostToDevice);
    const int threads = 256;
    const int blocks = (n + threads - 1) / threads;
    k_bgr_to_gray<<<blocks, threads>>>(d_bgr, d_gray, n);
    cudaMemcpy(gray, d_gray, static_cast<size_t>(n), cudaMemcpyDeviceToHost);
    cudaFree(d_bgr);
    cudaFree(d_gray);
}
