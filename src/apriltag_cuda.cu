#include <cuda_runtime.h>

__global__ void k_bgr_to_gray(const unsigned char* bgr, unsigned char* gray, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) {
        return;
    }
    const unsigned char* p = bgr + i * 3;
    gray[i] = static_cast<unsigned char>((p[0] * 29 + p[1] * 150 + p[2] * 77) >> 8);
}

// Returns 0 on success, non-zero on CUDA failure (caller should fall back to CPU).
extern "C" int calico_cuda_bgr_to_gray(const unsigned char* bgr, unsigned char* gray,
        int width, int height) {
    if (bgr == nullptr || gray == nullptr || width <= 0 || height <= 0) {
        return -1;
    }
    const int n = width * height;
    unsigned char* d_bgr = nullptr;
    unsigned char* d_gray = nullptr;
    if (cudaMalloc(&d_bgr, static_cast<size_t>(n) * 3) != cudaSuccess) {
        return -1;
    }
    if (cudaMalloc(&d_gray, static_cast<size_t>(n)) != cudaSuccess) {
        cudaFree(d_bgr);
        return -1;
    }
    if (cudaMemcpy(d_bgr, bgr, static_cast<size_t>(n) * 3, cudaMemcpyHostToDevice) != cudaSuccess) {
        cudaFree(d_bgr);
        cudaFree(d_gray);
        return -1;
    }
    const int threads = 256;
    const int blocks = (n + threads - 1) / threads;
    k_bgr_to_gray<<<blocks, threads>>>(d_bgr, d_gray, n);
    if (cudaGetLastError() != cudaSuccess ||
            cudaMemcpy(gray, d_gray, static_cast<size_t>(n), cudaMemcpyDeviceToHost) != cudaSuccess) {
        cudaFree(d_bgr);
        cudaFree(d_gray);
        return -1;
    }
    cudaFree(d_bgr);
    cudaFree(d_gray);
    return 0;
}
