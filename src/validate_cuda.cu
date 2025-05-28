#include <cuda_runtime.h>
#include <iostream>

__global__ void dummyKernel(float* data) {
    int idx = threadIdx.x + blockIdx.x * blockDim.x;
    if (idx == 0) data[0] = 3.1415f;
}

int main() {
    // 1. Device count
    int deviceCount = 0;
    cudaError_t err = cudaGetDeviceCount(&deviceCount);
    if (err != cudaSuccess || deviceCount == 0) {
        std::cerr << "No CUDA device found: " << cudaGetErrorString(err) << "\n";
        return 1;
    }
    cudaDeviceProp prop;
    cudaGetDeviceProperties(&prop, 0);
    std::cout << "Using GPU: " << prop.name << "\n";

    // 2. Allocate & launch kernel
    float* d_val = nullptr;
    cudaMalloc(&d_val, sizeof(float));
    cudaMemset(d_val, 0, sizeof(float));

    // Timing events
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    cudaEventRecord(start);

    dummyKernel<<<1, 32>>>(d_val);

    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    float ms = 0;
    cudaEventElapsedTime(&ms, start, stop);

    // 3. Copy back and print
    float h_val = 0;
    cudaMemcpy(&h_val, d_val, sizeof(float), cudaMemcpyDeviceToHost);
    std::cout << "Kernel time: " << ms << " ms, result: " << h_val << "\n";

    // 4. Memory info
    size_t freeMem = 0, totalMem = 0;
    cudaMemGetInfo(&freeMem, &totalMem);
    std::cout << "GPU Memory Used: "
              << (totalMem - freeMem) / 1024 / 1024 << " MB / "
              << totalMem / 1024 / 1024 << " MB\n";

    // Cleanup
    cudaFree(d_val);
    cudaEventDestroy(start);
    cudaEventDestroy(stop);
    return 0;
}
