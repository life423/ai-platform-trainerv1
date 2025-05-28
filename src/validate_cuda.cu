#include <cuda_runtime.h>
#include <iostream>
#include <cstdlib>

// Simple CUDA kernel to validate CUDA is working
__global__ void validateCudaKernel(int *result) {
    // This kernel can use threadIdx and blockIdx
    int idx = threadIdx.x + blockIdx.x * blockDim.x;
    if (idx == 0) {
        *result = 42;  // Magic number to verify kernel ran
    }
}

// Function to check CUDA error
#define CUDA_CHECK(call) \
    do { \
        cudaError_t error = call; \
        if (error != cudaSuccess) { \
            std::cerr << "CUDA error at " << __FILE__ << ":" << __LINE__ \
                     << " code=" << error << " \"" << cudaGetErrorString(error) << "\"" << std::endl; \
            exit(1); \
        } \
    } while(0)

int main() {
    std::cout << "=== CUDA Validation Test ===" << std::endl;
    
    // Check if CUDA is available
    int deviceCount = 0;
    cudaError_t error = cudaGetDeviceCount(&deviceCount);
    
    if (error != cudaSuccess) {
        std::cerr << "CUDA Error: " << cudaGetErrorString(error) << std::endl;
        return 1;
    }
    
    if (deviceCount == 0) {
        std::cerr << "No CUDA devices found!" << std::endl;
        return 1;
    }
    
    std::cout << "Found " << deviceCount << " CUDA device(s)" << std::endl;
    
    // Get device properties
    for (int i = 0; i < deviceCount; i++) {
        cudaDeviceProp prop;
        CUDA_CHECK(cudaGetDeviceProperties(&prop, i));
        
        std::cout << "\nDevice " << i << ": " << prop.name << std::endl;
        std::cout << "  Compute capability: " << prop.major << "." << prop.minor << std::endl;
        std::cout << "  Total memory: " << prop.totalGlobalMem / (1024*1024) << " MB" << std::endl;
        std::cout << "  Multiprocessors: " << prop.multiProcessorCount << std::endl;
    }
    
    // Test kernel execution
    std::cout << "\nTesting kernel execution..." << std::endl;
    
    int h_result = 0;
    int *d_result;
    
    // Allocate device memory
    CUDA_CHECK(cudaMalloc(&d_result, sizeof(int)));
    CUDA_CHECK(cudaMemcpy(d_result, &h_result, sizeof(int), cudaMemcpyHostToDevice));
    
    // Launch kernel
    validateCudaKernel<<<1, 1>>>(d_result);
    
    // Check for kernel launch errors
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());
    
    // Copy result back
    CUDA_CHECK(cudaMemcpy(&h_result, d_result, sizeof(int), cudaMemcpyDeviceToHost));
    
    // Free device memory
    CUDA_CHECK(cudaFree(d_result));
    
    // Verify result
    if (h_result == 42) {
        std::cout << "✓ CUDA kernel execution successful!" << std::endl;
        std::cout << "✓ CUDA is properly configured and working!" << std::endl;
        return 0;
    } else {
        std::cerr << "✗ CUDA kernel execution failed!" << std::endl;
        return 1;
    }
}
