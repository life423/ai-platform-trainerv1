#include "cuda_utils.h"
#ifdef CUDA_ENABLED
#include <cuda_runtime.h>
#endif
#include <iostream>
#include <cstring>

namespace cudarl {

bool is_cuda_available() {
#ifdef CUDA_ENABLED
    int device_count = 0;
    cudaError_t error = cudaGetDeviceCount(&device_count);
    return (error == cudaSuccess && device_count > 0);
#else
    return false;
#endif
}

CudaDeviceInfo get_cuda_device_info() {
    CudaDeviceInfo info = {};
    
#ifdef CUDA_ENABLED
    cudaGetDeviceCount(&info.device_count);
    if (info.device_count > 0) {
        cudaGetDevice(&info.current_device);
        
        cudaDeviceProp prop;
        cudaGetDeviceProperties(&prop, info.current_device);
        
        size_t free_mem, total_mem;
        cudaMemGetInfo(&free_mem, &total_mem);
        
        info.total_memory = total_mem;
        info.free_memory = free_mem;
        info.multiprocessor_count = prop.multiProcessorCount;
        strncpy(info.device_name, prop.name, sizeof(info.device_name) - 1);
    }
#else
    info.device_count = 0;
    strcpy(info.device_name, "CUDA not available");
#endif
    
    return info;
}

void* cuda_malloc(size_t size) {
#ifdef CUDA_ENABLED
    void* ptr = nullptr;
    CUDA_CHECK(cudaMalloc(&ptr, size));
    return ptr;
#else
    return nullptr;
#endif
}

void cuda_free(void* ptr) {
#ifdef CUDA_ENABLED
    if (ptr) {
        CUDA_CHECK(cudaFree(ptr));
    }
#endif
}

void cuda_memcpy_host_to_device(void* dst, const void* src, size_t size) {
#ifdef CUDA_ENABLED
    CUDA_CHECK(cudaMemcpy(dst, src, size, cudaMemcpyHostToDevice));
#endif
}

void cuda_memcpy_device_to_host(void* dst, const void* src, size_t size) {
#ifdef CUDA_ENABLED
    CUDA_CHECK(cudaMemcpy(dst, src, size, cudaMemcpyDeviceToHost));
#endif
}

} // namespace cudarl
