#pragma once

#ifdef __CUDACC__
#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#define CUDA_AVAILABLE 1
#else
#define CUDA_AVAILABLE 0
#endif

namespace cudarl {

/**
 * Check if CUDA is available and functioning
 */
bool is_cuda_available();

/**
 * Get CUDA device information
 */
struct CudaDeviceInfo {
    int device_count;
    int current_device;
    size_t total_memory;
    size_t free_memory;
    int multiprocessor_count;
    char device_name[256];
};

CudaDeviceInfo get_cuda_device_info();

/**
 * CUDA error checking utility
 */
#ifdef CUDA_ENABLED
#define CUDA_CHECK(call) \
    do { \
        cudaError_t err = call; \
        if (err != cudaSuccess) { \
            fprintf(stderr, "CUDA error at %s:%d - %s\n", __FILE__, __LINE__, cudaGetErrorString(err)); \
            exit(1); \
        } \
    } while(0)
#else
#define CUDA_CHECK(call) // No-op when CUDA not available
#endif

/**
 * Memory management utilities
 */
void* cuda_malloc(size_t size);
void cuda_free(void* ptr);
void cuda_memcpy_host_to_device(void* dst, const void* src, size_t size);
void cuda_memcpy_device_to_host(void* dst, const void* src, size_t size);

/**
 * CUDA availability and device information
 */
bool is_cuda_available();
CudaDeviceInfo get_cuda_device_info();

} // namespace cudarl
