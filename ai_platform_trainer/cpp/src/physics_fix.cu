#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <stdio.h>
#include <math.h>
#include <thrust/device_vector.h>
#include <thrust/host_vector.h>

#include "../include/physics.h"

namespace gpu_env {

// CUDA error checking helper macro
#define CUDA_CHECK(call) \
do { \
    cudaError_t err = call; \
    if (err != cudaSuccess) { \
        fprintf(stderr, "CUDA error in file '%s' in line %i: %s.\n", \
                __FILE__, __LINE__, cudaGetErrorString(err)); \
        exit(EXIT_FAILURE); \
    } \
} while (0)

// Rest of the file remains unchanged until the end

std::pair<float, float> PhysicsEngine::calculate_evasion_vector(
    float enemy_x,
    float enemy_y,
    const MissileBatch& missiles,
    int prediction_steps
) {
    if (!initialized_) {
        initialize();
    }
    
    const size_t missile_count = missiles.size();
    
    // Default result if no missiles
    if (missile_count == 0) {
        return std::make_pair(0.0f, 0.0f);
    }
    
    // Ensure we have enough device memory
    const size_t required_memory = missile_count * 4;
    if (required_memory > d_temp_array_size_) {
        allocate_device_memory(required_memory);
    }
    
    // Create device vectors for missile data
    thrust::device_vector<float> d_missiles_x(missiles.x.begin(), missiles.x.end());
    thrust::device_vector<float> d_missiles_y(missiles.y.begin(), missiles.y.end());
    thrust::device_vector<float> d_missiles_vx(missiles.vx.begin(), missiles.vx.end());
    thrust::device_vector<float> d_missiles_vy(missiles.vy.begin(), missiles.vy.end());
    
    // Prepare output variables
    float evasion_x = 0.0f;
    float evasion_y = 0.0f;
    
    // Call the CUDA kernel launcher
    cuda_calculate_evasion_vector(
        enemy_x, enemy_y,
        thrust::raw_pointer_cast(d_missiles_x.data()),
        thrust::raw_pointer_cast(d_missiles_y.data()),
        thrust::raw_pointer_cast(d_missiles_vx.data()),
        thrust::raw_pointer_cast(d_missiles_vy.data()),
        static_cast<int>(missile_count),
        prediction_steps,
        &evasion_x, &evasion_y
    );
    
    return std::make_pair(evasion_x, evasion_y);
}

// Add missing methods for memory management
void PhysicsEngine::allocate_device_memory(size_t size) {
    // Free existing memory if already allocated
    free_device_memory();
    
    // Allocate new memory
    CUDA_CHECK(cudaMalloc(&d_temp_float_array1_, size * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_temp_float_array2_, size * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_temp_bool_array_, size * sizeof(bool)));
    
    d_temp_array_size_ = size;
}

void PhysicsEngine::free_device_memory() {
    if (d_temp_float_array1_) {
        CUDA_CHECK(cudaFree(d_temp_float_array1_));
        d_temp_float_array1_ = nullptr;
    }
    
    if (d_temp_float_array2_) {
        CUDA_CHECK(cudaFree(d_temp_float_array2_));
        d_temp_float_array2_ = nullptr;
    }
    
    if (d_temp_bool_array_) {
        CUDA_CHECK(cudaFree(d_temp_bool_array_));
        d_temp_bool_array_ = nullptr;
    }
    
    d_temp_array_size_ = 0;
}

} // namespace gpu_env