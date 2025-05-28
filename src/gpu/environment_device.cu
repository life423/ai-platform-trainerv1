#include "kernels.cuh"
#include "../core/environment.h"
#include "../core/cuda_utils.h"
#include <curand_kernel.h>
#include <iostream>
#include <chrono>

namespace cudarl {

/**
 * GPU-accelerated environment operations
 * This file contains the device-specific environment logic
 */

#ifdef __CUDACC__

/**
 * Device function to validate environment state
 */
__device__ bool validate_agent_state(const AgentState& state, const EnvironmentConfig& config) {
    return (state.x >= 0 && state.x < config.grid_width &&
            state.y >= 0 && state.y < config.grid_height &&
            state.goal_x >= 0 && state.goal_x < config.grid_width &&
            state.goal_y >= 0 && state.goal_y < config.grid_height &&
            state.episode_step >= 0 && state.episode_step <= config.max_episode_steps);
}

/**
 * Device function to calculate reward for current state
 */
__device__ float calculate_reward_device(const AgentState& state, const EnvironmentConfig& config) {
    float reward = -0.01f;  // Base step penalty
    
    // Check if goal reached
    if (state.x == state.goal_x && state.y == state.goal_y) {
        return 10.0f * config.reward_scale;
    }
    
    // Check if episode timed out
    if (state.episode_step >= config.max_episode_steps) {
        return -1.0f * config.reward_scale;
    }
    
    // Distance-based shaping reward
    float distance = sqrtf(
        powf(state.x - state.goal_x, 2.0f) + 
        powf(state.y - state.goal_y, 2.0f)
    );
    reward -= distance * 0.01f;
    
    return reward * config.reward_scale;
}

/**
 * Batch environment reset with validation
 */
__global__ void batch_reset_with_validation_kernel(
    AgentState* agent_states,
    bool* reset_success,
    int batch_size,
    EnvironmentConfig config,
    unsigned int seed
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= batch_size) return;
    
    curandState local_state;
    curand_init(seed + idx, idx, 0, &local_state);
    
    AgentState& state = agent_states[idx];
    bool success = false;
    int attempts = 0;
    const int max_attempts = 100;
    
    // Try to find valid start and goal positions
    while (!success && attempts < max_attempts) {
        state.x = curand(&local_state) % config.grid_width;
        state.y = curand(&local_state) % config.grid_height;
        state.goal_x = curand(&local_state) % config.grid_width;
        state.goal_y = curand(&local_state) % config.grid_height;
        
        // Ensure start != goal and state is valid
        if ((state.x != state.goal_x || state.y != state.goal_y) &&
            validate_agent_state(state, config)) {
            success = true;
        }
        attempts++;
    }
    
    if (success) {
        state.episode_step = 0;
        state.cumulative_reward = 0.0f;
        state.done = false;
    }
    
    reset_success[idx] = success;
}

/**
 * Parallel Q-table initialization
 */
__global__ void initialize_q_tables_kernel(
    float* q_tables,
    int batch_size,
    EnvironmentConfig config,
    float init_value
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int num_states = config.grid_width * config.grid_height;
    int total_q_values = batch_size * num_states * config.num_actions;
    
    if (idx < total_q_values) {
        q_tables[idx] = init_value;
    }
}

#endif // __CUDACC__

/**
 * Host functions for device environment management
 */

bool cuda_batch_reset_environments(
    void* d_agent_states,
    int batch_size,
    const EnvironmentConfig& config
) {
#ifdef __CUDACC__
    // Allocate temporary success tracking
    bool* d_reset_success;
    bool* h_reset_success = new bool[batch_size];
    
    CUDA_CHECK(cudaMalloc(&d_reset_success, batch_size * sizeof(bool)));
    
    int block_size = get_optimal_block_size();
    dim3 grid_dim = get_grid_dim(batch_size, block_size);
    
    auto seed = std::chrono::high_resolution_clock::now().time_since_epoch().count();
    
    batch_reset_with_validation_kernel<<<grid_dim, block_size>>>(
        static_cast<AgentState*>(d_agent_states),
        d_reset_success,
        batch_size,
        config,
        static_cast<unsigned int>(seed)
    );
    
    CUDA_CHECK(cudaDeviceSynchronize());
    
    // Check if all resets were successful
    CUDA_CHECK(cudaMemcpy(h_reset_success, d_reset_success, 
                         batch_size * sizeof(bool), cudaMemcpyDeviceToHost));
    
    bool all_success = true;
    for (int i = 0; i < batch_size; ++i) {
        if (!h_reset_success[i]) {
            all_success = false;
            std::cerr << "Warning: Environment " << i << " failed to reset" << std::endl;
        }
    }
    
    cudaFree(d_reset_success);
    delete[] h_reset_success;
    
    return all_success;
#else
    return false;
#endif
}

bool cuda_initialize_q_tables(
    void* d_q_tables,
    int batch_size,
    const EnvironmentConfig& config,
    float init_value
) {
#ifdef __CUDACC__
    int num_states = config.grid_width * config.grid_height;
    int total_q_values = batch_size * num_states * config.num_actions;
    
    int block_size = get_optimal_block_size();
    dim3 grid_dim = get_grid_dim(total_q_values, block_size);
    
    initialize_q_tables_kernel<<<grid_dim, block_size>>>(
        static_cast<float*>(d_q_tables),
        batch_size,
        config,
        init_value
    );
    
    CUDA_CHECK(cudaDeviceSynchronize());
    return true;
#else
    return false;
#endif
}

void* cuda_allocate_agent_states(int batch_size) {
#ifdef __CUDACC__
    void* d_states = nullptr;
    size_t size = batch_size * sizeof(AgentState);
    CUDA_CHECK(cudaMalloc(&d_states, size));
    return d_states;
#else
    return nullptr;
#endif
}

void* cuda_allocate_q_tables(int batch_size, const EnvironmentConfig& config) {
#ifdef __CUDACC__
    void* d_q_tables = nullptr;
    int num_states = config.grid_width * config.grid_height;
    size_t size = batch_size * num_states * config.num_actions * sizeof(float);
    CUDA_CHECK(cudaMalloc(&d_q_tables, size));
    return d_q_tables;
#else
    return nullptr;
#endif
}

bool cuda_copy_agent_states_to_host(
    const void* d_agent_states,
    AgentState* h_agent_states,
    int batch_size
) {
#ifdef __CUDACC__
    size_t size = batch_size * sizeof(AgentState);
    CUDA_CHECK(cudaMemcpy(h_agent_states, d_agent_states, size, cudaMemcpyDeviceToHost));
    return true;
#else
    return false;
#endif
}

bool cuda_copy_agent_states_to_device(
    void* d_agent_states,
    const AgentState* h_agent_states,
    int batch_size
) {
#ifdef __CUDACC__
    size_t size = batch_size * sizeof(AgentState);
    CUDA_CHECK(cudaMemcpy(d_agent_states, h_agent_states, size, cudaMemcpyHostToDevice));
    return true;
#else
    return false;
#endif
}

} // namespace cudarl
