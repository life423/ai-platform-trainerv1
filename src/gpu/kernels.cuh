#pragma once

#ifdef __CUDACC__
#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#endif

namespace cudarl {

// Forward declarations
struct AgentState;
struct EnvironmentConfig;

/**
 * CUDA kernel declarations for Phase 2 implementation
 * For now, these are placeholders that will be implemented later
 */

#ifdef __CUDACC__

/**
 * Reset all environments to initial state
 */
__global__ void reset_environments_kernel(
    AgentState* agent_states,
    int batch_size,
    EnvironmentConfig config,
    unsigned int seed
);

/**
 * Step all environments with given actions
 */
__global__ void step_environments_kernel(
    AgentState* agent_states,
    const int* actions,
    int batch_size,
    EnvironmentConfig config
);

/**
 * Update Q-table values using Q-learning rule
 */
__global__ void update_q_values_kernel(
    float* q_tables,
    const int* states,
    const int* actions,
    const float* rewards,
    const int* next_states,
    int batch_size,
    EnvironmentConfig config
);

/**
 * Select actions using epsilon-greedy policy
 */
__global__ void select_actions_kernel(
    const float* q_tables,
    const int* states,
    int* actions,
    int batch_size,
    EnvironmentConfig config,
    unsigned int seed
);

#endif // __CUDACC__

/**
 * Host wrapper functions (callable from C++)
 */

void cuda_reset_environments(
    void* d_agent_states,
    int batch_size,
    const EnvironmentConfig& config
);

void cuda_step_environments(
    void* d_agent_states,
    const int* actions,
    int batch_size,
    const EnvironmentConfig& config
);

void cuda_update_q_values(
    void* d_q_tables,
    const int* states,
    const int* actions,
    const float* rewards,
    const int* next_states,
    int batch_size,
    const EnvironmentConfig& config
);

void cuda_select_actions(
    const void* d_q_tables,
    const int* states,
    int* actions,
    int batch_size,
    const EnvironmentConfig& config
);

/**
 * Utility functions
 */
int get_optimal_block_size();
dim3 get_grid_dim(int total_threads, int block_size);

} // namespace cudarl
