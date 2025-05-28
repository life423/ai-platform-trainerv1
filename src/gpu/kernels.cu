#include "kernels.cuh"
#include "../core/environment.h"
#include <curand_kernel.h>
#include <cmath>

namespace cudarl {

#ifdef __CUDACC__

// Device utility functions
__device__ int state_to_index_device(int x, int y, int grid_width) {
    return y * grid_width + x;
}

__device__ void index_to_state_device(int index, int& x, int& y, int grid_width) {
    x = index % grid_width;
    y = index / grid_width;
}

__device__ float calculate_distance_device(int x1, int y1, int x2, int y2) {
    float dx = x1 - x2;
    float dy = y1 - y2;
    return sqrtf(dx * dx + dy * dy);
}

/**
 * Initialize curand states for random number generation
 */
__global__ void init_curand_states(curandState* states, unsigned int seed, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        curand_init(seed, idx, 0, &states[idx]);
    }
}

/**
 * Reset all environments to initial state
 */
__global__ void reset_environments_kernel(
    AgentState* agent_states,
    int batch_size,
    EnvironmentConfig config,
    unsigned int seed
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= batch_size) return;
    
    // Initialize local random state
    curandState local_state;
    curand_init(seed, idx, 0, &local_state);
    
    AgentState& state = agent_states[idx];
    
    // Random start position
    state.x = curand(&local_state) % config.grid_width;
    state.y = curand(&local_state) % config.grid_height;
    
    // Random goal position (different from start)
    do {
        state.goal_x = curand(&local_state) % config.grid_width;
        state.goal_y = curand(&local_state) % config.grid_height;
    } while (state.goal_x == state.x && state.goal_y == state.y);
    
    state.episode_step = 0;
    state.cumulative_reward = 0.0f;
    state.done = false;
}

/**
 * Step all environments with given actions
 */
__global__ void step_environments_kernel(
    AgentState* agent_states,
    const int* actions,
    int batch_size,
    EnvironmentConfig config
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= batch_size) return;
    
    AgentState& state = agent_states[idx];
    
    if (state.done) return;  // Skip finished episodes
    
    // Apply action: 0=up, 1=down, 2=left, 3=right
    int new_x = state.x;
    int new_y = state.y;
    
    switch (actions[idx]) {
        case 0: new_y = max(0, new_y - 1); break;           // up
        case 1: new_y = min(config.grid_height - 1, new_y + 1); break;  // down
        case 2: new_x = max(0, new_x - 1); break;           // left
        case 3: new_x = min(config.grid_width - 1, new_x + 1); break;   // right
    }
    
    state.x = new_x;
    state.y = new_y;
    state.episode_step++;
    
    // Calculate reward
    float distance_to_goal = calculate_distance_device(state.x, state.y, state.goal_x, state.goal_y);
    
    float reward = -0.01f;  // Small step penalty
    
    // Check if reached goal
    if (state.x == state.goal_x && state.y == state.goal_y) {
        reward = 10.0f;  // Large reward for reaching goal
        state.done = true;
    }
    // Time limit
    else if (state.episode_step >= config.max_episode_steps) {
        reward = -1.0f;  // Penalty for timeout
        state.done = true;
    }
    // Distance-based shaping
    else {
        reward -= distance_to_goal * 0.01f;  // Encourage getting closer
    }
    
    state.cumulative_reward += reward * config.reward_scale;
}

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
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= batch_size) return;
    
    int state = states[idx];
    int action = actions[idx];
    float reward = rewards[idx];
    int next_state = next_states[idx];
    
    int num_states = config.grid_width * config.grid_height;
    
    if (state >= num_states || next_state >= num_states || action >= config.num_actions) {
        return;  // Invalid indices
    }
    
    // Calculate Q-table indices
    int env_offset = idx * num_states * config.num_actions;
    int current_q_idx = env_offset + state * config.num_actions + action;
    
    float current_q = q_tables[current_q_idx];
    
    // Find max Q-value for next state
    float max_next_q = q_tables[env_offset + next_state * config.num_actions];
    for (int a = 1; a < config.num_actions; ++a) {
        float q_val = q_tables[env_offset + next_state * config.num_actions + a];
        max_next_q = fmaxf(max_next_q, q_val);
    }
    
    // Q-learning update: Q(s,a) = Q(s,a) + α[r + γ max_a' Q(s',a') - Q(s,a)]
    float target = reward + config.discount_factor * max_next_q;
    float new_q = current_q + config.learning_rate * (target - current_q);
    
    q_tables[current_q_idx] = new_q;
}

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
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= batch_size) return;
    
    // Initialize local random state
    curandState local_state;
    curand_init(seed, idx, 0, &local_state);
    
    int state = states[idx];
    int num_states = config.grid_width * config.grid_height;
    
    if (state >= num_states) {
        actions[idx] = curand(&local_state) % config.num_actions;  // Random action for invalid state
        return;
    }
    
    // Epsilon-greedy action selection
    float epsilon_roll = curand_uniform(&local_state);
    
    if (epsilon_roll < config.epsilon) {
        // Random action
        actions[idx] = curand(&local_state) % config.num_actions;
    } else {
        // Greedy action (highest Q-value)
        int env_offset = idx * num_states * config.num_actions;
        int best_action = 0;
        float best_q = q_tables[env_offset + state * config.num_actions];
        
        for (int a = 1; a < config.num_actions; ++a) {
            float q_val = q_tables[env_offset + state * config.num_actions + a];
            if (q_val > best_q) {
                best_q = q_val;
                best_action = a;
            }
        }
        
        actions[idx] = best_action;
    }
}

#endif // __CUDACC__

// Host wrapper functions

void cuda_reset_environments(
    void* d_agent_states,
    int batch_size,
    const EnvironmentConfig& config
) {
#ifdef __CUDACC__
    int block_size = get_optimal_block_size();
    dim3 grid_dim = get_grid_dim(batch_size, block_size);
    
    unsigned int seed = static_cast<unsigned int>(time(nullptr));
    
    reset_environments_kernel<<<grid_dim, block_size>>>(
        static_cast<AgentState*>(d_agent_states),
        batch_size,
        config,
        seed
    );
    
    cudaDeviceSynchronize();
#endif
}

void cuda_step_environments(
    void* d_agent_states,
    const int* actions,
    int batch_size,
    const EnvironmentConfig& config
) {
#ifdef __CUDACC__
    int block_size = get_optimal_block_size();
    dim3 grid_dim = get_grid_dim(batch_size, block_size);
    
    step_environments_kernel<<<grid_dim, block_size>>>(
        static_cast<AgentState*>(d_agent_states),
        actions,
        batch_size,
        config
    );
    
    cudaDeviceSynchronize();
#endif
}

void cuda_update_q_values(
    void* d_q_tables,
    const int* states,
    const int* actions,
    const float* rewards,
    const int* next_states,
    int batch_size,
    const EnvironmentConfig& config
) {
#ifdef __CUDACC__
    int block_size = get_optimal_block_size();
    dim3 grid_dim = get_grid_dim(batch_size, block_size);
    
    update_q_values_kernel<<<grid_dim, block_size>>>(
        static_cast<float*>(d_q_tables),
        states,
        actions,
        rewards,
        next_states,
        batch_size,
        config
    );
    
    cudaDeviceSynchronize();
#endif
}

void cuda_select_actions(
    const void* d_q_tables,
    const int* states,
    int* actions,
    int batch_size,
    const EnvironmentConfig& config
) {
#ifdef __CUDACC__
    int block_size = get_optimal_block_size();
    dim3 grid_dim = get_grid_dim(batch_size, block_size);
    
    unsigned int seed = static_cast<unsigned int>(time(nullptr));
    
    select_actions_kernel<<<grid_dim, block_size>>>(
        static_cast<const float*>(d_q_tables),
        states,
        actions,
        batch_size,
        config,
        seed
    );
    
    cudaDeviceSynchronize();
#endif
}

// Utility functions

int get_optimal_block_size() {
    return 256;  // Good default for most GPUs
}

dim3 get_grid_dim(int total_threads, int block_size) {
    int grid_size = (total_threads + block_size - 1) / block_size;
    return dim3(grid_size);
}

} // namespace cudarl
