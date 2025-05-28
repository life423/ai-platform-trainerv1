// simple_physics.cu - Simple CUDA physics kernel for RL training
#include <cuda_runtime.h>
#include <curand_kernel.h>

__global__ void updateEnemyPhysics(
    float* enemy_pos,      // [batch_size, 2] - enemy positions (x, y)
    float* player_pos,     // [batch_size, 2] - player positions (x, y) 
    float* actions,        // [batch_size, 2] - actions from RL agent (-1 to 1)
    float* rewards,        // [batch_size] - output rewards
    bool* dones,           // [batch_size] - episode done flags
    int batch_size,
    float dt = 0.016f      // 60 FPS timestep
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= batch_size) return;
    
    int pos_idx = idx * 2;
    
    // Update enemy position based on actions
    float action_x = actions[pos_idx];
    float action_y = actions[pos_idx + 1];
    
    // Clamp actions to [-1, 1] range
    action_x = fmaxf(-1.0f, fminf(1.0f, action_x));
    action_y = fmaxf(-1.0f, fminf(1.0f, action_y));
    
    // Apply actions as velocity (scaled by speed and timestep)
    float speed = 200.0f;  // pixels per second
    enemy_pos[pos_idx] += action_x * speed * dt;
    enemy_pos[pos_idx + 1] += action_y * speed * dt;
    
    // Keep enemy in bounds (800x600 game window with margin)
    enemy_pos[pos_idx] = fmaxf(25.0f, fminf(775.0f, enemy_pos[pos_idx]));
    enemy_pos[pos_idx + 1] = fmaxf(25.0f, fminf(575.0f, enemy_pos[pos_idx + 1]));
    
    // Calculate distance to player
    float dx = player_pos[pos_idx] - enemy_pos[pos_idx];
    float dy = player_pos[pos_idx + 1] - enemy_pos[pos_idx + 1];
    float distance = sqrtf(dx * dx + dy * dy);
    
    // Calculate reward
    // Reward structure:
    // - Negative distance (encourage getting closer)
    // - Bonus for being very close 
    // - Penalty for moving too fast (encourage smooth movement)
    float distance_reward = -distance / 100.0f;
    float proximity_bonus = (distance < 50.0f) ? 10.0f : 0.0f;
    float smoothness_penalty = -(fabsf(action_x) + fabsf(action_y)) * 0.1f;
    
    rewards[idx] = distance_reward + proximity_bonus + smoothness_penalty;
    
    // Episode done if enemy catches player (within 30 pixels)
    dones[idx] = distance < 30.0f;
}

__global__ void initializePositions(
    float* enemy_pos,
    float* player_pos,
    curandState* states,
    int batch_size
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= batch_size) return;
    
    int pos_idx = idx * 2;
    
    // Initialize random number generator state
    curand_init(1234 + idx, idx, 0, &states[idx]);
    
    // Random positions for enemy (avoiding edges)
    enemy_pos[pos_idx] = curand_uniform(&states[idx]) * 700.0f + 50.0f;      // x: 50-750
    enemy_pos[pos_idx + 1] = curand_uniform(&states[idx]) * 500.0f + 50.0f;  // y: 50-550
    
    // Random positions for player (avoiding edges)  
    player_pos[pos_idx] = curand_uniform(&states[idx]) * 700.0f + 50.0f;     // x: 50-750
    player_pos[pos_idx + 1] = curand_uniform(&states[idx]) * 500.0f + 50.0f; // y: 50-550
}

__global__ void updatePlayerPositions(
    float* player_pos,
    curandState* states,
    int batch_size,
    float dt = 0.016f
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= batch_size) return;
    
    int pos_idx = idx * 2;
    
    // Simple random walk for player (makes training more interesting)
    float player_speed = 150.0f;  // pixels per second
    float random_x = (curand_uniform(&states[idx]) - 0.5f) * 2.0f;  // -1 to 1
    float random_y = (curand_uniform(&states[idx]) - 0.5f) * 2.0f;  // -1 to 1
    
    player_pos[pos_idx] += random_x * player_speed * dt;
    player_pos[pos_idx + 1] += random_y * player_speed * dt;
    
    // Keep player in bounds
    player_pos[pos_idx] = fmaxf(25.0f, fminf(775.0f, player_pos[pos_idx]));
    player_pos[pos_idx + 1] = fmaxf(25.0f, fminf(575.0f, player_pos[pos_idx + 1]));
}

// C interface for Python binding
extern "C" {
    void cuda_update_physics(
        float* enemy_pos, 
        float* player_pos, 
        float* actions,
        float* rewards,
        bool* dones,
        int batch_size
    ) {
        int threads = 256;
        int blocks = (batch_size + threads - 1) / threads;
        
        updateEnemyPhysics<<<blocks, threads>>>(
            enemy_pos, player_pos, actions, rewards, dones, batch_size
        );
        
        cudaDeviceSynchronize();
    }
    
    void cuda_initialize_positions(
        float* enemy_pos,
        float* player_pos, 
        void* states,
        int batch_size
    ) {
        int threads = 256;
        int blocks = (batch_size + threads - 1) / threads;
        
        initializePositions<<<blocks, threads>>>(
            enemy_pos, player_pos, static_cast<curandState*>(states), batch_size
        );
        
        cudaDeviceSynchronize();
    }
    
    void cuda_update_player_positions(
        float* player_pos,
        void* states,
        int batch_size
    ) {
        int threads = 256;
        int blocks = (batch_size + threads - 1) / threads;
        
        updatePlayerPositions<<<blocks, threads>>>(
            player_pos, static_cast<curandState*>(states), batch_size
        );
        
        cudaDeviceSynchronize();
    }
    
    void* cuda_allocate_random_states(int batch_size) {
        curandState* states;
        cudaMalloc(&states, batch_size * sizeof(curandState));
        return states;
    }
    
    void cuda_free_random_states(void* states) {
        cudaFree(states);
    }
}
