#include "environment.h"
#include "cuda_utils.h"
#include <algorithm>
#include <random>
#include <cmath>
#include <iostream>

namespace cudarl {

Environment::Environment(const EnvironmentConfig& config)
    : config_(config), batch_size_(0), cuda_enabled_(false),
      d_agent_states_(nullptr), d_q_tables_(nullptr) {
    
    // Check if CUDA is available
    cuda_enabled_ = is_cuda_available();
    if (cuda_enabled_) {
        std::cout << "CUDA is available - will use GPU acceleration" << std::endl;
    } else {
        std::cout << "CUDA not available - using CPU fallback" << std::endl;
    }
}

Environment::~Environment() {
    cleanup_cuda_memory();
}

void Environment::initialize(int batch_size) {
    batch_size_ = batch_size;
    
    // Initialize CPU storage
    agent_states_.resize(batch_size_);
    q_tables_.resize(batch_size_);
    
    // Initialize Q-tables for each environment
    int num_states = config_.grid_width * config_.grid_height;
    for (int i = 0; i < batch_size_; ++i) {
        q_tables_[i].resize(num_states * config_.num_actions, 0.0f);
    }
    
    // Setup CUDA memory if available
    if (cuda_enabled_) {
        setup_cuda_memory();
    }
}

std::vector<Observation> Environment::reset() {
    if (cuda_enabled_) {
        // GPU implementation (Phase 2)
        // cuda_reset_environments(d_agent_states_, batch_size_, config_);
        cpu_reset();  // Fallback for now
    } else {
        cpu_reset();
    }
    
    return get_observations();
}

std::vector<Observation> Environment::step(const std::vector<int>& actions) {
    if (actions.size() != static_cast<size_t>(batch_size_)) {
        throw std::invalid_argument("Actions size must match batch size");
    }
    
    if (cuda_enabled_) {
        // GPU implementation (Phase 2)
        // cuda_step_environments(d_agent_states_, actions.data(), batch_size_, config_);
        cpu_step(actions);  // Fallback for now
    } else {
        cpu_step(actions);
    }
    
    return get_observations();
}

std::vector<Observation> Environment::get_observations() const {
    std::vector<Observation> observations;
    observations.reserve(batch_size_);
    
    for (int i = 0; i < batch_size_; ++i) {
        observations.push_back(create_observation(agent_states_[i]));
    }
    
    return observations;
}

void Environment::update_q_table(int env_id, int state, int action, float reward, int next_state) {
    if (env_id >= batch_size_ || state < 0 || action < 0) {
        return;  // Invalid parameters
    }
    
    if (cuda_enabled_) {
        // GPU implementation (Phase 2)
        // cuda_update_q_value(d_q_tables_, env_id, state, action, reward, next_state, config_);
        cpu_update_q_table(env_id, state, action, reward, next_state);
    } else {
        cpu_update_q_table(env_id, state, action, reward, next_state);
    }
}

int Environment::select_action_epsilon_greedy(int env_id, int state) {
    if (env_id >= batch_size_ || state < 0) {
        return 0;  // Invalid parameters, return safe action
    }
    
    if (cuda_enabled_) {
        // GPU implementation (Phase 2)
        // return cuda_select_action(d_q_tables_, env_id, state, config_);
        return cpu_select_action_epsilon_greedy(env_id, state);
    } else {
        return cpu_select_action_epsilon_greedy(env_id, state);
    }
}

// Private implementation methods

void Environment::setup_cuda_memory() {
    if (!cuda_enabled_) return;
    
    // Allocate GPU memory for agent states and Q-tables
    // This will be implemented in Phase 2 with actual CUDA kernels
    // For now, we keep everything on CPU
}

void Environment::cleanup_cuda_memory() {
    if (d_agent_states_) {
        // cudaFree(d_agent_states_); // Phase 2
        d_agent_states_ = nullptr;
    }
    if (d_q_tables_) {
        // cudaFree(d_q_tables_); // Phase 2
        d_q_tables_ = nullptr;
    }
}

Observation Environment::create_observation(const AgentState& state) const {
    Observation obs;
    
    // Create normalized features
    obs.features.resize(4);
    obs.features[0] = static_cast<float>(state.x) / config_.grid_width;
    obs.features[1] = static_cast<float>(state.y) / config_.grid_height;
    obs.features[2] = static_cast<float>(state.goal_x) / config_.grid_width;
    obs.features[3] = static_cast<float>(state.goal_y) / config_.grid_height;
    
    obs.reward = state.cumulative_reward;
    obs.done = state.done;
    obs.info_steps = state.episode_step;
    
    return obs;
}

int Environment::state_to_index(int x, int y) const {
    return y * config_.grid_width + x;
}

void Environment::index_to_state(int index, int& x, int& y) const {
    x = index % config_.grid_width;
    y = index / config_.grid_width;
}

// CPU fallback implementations

void Environment::cpu_reset() {
    static std::random_device rd;
    static std::mt19937 gen(rd());
    std::uniform_int_distribution<> x_dist(0, config_.grid_width - 1);
    std::uniform_int_distribution<> y_dist(0, config_.grid_height - 1);
    
    for (int i = 0; i < batch_size_; ++i) {
        auto& state = agent_states_[i];
        
        // Random start position
        state.x = x_dist(gen);
        state.y = y_dist(gen);
        
        // Random goal position (different from start)
        do {
            state.goal_x = x_dist(gen);
            state.goal_y = y_dist(gen);
        } while (state.goal_x == state.x && state.goal_y == state.y);
        
        state.episode_step = 0;
        state.cumulative_reward = 0.0f;
        state.done = false;
    }
}

void Environment::cpu_step(const std::vector<int>& actions) {
    for (int i = 0; i < batch_size_; ++i) {
        auto& state = agent_states_[i];
        
        if (state.done) continue;  // Skip finished episodes
        
        // Apply action: 0=up, 1=down, 2=left, 3=right
        int new_x = state.x;
        int new_y = state.y;
        
        switch (actions[i]) {
            case 0: new_y = std::max(0, new_y - 1); break;           // up
            case 1: new_y = std::min(config_.grid_height - 1, new_y + 1); break;  // down
            case 2: new_x = std::max(0, new_x - 1); break;           // left
            case 3: new_x = std::min(config_.grid_width - 1, new_x + 1); break;   // right
        }
        
        state.x = new_x;
        state.y = new_y;
        state.episode_step++;
        
        // Calculate reward
        float distance_to_goal = std::sqrt(
            std::pow(state.x - state.goal_x, 2) + 
            std::pow(state.y - state.goal_y, 2)
        );
        
        float reward = -0.01f;  // Small step penalty
        
        // Check if reached goal
        if (state.x == state.goal_x && state.y == state.goal_y) {
            reward = 10.0f;  // Large reward for reaching goal
            state.done = true;
        }
        // Time limit
        else if (state.episode_step >= config_.max_episode_steps) {
            reward = -1.0f;  // Penalty for timeout
            state.done = true;
        }
        // Distance-based shaping
        else {
            reward -= distance_to_goal * 0.01f;  // Encourage getting closer
        }
        
        state.cumulative_reward += reward * config_.reward_scale;
    }
}

void Environment::cpu_update_q_table(int env_id, int state, int action, float reward, int next_state) {
    auto& q_table = q_tables_[env_id];
    int num_states = config_.grid_width * config_.grid_height;
    
    if (state >= num_states || next_state >= num_states || action >= config_.num_actions) {
        return;  // Invalid indices
    }
    
    // Q-learning update: Q(s,a) = Q(s,a) + α[r + γ max_a' Q(s',a') - Q(s,a)]
    float current_q = q_table[state * config_.num_actions + action];
    
    // Find max Q-value for next state
    float max_next_q = q_table[next_state * config_.num_actions];
    for (int a = 1; a < config_.num_actions; ++a) {
        max_next_q = std::max(max_next_q, q_table[next_state * config_.num_actions + a]);
    }
    
    float target = reward + config_.discount_factor * max_next_q;
    float new_q = current_q + config_.learning_rate * (target - current_q);
    
    q_table[state * config_.num_actions + action] = new_q;
}

int Environment::cpu_select_action_epsilon_greedy(int env_id, int state) {
    static std::random_device rd;
    static std::mt19937 gen(rd());
    std::uniform_real_distribution<> epsilon_dist(0.0, 1.0);
    std::uniform_int_distribution<> action_dist(0, config_.num_actions - 1);
    
    auto& q_table = q_tables_[env_id];
    int num_states = config_.grid_width * config_.grid_height;
    
    if (state >= num_states) {
        return action_dist(gen);  // Random action for invalid state
    }
    
    // Epsilon-greedy action selection
    if (epsilon_dist(gen) < config_.epsilon) {
        return action_dist(gen);  // Random action
    } else {
        // Greedy action (highest Q-value)
        int best_action = 0;
        float best_q = q_table[state * config_.num_actions];
        
        for (int a = 1; a < config_.num_actions; ++a) {
            float q_val = q_table[state * config_.num_actions + a];
            if (q_val > best_q) {
                best_q = q_val;
                best_action = a;
            }
        }
        
        return best_action;
    }
}

} // namespace cudarl
