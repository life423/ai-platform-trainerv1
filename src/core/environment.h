#pragma once

#include <vector>
#include <array>
#include <memory>
#include <cstdint>

namespace cudarl {

/**
 * Core environment configuration
 */
struct EnvironmentConfig {
    int grid_width = 32;
    int grid_height = 32;
    int max_episode_steps = 1000;
    float reward_scale = 1.0f;
    
    // Q-learning specific
    int num_actions = 4;  // up, down, left, right
    float learning_rate = 0.1f;
    float discount_factor = 0.95f;
    float epsilon = 0.1f;
};

/**
 * Environment state for a single agent
 */
struct AgentState {
    int x, y;                    // Agent position
    int goal_x, goal_y;         // Goal position
    int episode_step;           // Current episode step
    float cumulative_reward;    // Total reward this episode
    bool done;                  // Episode finished flag
};

/**
 * Observation structure returned to Python
 */
struct Observation {
    std::vector<float> features;  // Flattened state features
    float reward;
    bool done;
    int info_steps;              // Additional info: steps taken
};

/**
 * Host-side (CPU) Environment class
 * 
 * This handles the core logic and interfaces with GPU kernels for acceleration
 */
class Environment {
public:
    explicit Environment(const EnvironmentConfig& config = EnvironmentConfig{});
    ~Environment();

    /**
     * Initialize the environment with specified batch size
     * @param batch_size Number of parallel environments
     */
    void initialize(int batch_size);

    /**
     * Reset all environments to initial state
     * @return Batch of initial observations
     */
    std::vector<Observation> reset();

    /**
     * Step all environments with given actions
     * @param actions Batch of actions (one per environment)
     * @return Batch of observations after stepping
     */
    std::vector<Observation> step(const std::vector<int>& actions);

    /**
     * Get current state of all environments
     * @return Batch of current observations
     */
    std::vector<Observation> get_observations() const;

    /**
     * Access to Q-learning functionality (Phase 2)
     */
    void update_q_table(int env_id, int state, int action, float reward, int next_state);
    int select_action_epsilon_greedy(int env_id, int state);
    
    // Configuration accessors
    const EnvironmentConfig& get_config() const { return config_; }
    int get_batch_size() const { return batch_size_; }
    bool is_cuda_enabled() const { return cuda_enabled_; }

private:
    EnvironmentConfig config_;
    int batch_size_;
    bool cuda_enabled_;
    
    // CPU fallback storage
    std::vector<AgentState> agent_states_;
    std::vector<std::vector<float>> q_tables_;  // Per-environment Q-tables
    
    // GPU storage (managed by CUDA functions)
    void* d_agent_states_;   // Device pointer to AgentState array
    void* d_q_tables_;       // Device pointer to Q-table data
    
    // Internal methods
    void setup_cuda_memory();
    void cleanup_cuda_memory();
    Observation create_observation(const AgentState& state) const;
    int state_to_index(int x, int y) const;
    void index_to_state(int index, int& x, int& y) const;
    
    // CPU fallback implementations
    void cpu_reset();
    void cpu_step(const std::vector<int>& actions);
    void cpu_update_q_table(int env_id, int state, int action, float reward, int next_state);
    int cpu_select_action_epsilon_greedy(int env_id, int state);
};

} // namespace cudarl
