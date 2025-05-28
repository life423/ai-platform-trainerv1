#include "../core/environment.h"
#include "../core/cuda_utils.h"
#include <iostream>
#include <chrono>
#include <vector>

using namespace cudarl;

void print_cuda_info() {
    std::cout << "=== CUDA Information ===" << std::endl;
    
    bool cuda_available = is_cuda_available();
    std::cout << "CUDA Available: " << (cuda_available ? "Yes" : "No") << std::endl;
    
    if (cuda_available) {
        CudaDeviceInfo info = get_cuda_device_info();
        std::cout << "Device Count: " << info.device_count << std::endl;
        std::cout << "Current Device: " << info.current_device << std::endl;
        std::cout << "Device Name: " << info.device_name << std::endl;
        std::cout << "Total Memory: " << info.total_memory / (1024*1024) << " MB" << std::endl;
        std::cout << "Free Memory: " << info.free_memory / (1024*1024) << " MB" << std::endl;
        std::cout << "Multiprocessors: " << info.multiprocessor_count << std::endl;
    }
    
    std::cout << std::endl;
}

void test_basic_environment() {
    std::cout << "=== Basic Environment Test ===" << std::endl;
    
    EnvironmentConfig config;
    config.grid_width = 8;
    config.grid_height = 8;
    config.max_episode_steps = 100;
    
    Environment env(config);
    
    int batch_size = 4;
    env.initialize(batch_size);
    
    std::cout << "Environment initialized with batch size: " << batch_size << std::endl;
    std::cout << "CUDA enabled: " << (env.is_cuda_enabled() ? "Yes" : "No") << std::endl;
    
    // Test reset
    auto observations = env.reset();
    std::cout << "Reset successful, got " << observations.size() << " observations" << std::endl;
    
    // Print initial states
    for (size_t i = 0; i < observations.size(); ++i) {
        const auto& obs = observations[i];
        std::cout << "  Env " << i << ": features=[";
        for (size_t j = 0; j < obs.features.size(); ++j) {
            std::cout << obs.features[j];
            if (j < obs.features.size() - 1) std::cout << ", ";
        }
        std::cout << "], reward=" << obs.reward << ", done=" << obs.done << std::endl;
    }
    
    // Test step
    std::vector<int> actions = {0, 1, 2, 3};  // up, down, left, right
    observations = env.step(actions);
    
    std::cout << "Step successful" << std::endl;
    for (size_t i = 0; i < observations.size(); ++i) {
        const auto& obs = observations[i];
        std::cout << "  Env " << i << ": reward=" << obs.reward 
                  << ", done=" << obs.done << ", steps=" << obs.info_steps << std::endl;
    }
    
    std::cout << std::endl;
}

void test_q_learning() {
    std::cout << "=== Q-Learning Test ===" << std::endl;
    
    EnvironmentConfig config;
    config.grid_width = 4;
    config.grid_height = 4;
    config.learning_rate = 0.1f;
    config.epsilon = 0.1f;
    
    Environment env(config);
    env.initialize(1);  // Single environment
    
    // Reset environment
    env.reset();
    
    // Test Q-learning update
    int state = 0;   // State (0, 0)
    int action = 1;  // Action: down
    float reward = -0.01f;
    int next_state = 4;  // Next state (0, 1)
    
    env.update_q_table(0, state, action, reward, next_state);
    std::cout << "Q-table update successful" << std::endl;
    
    // Test action selection
    int selected_action = env.select_action_epsilon_greedy(0, state);
    std::cout << "Selected action: " << selected_action << std::endl;
    
    std::cout << std::endl;
}

void benchmark_performance() {
    std::cout << "=== Performance Benchmark ===" << std::endl;
    
    EnvironmentConfig config;
    config.grid_width = 32;
    config.grid_height = 32;
    
    std::vector<int> batch_sizes = {1, 16, 64, 256};
    int num_steps = 1000;
    
    for (int batch_size : batch_sizes) {
        Environment env(config);
        env.initialize(batch_size);
        
        // Warm up
        env.reset();
        std::vector<int> actions(batch_size, 0);
        for (int i = 0; i < 10; ++i) {
            env.step(actions);
        }
        
        // Benchmark
        auto start = std::chrono::high_resolution_clock::now();
        
        env.reset();
        for (int step = 0; step < num_steps; ++step) {
            env.step(actions);
        }
        
        auto end = std::chrono::high_resolution_clock::now();
        auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end - start);
        
        double steps_per_second = (static_cast<double>(batch_size) * num_steps * 1000000.0) / duration.count();
        
        std::cout << "Batch size " << batch_size << ": " 
                  << static_cast<int>(steps_per_second) << " steps/sec" << std::endl;
    }
    
    std::cout << std::endl;
}

int main() {
    std::cout << "CudaRL-Arena Test Suite" << std::endl;
    std::cout << "======================" << std::endl << std::endl;
    
    try {
        print_cuda_info();
        test_basic_environment();
        test_q_learning();
        benchmark_performance();
        
        std::cout << "All tests completed successfully!" << std::endl;
        
    } catch (const std::exception& e) {
        std::cerr << "Error: " << e.what() << std::endl;
        return 1;
    }
    
    return 0;
}
