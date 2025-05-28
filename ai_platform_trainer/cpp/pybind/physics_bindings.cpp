// physics_bindings.cpp - PyBind11 wrapper for CUDA physics
#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <pybind11/stl.h>
#include <cuda_runtime.h>
#include <memory>

namespace py = pybind11;

// External C functions from simple_physics.cu
extern "C" {
    void cuda_update_physics(float* enemy_pos, float* player_pos, float* actions,
                            float* rewards, bool* dones, int batch_size);
    void cuda_initialize_positions(float* enemy_pos, float* player_pos, 
                                 void* states, int batch_size);
    void cuda_update_player_positions(float* player_pos, void* states, int batch_size);
    void* cuda_allocate_random_states(int batch_size);
    void cuda_free_random_states(void* states);
}

class CUDAPhysicsEngine {
public:
    CUDAPhysicsEngine(int batch_size) 
        : batch_size_(batch_size), random_states_(nullptr) {
        
        // Allocate GPU memory for positions, rewards, and dones
        size_t pos_size = batch_size * 2 * sizeof(float);  // 2D positions
        size_t reward_size = batch_size * sizeof(float);
        size_t done_size = batch_size * sizeof(bool);
        
        cudaMalloc(&d_enemy_pos_, pos_size);
        cudaMalloc(&d_player_pos_, pos_size);
        cudaMalloc(&d_actions_, pos_size);
        cudaMalloc(&d_rewards_, reward_size);
        cudaMalloc(&d_dones_, done_size);
        
        // Allocate random states for CUDA random number generation
        random_states_ = cuda_allocate_random_states(batch_size);
        
        // Initialize positions
        cuda_initialize_positions(d_enemy_pos_, d_player_pos_, random_states_, batch_size);
        
        // Initialize host arrays
        h_enemy_pos_.resize(batch_size * 2);
        h_player_pos_.resize(batch_size * 2);
        h_rewards_.resize(batch_size);
        h_dones_.resize(batch_size);
        
        // Copy initial positions to host
        cudaMemcpy(h_enemy_pos_.data(), d_enemy_pos_, pos_size, cudaMemcpyDeviceToHost);
        cudaMemcpy(h_player_pos_.data(), d_player_pos_, pos_size, cudaMemcpyDeviceToHost);
    }
    
    ~CUDAPhysicsEngine() {
        // Clean up GPU memory
        if (d_enemy_pos_) cudaFree(d_enemy_pos_);
        if (d_player_pos_) cudaFree(d_player_pos_);
        if (d_actions_) cudaFree(d_actions_);
        if (d_rewards_) cudaFree(d_rewards_);
        if (d_dones_) cudaFree(d_dones_);
        if (random_states_) cuda_free_random_states(random_states_);
    }
    
    // Reset environment - initialize new random positions
    py::array_t<float> reset() {
        cuda_initialize_positions(d_enemy_pos_, d_player_pos_, random_states_, batch_size_);
        
        // Copy positions to host
        size_t pos_size = batch_size_ * 2 * sizeof(float);
        cudaMemcpy(h_enemy_pos_.data(), d_enemy_pos_, pos_size, cudaMemcpyDeviceToHost);
        cudaMemcpy(h_player_pos_.data(), d_player_pos_, pos_size, cudaMemcpyDeviceToHost);
        
        return get_observations();
    }
    
    // Step environment with actions
    py::tuple step(py::array_t<float> actions) {
        // Copy actions to GPU
        auto buf = actions.request();
        if (buf.size != batch_size_ * 2) {
            throw std::runtime_error("Actions must have shape [batch_size, 2]");
        }
        
        float* action_ptr = static_cast<float*>(buf.ptr);
        size_t action_size = batch_size_ * 2 * sizeof(float);
        cudaMemcpy(d_actions_, action_ptr, action_size, cudaMemcpyHostToDevice);
        
        // Update player positions (random movement)
        cuda_update_player_positions(d_player_pos_, random_states_, batch_size_);
        
        // Run physics update
        cuda_update_physics(d_enemy_pos_, d_player_pos_, d_actions_, 
                           d_rewards_, d_dones_, batch_size_);
        
        // Copy results back to host
        size_t pos_size = batch_size_ * 2 * sizeof(float);
        size_t reward_size = batch_size_ * sizeof(float);
        size_t done_size = batch_size_ * sizeof(bool);
        
        cudaMemcpy(h_enemy_pos_.data(), d_enemy_pos_, pos_size, cudaMemcpyDeviceToHost);
        cudaMemcpy(h_player_pos_.data(), d_player_pos_, pos_size, cudaMemcpyDeviceToHost);
        cudaMemcpy(h_rewards_.data(), d_rewards_, reward_size, cudaMemcpyDeviceToHost);
        cudaMemcpy(h_dones_.data(), d_dones_, done_size, cudaMemcpyDeviceToHost);
        
        // Return observations, rewards, dones
        auto observations = get_observations();
        auto rewards = get_rewards();
        auto dones = get_dones();
        
        return py::make_tuple(observations, rewards, dones);
    }
    
    // Get current observations (relative positions + velocities)
    py::array_t<float> get_observations() {
        auto result = py::array_t<float>(batch_size_ * 4);  // [batch_size, 4]
        auto buf = result.request();
        float* ptr = static_cast<float*>(buf.ptr);
        
        for (int i = 0; i < batch_size_; i++) {
            int pos_idx = i * 2;
            int obs_idx = i * 4;
            
            // Relative position (player - enemy)
            ptr[obs_idx] = h_player_pos_[pos_idx] - h_enemy_pos_[pos_idx];         // rel_x
            ptr[obs_idx + 1] = h_player_pos_[pos_idx + 1] - h_enemy_pos_[pos_idx + 1]; // rel_y
            
            // Normalized positions (for better neural network training)
            ptr[obs_idx + 2] = h_enemy_pos_[pos_idx] / 800.0f;                    // norm_enemy_x
            ptr[obs_idx + 3] = h_enemy_pos_[pos_idx + 1] / 600.0f;                // norm_enemy_y
        }
        
        // Reshape to [batch_size, 4]
        result.resize({batch_size_, 4});
        return result;
    }
    
    // Get rewards array
    py::array_t<float> get_rewards() {
        auto result = py::array_t<float>(batch_size_);
        auto buf = result.request();
        float* ptr = static_cast<float*>(buf.ptr);
        
        std::copy(h_rewards_.begin(), h_rewards_.end(), ptr);
        return result;
    }
    
    // Get dones array
    py::array_t<bool> get_dones() {
        auto result = py::array_t<bool>(batch_size_);
        auto buf = result.request();
        bool* ptr = static_cast<bool*>(buf.ptr);
        
        std::copy(h_dones_.begin(), h_dones_.end(), ptr);
        return result;
    }
    
    // Get current positions for debugging/visualization
    py::dict get_positions() {
        py::dict result;
        
        // Enemy positions
        auto enemy_pos = py::array_t<float>(batch_size_ * 2);
        auto enemy_buf = enemy_pos.request();
        float* enemy_ptr = static_cast<float*>(enemy_buf.ptr);
        std::copy(h_enemy_pos_.begin(), h_enemy_pos_.end(), enemy_ptr);
        enemy_pos.resize({batch_size_, 2});
        
        // Player positions
        auto player_pos = py::array_t<float>(batch_size_ * 2);
        auto player_buf = player_pos.request();
        float* player_ptr = static_cast<float*>(player_buf.ptr);
        std::copy(h_player_pos_.begin(), h_player_pos_.end(), player_ptr);
        player_pos.resize({batch_size_, 2});
        
        result["enemy_positions"] = enemy_pos;
        result["player_positions"] = player_pos;
        return result;
    }
    
    int get_batch_size() const { return batch_size_; }
    
private:
    int batch_size_;
    
    // GPU memory pointers
    float* d_enemy_pos_;
    float* d_player_pos_;
    float* d_actions_;
    float* d_rewards_;
    bool* d_dones_;
    void* random_states_;
    
    // Host arrays for transferring data
    std::vector<float> h_enemy_pos_;
    std::vector<float> h_player_pos_;
    std::vector<float> h_rewards_;
    std::vector<bool> h_dones_;
};

// Simple test function to verify CUDA is working
py::array_t<float> test_cuda_add(py::array_t<float> a, py::array_t<float> b) {
    auto buf_a = a.request();
    auto buf_b = b.request();
    
    if (buf_a.size != buf_b.size) {
        throw std::runtime_error("Arrays must have the same size");
    }
    
    int n = static_cast<int>(buf_a.size);
    auto result = py::array_t<float>(n);
    auto buf_result = result.request();
    
    float* ptr_a = static_cast<float*>(buf_a.ptr);
    float* ptr_b = static_cast<float*>(buf_b.ptr);
    float* ptr_result = static_cast<float*>(buf_result.ptr);
    
    // Simple CPU addition for testing
    for (int i = 0; i < n; i++) {
        ptr_result[i] = ptr_a[i] + ptr_b[i];
    }
    
    return result;
}

PYBIND11_MODULE(physics_cuda_module, m) {
    m.doc() = "CUDA-accelerated physics engine for RL training";
    
    // Main physics engine class
    py::class_<CUDAPhysicsEngine>(m, "CUDAPhysicsEngine")
        .def(py::init<int>(), py::arg("batch_size") = 256)
        .def("reset", &CUDAPhysicsEngine::reset, "Reset environment with new random positions")
        .def("step", &CUDAPhysicsEngine::step, py::arg("actions"), "Step environment with actions")
        .def("get_observations", &CUDAPhysicsEngine::get_observations, "Get current observations")
        .def("get_rewards", &CUDAPhysicsEngine::get_rewards, "Get current rewards")
        .def("get_dones", &CUDAPhysicsEngine::get_dones, "Get episode done flags")
        .def("get_positions", &CUDAPhysicsEngine::get_positions, "Get current positions for debugging")
        .def("get_batch_size", &CUDAPhysicsEngine::get_batch_size, "Get batch size");
    
    // Test function
    m.def("test_cuda_add", &test_cuda_add, py::arg("a"), py::arg("b"), 
          "Test function for CUDA binding verification");
    
    // Utility function to check CUDA availability
    m.def("cuda_device_count", []() {
        int count;
        cudaGetDeviceCount(&count);
        return count;
    }, "Get number of CUDA devices");
    
    m.def("cuda_device_name", [](int device_id = 0) {
        cudaDeviceProp prop;
        cudaGetDeviceProperties(&prop, device_id);
        return std::string(prop.name);
    }, py::arg("device_id") = 0, "Get CUDA device name");
}
