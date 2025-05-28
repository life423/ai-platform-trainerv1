#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <pybind11/numpy.h>
#include <chrono>
#include <string>
#include "../../core/environment.h"
#include "../../core/cuda_utils.h"

namespace py = pybind11;
using namespace cudarl;

/**
 * Python bindings for CudaRL-Arena
 * 
 * This creates the direct Python interface to the C++/CUDA environment
 * without any bridge or compatibility layers.
 */

PYBIND11_MODULE(cudarl_core_python, m) {
    m.doc() = "CudaRL-Arena: CUDA-accelerated reinforcement learning environment";
    
    // Environment Configuration
    py::class_<EnvironmentConfig>(m, "EnvironmentConfig")
        .def(py::init<>())
        .def_readwrite("grid_width", &EnvironmentConfig::grid_width)
        .def_readwrite("grid_height", &EnvironmentConfig::grid_height)
        .def_readwrite("max_episode_steps", &EnvironmentConfig::max_episode_steps)
        .def_readwrite("reward_scale", &EnvironmentConfig::reward_scale)
        .def_readwrite("num_actions", &EnvironmentConfig::num_actions)
        .def_readwrite("learning_rate", &EnvironmentConfig::learning_rate)
        .def_readwrite("discount_factor", &EnvironmentConfig::discount_factor)
        .def_readwrite("epsilon", &EnvironmentConfig::epsilon)
        .def("__repr__", [](const EnvironmentConfig& config) {
            return "<EnvironmentConfig: " + 
                   std::to_string(config.grid_width) + "x" + 
                   std::to_string(config.grid_height) + " grid>";
        });
    
    // Agent State
    py::class_<AgentState>(m, "AgentState")
        .def(py::init<>())
        .def_readwrite("x", &AgentState::x)
        .def_readwrite("y", &AgentState::y)
        .def_readwrite("goal_x", &AgentState::goal_x)
        .def_readwrite("goal_y", &AgentState::goal_y)
        .def_readwrite("episode_step", &AgentState::episode_step)
        .def_readwrite("cumulative_reward", &AgentState::cumulative_reward)
        .def_readwrite("done", &AgentState::done)
        .def("__repr__", [](const AgentState& state) {
            return "<AgentState: (" + std::to_string(state.x) + "," + 
                   std::to_string(state.y) + ") -> (" + 
                   std::to_string(state.goal_x) + "," + 
                   std::to_string(state.goal_y) + ")>";
        });
    
    // Observation
    py::class_<Observation>(m, "Observation")
        .def(py::init<>())
        .def_readwrite("features", &Observation::features)
        .def_readwrite("reward", &Observation::reward)
        .def_readwrite("done", &Observation::done)
        .def_readwrite("info_steps", &Observation::info_steps)
        .def("__repr__", [](const Observation& obs) {
            return "<Observation: reward=" + std::to_string(obs.reward) + 
                   ", done=" + (obs.done ? "True" : "False") + ">";
        });
    
    // CUDA Device Information
    py::class_<CudaDeviceInfo>(m, "CudaDeviceInfo")
        .def(py::init<>())
        .def_readwrite("device_count", &CudaDeviceInfo::device_count)
        .def_readwrite("current_device", &CudaDeviceInfo::current_device)
        .def_readwrite("total_memory", &CudaDeviceInfo::total_memory)
        .def_readwrite("free_memory", &CudaDeviceInfo::free_memory)
        .def_readwrite("multiprocessor_count", &CudaDeviceInfo::multiprocessor_count)
        .def_property("device_name", 
            [](const CudaDeviceInfo& info) { return std::string(info.device_name); },
            [](CudaDeviceInfo& info, const std::string& name) {
                strncpy(info.device_name, name.c_str(), sizeof(info.device_name) - 1);
                info.device_name[sizeof(info.device_name) - 1] = '\0';
            })
        .def("__repr__", [](const CudaDeviceInfo& info) {
            return "<CudaDeviceInfo: " + std::string(info.device_name) + 
                   ", " + std::to_string(info.total_memory / (1024*1024)) + " MB>";
        });
    
    // Main Environment Class
    py::class_<Environment>(m, "Environment")
        .def(py::init<const EnvironmentConfig&>(), py::arg("config") = EnvironmentConfig{})
        .def("initialize", &Environment::initialize, 
             py::arg("batch_size"),
             "Initialize the environment with specified batch size")
        .def("reset", &Environment::reset,
             "Reset all environments to initial state")
        .def("step", &Environment::step,
             py::arg("actions"),
             "Step all environments with given actions")
        .def("get_observations", &Environment::get_observations,
             "Get current state of all environments")
        .def("update_q_table", &Environment::update_q_table,
             py::arg("env_id"), py::arg("state"), py::arg("action"), 
             py::arg("reward"), py::arg("next_state"),
             "Update Q-table for Q-learning")
        .def("select_action_epsilon_greedy", &Environment::select_action_epsilon_greedy,
             py::arg("env_id"), py::arg("state"),
             "Select action using epsilon-greedy policy")
        .def("get_config", &Environment::get_config, 
             py::return_value_policy::reference_internal,
             "Get environment configuration")
        .def("get_batch_size", &Environment::get_batch_size,
             "Get current batch size")
        .def("is_cuda_enabled", &Environment::is_cuda_enabled,
             "Check if CUDA acceleration is enabled")
        .def("__repr__", [](const Environment& env) {
            return "<Environment: batch_size=" + std::to_string(env.get_batch_size()) + 
                   ", cuda=" + (env.is_cuda_enabled() ? "enabled" : "disabled") + ">";
        });
    
    // Utility Functions
    m.def("is_cuda_available", &is_cuda_available,
          "Check if CUDA is available and functioning");
    
    m.def("get_cuda_device_info", &get_cuda_device_info,
          "Get information about available CUDA devices");
    
    // Convenience functions for Python integration
    m.def("create_environment", 
          [](int grid_width, int grid_height, int batch_size) {
              EnvironmentConfig config;
              config.grid_width = grid_width;
              config.grid_height = grid_height;
              
              auto env = std::make_unique<Environment>(config);
              env->initialize(batch_size);
              return env;
          },
          py::arg("grid_width") = 32, 
          py::arg("grid_height") = 32, 
          py::arg("batch_size") = 1,
          "Create and initialize an environment with default settings");
    
    m.def("benchmark_environment",
          [](int batch_size, int num_steps) {
              EnvironmentConfig config;
              Environment env(config);
              env.initialize(batch_size);
              
              auto start = std::chrono::high_resolution_clock::now();
              
              // Run benchmark
              env.reset();
              std::vector<int> actions(batch_size, 0);  // All agents move up
              
              for (int step = 0; step < num_steps; ++step) {
                  env.step(actions);
              }
              
              auto end = std::chrono::high_resolution_clock::now();
              auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start);
              
              double steps_per_second = (static_cast<double>(batch_size) * num_steps * 1000.0) / duration.count();
                py::dict result;
              result["steps_per_second"] = steps_per_second;
              result["total_steps"] = batch_size * num_steps;
              result["duration_ms"] = duration.count();
              result["cuda_enabled"] = env.is_cuda_enabled();
              return result;
          },
          py::arg("batch_size") = 32,
          py::arg("num_steps") = 1000,
          "Benchmark environment performance");
    
    // Module constants
    m.attr("VERSION") = "1.0.0";
    m.attr("CUDA_AVAILABLE") = is_cuda_available();
    
    // Action constants for convenience
    m.attr("ACTION_UP") = 0;
    m.attr("ACTION_DOWN") = 1;
    m.attr("ACTION_LEFT") = 2;
    m.attr("ACTION_RIGHT") = 3;
}
