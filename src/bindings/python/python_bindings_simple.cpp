#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <chrono>
#include <string>
#include <cstring>
#include "../../core/environment.h"
#include "../../core/cuda_utils.h"

namespace py = pybind11;
using namespace cudarl;

PYBIND11_MODULE(cudarl_core_python, m) {
    m.doc() = "CudaRL-Arena: CUDA-accelerated reinforcement learning environment";
    
    // Environment Configuration
    py::class_<EnvironmentConfig>(m, "EnvironmentConfig")
        .def(py::init<>())
        .def_readwrite("grid_width", &EnvironmentConfig::grid_width)
        .def_readwrite("grid_height", &EnvironmentConfig::grid_height)
        .def_readwrite("max_episode_steps", &EnvironmentConfig::max_episode_steps);
    
    // Agent State
    py::class_<AgentState>(m, "AgentState")
        .def(py::init<>())
        .def_readwrite("x", &AgentState::x)
        .def_readwrite("y", &AgentState::y);
    
    // Observation
    py::class_<Observation>(m, "Observation")
        .def(py::init<>())
        .def_readwrite("features", &Observation::features);
      // CUDA Device Info
    py::class_<CudaDeviceInfo>(m, "CudaDeviceInfo")
        .def(py::init<>())
        .def_readwrite("device_count", &CudaDeviceInfo::device_count)
        .def_property("device_name", 
            [](const CudaDeviceInfo& info) { return std::string(info.device_name); },
            [](CudaDeviceInfo& info, const std::string& name) { 
                strncpy(info.device_name, name.c_str(), sizeof(info.device_name) - 1);
                info.device_name[sizeof(info.device_name) - 1] = '\0';
            });
    
    // Environment
    py::class_<Environment>(m, "Environment")
        .def(py::init<const EnvironmentConfig&>())
        .def("initialize", &Environment::initialize)
        .def("reset", &Environment::reset)
        .def("step", &Environment::step)
        .def("is_cuda_enabled", &Environment::is_cuda_enabled);
    
    // Utility functions
    m.def("is_cuda_available", &is_cuda_available, "Check if CUDA is available");
    m.def("get_cuda_device_info", &get_cuda_device_info, "Get CUDA device information");
    
    // Simple create environment function
    m.def("create_environment", [](int grid_width, int grid_height, int batch_size) {
        EnvironmentConfig config;
        config.grid_width = grid_width;
        config.grid_height = grid_height;
        Environment env(config);
        env.initialize(batch_size);
        return env;
    }, py::arg("grid_width") = 32, py::arg("grid_height") = 32, py::arg("batch_size") = 1);
    
    // Simple benchmark function  
    m.def("benchmark_environment", [](int batch_size, int num_steps) {
        EnvironmentConfig config;
        Environment env(config);
        env.initialize(batch_size);
        
        auto start = std::chrono::high_resolution_clock::now();
        
        env.reset();
        std::vector<int> actions(batch_size, 0);
        
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
    }, py::arg("batch_size") = 32, py::arg("num_steps") = 1000);
    
    // Module constants
    m.attr("VERSION") = "1.0.0";
    m.attr("CUDA_AVAILABLE") = is_cuda_available();
    
    // Action constants
    m.attr("ACTION_UP") = 0;
    m.attr("ACTION_DOWN") = 1;
    m.attr("ACTION_LEFT") = 2;
    m.attr("ACTION_RIGHT") = 3;
}
