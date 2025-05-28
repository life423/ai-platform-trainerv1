cmake_minimum_required(VERSION 3.18 FATAL_ERROR)
project(CudaRL-Arena LANGUAGES CXX CUDA)

# Set C++ and CUDA standards
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)

# Find required packages
find_package(CUDA REQUIRED)
find_package(pybind11 REQUIRED)

# CUDA architecture settings
if(NOT DEFINED CMAKE_CUDA_ARCHITECTURES)
    set(CMAKE_CUDA_ARCHITECTURES 61 75 80 86 89)
endif()

# Include directories
include_directories(${CMAKE_CURRENT_SOURCE_DIR}/src)
include_directories(${CUDA_INCLUDE_DIRS})

# Source files
set(CORE_SOURCES
    src/core/environment.cpp
    src/core/cuda_utils.cpp
)

set(GPU_SOURCES
    src/gpu/kernels.cu
    src/gpu/environment_device.cu
)

set(BINDING_SOURCES
    src/bindings/python/python_bindings.cpp
)

# Create the Python module
pybind11_add_module(cudarl_core_python 
    ${CORE_SOURCES} 
    ${GPU_SOURCES} 
    ${BINDING_SOURCES}
)

# Set module properties
target_compile_definitions(cudarl_core_python PRIVATE VERSION_INFO="dev")
target_compile_features(cudarl_core_python PRIVATE cxx_std_17)

# CUDA-specific compilation flags
set_property(TARGET cudarl_core_python PROPERTY CUDA_SEPARABLE_COMPILATION ON)

# Link libraries
target_link_libraries(cudarl_core_python PRIVATE ${CUDA_LIBRARIES})

# Python module suffix
set_target_properties(cudarl_core_python PROPERTIES
    CUDA_RESOLVE_DEVICE_SYMBOLS ON
    POSITION_INDEPENDENT_CODE ON
)

# Optional: Create a separate executable for testing
add_executable(test_cuda_env
    src/core/environment.cpp
    src/gpu/kernels.cu
    src/gpu/environment_device.cu
    src/test/test_main.cpp
)

target_link_libraries(test_cuda_env ${CUDA_LIBRARIES})
set_property(TARGET test_cuda_env PROPERTY CUDA_SEPARABLE_COMPILATION ON)
