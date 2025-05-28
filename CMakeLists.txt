cmake_minimum_required(VERSION 3.18 FATAL_ERROR)
project(gpu_game_environment LANGUAGES CXX CUDA)

# Set C++ standard
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Enable CUDA
enable_language(CUDA)
set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)

# Find CUDA toolkit
find_package(CUDAToolkit REQUIRED)

# Add the CUDA validation executable
add_executable(validate_cuda src/validate_cuda.cu)

# Ensure CUDA is enabled
target_link_libraries(validate_cuda PRIVATE CUDA::cudart)

# Set standards
set_target_properties(validate_cuda PROPERTIES
    CXX_STANDARD 17
    CUDA_SEPARABLE_COMPILATION ON
)
