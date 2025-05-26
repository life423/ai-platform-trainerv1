# CUDA Acceleration for AI Platform Trainer

This document explains how to use CUDA acceleration for the AI Platform Trainer.

## Current Status

The AI Platform Trainer supports two approaches for CUDA acceleration:

1. **PyTorch with CUDA** - Uses PyTorch's built-in CUDA support for training
2. **Custom C++/CUDA Extensions** - Uses custom CUDA kernels for physics simulation (currently has build issues)

## Using PyTorch with CUDA

The simplest way to leverage your GPU is to use PyTorch's built-in CUDA support:

```bash
# Train with GPU acceleration (if available)
python train_rl_with_gpu.py --device cuda
```

This approach doesn't require building the custom C++/CUDA extensions and works with any CUDA-compatible GPU.

## Building Custom C++/CUDA Extensions (Advanced)

The custom C++/CUDA extensions provide additional acceleration for physics simulation but require building from source.

### Prerequisites

- NVIDIA CUDA Toolkit 11.0+
- C++ compiler compatible with your CUDA version
- CMake 3.18+
- Python 3.8+
- PyBind11

### Building Steps

1. Navigate to the cpp directory:
   ```bash
   cd ai_platform_trainer/cpp
   ```

2. Build the extensions:
   ```bash
   python setup.py build_ext --inplace
   ```

3. If you encounter build errors, check:
   - CUDA architecture compatibility
   - Missing header files
   - Syntax errors in CUDA code

### Troubleshooting

If you encounter build errors:

1. Check that your CUDA toolkit is properly installed:
   ```bash
   nvcc --version
   ```

2. Verify that your GPU is detected:
   ```bash
   nvidia-smi
   ```

3. Update the CUDA architecture in `CMakeLists.txt` to match your GPU:
   ```cmake
   set(CMAKE_CUDA_ARCHITECTURES 86 89)  # For RTX 30xx and RTX 40xx/50xx series
   ```

## RTX 5070 Specific Configuration

For the RTX 5070 GPU, use the following CUDA architecture in `CMakeLists.txt`:

```cmake
set(CMAKE_CUDA_ARCHITECTURES 89)  # For Ada Lovelace architecture (RTX 40xx/50xx)
```

## Performance Comparison

| Training Method | Hardware | Time for 100 Episodes |
|----------------|----------|----------------------|
| CPU Only       | Intel i7 | ~25 seconds          |
| PyTorch CUDA   | RTX 5070 | ~10 seconds (est.)   |
| Custom CUDA    | RTX 5070 | ~5 seconds (est.)    |

## Future Improvements

- Fix build issues with custom C++/CUDA extensions
- Optimize CUDA kernels for better performance
- Add support for tensor cores on RTX GPUs
- Implement multi-GPU training