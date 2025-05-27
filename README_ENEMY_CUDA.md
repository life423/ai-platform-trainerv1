# Training Enemy Agent with Custom CUDA

This guide explains how to train the enemy agent using custom C++/CUDA modules on your NVIDIA GPU.

## Prerequisites

- NVIDIA GPU with CUDA support
- NVIDIA CUDA Toolkit installed
- C++ compiler compatible with your CUDA version
- Python 3.7+

## Quick Start

### Windows

```bash
# Run verification and training
run_enemy_cuda_training.bat

# Or with custom parameters
run_enemy_cuda_training.bat --episodes 1000 --output models/enemy_rl/my_model.npz
```

### Linux/macOS

```bash
# Make the script executable
chmod +x run_enemy_cuda_training.sh

# Run verification and training
./run_enemy_cuda_training.sh

# Or with custom parameters
./run_enemy_cuda_training.sh --episodes 1000 --output models/enemy_rl/my_model.npz
```

## Manual Steps

### 1. Verify CUDA Extensions

First, verify that your CUDA extensions are built and working correctly:

```bash
python verify_cuda_extensions.py
```

This script will:
- Check if an NVIDIA GPU is available
- Check if the NVIDIA CUDA compiler (nvcc) is available
- Build the custom C++/CUDA extensions
- Test if the extensions are working correctly

### 2. Train Enemy Agent

Once the verification is successful, you can train the enemy agent:

```bash
python train_enemy_cuda.py
```

Optional parameters:
- `--episodes`: Number of training episodes (default: 500)
- `--output`: Output model path (default: models/enemy_rl/cuda_model.npz)
- `--skip-build`: Skip building extensions (use if already built)

Example:
```bash
python train_enemy_cuda.py --episodes 1000 --output models/enemy_rl/my_model.npz
```

## Troubleshooting

If you encounter issues:

1. Make sure your NVIDIA GPU is detected:
   ```bash
   nvidia-smi
   ```

2. Check if CUDA compiler is available:
   ```bash
   nvcc --version
   ```

3. Try building the extensions manually:
   ```bash
   cd ai_platform_trainer/cpp
   python setup.py build_ext --inplace
   ```

4. Check if the extensions can be imported:
   ```bash
   cd ai_platform_trainer/cpp
   python check_cuda_extensions.py
   ```