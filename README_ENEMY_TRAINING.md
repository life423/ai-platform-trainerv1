# Enemy Agent Training with GPU Acceleration

This guide explains how to train the enemy agent using GPU acceleration for improved performance.

## Overview

The AI Platform Trainer includes scripts to train the enemy agent using different hardware acceleration methods:

1. **Custom C++/CUDA Modules** - Uses custom CUDA kernels for physics simulation and environment stepping
2. **PyTorch with CUDA** - Uses PyTorch's built-in CUDA support for neural network training
3. **CPU Fallback** - Uses CPU for training when GPU acceleration is not available

## Quick Start

The easiest way to train the enemy agent is to use the unified training script, which automatically detects and uses the best available hardware:

```bash
python train_enemy_agent.py --episodes 500
```

This script will:
1. Check if custom C++/CUDA extensions are built and working
2. Check if PyTorch with CUDA is available
3. Fall back to CPU if neither GPU option is available
4. Train the enemy agent using the best available method

## Building Custom C++/CUDA Extensions

To use the custom C++/CUDA extensions for maximum performance:

```bash
# Build the extensions
python build_cuda_extensions.py

# Train using the custom CUDA extensions
python train_enemy_agent.py --force-method cuda
```

## Using PyTorch with CUDA

If you prefer to use PyTorch's CUDA support:

```bash
python train_enemy_agent.py --force-method pytorch
```

## Command-Line Options

The `train_enemy_agent.py` script supports the following options:

- `--episodes`: Number of training episodes (default: 500)
- `--output`: Output model path prefix (default: models/enemy_rl/model)
- `--force-method`: Force a specific training method (cuda, pytorch, or cpu)

## Troubleshooting

If you encounter issues with GPU training:

1. Verify your NVIDIA GPU is detected:
   ```bash
   nvidia-smi
   ```

2. Check if PyTorch can use CUDA:
   ```python
   import torch
   print(torch.cuda.is_available())
   ```

3. Try rebuilding the custom C++/CUDA extensions:
   ```bash
   python build_cuda_extensions.py
   ```

4. If all else fails, fall back to CPU training:
   ```bash
   python train_enemy_agent.py --force-method cpu
   ```