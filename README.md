# AI Platform Trainer

An enterprise-grade platform for training and evaluating AI agents using reinforcement learning in a game environment.

## Overview

AI Platform Trainer is a 2D game environment designed for training and evaluating AI agents using deep reinforcement learning. The platform includes neural network-based enemy AI, reinforcement learning training capabilities, real-time visualizations, and a high-performance C++/CUDA backend for accelerated training.

## Quick Start

The simplest way to run the application:

```bash
python unified_launcher.py
```

## Features

- **Game Environment**: A 2D game environment built with PyGame where entities can interact
- **Neural Network Models**: Pre-trained models for missile trajectory prediction and enemy movement
- **Reinforcement Learning**: GPU-accelerated reinforcement learning using PPO for training enemy behaviors
- **C++/CUDA Integration**: High-performance physics simulation with Python bindings
- **Visualizations**: Real-time training visualizations and performance metrics
- **Cross-platform**: Support for both CPU and GPU environments with automatic detection

## Installation

### Basic Installation

```bash
# Install the package in development mode
pip install -e .
```

### CPU Environment

```bash
# Create a conda environment
conda env create -f config/environment-cpu.yml
conda activate ai-platform-cpu

# Install the package
pip install -e .
```

### GPU Environment (recommended for training)

```bash
# Create a conda environment with CUDA support
conda env create -f config/environment-gpu.yml
conda activate ai-platform-gpu

# Install the package
pip install -e .

# Build the C++ extensions
cd ai_platform_trainer/cpp
python setup.py build_ext --inplace
```

## Game Controls

- **Arrow Keys/WASD**: Move the player
- **Space**: Shoot missiles
- **F**: Toggle fullscreen
- **M**: Return to menu
- **Escape**: Exit game

## Development

This project is currently undergoing a major cleanup and refactoring. See `cleanup_plan.md` for details on the ongoing improvements.

## License

[MIT License](LICENSE)