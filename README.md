# AI Platform Trainer

A game-based platform for training and testing AI models, with a focus on reinforcement learning.

## Features

- Play mode with AI-controlled enemy
- Training mode for collecting data
- Reinforcement learning integration
- CUDA acceleration for training (when available)
- Headless training mode for faster iterations

## Getting Started

### Prerequisites

- Python 3.8+
- PyTorch
- Pygame
- Stable Baselines 3 (optional, for RL training)

### Installation

```bash
# Clone the repository
git clone https://github.com/yourusername/ai-platform-trainerv1.git
cd ai-platform-trainerv1

# Install dependencies
pip install -r requirements.txt
```

### Running the Game

```bash
python run_game.py
```

## Training AI Models

### Training with Reinforcement Learning

```bash
# Train with unified script (auto-selects best method)
python train_rl.py

# Train with specific options
python train_rl.py --method pytorch --device cuda --episodes 500
```

Options:
- `--method`: Training method (`auto`, `sb3`, or `pytorch`)
- `--device`: Device to train on (`auto`, `cuda`, or `cpu`)
- `--episodes`: Number of episodes for PyTorch training
- `--timesteps`: Number of timesteps for Stable Baselines training
- `--output`: Output path prefix for the model

### CUDA Acceleration

If you have an NVIDIA GPU, you can build the CUDA extensions for faster training:

```bash
cd ai_platform_trainer/cpp
python setup.py build_ext --inplace
```

## Controls

- **Arrow keys** or **WASD**: Move player
- **Spacebar**: Shoot missile
- **F**: Toggle fullscreen
- **M**: Return to menu
- **Escape**: Exit game

## Project Structure

- `ai_platform_trainer/`: Main package
  - `ai/`: AI models and training code
  - `core/`: Core game functionality
  - `cpp/`: C++/CUDA extensions
  - `entities/`: Game entities (player, enemy, missiles)
  - `gameplay/`: Game mechanics and modes
  - `utils/`: Utility functions
- `models/`: Saved AI models
- `train_rl.py`: Unified RL training script
- `run_game.py`: Game launcher

## License

This project is licensed under the MIT License - see the LICENSE file for details.