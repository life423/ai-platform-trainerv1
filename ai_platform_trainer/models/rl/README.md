# Reinforcement Learning Models

This directory will contain trained RL models and checkpoints for different difficulty levels.

## Planned Model Files

- `pretrained_easy.dat` - Easy difficulty RL agent
- `pretrained_medium.dat` - Medium difficulty RL agent  
- `pretrained_hard.dat` - Hard difficulty RL agent
- `checkpoints/` - Live training session checkpoints

## Usage

Models will be loaded by the RLAgent class (Phase 2 implementation):
`ai_platform_trainer/agents/rl/agent.py`

## Model Format

RL models will use a custom binary format optimized for C++/CUDA loading.

## Training

Live training occurs during gameplay in "Play Against Learning RL Agent" mode.
Offline training scripts will be added in Phase 2.

## Status

**Phase 1**: ✅ Directory structure created  
**Phase 2**: 🚧 C++/CUDA implementation and model training (pending)
