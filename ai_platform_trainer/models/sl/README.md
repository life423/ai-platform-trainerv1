# Supervised Learning Models

This directory contains trained models for the supervised learning enemy AI.

## Model Files

- `enemy_ai_model.pth` - Main enemy AI neural network (PyTorch format)
- `missile_model.pth` - Missile AI model (if applicable)

## Usage

Models in this directory are loaded by the SupervisedAgent class in:
`ai_platform_trainer/agents/sl/agent.py`

## Model Format

The models are typically PyTorch `.pth` files containing trained neural network weights.

## Training

To retrain models, use the training scripts in:
- `ai_platform_trainer/ai/training/`
- `train_enemy_agent.py` (root level)
