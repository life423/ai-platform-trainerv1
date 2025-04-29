# Reinforcement Learning for AI Platform Trainer

This extension adds reinforcement learning capabilities to the AI Platform Trainer game, enabling more intelligent and adaptive enemy behavior.

## Features

- **Deep Reinforcement Learning**: Uses Proximal Policy Optimization (PPO) from Stable Baselines3 for the enemy AI
- **Intelligent Enemy Behavior**: The enemy learns to chase, avoid missiles, and employ strategic movement
- **Automatic Fallback**: Falls back to the traditional neural network if the RL model is not available
- **Training Tools**: Includes a script for easy training with various options

## Installation

The reinforcement learning functionality is already integrated into the codebase. You just need to install the required dependencies:

```bash
pip install stable-baselines3[extra]
```

## Training a New Model

To train a new reinforcement learning model for the enemy:

```bash
# Quick training (for testing)
python train_enemy_rl_model.py --timesteps 100000

# More extensive training for better results
python train_enemy_rl_model.py --timesteps 1000000 --headless
```

Training options:

- `--timesteps`: Number of training steps to run (default: 500000)
- `--headless`: Run training without visualization (faster training)
- `--save-path`: Directory to save models to (default: models/enemy_rl)
- `--log-path`: Directory to save TensorBoard logs to (default: logs/enemy_rl)

## Monitoring Training

You can monitor the training progress with TensorBoard:

```bash
# Install TensorBoard if you don't have it
pip install tensorboard

# Launch TensorBoard
tensorboard --logdir=logs/enemy_rl
```

Then open your browser at http://localhost:6006 to view training metrics including rewards and learning progress.

## Playing with the Trained AI

After training, the model is automatically saved to `models/enemy_rl/final_model.zip`. The game will automatically use this model if it exists when you run:

```bash
python -m ai_platform_trainer.main
```

## Architecture

The reinforcement learning implementation consists of:

1. **Custom Gym Environment** (`EnemyGameEnv`): Wraps the game state to provide a reinforcement learning interface with observations, actions, and rewards.

2. **PPO Model**: A neural network policy that learns to map game states to optimal enemy movement.

3. **Integration Layer**: Modifications to the `EnemyPlay` class that enable it to use the trained RL model while maintaining compatibility with the existing neural network approach.

## Reward Structure

The enemy is rewarded for:
- Getting closer to the player
- Successfully hitting the player

The enemy is penalized for:
- Being hit by a missile

This reward structure encourages aggressive yet strategic behavior.

## Future Improvements

Possible enhancements to the reinforcement learning system:

1. **Curriculum Learning**: Gradually increase difficulty during training
2. **Multi-agent Training**: Have multiple enemies train simultaneously, potentially developing team strategies
3. **Imitation Learning**: Initialize RL training with demonstrations from human players
4. **Meta-Learning**: Train the enemy to adapt to different player styles in real-time
