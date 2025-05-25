#!/usr/bin/env python
"""
Train an RL model for enemy behavior using Stable Baselines3.

This script trains a reinforcement learning model for the enemy AI
using either the GPU-accelerated environment or a CPU fallback.
"""
import os
import logging
import numpy as np
import torch
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv

# Configure logging
logging.basicConfig(level=logging.INFO, 
                   format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger("RL-Training")

# Check for CUDA
if torch.cuda.is_available():
    device = "cuda"
    logger.info(f"Using CUDA: {torch.cuda.get_device_name(0)}")
else:
    device = "cpu"
    logger.info("CUDA not available, using CPU")

# Try to import GPU environment
try:
    from ai_platform_trainer.cpp.gpu_environment import GPUGameEnv
    use_gpu_env = True
    logger.info("Using GPU-accelerated environment")
except ImportError:
    use_gpu_env = False
    logger.info("GPU environment not available, using CPU environment")
    from ai_platform_trainer.ai.models.enemy_rl_agent import EnemyGameEnv

# Create directory for saving models
os.makedirs("models/enemy_rl", exist_ok=True)

# Create and wrap the environment
if use_gpu_env:
    env = GPUGameEnv()
else:
    env = EnemyGameEnv()
env = DummyVecEnv([lambda: env])

# Initialize the agent
model = PPO("MlpPolicy", env, verbose=1, 
            learning_rate=0.0003,
            n_steps=2048,
            batch_size=64,
            n_epochs=10,
            gamma=0.99,
            gae_lambda=0.95,
            clip_range=0.2,
            device=device)

# Train the agent
logger.info("Starting RL model training...")
model.learn(total_timesteps=100000)

# Save the trained model
model.save("models/enemy_rl/final_model.zip")
logger.info("Model saved to models/enemy_rl/final_model.zip")