#!/usr/bin/env python
"""
Train an RL model for enemy behavior using Stable Baselines3.

This script trains a reinforcement learning model for the enemy AI
using either the GPU-accelerated environment or a CPU fallback.
"""
import os
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from tqdm import tqdm
import logging
import gymnasium as gym
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv

# Configure logging
logging.basicConfig(level=logging.INFO, 
                   format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger("RL-Training")

# Import our environment
from ai_platform_trainer.core.render_mode import RenderMode
from ai_platform_trainer.ai.models.game_environment import GameEnvironment

def main():
    """Train an RL model for enemy behavior."""
    # Create directory for saving models
    os.makedirs("models/enemy_rl", exist_ok=True)
    
    # Check for CUDA
    if torch.cuda.is_available():
        device = "cuda"
        logger.info(f"Using CUDA: {torch.cuda.get_device_name(0)}")
    else:
        device = "cpu"
        logger.info("CUDA not available, using CPU")
    
    # Create vectorized environment
    env = DummyVecEnv([lambda: GameEnvironment(render_mode='none')])
    
    # Initialize agent
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
    model.learn(total_timesteps=50000)
    
    # Save the trained model
    model_path = "models/enemy_rl/final_model.zip"
    model.save(model_path)
    logger.info(f"Model saved to {model_path}")

if __name__ == "__main__":
    main()