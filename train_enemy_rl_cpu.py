#!/usr/bin/env python
"""
Train an RL model for enemy behavior using CPU only.

This script provides a simplified training approach that doesn't require
the CUDA extensions to be built.
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
logger = logging.getLogger("RL-Training-CPU")

# Import the environment
from ai_platform_trainer.ai.models.enemy_rl_agent import EnemyGameEnv

# Create directory for saving models
os.makedirs("models/enemy_rl", exist_ok=True)

# Create and wrap the environment
env = EnemyGameEnv()  # No game instance - will run in standalone mode
env = DummyVecEnv([lambda: env])

# Initialize the agent (force CPU)
model = PPO("MlpPolicy", env, verbose=1, device="cpu")

# Train the agent
logger.info("Starting RL model training (CPU only)...")
model.learn(total_timesteps=50000)

# Save the trained model
model.save("models/enemy_rl/final_model.zip")
logger.info("Model saved to models/enemy_rl/final_model.zip")