#!/usr/bin/env python
"""
Train an RL model for enemy behavior using PyTorch (CPU-only).

This script provides a CPU-only implementation for training the enemy AI
when CUDA is not available or when Stable Baselines3 is not installed.
"""
import os
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from tqdm import tqdm
import logging

# Configure logging
logging.basicConfig(level=logging.INFO, 
                   format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger("RL-Training-CPU")

# Import our environment and policy network
from ai_platform_trainer.core.render_mode import RenderMode
from ai_platform_trainer.ai.models.game_environment import GameEnvironment
from ai_platform_trainer.ai.models.policy_network import PolicyNetwork

def train_policy_network(env, policy_net, optimizer, num_episodes=500):
    """
    Train the policy network using REINFORCE algorithm.
    
    Args:
        env: The game environment
        policy_net: The policy network to train
        optimizer: The optimizer to use
        num_episodes: Number of episodes to train for
        
    Returns:
        List of episode rewards
    """
    episode_rewards = []
    
    for episode in tqdm(range(num_episodes), desc="Training"):
        # Reset environment
        state, _ = env.reset()
        episode_reward = 0
        
        # Lists to store episode data
        log_probs = []
        rewards = []
        
        # Episode loop
        done = False
        truncated = False
        while not (done or truncated):
            # Convert state to tensor
            state_tensor = torch.FloatTensor(state).unsqueeze(0)
            
            # Get action from policy network
            action_mean = policy_net(state_tensor)
            
            # Add exploration noise
            action_std = torch.ones_like(action_mean) * 0.1
            action_dist = torch.distributions.Normal(action_mean, action_std)
            action = action_dist.sample()
            action = torch.clamp(action, -1.0, 1.0)
            
            # Store log probability
            log_prob = action_dist.log_prob(action).sum()
            log_probs.append(log_prob)
            
            # Take action in environment
            action_np = action.squeeze().detach().numpy()
            next_state, reward, done, truncated, _ = env.step(action_np)
            
            # Store reward
            rewards.append(reward)
            episode_reward += reward
            
            # Update state
            state = next_state
        
        # Update policy network
        optimizer.zero_grad()
        
        # Calculate returns
        returns = []
        R = 0
        for r in reversed(rewards):
            R = r + 0.99 * R  # Gamma = 0.99
            returns.insert(0, R)
        returns = torch.tensor(returns)
        
        # Normalize returns
        if len(returns) > 1:
            returns = (returns - returns.mean()) / (returns.std() + 1e-8)
        
        # Calculate loss
        policy_loss = []
        for log_prob, R in zip(log_probs, returns):
            policy_loss.append(-log_prob * R)
        
        if policy_loss:
            policy_loss = torch.cat(policy_loss).sum()
            
            # Backpropagate
            policy_loss.backward()
            optimizer.step()
        
        # Log progress
        episode_rewards.append(episode_reward)
        if episode % 10 == 0:
            logger.info(f"Episode {episode}: Reward = {episode_reward:.2f}")
    
    return episode_rewards

def main():
    """Train an RL model for enemy behavior using CPU."""
    # Create directory for saving models
    os.makedirs("models/enemy_rl", exist_ok=True)
    
    # Create environment
    env = GameEnvironment(render_mode='none')
    
    # Create policy network
    input_size = env.observation_space.shape[0]
    output_size = env.action_space.shape[0]
    policy_net = PolicyNetwork(input_size, 64, output_size)
    
    # Create optimizer
    optimizer = optim.Adam(policy_net.parameters(), lr=0.001)
    
    # Train policy network
    logger.info("Starting policy network training (CPU only)...")
    train_policy_network(env, policy_net, optimizer, num_episodes=500)
    
    # Save trained model
    model_path = "models/enemy_rl/final_model.pth"
    policy_net.save(model_path)
    logger.info(f"Model saved to {model_path}")

if __name__ == "__main__":
    main()