#!/usr/bin/env python
"""
Train an RL model for enemy behavior using PyTorch with CUDA support.

This script provides a simplified approach that uses PyTorch's CUDA support
without requiring the custom C++ extensions to be built.
"""
import os
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import logging
from tqdm import tqdm
import argparse

# Configure logging
logging.basicConfig(level=logging.INFO, 
                   format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger("RL-Training-GPU")

# Import our environment and policy network
from ai_platform_trainer.core.render_mode import RenderMode
from ai_platform_trainer.ai.models.game_environment import GameEnvironment
from ai_platform_trainer.ai.models.policy_network import PolicyNetwork

def train_policy_network(env, policy_net, optimizer, num_episodes=500, device="cuda"):
    """
    Train the policy network using REINFORCE algorithm with GPU acceleration.
    
    Args:
        env: The game environment
        policy_net: The policy network to train
        optimizer: The optimizer to use
        num_episodes: Number of episodes to train for
        device: Device to train on ("cuda" or "cpu")
        
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
            # Convert state to tensor and move to device
            state_tensor = torch.FloatTensor(state).unsqueeze(0).to(device)
            
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
            
            # Take action in environment (move back to CPU for numpy)
            action_np = action.squeeze().detach().cpu().numpy()
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
        
        if len(returns) > 0:  # Only proceed if we have returns
            # Convert to tensor and move to device
            returns = torch.tensor(returns, device=device)
            
            # Normalize returns
            if len(returns) > 1:
                returns = (returns - returns.mean()) / (returns.std() + 1e-8)
            
            # Calculate loss
            policy_loss = []
            for log_prob, R in zip(log_probs, returns):
                policy_loss.append(-log_prob * R)
            
            if policy_loss:  # Only proceed if we have policy losses
                policy_loss = torch.stack(policy_loss).sum()
                
                # Backpropagate
                policy_loss.backward()
                optimizer.step()
        
        # Log progress
        episode_rewards.append(episode_reward)
        if episode % 10 == 0:
            logger.info(f"Episode {episode}: Reward = {episode_reward:.2f}")
    
    return episode_rewards

def main():
    """Train an RL model for enemy behavior using GPU."""
    parser = argparse.ArgumentParser(description="Train RL model for enemy behavior with GPU")
    parser.add_argument("--episodes", type=int, default=500,
                      help="Number of episodes for training (default: 500)")
    parser.add_argument("--device", choices=["auto", "cuda", "cpu"], default="auto",
                      help="Device to train on (default: auto)")
    parser.add_argument("--output", type=str, default="models/enemy_rl/final_model.pth",
                      help="Output path for the model (default: models/enemy_rl/final_model.pth)")
    args = parser.parse_args()
    
    # Create directory for saving models
    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    
    # Determine device
    if args.device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"
    else:
        device = args.device
    
    # Log device info
    if device == "cuda":
        logger.info(f"Using CUDA: {torch.cuda.get_device_name(0)}")
        logger.info(f"CUDA version: {torch.version.cuda}")
    else:
        logger.info("Using CPU for training")
    
    # Create environment
    env = GameEnvironment(render_mode='none')
    
    # Create policy network
    input_size = env.observation_space.shape[0]
    output_size = env.action_space.shape[0]
    policy_net = PolicyNetwork(input_size, 64, output_size)
    policy_net.to(device)
    
    # Create optimizer
    optimizer = optim.Adam(policy_net.parameters(), lr=0.001)
    
    # Train policy network
    logger.info(f"Starting policy network training on {device}...")
    train_policy_network(env, policy_net, optimizer, num_episodes=args.episodes, device=device)
    
    # Save trained model
    policy_net.save(args.output)
    logger.info(f"Model saved to {args.output}")

if __name__ == "__main__":
    main()