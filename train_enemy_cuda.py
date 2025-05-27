#!/usr/bin/env python
"""
Train enemy agent using custom C++/CUDA modules on NVIDIA GPU.

This script ensures the enemy agent is trained using the custom C++/CUDA
implementation for maximum performance.
"""
import os
import sys
import numpy as np
import logging
import argparse
from tqdm import tqdm

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger("CUDA-Training")

def verify_gpu():
    """Verify NVIDIA GPU is available."""
    try:
        import subprocess
        result = subprocess.run(['nvidia-smi'], capture_output=True, text=True)
        if result.returncode == 0:
            logger.info("NVIDIA GPU detected")
            return True
        else:
            logger.error("NVIDIA GPU not detected")
            return False
    except Exception as e:
        logger.error(f"Error checking GPU: {e}")
        return False

def train_enemy_with_cuda(episodes=500, output_path="models/enemy_rl/cuda_model.npz", 
                          num_envs=4, headless=True):
    """Train enemy agent using custom CUDA implementation."""
    # Add cpp directory to path
    cpp_dir = os.path.abspath(os.path.join("ai_platform_trainer", "cpp"))
    if cpp_dir not in sys.path:
        sys.path.append(cpp_dir)
    
    # Import custom GPU environment
    try:
        import gpu_environment
        logger.info("Successfully imported custom CUDA environment")
    except ImportError as e:
        logger.error(f"Failed to import custom GPU environment: {e}")
        logger.error("Make sure you've built the extensions with:")
        logger.error("cd ai_platform_trainer/cpp && python setup.py build_ext --inplace")
        return False
    
    # Create vectorized environment for parallel training
    try:
        # Use the vectorized environment from the CUDA module
        envs = gpu_environment.create_vectorized_env(num_envs=num_envs)
        logger.info(f"Created {num_envs} parallel environments using custom CUDA")
    except Exception as e:
        logger.error(f"Failed to create vectorized environment: {e}")
        return False
    
    # Try to import PyTorch for the policy network
    try:
        import torch
        import torch.nn as nn
        import torch.optim as optim
        
        # Define a simple policy network
        class PolicyNetwork(nn.Module):
            def __init__(self, input_size, hidden_size, output_size):
                super().__init__()
                self.fc1 = nn.Linear(input_size, hidden_size)
                self.fc2 = nn.Linear(hidden_size, hidden_size)
                self.fc3 = nn.Linear(hidden_size, output_size)
                
            def forward(self, x):
                x = torch.relu(self.fc1(x))
                x = torch.relu(self.fc2(x))
                return torch.tanh(self.fc3(x))
            
            def save(self, path):
                torch.save(self.state_dict(), path)
                logger.info(f"Model saved to {path}")
        
        # Create policy network on GPU
        obs_shape = envs.observation_shape
        action_shape = envs.action_shape
        input_size = obs_shape[0]
        output_size = action_shape[0]
        
        policy_net = PolicyNetwork(input_size, 128, output_size).cuda()
        optimizer = optim.Adam(policy_net.parameters(), lr=0.001)
        
        logger.info(f"Created policy network on GPU with input size {input_size}, output size {output_size}")
        
        # Training loop
        logger.info(f"Starting training for {episodes} episodes using custom CUDA...")
        
        for episode in tqdm(range(episodes)):
            # Reset environments
            observations = envs.reset()
            episode_rewards = [0] * num_envs
            
            # Lists to store episode data
            log_probs_batch = [[] for _ in range(num_envs)]
            rewards_batch = [[] for _ in range(num_envs)]
            
            # Episode loop
            dones = [False] * num_envs
            
            while not all(dones):
                # Convert observations to tensor on GPU
                obs_tensor = torch.FloatTensor(observations).cuda()
                
                # Get actions from policy network
                with torch.no_grad():
                    action_means = policy_net(obs_tensor)
                
                # Add exploration noise
                action_std = torch.ones_like(action_means) * 0.1
                action_dist = torch.distributions.Normal(action_means, action_std)
                actions = action_dist.sample()
                actions = torch.clamp(actions, -1.0, 1.0)
                
                # Store log probabilities
                log_probs = action_dist.log_prob(actions).sum(dim=1)
                
                # Take actions in environments
                actions_np = actions.cpu().numpy()
                next_observations, rewards, new_dones, truncateds, infos = envs.step(actions_np)
                
                # Update dones
                dones = [d or t for d, t in zip(new_dones, truncateds)]
                
                # Store rewards and log probs
                for i in range(num_envs):
                    if not dones[i]:
                        rewards_batch[i].append(rewards[i])
                        log_probs_batch[i].append(log_probs[i])
                        episode_rewards[i] += rewards[i]
                
                # Update observations
                observations = next_observations
            
            # Update policy network
            optimizer.zero_grad()
            
            # Calculate loss for each environment
            policy_loss = 0
            for env_idx in range(num_envs):
                if len(rewards_batch[env_idx]) > 0:
                    # Calculate returns
                    returns = []
                    R = 0
                    for r in reversed(rewards_batch[env_idx]):
                        R = r + 0.99 * R  # Gamma = 0.99
                        returns.insert(0, R)
                    
                    # Convert to tensor on GPU
                    returns = torch.FloatTensor(returns).cuda()
                    
                    # Normalize returns
                    if len(returns) > 1:
                        returns = (returns - returns.mean()) / (returns.std() + 1e-8)
                    
                    # Calculate loss
                    env_log_probs = torch.stack(log_probs_batch[env_idx])
                    env_policy_loss = -(env_log_probs * returns).sum()
                    policy_loss += env_policy_loss
            
            # Backpropagate
            if policy_loss != 0:
                policy_loss.backward()
                optimizer.step()
            
            # Log progress
            avg_reward = sum(episode_rewards) / num_envs
            if episode % 10 == 0:
                logger.info(f"Episode {episode}: Avg Reward = {avg_reward:.2f}")
        
        # Save trained model
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        policy_net.save(output_path)
        
        return True
    
    except ImportError:
        logger.error("PyTorch not available. Using NumPy implementation instead.")
        # Implement NumPy-based training here if needed
        return False

def main():
    """Main function."""
    parser = argparse.ArgumentParser(description="Train enemy agent with custom CUDA")
    parser.add_argument("--episodes", type=int, default=500, help="Number of episodes")
    parser.add_argument("--num-envs", type=int, default=4, help="Number of parallel environments")
    parser.add_argument("--output", type=str, default="models/enemy_rl/cuda_model.pth", 
                        help="Output model path")
    args = parser.parse_args()
    
    # Step 1: Verify GPU is available
    if not verify_gpu():
        logger.error("GPU verification failed. Cannot proceed.")
        sys.exit(1)
    
    # Step 2: Train enemy agent with custom CUDA
    if not train_enemy_with_cuda(
        episodes=args.episodes, 
        output_path=args.output,
        num_envs=args.num_envs
    ):
        logger.error("Training with custom CUDA failed.")
        sys.exit(1)
    
    logger.info("\n=== Training Complete ===")
    logger.info("Enemy agent was trained using custom C++/CUDA implementation.")

if __name__ == "__main__":
    main()