#!/usr/bin/env python
"""
Train enemy agent using PyTorch with GPU acceleration.

This script ensures the enemy agent is trained using PyTorch's CUDA support
for maximum performance.
"""
import os
import sys
import numpy as np
import logging
import argparse
import torch
from tqdm import tqdm

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger("GPU-Training")

def verify_gpu():
    """Verify CUDA is available and working."""
    if not torch.cuda.is_available():
        logger.error("CUDA is not available! Cannot use GPU.")
        return False
    
    logger.info(f"CUDA is available: {torch.cuda.is_available()}")
    logger.info(f"GPU: {torch.cuda.get_device_name(0)}")
    logger.info(f"CUDA version: {torch.version.cuda}")
    
    # Test CUDA with a simple operation
    try:
        x = torch.rand(100, 100).cuda()
        y = torch.matmul(x, x)
        del x, y
        torch.cuda.empty_cache()
        return True
    except Exception as e:
        logger.error(f"Error testing CUDA: {e}")
        return False

def train_enemy_with_gpu(episodes=500, output_path="models/enemy_rl/gpu_model.pth"):
    """Train enemy agent using PyTorch with GPU acceleration."""
    # Import the game environment
    try:
        from ai_platform_trainer.core.render_mode import RenderMode
        from ai_platform_trainer.ai.models.game_environment import GameEnvironment
        from ai_platform_trainer.ai.models.policy_network import PolicyNetwork
        
        logger.info("Successfully imported game environment and policy network")
    except ImportError as e:
        logger.error(f"Failed to import required modules: {e}")
        return False
    
    # Create environment in headless mode
    env = GameEnvironment(render_mode='none')
    logger.info("Created game environment in headless mode")
    
    # Create policy network on GPU
    input_size = env.observation_space.shape[0]
    output_size = env.action_space.shape[0]
    policy_net = PolicyNetwork(input_size, 128, output_size).cuda()
    optimizer = torch.optim.Adam(policy_net.parameters(), lr=0.001)
    
    logger.info(f"Created policy network on GPU with input size {input_size}, output size {output_size}")
    
    # Training loop
    logger.info(f"Starting training for {episodes} episodes using PyTorch CUDA...")
    episode_rewards = []
    
    for episode in tqdm(range(episodes)):
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
            # Convert state to tensor on GPU
            state_tensor = torch.FloatTensor(state).unsqueeze(0).cuda()
            
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
        
        if len(returns) > 0:
            # Convert to tensor on GPU
            returns = torch.tensor(returns, device="cuda")
            
            # Normalize returns
            if len(returns) > 1:
                returns = (returns - returns.mean()) / (returns.std() + 1e-8)
            
            # Calculate loss
            policy_loss = []
            for log_prob, R in zip(log_probs, returns):
                policy_loss.append(-log_prob * R)
            
            if policy_loss:
                policy_loss = torch.stack(policy_loss).sum()
                
                # Backpropagate
                policy_loss.backward()
                optimizer.step()
        
        # Log progress
        episode_rewards.append(episode_reward)
        if episode % 10 == 0:
            logger.info(f"Episode {episode}: Reward = {episode_reward:.2f}")
            # Log GPU memory usage to confirm GPU is being used
            logger.info(f"GPU memory: {torch.cuda.memory_allocated() / 1024**2:.1f} MB")
    
    # Save trained model
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    policy_net.save(output_path)
    logger.info(f"Model saved to {output_path}")
    
    return True

def main():
    """Main function."""
    parser = argparse.ArgumentParser(description="Train enemy agent with GPU acceleration")
    parser.add_argument("--episodes", type=int, default=500, help="Number of episodes")
    parser.add_argument("--output", type=str, default="models/enemy_rl/gpu_model.pth", 
                        help="Output model path")
    args = parser.parse_args()
    
    # Step 1: Verify GPU is available
    if not verify_gpu():
        logger.error("GPU verification failed. Cannot proceed.")
        sys.exit(1)
    
    # Step 2: Train enemy agent with GPU
    if not train_enemy_with_gpu(episodes=args.episodes, output_path=args.output):
        logger.error("Training with GPU failed.")
        sys.exit(1)
    
    # Step 3: Verify GPU was used
    logger.info("\nVerifying GPU usage:")
    allocated = torch.cuda.memory_allocated(0)
    max_allocated = torch.cuda.max_memory_allocated(0)
    logger.info(f"CUDA memory currently allocated: {allocated/1024**2:.1f} MB")
    logger.info(f"Max CUDA memory allocated: {max_allocated/1024**2:.1f} MB")
    
    if max_allocated > 10*1024*1024:  # More than 10MB used
        logger.info("✓ GPU was successfully used for training!")
    else:
        logger.warning("! Low GPU memory usage. Training might not have used GPU effectively.")
    
    logger.info("\n=== Training Complete ===")
    logger.info("Enemy agent was trained using PyTorch with GPU acceleration.")

if __name__ == "__main__":
    main()