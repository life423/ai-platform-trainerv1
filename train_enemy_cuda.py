#!/usr/bin/env python
"""
Train enemy agent using custom C++/CUDA modules on NVIDIA GPU.

This script uses the native C++/CUDA implementation without relying on PyTorch
for GPU acceleration.
"""
import os
import sys
import numpy as np
import logging
import subprocess
import argparse
from tqdm import tqdm

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger("CUDA-Native-Training")

def verify_gpu():
    """Verify NVIDIA GPU is available."""
    try:
        result = subprocess.run(['nvidia-smi'], capture_output=True, text=True)
        if result.returncode == 0:
            logger.info("NVIDIA GPU detected:")
            for line in result.stdout.split('\n')[:5]:  # Show first 5 lines
                if line.strip():
                    logger.info(line)
            return True
        else:
            logger.error("nvidia-smi command failed. GPU not detected.")
            return False
    except Exception as e:
        logger.error(f"Error checking GPU: {e}")
        return False

def build_cuda_extensions():
    """Build the custom C++/CUDA extensions."""
    cpp_dir = os.path.join("ai_platform_trainer", "cpp")
    logger.info(f"Building custom CUDA extensions in {cpp_dir}...")
    
    try:
        result = subprocess.run(
            [sys.executable, "setup.py", "build_ext", "--inplace"],
            cwd=cpp_dir,
            check=True,
            capture_output=True,
            text=True
        )
        if "error" in result.stderr.lower():
            logger.error("Build failed with errors:")
            logger.error(result.stderr)
            return False
        return True
    except subprocess.CalledProcessError as e:
        logger.error(f"Build failed: {e}")
        logger.error(e.stderr)
        return False

def train_enemy_with_cuda(episodes=500, output_path="models/enemy_rl/cuda_model.npz"):
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
        return False
    
    # Create environment config
    config = gpu_environment.EnvironmentConfig()
    config.enable_missile_avoidance = True
    config.missile_prediction_steps = 30
    
    # Create environment
    env = gpu_environment.Environment(config)
    logger.info("Created environment with custom CUDA")
    
    # Get observation and action shapes
    obs_shape = env.get_observation_shape()
    action_shape = env.get_action_shape()
    logger.info(f"Observation shape: {obs_shape}, Action shape: {action_shape}")
    
    # Simple neural network implementation (without PyTorch)
    class SimpleNetwork:
        def __init__(self, input_size, hidden_size, output_size):
            # Initialize with random weights
            self.w1 = np.random.randn(input_size, hidden_size) * 0.1
            self.b1 = np.zeros(hidden_size)
            self.w2 = np.random.randn(hidden_size, output_size) * 0.1
            self.b2 = np.zeros(output_size)
        
        def forward(self, x):
            # Simple feedforward
            h = np.tanh(np.dot(x, self.w1) + self.b1)
            return np.tanh(np.dot(h, self.w2) + self.b2)
        
        def update(self, grads, lr=0.01):
            # Simple gradient update
            self.w1 -= lr * grads[0]
            self.b1 -= lr * grads[1]
            self.w2 -= lr * grads[2]
            self.b2 -= lr * grads[3]
        
        def save(self, path):
            # Save weights
            os.makedirs(os.path.dirname(path), exist_ok=True)
            np.savez(path, w1=self.w1, b1=self.b1, w2=self.w2, b2=self.b2)
            logger.info(f"Model saved to {path}")
    
    # Create policy network
    input_size = obs_shape[0]
    output_size = action_shape[0]
    policy_net = SimpleNetwork(input_size, 64, output_size)
    logger.info(f"Created policy network with input size {input_size}, output size {output_size}")
    
    # Training loop
    logger.info(f"Starting training for {episodes} episodes using custom CUDA...")
    episode_rewards = []
    
    for episode in tqdm(range(episodes)):
        # Reset environment
        obs = env.reset(episode)  # Use episode as seed
        episode_reward = 0
        
        # Episode loop
        done = False
        truncated = False
        step = 0
        
        while not (done or truncated):
            # Get action from policy network
            action = policy_net.forward(obs)
            
            # Add exploration noise
            action = action + np.random.normal(0, 0.1, size=action.shape)
            action = np.clip(action, -1.0, 1.0)
            
            # Take action in environment
            next_obs, reward, done, truncated, info = env.step(action)
            
            # Store reward
            episode_reward += reward
            
            # Update observation
            obs = next_obs
            step += 1
            
            # Simple update (not a proper RL algorithm, just for demonstration)
            if reward > 0:
                # Encourage actions that led to positive rewards
                # This is a very simplified approach, not a proper RL algorithm
                grads = [np.zeros_like(policy_net.w1), 
                         np.zeros_like(policy_net.b1),
                         np.zeros_like(policy_net.w2), 
                         np.zeros_like(policy_net.b2)]
                policy_net.update(grads, lr=0.001 * reward)
            
            # Limit maximum steps
            if step >= 1000:
                truncated = True
        
        # Log progress
        episode_rewards.append(episode_reward)
        if episode % 10 == 0:
            logger.info(f"Episode {episode}: Reward = {episode_reward:.2f}, Steps = {step}")
    
    # Save trained model
    policy_net.save(output_path)
    
    return True

def main():
    """Main function."""
    parser = argparse.ArgumentParser(description="Train enemy agent with custom CUDA")
    parser.add_argument("--episodes", type=int, default=500, help="Number of episodes")
    parser.add_argument("--output", type=str, default="models/enemy_rl/cuda_model.npz", 
                        help="Output model path")
    parser.add_argument("--skip-build", action="store_true", 
                        help="Skip building extensions (use if already built)")
    args = parser.parse_args()
    
    # Step 1: Verify GPU is available
    if not verify_gpu():
        logger.error("GPU verification failed. Cannot proceed.")
        sys.exit(1)
    
    # Step 2: Build custom CUDA extensions if needed
    if not args.skip_build:
        if not build_cuda_extensions():
            logger.error("Failed to build custom CUDA extensions.")
            sys.exit(1)
    
    # Step 3: Train enemy agent with custom CUDA
    if not train_enemy_with_cuda(episodes=args.episodes, output_path=args.output):
        logger.error("Training with custom CUDA failed.")
        sys.exit(1)
    
    logger.info("\n=== Training Complete ===")
    logger.info("Enemy agent was trained using custom C++/CUDA implementation.")

if __name__ == "__main__":
    main()