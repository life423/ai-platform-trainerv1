"""
GPU-Accelerated RL Environment for Enemy Agent Training

Pure NumPy implementation that interfaces with CUDA C++ backend.
No PyTorch dependency - uses existing gpu_environment module.
"""
import logging
import os
import sys
import time
from typing import Dict, List, Optional, Tuple

import numpy as np

# Add the cpp directory to the Python path for gpu_environment module
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', 'cpp'))

logger = logging.getLogger(__name__)


class GPURLEnvironment:
    """GPU-accelerated environment using pure NumPy and CUDA C++ backend."""
    
    def __init__(self, batch_size: int = 256):
        """Initialize GPU environment.
        
        Args:
            batch_size: Number of parallel environments
        """
        self.batch_size = batch_size
        
        logger.info(f"Initializing GPURLEnvironment with batch_size={batch_size}")
        
        # Try to import CUDA environment module
        try:
            import gpu_environment
            self.gpu_env = gpu_environment
            self.cuda_available = True
            logger.info("CUDA environment module loaded successfully")
        except ImportError as e:
            logger.warning(f"CUDA environment module not found: {e}")
            logger.info("Using CPU fallback implementation")
            self.gpu_env = None
            self.cuda_available = False
        
        # Initialize environments
        if self.cuda_available:
            # Configure CUDA environment
            self.config = self.gpu_env.EnvironmentConfig()
            self.config.max_steps = 200  # Shorter episodes for faster training
            self.config.screen_width = 800
            self.config.screen_height = 600
            
            # Create environment instances for batch processing
            self.envs = [self.gpu_env.Environment(self.config) for _ in range(self.batch_size)]
            logger.info(f"Created {self.batch_size} CUDA environment instances")
        else:
            # CPU fallback initialization
            self.enemy_positions = np.random.rand(self.batch_size, 2) * 700 + 50
            self.player_positions = np.random.rand(self.batch_size, 2) * 700 + 50
            self.episode_steps = np.zeros(self.batch_size, dtype=np.int32)
            self.max_steps = 200
        
        # Reset environments
        self.reset()
    
    def reset(self) -> np.ndarray:
        """Reset all environments in the batch.
        
        Returns:
            Initial observations [batch_size, obs_size]
        """
        if self.cuda_available:
            # Use CUDA batch reset
            seeds = list(range(self.batch_size))
            observations = []
            
            for i, env in enumerate(self.envs):
                obs = env.reset(seeds[i])
                observations.append(np.array(obs))
            
            self.observations = np.array(observations, dtype=np.float32)
        else:
            # CPU fallback
            self.enemy_positions = np.random.rand(self.batch_size, 2) * 700 + 50
            self.player_positions = np.random.rand(self.batch_size, 2) * 700 + 50
            self.episode_steps.fill(0)
            self.observations = self._get_observations()
        
        return self.observations
    
    def step(self, actions: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Step all environments with given actions.
        
        Args:
            actions: Actions for all environments [batch_size, 2]
            
        Returns:
            Tuple of (observations, rewards, dones)
        """
        if self.cuda_available:
            # Use CUDA environments
            observations = []
            rewards = []
            dones = []
            
            for i, env in enumerate(self.envs):
                action = actions[i].tolist()
                obs, reward, done, truncated, info = env.step(action)
                
                observations.append(np.array(obs))
                rewards.append(reward)
                dones.append(done or truncated)
            
            self.observations = np.array(observations, dtype=np.float32)
            rewards_array = np.array(rewards, dtype=np.float32)
            dones_array = np.array(dones, dtype=bool)
            
            return self.observations, rewards_array, dones_array
        else:
            # CPU fallback
            return self._cpu_step(actions)
    
    def _cpu_step(self, actions: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """CPU fallback implementation for physics step."""
        # Clamp actions to [-1, 1]
        actions = np.clip(actions, -1.0, 1.0)
        
        # Update enemy positions
        speed = 200.0  # pixels per second
        dt = 0.016  # 60 FPS
        self.enemy_positions += actions * speed * dt
        
        # Keep enemies in bounds
        self.enemy_positions[:, 0] = np.clip(self.enemy_positions[:, 0], 25, 775)
        self.enemy_positions[:, 1] = np.clip(self.enemy_positions[:, 1], 25, 575)
        
        # Update player positions (random walk)
        player_speed = 150.0
        random_actions = np.random.randn(*self.player_positions.shape) * 0.5
        self.player_positions += random_actions * player_speed * dt
        
        # Keep players in bounds
        self.player_positions[:, 0] = np.clip(self.player_positions[:, 0], 25, 775)
        self.player_positions[:, 1] = np.clip(self.player_positions[:, 1], 25, 575)
        
        # Calculate distances
        distances = np.linalg.norm(self.player_positions - self.enemy_positions, axis=1)
        
        # Calculate rewards
        distance_reward = -distances / 100.0
        proximity_bonus = (distances < 50.0).astype(np.float32) * 10.0
        smoothness_penalty = -np.linalg.norm(actions, axis=1) * 0.1
        
        rewards = distance_reward + proximity_bonus + smoothness_penalty
        
        # Check if episodes are done
        dones = distances < 30.0
        self.episode_steps += 1
        max_steps_reached = self.episode_steps >= self.max_steps
        dones = dones | max_steps_reached
        
        # Reset done environments
        done_indices = np.where(dones)[0]
        if len(done_indices) > 0:
            self.enemy_positions[done_indices] = np.random.rand(len(done_indices), 2) * 700 + 50
            self.player_positions[done_indices] = np.random.rand(len(done_indices), 2) * 700 + 50
            self.episode_steps[done_indices] = 0
        
        self.observations = self._get_observations()
        return self.observations, rewards, dones
    
    def _get_observations(self) -> np.ndarray:
        """Get observations for CPU fallback.
        
        Returns:
            Observations [batch_size, 4]
        """
        # Relative position (player - enemy)
        relative_pos = self.player_positions - self.enemy_positions
        
        # Normalized enemy position
        normalized_enemy = self.enemy_positions.copy()
        normalized_enemy[:, 0] /= 800.0
        normalized_enemy[:, 1] /= 600.0
        
        # Concatenate observations
        observations = np.concatenate([relative_pos, normalized_enemy], axis=1)
        return observations.astype(np.float32)
    
    def get_observation_shape(self) -> Tuple[int, ...]:
        """Return the shape of observations."""
        if self.cuda_available:
            return self.envs[0].get_observation_shape()
        else:
            return (4,)  # relative_pos (2) + normalized_enemy (2)
    
    def get_action_shape(self) -> Tuple[int, ...]:
        """Return the shape of actions."""
        if self.cuda_available:
            return self.envs[0].get_action_shape()
        else:
            return (2,)  # x, y velocity
    
    def get_positions(self) -> Dict[str, np.ndarray]:
        """Get current positions for debugging/visualization."""
        if self.cuda_available:
            # Get positions from first environment as example
            debug_data = self.envs[0].get_debug_data()
            return debug_data
        else:
            return {
                'enemy_positions': self.enemy_positions.copy(),
                'player_positions': self.player_positions.copy()
            }
    
    def benchmark(self, num_steps: int = 1000) -> Dict[str, float]:
        """Benchmark the environment performance.
        
        Args:
            num_steps: Number of steps to benchmark
            
        Returns:
            Performance statistics
        """
        logger.info(f"Benchmarking environment for {num_steps} steps...")
        
        # Reset environment
        obs = self.reset()
        
        # Measure step time
        start_time = time.time()
        
        for _ in range(num_steps):
            # Random actions
            actions = np.random.randn(self.batch_size, 2)
            obs, rewards, dones = self.step(actions)
        
        end_time = time.time()
        
        # Calculate statistics
        total_time = end_time - start_time
        steps_per_second = num_steps / total_time
        env_steps_per_second = steps_per_second * self.batch_size
        
        stats = {
            'total_time': total_time,
            'steps_per_second': steps_per_second,
            'env_steps_per_second': env_steps_per_second,
            'batch_size': self.batch_size,
            'cuda_available': self.cuda_available
        }
        
        logger.info(f"Benchmark results:")
        logger.info(f"  Total time: {total_time:.2f}s")
        logger.info(f"  Steps/sec: {steps_per_second:.1f}")
        logger.info(f"  Env-steps/sec: {env_steps_per_second:.1f}")
        logger.info(f"  Using CUDA: {self.cuda_available}")
        
        return stats


def test_gpu_environment():
    """Test the GPU environment."""
    logger.info("=== Testing GPU Environment ===")
    
    try:
        # Test with small batch size first
        batch_size = 4
        env = GPURLEnvironment(batch_size=batch_size)
        
        logger.info(f"Created environment with batch_size={batch_size}")
        
        # Test reset
        obs = env.reset()
        logger.info(f"Reset successful, observation shape: {obs.shape}")
        
        # Test a few steps
        for i in range(5):
            actions = np.random.randn(batch_size, 2)
            obs, rewards, dones = env.step(actions)
            
            logger.info(f"Step {i+1}:")
            logger.info(f"  Observation shape: {obs.shape}")
            logger.info(f"  Reward range: [{rewards.min():.3f}, {rewards.max():.3f}]")
            logger.info(f"  Done count: {dones.sum()}")
        
        # Test observation and action shapes
        obs_shape = env.get_observation_shape()
        action_shape = env.get_action_shape()
        logger.info(f"Observation shape: {obs_shape}")
        logger.info(f"Action shape: {action_shape}")
        
        # Test positions
        positions = env.get_positions()
        logger.info(f"Position data available: {list(positions.keys())}")
        
        # Quick benchmark
        stats = env.benchmark(num_steps=50)
        
        logger.info("GPU environment test completed successfully!")
        return True
        
    except Exception as e:
        logger.error(f"GPU environment test failed: {e}")
        return False


def benchmark_performance():
    """Benchmark CPU vs different batch sizes."""
    logger.info("=== Benchmarking Performance ===")
    
    batch_sizes = [32, 64, 128, 256]
    num_steps = 100
    
    results = []
    
    for batch_size in batch_sizes:
        logger.info(f"Testing batch size {batch_size}")
        
        try:
            env = GPURLEnvironment(batch_size=batch_size)
            stats = env.benchmark(num_steps=num_steps)
            results.append(stats)
            
        except Exception as e:
            logger.error(f"Failed to benchmark batch size {batch_size}: {e}")
    
    # Log summary
    logger.info("\nPerformance Summary:")
    logger.info("Batch Size | Env-Steps/Sec | CUDA")
    logger.info("-" * 35)
    for stats in results:
        cuda_str = "Yes" if stats['cuda_available'] else "No"
        logger.info(f"{stats['batch_size']:9d} | {stats['env_steps_per_second']:12.0f} | {cuda_str}")
    
    return results


if __name__ == "__main__":
    # Run tests
    success = test_gpu_environment()
    if success:
        benchmark_performance()
    else:
        logger.error("Skipping benchmark due to test failure")
        logger.error("Skipping benchmark due to test failure")
