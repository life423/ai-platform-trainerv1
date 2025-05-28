"""
GPU-accelerated reinforcement learning environment wrapper.

This module provides a Gym-compatible wrapper around the native C++ CUDA implementation.
It handles the interface between Python RL libraries and the high-performance C++ backend.
"""
import os
import sys
import platform
import gym
import numpy as np
import torch
from typing import Dict, Any, Tuple, List, Optional, Union
import logging

# Set up logging
logger = logging.getLogger(__name__)

# Setup DLL loading for Windows BEFORE any imports
def setup_cuda_dlls():
    """Setup CUDA DLL directories for Windows"""
    if platform.system() == "Windows":
        cuda_paths = []
        
        # Check CUDA_PATH environment variable
        cuda_path = os.environ.get('CUDA_PATH')
        if cuda_path:
            cuda_paths.append(os.path.join(cuda_path, 'bin'))
        
        # Common CUDA installation paths
        cuda_paths.extend([
            r"C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.9\bin",
            r"C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.1\bin",
            r"C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v11.8\bin",
        ])
        
        for path in cuda_paths:
            if os.path.exists(path):
                try:
                    os.add_dll_directory(path)
                    logger.debug(f"Added CUDA DLL directory: {path}")
                except Exception as e:
                    logger.warning(f"Failed to add DLL directory {path}: {e}")

# Setup DLLs before importing the extension
setup_cuda_dlls()

# CRITICAL FIX: Import the C++ extension from the Release directory
# This prevents circular imports and ensures we get the compiled extension
def import_gpu_extension():
    """Import the GPU extension from the correct location"""
    # Get the directory of this file
    current_dir = os.path.dirname(os.path.abspath(__file__))
    
    # Look for the extension in the Release directory ONLY
    release_dir = os.path.join(os.path.dirname(current_dir), "Release")
    
    # Remove any existing module to force reimport
    if 'gpu_environment' in sys.modules:
        del sys.modules['gpu_environment']
    
    # Temporarily add ONLY Release directory (avoid circular import)
    original_path = sys.path.copy()
    
    try:
        # Clear path and add only Release directory
        sys.path = [release_dir] + [p for p in sys.path if p != current_dir]
        
        # Import the C++ extension (.pyd file)
        import gpu_environment as gpu_ext
        
        # Verify it's the C++ extension by checking for EnvironmentConfig
        if not hasattr(gpu_ext, 'EnvironmentConfig'):
            raise ImportError("Imported module is not the C++ extension")
        
        logger.info(f"Loaded GPU extension from: {gpu_ext.__file__}")
        return gpu_ext
        
    except ImportError as e:
        logger.error(f"Failed to import GPU extension: {e}")
        raise
    finally:
        # Restore original path
        sys.path = original_path

# Import the native extension
try:
    native_env = import_gpu_extension()
    HAS_GPU_ENV = True
    
    # Verify the interface
    required_attrs = ['EnvironmentConfig', 'Environment']
    for attr in required_attrs:
        if not hasattr(native_env, attr):
            raise ImportError(f"GPU environment module is missing required attribute: {attr}")
    
    logger.info("GPU environment loaded successfully")
    
except ImportError as e:
    logger.error(f"Native GPU environment not found: {e}")
    logger.info("You need to build the C++ extension first.")
    logger.info("Run: cd ai_platform_trainer/cpp && python setup.py build_ext --inplace")
    HAS_GPU_ENV = False
    
    # Create dummy classes for fallback
    class DummyConfig:
        def __init__(self):
            self.screen_width = 800
            self.screen_height = 600
            self.max_missiles = 5
            self.player_size = 50.0
            self.enemy_size = 50.0
            self.missile_size = 10.0
            self.player_speed = 5.0
            self.enemy_speed = 5.0
            self.missile_speed = 5.0
            self.missile_lifespan = 10000.0
            self.respawn_delay = 500.0
            self.max_steps = 1000
            self.enable_missile_avoidance = True
            self.missile_prediction_steps = 30
            self.missile_detection_radius = 250.0
            self.missile_danger_radius = 150.0
            self.evasion_strength = 2.5
    
    class DummyEnvironment:
        def __init__(self, config):
            self.config = config
            
        def reset(self, seed=0):
            return np.zeros(10, dtype=np.float32)
            
        def step(self, action):
            obs = np.zeros(10, dtype=np.float32)
            return obs, 0.0, False, False, {}
            
        def get_observation_shape(self):
            return [10]
            
        def get_action_shape(self):
            return [2]
    
    class DummyModule:
        EnvironmentConfig = DummyConfig
        Environment = DummyEnvironment
    
    native_env = DummyModule()


# Define a gym-compatible wrapper
class GPUGameEnv(gym.Env):
    """
    A gym environment that wraps the CUDA-accelerated game environment.
    
    This class provides a standard gym interface for training RL agents
    with GPU-accelerated physics simulations for missile avoidance.
    """
    metadata = {'render.modes': ['human', 'rgb_array']}
    
    def __init__(self, config=None):
        """
        Initialize the GPU game environment.
        
        Args:
            config: Configuration parameters for the environment
        """
        if not HAS_GPU_ENV:
            logger.warning("GPU environment extension not available - using CPU fallback")
        
        # Create config object if none provided
        if config is None:
            config = self._default_config()
            
        # Create the native environment
        try:
            self.env = native_env.Environment(config)
            logger.info("Created GPU-accelerated environment")
        except Exception as e:
            logger.error(f"Failed to create environment: {e}")
            raise RuntimeError(f"Failed to create GPU environment: {e}")
        
        # Set up gym spaces
        obs_shape = self.env.get_observation_shape()
        action_shape = self.env.get_action_shape()
        
        self.observation_space = gym.spaces.Box(
            low=-float('inf'),
            high=float('inf'),
            shape=obs_shape,
            dtype=np.float32
        )
        
        self.action_space = gym.spaces.Box(
            low=-1.0,
            high=1.0,
            shape=action_shape,
            dtype=np.float32
        )
        
        # State variables
        self.current_obs = None
        self.steps = 0
        self.episode_reward = 0.0
        
    def reset(self, seed=None, options=None):
        """Reset the environment and return initial observation"""
        seed = seed if seed is not None else np.random.randint(0, 2**31 - 1)
        
        try:
            self.current_obs = self.env.reset(int(seed))
            self.steps = 0
            self.episode_reward = 0.0
            return self.current_obs
        except Exception as e:
            logger.error(f"Failed to reset environment: {e}")
            raise
    
    def step(self, action):
        """Take a step in the environment"""
        # Convert action to numpy array if needed
        if isinstance(action, torch.Tensor):
            action = action.cpu().numpy()
        
        action = np.asarray(action, dtype=np.float32)
        
        try:
            # Take step in native environment
            obs, reward, done, truncated, info = self.env.step(action)
            
            # Update state variables
            self.current_obs = obs
            self.steps += 1
            self.episode_reward += reward
            info['episode_reward'] = self.episode_reward
            info['steps'] = self.steps
            
            return obs, reward, done, truncated, info
            
        except Exception as e:
            logger.error(f"Failed to step environment: {e}")
            raise
    
    def render(self, mode='human'):
        """Render the environment (not implemented for headless training)"""
        if mode == 'rgb_array':
            # Could implement visualization here
            return np.zeros((600, 800, 3), dtype=np.uint8)
        return None
    
    def close(self):
        """Clean up environment resources"""
        pass
    
    def seed(self, seed=None):
        """Set random seed"""
        self._seed = seed if seed is not None else np.random.randint(0, 2**31 - 1)
        return [self._seed]
    
    @staticmethod
    def _default_config():
        """Create default environment configuration"""
        config = native_env.EnvironmentConfig()
        config.screen_width = 800
        config.screen_height = 600
        config.max_missiles = 5
        config.player_size = 50.0
        config.enemy_size = 50.0
        config.missile_size = 10.0
        config.player_speed = 5.0
        config.enemy_speed = 5.0
        config.missile_speed = 5.0
        config.missile_lifespan = 10000.0
        config.respawn_delay = 500.0
        config.max_steps = 1000
        config.enable_missile_avoidance = True
        config.missile_prediction_steps = 30
        config.missile_detection_radius = 250.0
        config.missile_danger_radius = 150.0
        config.evasion_strength = 2.5
        return config


def make_env(config=None):
    """Create a GPU-accelerated game environment"""
    return GPUGameEnv(config)


def create_vectorized_env(num_envs=4, config=None):
    """Create multiple environments for parallel training"""
    from stable_baselines3.common.vec_env import DummyVecEnv
    
    def make_env_fn():
        return make_env(config)
    
    return DummyVecEnv([make_env_fn for _ in range(num_envs)])


# Quick test if run directly
if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    
    print("Testing GPU environment wrapper...")
    
    if HAS_GPU_ENV:
        # Test basic functionality
        env = make_env()
        print(f"✓ Created environment")
        
        obs = env.reset()
        print(f"✓ Reset: observation shape = {obs.shape}")
        
        for i in range(5):
            action = env.action_space.sample()
            obs, reward, done, truncated, info = env.step(action)
            print(f"  Step {i}: reward = {reward:.3f}, done = {done}")
            
            if done:
                obs = env.reset()
        
        print("\n✓ GPU environment wrapper is working!")
    else:
        print("✗ GPU environment not available")
