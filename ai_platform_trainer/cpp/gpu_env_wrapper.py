"""
GPU-accelerated reinforcement learning environment wrapper.
Fixed version with Gymnasium compatibility and proper reset signature.

Save this as: gpu_env_wrapper.py (replacing the existing one)
"""
import os
import sys
import platform
import gymnasium as gym  # Use gymnasium instead of gym
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

# Add the Release directory to Python path
current_dir = os.path.dirname(os.path.abspath(__file__))
release_dir = os.path.join(os.path.dirname(current_dir), "Release")

if os.path.exists(release_dir) and release_dir not in sys.path:
    sys.path.insert(0, release_dir)
    logger.debug(f"Added Release directory to path: {release_dir}")

# Now import the C++ extension directly
try:
    # This will import the .pyd file, not any .py file
    import gpu_environment as native_env
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


# Define a gymnasium-compatible wrapper
class GPUGameEnv(gym.Env):
    """
    A gymnasium environment that wraps the CUDA-accelerated game environment.
    """
    metadata = {'render_modes': ['human', 'rgb_array']}
    
    def __init__(self, config=None):
        """Initialize the GPU game environment"""
        super().__init__()
        
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
        
        # Set up gymnasium spaces
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
        """
        Reset the environment and return initial observation and info.
        
        This follows the Gymnasium API which expects (observation, info).
        """
        super().reset(seed=seed)
        
        if seed is None:
            seed = np.random.randint(0, 2**31 - 1)
        
        self.current_obs = self.env.reset(int(seed))
        self.steps = 0
        self.episode_reward = 0.0
        
        # Return observation and info dict (Gymnasium API)
        info = {}
        return self.current_obs, info
    
    def step(self, action):
        """
        Take a step in the environment.
        
        Returns: (observation, reward, terminated, truncated, info)
        Following the Gymnasium API.
        """
        # Convert action to numpy array if needed
        if isinstance(action, torch.Tensor):
            action = action.cpu().numpy()
        
        action = np.asarray(action, dtype=np.float32)
        
        # Take step in native environment
        obs, reward, done, truncated, info = self.env.step(action)
        
        # Update state variables
        self.current_obs = obs
        self.steps += 1
        self.episode_reward += reward
        info['episode_reward'] = self.episode_reward
        info['steps'] = self.steps
        
        # In Gymnasium, 'done' is split into 'terminated' and 'truncated'
        terminated = done and not truncated
        
        return obs, reward, terminated, truncated, info
    
    def render(self):
        """Render the environment"""
        # Not implemented for headless training
        return None
    
    def close(self):
        """Clean up environment resources"""
        pass
    
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
        
        # Test reset with new API
        obs, info = env.reset()
        print(f"✓ Reset: observation shape = {obs.shape}, info = {info}")
        
        for i in range(5):
            action = env.action_space.sample()
            obs, reward, terminated, truncated, info = env.step(action)
            print(f"  Step {i}: reward = {reward:.3f}, terminated = {terminated}, truncated = {truncated}")
            
            if terminated or truncated:
                obs, info = env.reset()
        
        print("\n✓ GPU environment wrapper is working with Gymnasium API!")
    else:
        print("✗ GPU environment not available")
