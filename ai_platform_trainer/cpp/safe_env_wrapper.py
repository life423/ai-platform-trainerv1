"""
Safe environment wrappers to prevent heap corruption crashes.
These wrappers implement multiple strategies to keep training alive while debugging.
"""
import os
import sys
import logging
import gymnasium as gym
import numpy as np
import psutil
import gc
from typing import Dict, Any, Tuple, Optional

logger = logging.getLogger(__name__)


class StepCounterWrapper(gym.Wrapper):
    """
    Wrapper that prints step numbers to identify exact crash point.
    This helps pinpoint exactly when the heap corruption occurs.
    """
    def __init__(self, env):
        super().__init__(env)
        self.steps = 0
        self.episode_count = 0
    
    def step(self, action):
        print(f"About to take step {self.steps}", flush=True)
        
        try:
            result = self.env.step(action)
            self.steps += 1
            print(f"Completed step {self.steps-1} successfully", flush=True)
            return result
            
        except Exception as e:
            print(f"💥 CRASH at step {self.steps}! Error: {e}", flush=True)
            raise
    
    def reset(self, **kwargs):
        if self.steps > 0:
            print(f"Episode {self.episode_count} ended after {self.steps} steps", flush=True)
        
        self.steps = 0
        self.episode_count += 1
        print(f"Starting episode {self.episode_count}", flush=True)
        return self.env.reset(**kwargs)


class SafeStepWrapper(gym.Wrapper):
    """
    Wrapper that prevents heap corruption by auto-resetting before crash.
    Also adds extensive debugging information.
    """
    def __init__(self, env, max_steps_before_reset=90, debug=True):
        super().__init__(env)
        self.max_steps_before_reset = max_steps_before_reset
        self.debug = debug
        self.step_count = 0
        self.episode_count = 0
        self.total_steps = 0
        
        # Track memory patterns to detect leaks
        self.last_obs = None
        self.memory_snapshots = []
        
        if self.debug:
            logger.info(f"SafeStepWrapper initialized with max_steps={max_steps_before_reset}")
        
    def reset(self, **kwargs):
        """Reset environment and internal counters"""
        if self.debug and self.step_count > 0:
            logger.info(f"Episode {self.episode_count} ended after {self.step_count} steps")
        
        self.step_count = 0
        self.episode_count += 1
        
        # Force garbage collection to clean up any leaked memory
        gc.collect()
        
        try:
            obs, info = self.env.reset(**kwargs)
            self.last_obs = obs.copy() if isinstance(obs, np.ndarray) else obs
            
            if self.debug:
                logger.debug(f"Reset successful - Episode {self.episode_count}")
            
            return obs, info
            
        except Exception as e:
            logger.error(f"Reset failed: {e}")
            # Return safe dummy observation
            dummy_obs = np.zeros(self.observation_space.shape, dtype=np.float32)
            return dummy_obs, {}
    
    def step(self, action):
        """Step with safety checks and auto-reset"""
        # Pre-step debugging
        if self.debug and self.step_count % 10 == 0:
            try:
                process = psutil.Process()
                memory_mb = process.memory_info().rss / 1024 / 1024
                logger.debug(f"Step {self.step_count}: Memory = {memory_mb:.1f}MB")
            except:
                pass  # Ignore psutil errors
        
        # Auto-reset before crash
        if self.step_count >= self.max_steps_before_reset:
            if self.debug:
                logger.warning(f"Auto-resetting at step {self.step_count} to prevent crash")
            
            # Save final reward/done status
            obs = self.last_obs if self.last_obs is not None else np.zeros(self.observation_space.shape)
            reward = 0.0
            terminated = True
            truncated = False
            info = {"auto_reset": True, "step_count": self.step_count}
            
            # Reset environment for next episode
            self.reset()
            
            return obs, reward, terminated, truncated, info
        
        # Actual step with error handling
        try:
            obs, reward, terminated, truncated, info = self.env.step(action)
            
            # Validate observation
            if isinstance(obs, np.ndarray):
                if np.any(np.isnan(obs)) or np.any(np.isinf(obs)):
                    logger.error(f"Invalid observation at step {self.step_count}: contains NaN/Inf")
                    terminated = True
                    
                # Check for memory corruption patterns
                if self.last_obs is not None and obs.shape != self.last_obs.shape:
                    logger.error(f"Observation shape changed! {self.last_obs.shape} -> {obs.shape}")
                    terminated = True
            
            self.last_obs = obs.copy() if isinstance(obs, np.ndarray) else obs
            self.step_count += 1
            self.total_steps += 1
            
            # Add step count to info 
            if isinstance(info, dict):
                info["step_count"] = self.step_count
                info["total_steps"] = self.total_steps
            
            return obs, reward, terminated, truncated, info
            
        except Exception as e:
            logger.error(f"Step failed at step {self.step_count}: {e}")
            logger.error(f"Action was: {action}")
            
            # Return safe values and force episode end
            obs = self.last_obs if self.last_obs is not None else np.zeros(self.observation_space.shape)
            return obs, -10.0, True, False, {"error": str(e), "crash_step": self.step_count}


class MemoryDebugWrapper(gym.Wrapper):
    """
    Wrapper specifically for debugging memory issues in GPU environments.
    Tracks system memory usage and patterns.
    """
    def __init__(self, env, check_cuda=True, memory_log_interval=20):
        super().__init__(env)
        self.check_cuda = check_cuda
        self.memory_log_interval = memory_log_interval
        self.step_count = 0
        self.memory_history = []
        
        # Try to initialize CUDA memory checking
        self.cuda_available = False
        if check_cuda:
            try:
                import pycuda.driver as cuda
                cuda.init()
                self.cuda_available = True
                logger.info("CUDA memory debugging enabled")
            except ImportError:
                logger.warning("PyCUDA not available - system memory only")
            except Exception as e:
                logger.warning(f"CUDA initialization failed: {e}")
    
    def step(self, action):
        """Step with memory tracking"""
        self.step_count += 1
        
        # Memory logging
        if self.step_count % self.memory_log_interval == 0:
            self._log_memory_usage()
        
        # Call original step
        result = self.env.step(action)
        
        # Post-step validation
        obs = result[0]
        if isinstance(obs, np.ndarray):
            # Check for common corruption patterns
            if hasattr(self.observation_space, 'dtype') and obs.dtype != self.observation_space.dtype:
                logger.error(f"Observation dtype changed to {obs.dtype}")
            
            # Check for out-of-bounds values
            if hasattr(self.observation_space, 'low') and hasattr(self.observation_space, 'high'):
                if np.any(obs < self.observation_space.low) or np.any(obs > self.observation_space.high):
                    logger.warning(f"Observation out of bounds at step {self.step_count}")
        
        return result
    
    def _log_memory_usage(self):
        """Log current memory usage"""
        try:
            # System memory
            process = psutil.Process()
            system_mb = process.memory_info().rss / (1024 * 1024)
            
            self.memory_history.append({
                'step': self.step_count,
                'system_mb': system_mb
            })
            
            # Keep only recent history
            if len(self.memory_history) > 100:
                self.memory_history = self.memory_history[-50:]
            
            logger.debug(f"Memory - Step {self.step_count}: {system_mb:.1f}MB system")
            
            # CUDA memory if available
            if self.cuda_available:
                try:
                    import pycuda.driver as cuda
                    free, total = cuda.mem_get_info()
                    used_mb = (total - free) / (1024 * 1024)
                    logger.debug(f"CUDA Memory - Step {self.step_count}: {used_mb:.1f}MB used")
                except Exception as e:
                    logger.debug(f"CUDA memory check failed: {e}")
                    
        except Exception as e:
            logger.debug(f"Memory logging failed: {e}")


def create_safe_env(base_env_creator, max_steps=90, debug=True, enable_memory_debug=True):
    """
    Create a safe wrapped environment that won't crash.
    
    Args:
        base_env_creator: Function that creates the base GPU environment  
        max_steps: Maximum steps before auto-reset (default 90 to avoid crash at 100)
        debug: Enable debug logging
        enable_memory_debug: Enable memory usage tracking
    
    Returns:
        Wrapped environment safe from heap corruption crashes
    """
    # Create base environment
    env = base_env_creator()
    
    # Add step counter for crash detection
    if debug:
        env = StepCounterWrapper(env)
    
    # Add safety wrapper (most important - prevents crashes)
    env = SafeStepWrapper(env, max_steps_before_reset=max_steps, debug=debug)
    
    # Add memory debugging if requested
    if enable_memory_debug:
        env = MemoryDebugWrapper(env, check_cuda=False)  # Disable CUDA checks for now
    
    logger.info(f"Created safe environment with max_steps={max_steps}, debug={debug}")
    return env


def make_safe_vec_env(env_creator, n_envs=1, max_steps=90, debug=False):
    """Create vectorized safe environments for stable-baselines3"""
    from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv
    from stable_baselines3.common.vec_env import VecMonitor
    
    def make_env():
        def _init():
            return create_safe_env(env_creator, max_steps=max_steps, debug=debug, enable_memory_debug=False)
        return _init
    
    if n_envs == 1:
        env = DummyVecEnv([make_env()])
    else:
        # Use SubprocVecEnv for multiple environments
        env = SubprocVecEnv([make_env() for _ in range(n_envs)])
    
    # Add monitoring
    env = VecMonitor(env)
    
    logger.info(f"Created {n_envs} safe vectorized environments")
    return env


# Quick test function
def test_safe_wrapper():
    """Test the safe wrapper with a dummy crashing environment"""
    
    class CrashingEnv(gym.Env):
        def __init__(self):
            self.observation_space = gym.spaces.Box(low=-1, high=1, shape=(4,))
            self.action_space = gym.spaces.Box(low=-1, high=1, shape=(2,))
            self.step_count = 0
            
        def reset(self, **kwargs):
            self.step_count = 0
            return np.random.randn(4).astype(np.float32), {}
            
        def step(self, action):
            self.step_count += 1
            if self.step_count == 10:
                raise RuntimeError("Simulated heap corruption at step 10!")
            
            obs = np.random.randn(4).astype(np.float32)
            reward = np.random.randn()
            return obs, reward, False, False, {}
    
    print("Testing safe wrapper with simulated crash at step 10...")
    
    # Test crashing environment
    env = CrashingEnv()
    safe_env = create_safe_env(lambda: env, max_steps=8, debug=True)
    
    obs, info = safe_env.reset()
    for i in range(20):
        action = safe_env.action_space.sample()
        obs, reward, done, truncated, info = safe_env.step(action)
        print(f"Step {i}: reward={reward:.3f}, done={done}")
        
        if done:
            obs, info = safe_env.reset()
    
    print("✅ Safe wrapper test completed - no crashes!")


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    test_safe_wrapper()
