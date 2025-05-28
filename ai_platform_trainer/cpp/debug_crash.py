"""
Crash detection and debugging script for GPU training.
Identifies exactly where and why the training crashes.
"""
import os
import sys
import time
import logging
import traceback
import faulthandler
import psutil
import gc
import gymnasium as gym
from typing import Optional

# Enable faulthandler for crash detection
faulthandler.enable()

# Setup detailed logging
logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler('debug_crash.log')
    ]
)
logger = logging.getLogger(__name__)

# Add paths
script_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, script_dir)
sys.path.insert(0, os.path.join(os.path.dirname(script_dir), "Release"))

# Import modules
from gpu_env_wrapper import make_env, HAS_GPU_ENV
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv


class CrashDetectionWrapper(gym.Wrapper):
    """Wrapper to detect and log exactly where crashes occur"""
    
    def __init__(self, env):
        super().__init__(env)
        self.step_count = 0
        self.episode_count = 0
        self.reset_count = 0
        
    def reset(self, **kwargs):
        self.reset_count += 1
        logger.debug(f"RESET #{self.reset_count} called")
        try:
            result = self.env.reset(**kwargs)
            logger.debug(f"RESET #{self.reset_count} successful")
            return result
        except Exception as e:
            logger.error(f"RESET #{self.reset_count} FAILED: {e}")
            raise
    
    def step(self, action):
        self.step_count += 1
        if self.step_count % 10 == 0:
            logger.debug(f"Step {self.step_count} - Memory: {self._get_memory_usage():.1f}MB")
        
        try:
            result = self.env.step(action)
            obs, reward, terminated, truncated, info = result
            
            if terminated or truncated:
                self.episode_count += 1
                logger.debug(f"Episode {self.episode_count} ended at step {self.step_count}")
            
            return result
        except Exception as e:
            logger.error(f"STEP {self.step_count} FAILED: {e}")
            traceback.print_exc()
            raise
    
    def _get_memory_usage(self):
        """Get current memory usage in MB"""
        process = psutil.Process()
        return process.memory_info().rss / 1024 / 1024


class PPOCrashDetector:
    """Monitor PPO training for crashes"""
    
    def __init__(self):
        self.iteration_count = 0
        self.timestep_count = 0
        
    def on_training_start(self):
        logger.info("PPO training starting...")
        
    def on_step(self):
        self.timestep_count += 1
        if self.timestep_count % 64 == 0:  # Every batch
            self.iteration_count += 1
            logger.info(f"Completed iteration {self.iteration_count} ({self.timestep_count} total steps)")
            
            # Force garbage collection
            gc.collect()
            
            # Log memory usage
            process = psutil.Process()
            memory_mb = process.memory_info().rss / 1024 / 1024
            logger.info(f"Memory usage: {memory_mb:.1f}MB")
            
            # Check for potential issues
            if memory_mb > 2000:  # 2GB threshold
                logger.warning(f"High memory usage detected: {memory_mb:.1f}MB")


def test_progressive_timesteps():
    """Test increasing timestep counts to find the breaking point"""
    logger.info("=== PROGRESSIVE TIMESTEP TESTING ===")
    
    if not HAS_GPU_ENV:
        logger.error("GPU environment not available!")
        return False
    
    # Test these timestep counts progressively
    test_timesteps = [64, 100, 128, 200, 256, 500, 1000]
    
    for timesteps in test_timesteps:
        logger.info(f"\n--- Testing {timesteps} timesteps ---")
        
        try:
            # Create fresh environment
            env = DummyVecEnv([lambda: CrashDetectionWrapper(make_env())])
            
            # Create PPO with minimal settings
            model = PPO(
                "MlpPolicy",
                env,
                verbose=0,  # Reduce noise
                learning_rate=3e-4,
                n_steps=64,
                batch_size=32,
                device="cpu"
            )
            
            detector = PPOCrashDetector()
            detector.on_training_start()
            
            logger.info(f"Starting training for {timesteps} timesteps...")
            start_time = time.time()
            
            # Train with comprehensive error handling
            model.learn(
                total_timesteps=timesteps,
                progress_bar=False
            )
            
            training_time = time.time() - start_time
            logger.info(f"✅ {timesteps} timesteps SUCCESSFUL in {training_time:.2f}s")
            
            # Clean up
            env.close()
            del model
            del env
            gc.collect()
            
            # Brief pause between tests
            time.sleep(1)
            
        except Exception as e:
            logger.error(f"❌ {timesteps} timesteps FAILED: {e}")
            logger.error(f"Crash details: {traceback.format_exc()}")
            return timesteps  # Return the failing timestep count
        
        except SystemExit:
            logger.error(f"❌ {timesteps} timesteps CRASHED (SystemExit)")
            return timesteps
            
        except KeyboardInterrupt:
            logger.error(f"❌ {timesteps} timesteps INTERRUPTED")
            return timesteps
    
    logger.info("✅ All timestep tests passed!")
    return None


def test_environment_isolation():
    """Test the environment in isolation without PPO"""
    logger.info("=== ENVIRONMENT ISOLATION TEST ===")
    
    try:
        env = CrashDetectionWrapper(make_env())
        
        logger.info("Testing 1000 environment steps...")
        obs, info = env.reset()
        
        for i in range(1000):
            action = env.action_space.sample()
            obs, reward, terminated, truncated, info = env.step(action)
            
            if terminated or truncated:
                obs, info = env.reset()
        
        logger.info("✅ Environment isolation test PASSED")
        return True
        
    except Exception as e:
        logger.error(f"❌ Environment isolation test FAILED: {e}")
        traceback.print_exc()
        return False


def main():
    """Main debugging routine"""
    logger.info("🔍 Starting Comprehensive Crash Detection")
    logger.info("=" * 60)
    
    # Test 1: Environment isolation
    logger.info("\n1. Testing environment in isolation...")
    env_ok = test_environment_isolation()
    
    if not env_ok:
        logger.error("Environment has issues - stopping here")
        return
    
    # Test 2: Progressive timestep testing
    logger.info("\n2. Testing progressive timesteps...")
    failing_timesteps = test_progressive_timesteps()
    
    if failing_timesteps:
        logger.error(f"🎯 CRASH IDENTIFIED: Training fails at {failing_timesteps} timesteps")
        logger.error("This suggests the issue occurs during PPO's 3rd iteration")
        logger.error("Likely causes:")
        logger.error("  - CUDA memory corruption")
        logger.error("  - C++ object lifetime issues")
        logger.error("  - GPU context problems")
        logger.error("  - Buffer overflow in environment")
    else:
        logger.info("🎉 No crashes detected in progressive testing!")
    
    logger.info("\n🔍 Crash detection complete. Check debug_crash.log for details.")


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        logger.error(f"Debug script itself crashed: {e}")
        traceback.print_exc()
    finally:
        logger.info("Debug session ended.")
