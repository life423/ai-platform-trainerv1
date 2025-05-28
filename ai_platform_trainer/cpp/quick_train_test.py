"""
Quick training test with all fixes applied.
Save as: quick_train_test.py
"""
import os
import sys
import logging

# Setup paths
script_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, script_dir)
sys.path.insert(0, os.path.join(os.path.dirname(script_dir), "Release"))

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def train_simple():
    """Simple training without multiprocessing to test"""
    from gpu_env_wrapper import make_env, HAS_GPU_ENV, native_env
    from stable_baselines3 import PPO
    from stable_baselines3.common.vec_env import DummyVecEnv
    import torch
    
    if not HAS_GPU_ENV:
        logger.error("GPU environment not available!")
        return
    
    # Create single environment (no multiprocessing)
    logger.info("Creating environment...")
    env = DummyVecEnv([lambda: make_env()])
    
    # Use CPU for PPO (it's faster for MLP policies)
    device = "cpu"  # PPO with MLP is actually faster on CPU
    logger.info(f"Using device: {device}")
    
    # Create simple PPO model
    model = PPO(
        "MlpPolicy",
        env,
        verbose=1,
        learning_rate=3e-4,
        n_steps=128,  # Smaller for quick test
        batch_size=64,
        device=device,
        tensorboard_log=None  # Disable tensorboard for now
    )
    
    # Train for a short time
    logger.info("Starting training...")
    try:
        model.learn(total_timesteps=1000)
        logger.info("✅ Training successful!")
        
        # Test the trained model
        obs = env.reset()
        for i in range(10):
            action, _ = model.predict(obs)
            obs, reward, done, info = env.step(action)
            logger.info(f"Step {i}: reward = {reward[0]:.3f}")
            if done[0]:
                obs = env.reset()
                
    except Exception as e:
        logger.error(f"Training failed: {e}")
        import traceback
        traceback.print_exc()
    finally:
        env.close()

if __name__ == "__main__":
    print("=" * 60)
    print("Quick GPU Training Test")
    print("=" * 60)
    train_simple()
