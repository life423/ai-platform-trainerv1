"""
Working training script for missile avoidance with all fixes.
Save as: train_gpu_fixed.py
"""
import os
import sys
import time
import argparse
import numpy as np
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import CheckpointCallback
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize
import torch
import logging

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Add paths
script_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, script_dir)
sys.path.insert(0, os.path.join(os.path.dirname(script_dir), "Release"))

# Import from wrapper
from gpu_env_wrapper import make_env, HAS_GPU_ENV, native_env


def train_missile_avoidance(
    num_envs=4,
    total_timesteps=100000,
    save_dir="models/gpu_rl",
    use_tensorboard=False
):
    """Train a PPO model for missile avoidance"""
    
    if not HAS_GPU_ENV:
        raise RuntimeError("GPU environment not available!")
    
    # Create directories
    os.makedirs(save_dir, exist_ok=True)
    
    # Create environment(s)
    logger.info(f"Creating {num_envs} environments...")
    
    if num_envs == 1:
        # Single environment
        env = DummyVecEnv([lambda: make_env()])
    else:
        # Multiple environments (use DummyVecEnv to avoid multiprocessing issues)
        env = DummyVecEnv([lambda: make_env() for _ in range(num_envs)])
    
    # Normalize observations and rewards
    env = VecNormalize(env, norm_obs=True, norm_reward=True)
    
    # PPO runs better on CPU for MLP policies
    device = "cpu"
    logger.info(f"Using device: {device} (PPO with MLP is faster on CPU)")
    
    # Create PPO model
    model = PPO(
        "MlpPolicy",
        env,
        verbose=1,
        learning_rate=3e-4,
        n_steps=2048 // num_envs,  # Adjust for number of environments
        batch_size=64,
        n_epochs=10,
        gamma=0.99,
        gae_lambda=0.95,
        clip_range=0.2,
        ent_coef=0.01,
        device=device,
        tensorboard_log="./logs/" if use_tensorboard else None,
    )
    
    # Setup callbacks
    checkpoint_callback = CheckpointCallback(
        save_freq=10000 // num_envs,
        save_path=save_dir,
        name_prefix="missile_avoidance",
        save_vecnormalize=True
    )
    
    # Train
    logger.info(f"Starting training for {total_timesteps} timesteps...")
    start_time = time.time()
    
    try:
        model.learn(
            total_timesteps=total_timesteps,
            callback=checkpoint_callback,
            progress_bar=True
        )
        
        training_time = time.time() - start_time
        logger.info(f"✅ Training completed in {training_time:.1f} seconds")
        logger.info(f"Average: {total_timesteps/training_time:.0f} steps/second")
        
        # Save final model
        final_path = os.path.join(save_dir, "final_model")
        model.save(final_path)
        env.save(os.path.join(save_dir, "vec_normalize.pkl"))
        logger.info(f"Model saved to: {final_path}")
        
        # Quick evaluation
        logger.info("\nQuick evaluation...")
        obs = env.reset()
        total_reward = 0
        for i in range(100):
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, done, info = env.step(action)
            total_reward += reward[0]
            if done[0]:
                logger.info(f"Episode reward: {total_reward:.2f}")
                total_reward = 0
                obs = env.reset()
        
    except Exception as e:
        logger.error(f"Training error: {e}")
        import traceback
        traceback.print_exc()
    finally:
        env.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--num-envs', type=int, default=1, help='Number of parallel environments')
    parser.add_argument('--timesteps', type=int, default=10000, help='Total timesteps')
    parser.add_argument('--save-dir', type=str, default='models/gpu_rl')
    parser.add_argument('--tensorboard', action='store_true', help='Enable tensorboard logging')
    
    args = parser.parse_args()
    
    logger.info("GPU-Accelerated Missile Avoidance Training")
    logger.info("=" * 50)
    
    # Note about GPU usage
    logger.info("NOTE: The GPU accelerates the environment physics simulation.")
    logger.info("      PPO training (neural network) runs on CPU by default")
    logger.info("      as it's faster for small networks.")
    
    train_missile_avoidance(
        num_envs=args.num_envs,
        total_timesteps=args.timesteps,
        save_dir=args.save_dir,
        use_tensorboard=args.tensorboard
    )
