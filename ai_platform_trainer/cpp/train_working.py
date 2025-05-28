"""
Working training script based on debug findings.
No progress bars, optimal batch sizes, no VecNormalize issues.
"""
import os
import sys
import time
import argparse
import logging

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Add paths
script_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, script_dir)
sys.path.insert(0, os.path.join(os.path.dirname(script_dir), "Release"))

# Import from wrapper
from gpu_env_wrapper import make_env, HAS_GPU_ENV
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.common.callbacks import CheckpointCallback


def train_missile_avoidance(
    num_envs=1,
    total_timesteps=10000,
    save_dir="models/gpu_rl"
):
    """Train a PPO model for missile avoidance - optimized version"""
    
    if not HAS_GPU_ENV:
        raise RuntimeError("GPU environment not available!")
    
    # Create directories
    os.makedirs(save_dir, exist_ok=True)
    
    logger.info(f"Creating {num_envs} environments...")
    if num_envs == 1:
        env = DummyVecEnv([lambda: make_env()])
    else:
        env = DummyVecEnv([lambda: make_env() for _ in range(num_envs)])
    
    # Don't use VecNormalize - it can cause hanging
    logger.info("Using device: cpu (optimal for MLP)")
    
    # Create PPO model with conservative settings that work
    model = PPO(
        "MlpPolicy",
        env,
        verbose=1,
        learning_rate=3e-4,
        n_steps=64 * num_envs,  # Small, safe batch size
        batch_size=32,          # Small batch
        n_epochs=4,             # Fewer epochs
        gamma=0.99,
        gae_lambda=0.95,
        clip_range=0.2,
        ent_coef=0.01,
        device="cpu"
    )
    
    # Setup checkpoint callback
    checkpoint_callback = CheckpointCallback(
        save_freq=max(1000 // num_envs, 100),
        save_path=save_dir,
        name_prefix="missile_avoidance"
    )
    
    # Train WITHOUT progress bar (this was causing the hang)
    logger.info(f"Starting training for {total_timesteps} timesteps...")
    logger.info("Note: No progress bar to avoid hanging issues")
    
    start_time = time.time()
    
    try:
        model.learn(
            total_timesteps=total_timesteps,
            callback=checkpoint_callback,
            progress_bar=False  # This was the culprit!
        )
        
        training_time = time.time() - start_time
        logger.info(f"✅ Training completed in {training_time:.1f} seconds")
        logger.info(f"Average: {total_timesteps/training_time:.0f} steps/second")
        
        # Save final model
        final_path = os.path.join(save_dir, "final_model")
        model.save(final_path)
        logger.info(f"Model saved to: {final_path}")
        
        # Quick evaluation
        logger.info("\nEvaluating trained model...")
        obs = env.reset()
        episode_rewards = []
        current_reward = 0
        
        for i in range(200):
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, done, info = env.step(action)
            current_reward += reward[0]
            
            if done[0]:
                episode_rewards.append(current_reward)
                logger.info(f"Episode {len(episode_rewards)}: reward = {current_reward:.2f}")
                current_reward = 0
                obs = env.reset()
                
                if len(episode_rewards) >= 5:  # Evaluate 5 episodes
                    break
        
        if episode_rewards:
            avg_reward = sum(episode_rewards) / len(episode_rewards)
            logger.info(f"Average reward over {len(episode_rewards)} episodes: {avg_reward:.2f}")
        
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
    parser.add_argument('--save-dir', type=str, default='models/gpu_rl', help='Save directory')
    
    args = parser.parse_args()
    
    logger.info("🚀 GPU-Accelerated Missile Avoidance Training")
    logger.info("=" * 50)
    logger.info("✅ Environment: GPU-accelerated physics")
    logger.info("✅ Neural Network: CPU-optimized PPO")
    logger.info("✅ Configuration: Stable, no-hang settings")
    logger.info("")
    
    train_missile_avoidance(
        num_envs=args.num_envs,
        total_timesteps=args.timesteps,
        save_dir=args.save_dir
    )
    
    logger.info("\n🎯 Training complete! Your AI agent is ready.")
