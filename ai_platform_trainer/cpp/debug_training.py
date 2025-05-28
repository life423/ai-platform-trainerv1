"""
Debug script to find where training is hanging.
"""
import os
import sys
import time
import logging

# Setup logging with more detail
logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Add paths
script_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, script_dir)
sys.path.insert(0, os.path.join(os.path.dirname(script_dir), "Release"))

def test_environment():
    """Test basic environment functionality"""
    logger.info("=== Testing Environment ===")
    
    try:
        from gpu_env_wrapper import make_env, HAS_GPU_ENV
        
        if not HAS_GPU_ENV:
            logger.error("GPU environment not available!")
            return False
        
        logger.info("1. Creating environment...")
        env = make_env()
        logger.info("✓ Environment created")
        
        logger.info("2. Testing reset...")
        obs, info = env.reset()
        logger.info(f"✓ Reset successful: obs shape = {obs.shape}")
        
        logger.info("3. Testing single step...")
        action = env.action_space.sample()
        obs, reward, terminated, truncated, info = env.step(action)
        logger.info(f"✓ Step successful: reward = {reward:.3f}")
        
        logger.info("4. Testing multiple steps...")
        for i in range(10):
            action = env.action_space.sample()
            obs, reward, terminated, truncated, info = env.step(action)
            logger.info(f"   Step {i}: reward = {reward:.3f}, done = {terminated or truncated}")
            if terminated or truncated:
                obs, info = env.reset()
                logger.info("   Reset after episode end")
        
        logger.info("✓ Environment test complete")
        return True
        
    except Exception as e:
        logger.error(f"Environment test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_vectorized_env():
    """Test vectorized environment"""
    logger.info("=== Testing Vectorized Environment ===")
    
    try:
        from gpu_env_wrapper import make_env
        from stable_baselines3.common.vec_env import DummyVecEnv
        
        logger.info("1. Creating vectorized environment...")
        env = DummyVecEnv([lambda: make_env()])
        logger.info("✓ Vectorized environment created")
        
        logger.info("2. Testing vectorized reset...")
        obs = env.reset()
        logger.info(f"✓ Vectorized reset: obs shape = {obs.shape}")
        
        logger.info("3. Testing vectorized steps...")
        for i in range(5):
            action = [env.action_space.sample()]
            obs, reward, done, info = env.step(action)
            logger.info(f"   Step {i}: reward = {reward[0]:.3f}, done = {done[0]}")
            if done[0]:
                obs = env.reset()
                logger.info("   Reset after episode end")
        
        env.close()
        logger.info("✓ Vectorized environment test complete")
        return True
        
    except Exception as e:
        logger.error(f"Vectorized environment test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_ppo_creation():
    """Test PPO model creation"""
    logger.info("=== Testing PPO Model Creation ===")
    
    try:
        from gpu_env_wrapper import make_env
        from stable_baselines3.common.vec_env import DummyVecEnv
        from stable_baselines3 import PPO
        
        logger.info("1. Creating environment for PPO...")
        env = DummyVecEnv([lambda: make_env()])
        
        logger.info("2. Creating PPO model...")
        model = PPO(
            "MlpPolicy",
            env,
            verbose=1,
            learning_rate=3e-4,
            n_steps=64,  # Very small for quick test
            batch_size=32,
            device="cpu"
        )
        logger.info("✓ PPO model created")
        
        logger.info("3. Testing single PPO step...")
        obs = env.reset()
        action, _ = model.predict(obs)
        logger.info("✓ PPO prediction works")
        
        env.close()
        logger.info("✓ PPO creation test complete")
        return True
        
    except Exception as e:
        logger.error(f"PPO creation test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_minimal_training():
    """Test very minimal training loop"""
    logger.info("=== Testing Minimal Training ===")
    
    try:
        from gpu_env_wrapper import make_env
        from stable_baselines3.common.vec_env import DummyVecEnv
        from stable_baselines3 import PPO
        
        logger.info("1. Setting up minimal training...")
        env = DummyVecEnv([lambda: make_env()])
        
        model = PPO(
            "MlpPolicy",
            env,
            verbose=1,
            learning_rate=3e-4,
            n_steps=32,   # Very small
            batch_size=16,
            device="cpu"
        )
        
        logger.info("2. Starting minimal training (100 steps)...")
        start_time = time.time()
        
        # Train for just 100 steps - should complete quickly
        model.learn(
            total_timesteps=100,
            progress_bar=False  # No progress bar to avoid hanging
        )
        
        training_time = time.time() - start_time
        logger.info(f"✓ Minimal training completed in {training_time:.2f} seconds")
        
        env.close()
        return True
        
    except Exception as e:
        logger.error(f"Minimal training failed: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    logger.info("Starting GPU Training Debug Session")
    logger.info("=" * 50)
    
    tests = [
        ("Environment", test_environment),
        ("Vectorized Environment", test_vectorized_env),
        ("PPO Creation", test_ppo_creation),
        ("Minimal Training", test_minimal_training)
    ]
    
    for test_name, test_func in tests:
        logger.info(f"\n{'='*20} {test_name} {'='*20}")
        
        try:
            success = test_func()
            if success:
                logger.info(f"✅ {test_name} PASSED")
            else:
                logger.error(f"❌ {test_name} FAILED")
                break
        except Exception as e:
            logger.error(f"❌ {test_name} CRASHED: {e}")
            break
        
        time.sleep(1)  # Brief pause between tests
    
    logger.info("\nDebug session complete!")
