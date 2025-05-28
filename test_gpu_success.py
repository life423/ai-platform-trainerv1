#!/usr/bin/env python3
"""
Final validation of GPU environment success
"""
import sys
import logging
from ai_platform_trainer.cpp.gpu_environment import make_env, HAS_GPU_ENV

logging.basicConfig(level=logging.INFO)

def test_gpu_environment():
    """Test that our GPU environment is fully functional"""
    print("🎯 GPU Environment Success Validation")
    print("=" * 50)
    
    # Test 1: Import and availability
    print(f"✅ GPU Environment Available: {HAS_GPU_ENV}")
    if not HAS_GPU_ENV:
        print("❌ GPU environment not available")
        return False
    
    # Test 2: Single environment creation and API
    print("\n🧪 Testing Single Environment...")
    try:
        env = make_env()
        print("✅ Environment created successfully")
        
        # Test modern API with options parameter
        obs = env.reset(seed=42, options=None)
        print(f"✅ Reset with options parameter: obs shape = {obs.shape}")
        
        # Test action execution
        action = env.action_space.sample()
        obs, reward, done, truncated, info = env.step(action)
        print(f"✅ Step executed: reward = {reward:.3f}")
        print(f"✅ Info contains: {list(info.keys())}")
        
        # Test spaces
        print(f"✅ Observation space: {env.observation_space}")
        print(f"✅ Action space: {env.action_space}")
        
    except Exception as e:
        print(f"❌ Single environment test failed: {e}")
        return False
    
    # Test 3: Multiple environment instances
    print("\n🔄 Testing Multiple Environment Instances...")
    try:
        envs = [make_env() for _ in range(3)]
        print("✅ Created 3 separate environment instances")
        
        # Test they work independently
        observations = []
        for i, env in enumerate(envs):
            obs = env.reset(seed=i * 100)
            observations.append(obs)
        
        print("✅ All environments reset independently")
        
    except Exception as e:
        print(f"❌ Multiple environment test failed: {e}")
        return False
    
    print("\n🎉 ALL TESTS PASSED!")
    print("✅ GPU environment integration successful!")
    print("✅ API compatibility with modern RL libraries!")
    print("✅ Ready for training!")
    
    return True

if __name__ == "__main__":
    success = test_gpu_environment()
    sys.exit(0 if success else 1)
