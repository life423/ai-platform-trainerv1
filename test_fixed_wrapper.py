#!/usr/bin/env python
"""Test the fixed GPU environment wrapper"""

print("Testing fixed GPU environment wrapper...")

try:
    # Test import from the module path
    from ai_platform_trainer.cpp.gpu_environment import make_env, HAS_GPU_ENV, create_vectorized_env
    print(f"✓ Import successful! HAS_GPU_ENV: {HAS_GPU_ENV}")
    
    if HAS_GPU_ENV:
        # Test single environment
        print("\n1. Testing single environment...")
        env = make_env()
        print("✓ Environment created")
        
        obs = env.reset()
        print(f"✓ Reset successful, obs shape: {obs.shape}")
        print(f"✓ Observation space: {env.observation_space}")
        print(f"✓ Action space: {env.action_space}")
        
        # Test a few steps
        for i in range(3):
            action = env.action_space.sample()
            obs, reward, done, truncated, info = env.step(action)
            print(f"  Step {i}: reward={reward:.3f}, done={done}, info_keys={list(info.keys())}")
        
        env.close()
        print("✓ Single environment test passed!")
        
        # Test vectorized environment
        print("\n2. Testing vectorized environment...")
        vec_env = create_vectorized_env(num_envs=2)
        print("✓ Vectorized environment created")
        
        obs = vec_env.reset()
        print(f"✓ Vectorized reset successful, obs shape: {len(obs)} envs")
        
        # Test vectorized step
        actions = [vec_env.action_space.sample() for _ in range(vec_env.num_envs)]
        obs, rewards, dones, infos = vec_env.step(actions)
        print(f"✓ Vectorized step successful, rewards: {rewards}")
        
        vec_env.close()
        print("✓ Vectorized environment test passed!")
        
        print("\n🎉 All tests passed! GPU environment wrapper is fully functional!")
    else:
        print("❌ GPU environment not available")
        
except Exception as e:
    print(f"❌ Error: {e}")
    import traceback
    traceback.print_exc()
