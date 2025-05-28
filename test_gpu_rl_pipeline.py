"""
GPU RL Pipeline Test Script

Tests the complete GPU RL pipeline without PyTorch dependency.
Uses pure NumPy + CUDA C++ backend.
"""
import logging
import numpy as np
import time
import os

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def test_cuda_extension():
    """Test CUDA extension import and basic functionality."""
    logger.info("Testing CUDA extension...")
    
    try:
        import gpu_environment
        config = gpu_environment.EnvironmentConfig()
        env = gpu_environment.Environment(config)
        obs = env.reset(42)
        logger.info("CUDA Extension: SUCCESS")
        logger.info(f"Observation shape: {len(obs)}")
        return True
    except ImportError as e:
        logger.warning(f"CUDA Extension: NOT AVAILABLE - {e}")
        logger.info("Will use CPU fallback")
        return False
    except Exception as e:
        logger.error(f"CUDA Extension: FAILED - {e}")
        return False


def test_gpu_environment():
    """Test GPU environment wrapper."""
    logger.info("Testing GPU environment...")
    
    try:
        from ai_platform_trainer.ai.models.gpu_rl_environment import GPURLEnvironment
        
        env = GPURLEnvironment(batch_size=10)
        obs = env.reset()
        
        logger.info(f"Reset successful, observation shape: {obs.shape}")
        
        actions = np.random.randn(10, 2)
        next_obs, rewards, dones = env.step(actions)
        
        logger.info(f"Step successful:")
        logger.info(f"  Obs shape: {next_obs.shape}")
        logger.info(f"  Rewards shape: {rewards.shape}")
        logger.info(f"  Dones shape: {dones.shape}")
        logger.info("GPU Environment: SUCCESS")
        return True
        
    except Exception as e:
        logger.error(f"GPU Environment: FAILED - {e}")
        return False


def test_neural_network():
    """Test pure NumPy neural network."""
    logger.info("Testing neural network...")
    
    try:
        from ai_platform_trainer.ai.models.train_enemy_rl_gpu import SimpleEnemyAgent
        
        agent = SimpleEnemyAgent(obs_dim=4, action_dim=2, hidden_dim=32)
        
        # Test forward pass
        obs = np.random.randn(5, 4)
        actions = agent.forward(obs)
        
        logger.info(f"Forward pass successful:")
        logger.info(f"  Input shape: {obs.shape}")
        logger.info(f"  Output shape: {actions.shape}")
        logger.info(f"  Action range: [{actions.min():.3f}, {actions.max():.3f}]")
        
        # Test weight operations
        weights = agent.get_weights()
        logger.info(f"  Total parameters: {len(weights)}")
        
        # Test save/load
        test_path = 'test_model.npz'
        agent.save(test_path)
        
        new_agent = SimpleEnemyAgent(obs_dim=4, action_dim=2, hidden_dim=32)
        new_agent.load(test_path)
        
        # Cleanup
        if os.path.exists(test_path):
            os.remove(test_path)
        
        logger.info("Neural Network: SUCCESS")
        return True
        
    except Exception as e:
        logger.error(f"Neural Network: FAILED - {e}")
        return False


def test_rl_controller():
    """Test RL enemy controller."""
    logger.info("Testing RL controller...")
    
    try:
        from ai_platform_trainer.gameplay.rl_enemy_controller import RLEnemyController
        
        controller = RLEnemyController()
        status = controller.get_status()
        
        logger.info(f"Controller status: {status}")
        
        # Test action generation
        enemy_pos = (400.0, 300.0)
        player_pos = (200.0, 150.0)
        
        action = controller.get_action(enemy_pos, player_pos)
        logger.info(f"Generated action: {action}")
        
        logger.info("RL Controller: SUCCESS")
        return True
        
    except Exception as e:
        logger.error(f"RL Controller: FAILED - {e}")
        return False


def test_quick_training():
    """Test quick training run."""
    logger.info("Testing quick training...")
    
    try:
        from ai_platform_trainer.ai.models.train_enemy_rl_gpu import train_enemy_rl_gpu
        
        # Quick training run
        agent = train_enemy_rl_gpu(
            num_generations=5,
            batch_size=16,
            population_size=10,
            learning_rate=0.05
        )
        
        logger.info("Quick Training: SUCCESS")
        return True
        
    except Exception as e:
        logger.error(f"Quick Training: FAILED - {e}")
        return False


def benchmark_performance():
    """Benchmark environment performance."""
    logger.info("Benchmarking performance...")
    
    try:
        from ai_platform_trainer.ai.models.gpu_rl_environment import GPURLEnvironment
        
        batch_sizes = [32, 64, 128]
        num_steps = 50
        
        results = []
        
        for batch_size in batch_sizes:
            logger.info(f"Testing batch size {batch_size}")
            
            env = GPURLEnvironment(batch_size=batch_size)
            
            start_time = time.time()
            obs = env.reset()
            
            for _ in range(num_steps):
                actions = np.random.randn(batch_size, 2)
                obs, rewards, dones = env.step(actions)
            
            total_time = time.time() - start_time
            steps_per_second = (batch_size * num_steps) / total_time
            
            results.append({
                'batch_size': batch_size,
                'steps_per_second': steps_per_second,
                'cuda_available': env.cuda_available
            })
            
            logger.info(f"  {steps_per_second:.0f} env-steps/second")
        
        # Summary
        logger.info("\nPerformance Summary:")
        logger.info("Batch Size | Env-Steps/Sec | CUDA")
        logger.info("-" * 35)
        for result in results:
            cuda_str = "Yes" if result['cuda_available'] else "No"
            logger.info(f"{result['batch_size']:9d} | "
                       f"{result['steps_per_second']:12.0f} | {cuda_str}")
        
        return True
        
    except Exception as e:
        logger.error(f"Benchmark: FAILED - {e}")
        return False


def main():
    """Run all tests."""
    logger.info("=" * 60)
    logger.info("GPU RL Pipeline Test Suite")
    logger.info("=" * 60)
    
    tests = [
        ("CUDA Extension", test_cuda_extension),
        ("GPU Environment", test_gpu_environment),
        ("Neural Network", test_neural_network),
        ("RL Controller", test_rl_controller),
        ("Quick Training", test_quick_training),
        ("Performance", benchmark_performance)
    ]
    
    results = {}
    
    for test_name, test_func in tests:
        logger.info(f"\n--- {test_name} ---")
        try:
            results[test_name] = test_func()
        except Exception as e:
            logger.error(f"Test {test_name} crashed: {e}")
            results[test_name] = False
    
    # Final summary
    logger.info("\n" + "=" * 60)
    logger.info("FINAL RESULTS")
    logger.info("=" * 60)
    
    passed = 0
    total = len(tests)
    
    for test_name, result in results.items():
        status = "PASS" if result else "FAIL"
        logger.info(f"  {test_name:20s}: {status}")
        if result:
            passed += 1
    
    logger.info(f"\nSummary: {passed}/{total} tests passed")
    
    if passed == total:
        logger.info("All tests passed! GPU RL pipeline is ready.")
    else:
        logger.warning("Some tests failed. Check the logs above.")
    
    return passed == total


if __name__ == "__main__":
    success = main()
    exit(0 if success else 1)
