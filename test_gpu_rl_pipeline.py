"""
Complete GPU RL Pipeline Test

This script tests the entire GPU-accelerated RL pipeline:
1. Build CUDA extension
2. Test GPU environment
3. Train RL agent (quick test)
4. Test trained model
5. Integration test
"""
import os
import sys
import logging
import time

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def test_cuda_extension():
    """Test building and loading CUDA extension."""
    logger.info("=== Testing CUDA Extension ===")
    
    try:
        # Import and run the build script
        from build_cuda_extensions import main as build_main
        physics_module = build_main()
        
        if physics_module is None:
            logger.error("Failed to build CUDA extension")
            return False
        
        logger.info("✓ CUDA extension built successfully")
        return True
        
    except Exception as e:
        logger.error(f"CUDA extension test failed: {e}")
        return False


def test_gpu_environment():
    """Test GPU RL environment."""
    logger.info("\n=== Testing GPU Environment ===")
    
    try:
        from ai_platform_trainer.ai.models.gpu_rl_environment import test_gpu_environment
        env = test_gpu_environment()
        
        if env is None:
            logger.error("Failed to create GPU environment")
            return False
        
        logger.info("✓ GPU environment test passed")
        return True
        
    except Exception as e:
        logger.error(f"GPU environment test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_quick_training():
    """Test quick RL training."""
    logger.info("\n=== Testing Quick RL Training ===")
    
    try:
        from ai_platform_trainer.ai.models.train_enemy_rl_gpu import train_enemy_rl_gpu
        
        # Quick training run
        logger.info("Running quick training (25 episodes)...")
        agent = train_enemy_rl_gpu(
            num_episodes=25,
            batch_size=64,  # Smaller batch for testing
            log_interval=10
        )
        
        if agent is None:
            logger.error("Training failed")
            return False
        
        logger.info("✓ Quick training completed successfully")
        return True
        
    except Exception as e:
        logger.error(f"Training test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_model_loading():
    """Test loading and using trained model."""
    logger.info("\n=== Testing Model Loading ===")
    
    try:
        from ai_platform_trainer.gameplay.rl_enemy_controller import test_rl_controller
        test_rl_controller()
        
        logger.info("✓ Model loading test passed")
        return True
        
    except Exception as e:
        logger.error(f"Model loading test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_integration():
    """Test full integration."""
    logger.info("\n=== Testing Integration ===")
    
    try:
        from ai_platform_trainer.gameplay.rl_enemy_controller import EnemyRLIntegration
        
        # Create integration object
        integration = EnemyRLIntegration()
        
        # Test status
        status = integration.get_status()
        logger.info(f"Integration status: {status}")
        
        # Create mock enemy and player objects
        class MockEntity:
            def __init__(self, x, y):
                self.pos = [x, y]
                self.speed = 200.0
        
        enemy = MockEntity(400, 300)
        player = MockEntity(500, 400)
        
        logger.info(f"Initial enemy pos: {enemy.pos}")
        
        # Test controlling enemy
        for i in range(5):
            integration.control_enemy(enemy, player)
            logger.info(f"Step {i+1} enemy pos: {enemy.pos}")
        
        logger.info("✓ Integration test passed")
        return True
        
    except Exception as e:
        logger.error(f"Integration test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def benchmark_performance():
    """Benchmark GPU vs CPU performance."""
    logger.info("\n=== Performance Benchmark ===")
    
    try:
        from ai_platform_trainer.ai.models.gpu_rl_environment import GPURLEnvironment
        
        # Test different configurations
        configs = [
            ("CPU Small", {'batch_size': 32, 'device': 'cpu'}),
            ("CPU Large", {'batch_size': 128, 'device': 'cpu'}),
        ]
        
        # Add GPU tests if CUDA available
        try:
            import torch
            if torch.cuda.is_available():
                configs.extend([
                    ("GPU Small", {'batch_size': 32, 'device': 'cuda'}),
                    ("GPU Large", {'batch_size': 256, 'device': 'cuda'}),
                ])
        except ImportError:
            pass
        
        results = []
        
        for name, config in configs:
            logger.info(f"Benchmarking {name}...")
            
            try:
                env = GPURLEnvironment(**config)
                stats = env.benchmark(num_steps=100)
                
                results.append((name, stats))
                logger.info(f"  {stats['env_steps_per_second']:.1f} env-steps/sec")
                
            except Exception as e:
                logger.warning(f"  {name} benchmark failed: {e}")
        
        # Print summary
        logger.info("\nBenchmark Results:")
        for name, stats in results:
            logger.info(f"  {name}: {stats['env_steps_per_second']:.1f} env-steps/sec")
        
        return True
        
    except Exception as e:
        logger.error(f"Benchmark failed: {e}")
        return False


def create_models_directory():
    """Ensure models directory exists."""
    models_dir = "ai_platform_trainer/models"
    os.makedirs(models_dir, exist_ok=True)
    logger.info(f"Created models directory: {models_dir}")


def main():
    """Run complete pipeline test."""
    logger.info("🚀 GPU RL Pipeline Complete Test")
    logger.info("=" * 60)
    
    start_time = time.time()
    
    # Create necessary directories
    create_models_directory()
    
    # Test results
    results = {}
    
    # Run tests in sequence
    tests = [
        ("CUDA Extension", test_cuda_extension),
        ("GPU Environment", test_gpu_environment),
        ("Quick Training", test_quick_training),
        ("Model Loading", test_model_loading),
        ("Integration", test_integration),
    ]
    
    for test_name, test_func in tests:
        logger.info(f"\n{'='*20} {test_name} {'='*20}")
        
        try:
            result = test_func()
            results[test_name] = result
            
            if result:
                logger.info(f"✓ {test_name} PASSED")
            else:
                logger.error(f"❌ {test_name} FAILED")
                
        except Exception as e:
            logger.error(f"❌ {test_name} CRASHED: {e}")
            results[test_name] = False
    
    # Run benchmark (non-critical)
    logger.info(f"\n{'='*20} Performance Benchmark {'='*20}")
    benchmark_performance()
    
    # Final summary
    total_time = time.time() - start_time
    passed = sum(results.values())
    total = len(results)
    
    logger.info("\n" + "=" * 60)
    logger.info("🎯 FINAL RESULTS")
    logger.info("=" * 60)
    
    for test_name, result in results.items():
        status = "✓ PASS" if result else "❌ FAIL"
        logger.info(f"  {test_name}: {status}")
    
    logger.info(f"\nSummary: {passed}/{total} tests passed")
    logger.info(f"Total time: {total_time:.1f}s")
    
    if passed == total:
        logger.info("🎉 ALL TESTS PASSED! GPU RL pipeline is ready!")
        logger.info("\nNext steps:")
        logger.info("1. Run: python ai_platform_trainer/ai/models/train_enemy_rl_gpu.py")
        logger.info("2. Integrate RL controller into your game loop")
        logger.info("3. Enjoy GPU-accelerated enemy AI!")
        return 0
    else:
        logger.error(f"❌ {total - passed} tests failed. Please check the logs above.")
        return 1


if __name__ == "__main__":
    sys.exit(main())
