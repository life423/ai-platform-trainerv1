"""
Build and Test GPU RL Pipeline

Builds CUDA extension and tests the complete pipeline.
Works with or without CUDA available.
"""
import os
import sys
import subprocess
import logging
import time
from pathlib import Path

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def check_cuda_available():
    """Check if CUDA is available on the system."""
    try:
        result = subprocess.run(['nvcc', '--version'], 
                              capture_output=True, text=True)
        if result.returncode == 0:
            logger.info("CUDA compiler found")
            return True
        else:
            logger.warning("CUDA compiler not found")
            return False
    except FileNotFoundError:
        logger.warning("nvcc not found in PATH")
        return False


def build_cuda_extension():
    """Build the CUDA extension using CMake."""
    logger.info("Building CUDA extension...")
    
    cpp_dir = Path("ai_platform_trainer/cpp")
    build_dir = cpp_dir / "build"
    
    if not cpp_dir.exists():
        logger.error(f"C++ directory not found: {cpp_dir}")
        return False
    
    try:
        # Create build directory
        build_dir.mkdir(exist_ok=True)
        
        # Change to build directory
        original_dir = os.getcwd()
        os.chdir(build_dir)
        
        # Configure with CMake
        logger.info("Running CMake configure...")
        result = subprocess.run([
            'cmake', '..', 
            '-DCMAKE_BUILD_TYPE=Release'
        ], capture_output=True, text=True)
        
        if result.returncode != 0:
            logger.error(f"CMake configure failed: {result.stderr}")
            return False
        
        # Build
        logger.info("Running CMake build...")
        result = subprocess.run([
            'cmake', '--build', '.', '--config', 'Release'
        ], capture_output=True, text=True)
        
        if result.returncode != 0:
            logger.error(f"CMake build failed: {result.stderr}")
            return False
        
        logger.info("CUDA extension built successfully")
        return True
        
    except Exception as e:
        logger.error(f"Build failed: {e}")
        return False
    finally:
        os.chdir(original_dir)


def setup_python_path():
    """Setup Python path to find modules."""
    # Add project root to Python path
    project_root = os.path.abspath('.')
    if project_root not in sys.path:
        sys.path.insert(0, project_root)
    
    # Add cpp directory for gpu_environment module
    cpp_dir = os.path.join(project_root, 'ai_platform_trainer', 'cpp')
    if cpp_dir not in sys.path:
        sys.path.insert(0, cpp_dir)
    
    logger.info("Python path configured")


def create_models_directory():
    """Create models directory if it doesn't exist."""
    models_dir = Path("ai_platform_trainer/models")
    models_dir.mkdir(parents=True, exist_ok=True)
    logger.info(f"Models directory ready: {models_dir}")


def run_tests():
    """Run the test pipeline."""
    logger.info("Running GPU RL pipeline tests...")
    
    try:
        # Import and run test script
        import test_gpu_rl_pipeline
        success = test_gpu_rl_pipeline.main()
        return success
    except Exception as e:
        logger.error(f"Test execution failed: {e}")
        return False


def quick_training_demo():
    """Run a quick training demonstration."""
    logger.info("Running quick training demonstration...")
    
    try:
        from ai_platform_trainer.ai.models.train_enemy_rl_gpu import train_enemy_rl_gpu
        
        logger.info("Starting quick training (10 generations)...")
        agent = train_enemy_rl_gpu(
            num_generations=10,
            batch_size=32,
            population_size=20,
            learning_rate=0.03
        )
        
        logger.info("Quick training completed successfully")
        return True
        
    except Exception as e:
        logger.error(f"Quick training failed: {e}")
        return False


def integration_test():
    """Test integration with game controller."""
    logger.info("Testing game integration...")
    
    try:
        from ai_platform_trainer.gameplay.rl_enemy_controller import RLEnemyController
        
        controller = RLEnemyController()
        status = controller.get_status()
        
        logger.info(f"Controller status: {status}")
        
        # Test multiple actions
        for i in range(5):
            enemy_pos = (400.0 + i * 50, 300.0)
            player_pos = (200.0, 150.0)
            action = controller.get_action(enemy_pos, player_pos)
            logger.info(f"Action {i+1}: {action}")
        
        logger.info("Game integration test completed")
        return True
        
    except Exception as e:
        logger.error(f"Integration test failed: {e}")
        return False


def main():
    """Main build and test pipeline."""
    logger.info("=" * 60)
    logger.info("GPU RL Pipeline Build and Test")
    logger.info("=" * 60)
    
    start_time = time.time()
    
    # Setup
    setup_python_path()
    create_models_directory()
    
    # Check CUDA
    cuda_available = check_cuda_available()
    if cuda_available:
        logger.info("CUDA is available - will build GPU acceleration")
    else:
        logger.info("CUDA not available - will use CPU fallback")
    
    # Build CUDA extension (try even without CUDA for completeness)
    build_success = True
    if cuda_available:
        build_success = build_cuda_extension()
        if not build_success:
            logger.warning("CUDA build failed, continuing with CPU fallback")
    else:
        logger.info("Skipping CUDA build (CUDA not available)")
    
    # Run tests
    logger.info("\n" + "=" * 40)
    logger.info("RUNNING TESTS")
    logger.info("=" * 40)
    
    test_success = run_tests()
    
    if test_success:
        logger.info("\n" + "=" * 40)
        logger.info("RUNNING DEMONSTRATIONS")
        logger.info("=" * 40)
        
        # Quick training demo
        training_success = quick_training_demo()
        
        # Integration test
        integration_success = integration_test()
        
        overall_success = training_success and integration_success
    else:
        logger.error("Tests failed, skipping demonstrations")
        overall_success = False
    
    # Final summary
    total_time = time.time() - start_time
    
    logger.info("\n" + "=" * 60)
    logger.info("FINAL SUMMARY")
    logger.info("=" * 60)
    logger.info(f"CUDA Available:     {'Yes' if cuda_available else 'No'}")
    logger.info(f"CUDA Build:         {'Success' if build_success else 'Failed'}")
    logger.info(f"Tests:              {'Success' if test_success else 'Failed'}")
    logger.info(f"Overall:            {'SUCCESS' if overall_success else 'FAILED'}")
    logger.info(f"Total Time:         {total_time:.2f}s")
    
    if overall_success:
        logger.info("\nYour GPU RL pipeline is ready!")
        logger.info("Next steps:")
        logger.info("1. Run 'python test_gpu_rl_pipeline.py' to test components")
        logger.info("2. Run training with: python -m ai_platform_trainer.ai.models.train_enemy_rl_gpu")
        logger.info("3. Integrate with your game using RLEnemyController")
        
        if cuda_available and build_success:
            logger.info("4. Deploy to GPU machine for maximum performance")
    else:
        logger.error("\nSome components failed. Check the logs above.")
    
    return overall_success


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
