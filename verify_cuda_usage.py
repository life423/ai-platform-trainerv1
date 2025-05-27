#!/usr/bin/env python
"""
Verify that CUDA is being used for training with custom C++ modules.

This script checks if CUDA is available, builds the custom C++ extensions,
and verifies that they're correctly using the GPU.
"""
import os
import sys
import torch
import subprocess
import logging

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger("CUDA-Verification")

def verify_cuda_available():
    """Check if CUDA is available and working."""
    logger.info("Checking CUDA availability...")
    
    if not torch.cuda.is_available():
        logger.error("CUDA is not available! Cannot use GPU.")
        return False
    
    logger.info(f"CUDA is available: {torch.cuda.is_available()}")
    logger.info(f"GPU: {torch.cuda.get_device_name(0)}")
    logger.info(f"CUDA version: {torch.version.cuda}")
    
    # Run a simple CUDA test
    try:
        x = torch.rand(1000, 1000).cuda()
        y = torch.matmul(x, x)
        del x, y
        torch.cuda.empty_cache()
        logger.info("CUDA test successful!")
        return True
    except Exception as e:
        logger.error(f"CUDA test failed: {e}")
        return False

def check_nvidia_smi():
    """Check if nvidia-smi is working."""
    logger.info("Checking nvidia-smi...")
    
    try:
        result = subprocess.run(
            ['nvidia-smi'], 
            capture_output=True, 
            text=True, 
            check=True
        )
        logger.info("nvidia-smi output:")
        for line in result.stdout.split('\n')[:5]:  # Show first 5 lines
            if line.strip():
                logger.info(line)
        return True
    except Exception as e:
        logger.error(f"nvidia-smi check failed: {e}")
        return False

def verify_custom_cuda_modules():
    """Verify that custom CUDA modules are built and working."""
    logger.info("Verifying custom CUDA modules...")
    
    # Add cpp directory to path
    cpp_dir = os.path.abspath(os.path.join("ai_platform_trainer", "cpp"))
    if cpp_dir not in sys.path:
        sys.path.append(cpp_dir)
    
    # Try to import the module
    try:
        import gpu_environment
        logger.info("Custom CUDA module imported successfully!")
        
        # Create a test environment
        config = gpu_environment.EnvironmentConfig()
        env = gpu_environment.Environment(config)
        logger.info("Created environment with custom CUDA!")
        
        # Test reset and step
        import numpy as np
        obs = env.reset(42)
        action = np.array([0.5, 0.5], dtype=np.float32)
        next_obs, reward, done, truncated, info = env.step(action)
        
        logger.info("Successfully executed step in custom CUDA environment!")
        return True
    except ImportError as e:
        logger.error(f"Failed to import custom CUDA module: {e}")
        logger.info("Attempting to build the extensions...")
        
        try:
            result = subprocess.run(
                [sys.executable, "setup.py", "build_ext", "--inplace"],
                cwd=cpp_dir,
                check=True,
                capture_output=True,
                text=True
            )
            logger.info("Build completed. Trying to import again...")
            
            # Try importing again
            import gpu_environment
            logger.info("Custom CUDA module imported successfully after build!")
            return True
        except Exception as build_error:
            logger.error(f"Build failed: {build_error}")
            return False
    except Exception as e:
        logger.error(f"Error testing custom CUDA module: {e}")
        return False

def main():
    """Run all verification steps."""
    logger.info("=== CUDA and Custom C++ Module Verification ===")
    
    # Step 1: Check if CUDA is available
    if not verify_cuda_available():
        logger.error("CUDA verification failed!")
        return False
    
    # Step 2: Check nvidia-smi
    if not check_nvidia_smi():
        logger.warning("nvidia-smi check failed, but continuing...")
    
    # Step 3: Verify custom CUDA modules
    if not verify_custom_cuda_modules():
        logger.error("Custom CUDA module verification failed!")
        return False
    
    logger.info("\n=== Verification Successful! ===")
    logger.info("Your system is correctly set up to use NVIDIA GPU with custom C++ CUDA modules.")
    logger.info("To train the enemy agent with CUDA, run:")
    logger.info("python train_enemy_cuda.py")
    
    return True

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)