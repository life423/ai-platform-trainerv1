#!/usr/bin/env python
"""
Build the custom C++/CUDA extensions for the AI Platform Trainer.

This script builds the custom C++/CUDA extensions that accelerate the
environment simulation on NVIDIA GPUs.
"""
import os
import sys
import subprocess
import logging

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger("CUDA-Build")

def check_nvidia_gpu():
    """Check if NVIDIA GPU is available."""
    try:
        result = subprocess.run(['nvidia-smi'], capture_output=True, text=True)
        if result.returncode == 0:
            logger.info("NVIDIA GPU detected:")
            for line in result.stdout.split('\n')[:5]:  # Show first 5 lines
                if line.strip():
                    logger.info(line)
            return True
        else:
            logger.warning("nvidia-smi command failed. GPU not detected.")
            logger.warning("Building for CPU-only mode.")
            return False
    except Exception as e:
        logger.warning(f"Error checking GPU: {e}")
        logger.warning("Building for CPU-only mode.")
        return False

def check_nvcc():
    """Check if NVIDIA CUDA compiler (nvcc) is available."""
    try:
        result = subprocess.run(['nvcc', '--version'], capture_output=True, text=True)
        if result.returncode == 0:
            logger.info("NVIDIA CUDA compiler (nvcc) detected:")
            logger.info(result.stdout.strip())
            return True
        else:
            logger.warning("nvcc command failed. CUDA compiler not detected.")
            logger.warning("Building for CPU-only mode.")
            return False
    except Exception as e:
        logger.warning(f"Error checking nvcc: {e}")
        logger.warning("Building for CPU-only mode.")
        return False

def build_extensions():
    """Build the custom C++/CUDA extensions."""
    cpp_dir = os.path.join("ai_platform_trainer", "cpp")
    logger.info(f"Building custom CUDA extensions in {cpp_dir}...")
    
    if not os.path.exists(cpp_dir):
        logger.error(f"Directory not found: {cpp_dir}")
        return False
    
    try:
        result = subprocess.run(
            [sys.executable, "setup.py", "build_ext", "--inplace"],
            cwd=cpp_dir,
            check=True,
            capture_output=True,
            text=True
        )
        
        logger.info("Build output:")
        logger.info(result.stdout)
        
        if "error" in result.stderr.lower():
            logger.error("Build completed with errors:")
            logger.error(result.stderr)
            return False
        
        logger.info("Build completed successfully.")
        return True
    except subprocess.CalledProcessError as e:
        logger.error(f"Build failed: {e}")
        logger.error(e.stderr)
        return False

def test_extensions():
    """Test if the extensions are working."""
    cpp_dir = os.path.abspath(os.path.join("ai_platform_trainer", "cpp"))
    if cpp_dir not in sys.path:
        sys.path.append(cpp_dir)
    
    try:
        import gpu_environment
        logger.info("Successfully imported gpu_environment module.")
        
        # Create a test environment
        config = gpu_environment.EnvironmentConfig()
        env = gpu_environment.Environment(config)
        logger.info("Successfully created environment.")
        
        # Test reset and step
        import numpy as np
        obs = env.reset(42)
        logger.info(f"Reset successful. Observation shape: {obs.shape}")
        
        action = np.array([0.5, 0.5], dtype=np.float32)
        next_obs, reward, done, truncated, info = env.step(action)
        logger.info(f"Step successful. Reward: {reward}")
        
        logger.info("Custom CUDA extensions are working correctly!")
        return True
    except ImportError as e:
        logger.error(f"Failed to import gpu_environment module: {e}")
        return False
    except Exception as e:
        logger.error(f"Error testing CUDA extensions: {e}")
        return False

def main():
    """Main function."""
    logger.info("=== Building CUDA Extensions ===")
    
    # Step 1: Check for NVIDIA GPU
    has_gpu = check_nvidia_gpu()
    
    # Step 2: Check for NVIDIA CUDA compiler
    if has_gpu:
        has_nvcc = check_nvcc()
    else:
        has_nvcc = False
    
    # Step 3: Build extensions
    if not build_extensions():
        logger.error("Failed to build extensions.")
        return False
    
    # Step 4: Test extensions
    if not test_extensions():
        logger.error("Extensions test failed.")
        return False
    
    logger.info("\n=== Build Successful! ===")
    logger.info("CUDA extensions are built and working correctly.")
    logger.info("You can now train the enemy agent using custom CUDA with:")
    logger.info("python train_enemy_cuda.py")
    
    return True

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)