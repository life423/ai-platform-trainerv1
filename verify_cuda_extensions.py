#!/usr/bin/env python
"""
Verify that CUDA extensions are built and working correctly.

This script checks if the NVIDIA GPU is available and if the custom C++/CUDA
extensions are built and functioning properly.
"""
import os
import sys
import subprocess
import logging

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger("CUDA-Verification")

def check_nvidia_gpu():
    """Check if NVIDIA GPU is available using nvidia-smi."""
    logger.info("Checking for NVIDIA GPU...")
    try:
        result = subprocess.run(['nvidia-smi'], capture_output=True, text=True)
        if result.returncode == 0:
            logger.info("NVIDIA GPU detected:")
            for line in result.stdout.split('\n')[:5]:  # Show first 5 lines
                if line.strip():
                    logger.info(line)
            return True
        else:
            logger.error("nvidia-smi command failed. GPU not detected.")
            return False
    except Exception as e:
        logger.error(f"Error checking GPU: {e}")
        return False

def check_nvcc():
    """Check if NVIDIA CUDA compiler (nvcc) is available."""
    logger.info("Checking for NVIDIA CUDA compiler (nvcc)...")
    try:
        result = subprocess.run(['nvcc', '--version'], capture_output=True, text=True)
        if result.returncode == 0:
            logger.info("NVIDIA CUDA compiler (nvcc) detected:")
            logger.info(result.stdout.strip())
            return True
        else:
            logger.error("nvcc command failed. CUDA compiler not detected.")
            return False
    except Exception as e:
        logger.error(f"Error checking nvcc: {e}")
        return False

def build_cuda_extensions():
    """Build the custom C++/CUDA extensions."""
    cpp_dir = os.path.join("ai_platform_trainer", "cpp")
    logger.info(f"Building custom CUDA extensions in {cpp_dir}...")
    
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

def test_cuda_extensions():
    """Test if the custom CUDA extensions are working."""
    logger.info("Testing custom CUDA extensions...")
    
    # Add cpp directory to path
    cpp_dir = os.path.abspath(os.path.join("ai_platform_trainer", "cpp"))
    if cpp_dir not in sys.path:
        sys.path.append(cpp_dir)
    
    # Try to import the module
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
    logger.info("=== Verifying CUDA Extensions ===")
    
    # Step 1: Check for NVIDIA GPU
    if not check_nvidia_gpu():
        logger.error("NVIDIA GPU not detected. Cannot proceed.")
        return False
    
    # Step 2: Check for NVIDIA CUDA compiler
    if not check_nvcc():
        logger.warning("NVIDIA CUDA compiler not detected. Build may fail.")
    
    # Step 3: Build CUDA extensions
    if not build_cuda_extensions():
        logger.error("Failed to build CUDA extensions.")
        return False
    
    # Step 4: Test CUDA extensions
    if not test_cuda_extensions():
        logger.error("CUDA extensions test failed.")
        return False
    
    logger.info("\n=== Verification Successful! ===")
    logger.info("CUDA extensions are built and working correctly.")
    logger.info("You can now train the enemy agent using custom CUDA with:")
    logger.info("python train_enemy_cuda.py")
    
    return True

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)