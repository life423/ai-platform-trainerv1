#!/usr/bin/env python
"""
Verify that CUDA is available and our C++/CUDA validation works correctly.

This script checks CUDA availability and runs our C++ validation executable.
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

def build_cuda_validation():
    """Build the C++ CUDA validation executable."""
    logger.info("Building C++ CUDA validation...")
    
    build_dir = "build"
    if not os.path.exists(build_dir):
        logger.info("Creating build directory...")
        try:
            os.makedirs(build_dir)
        except Exception as e:
            logger.error(f"Failed to create build directory: {e}")
            return False
    
    try:
        # Configure CMake
        logger.info("Configuring CMake...")
        result = subprocess.run([
            'cmake', '..', '-G', 'Visual Studio 17 2022', '-A', 'x64'
        ], cwd=build_dir, capture_output=True, text=True, check=True)
        
        # Build the target
        logger.info("Building validate_cuda target...")
        result = subprocess.run([
            'cmake', '--build', '.', '--config', 'Release', '--target', 'validate_cuda'
        ], cwd=build_dir, capture_output=True, text=True, check=True)
        
        logger.info("Build completed successfully.")
        return True
    except subprocess.CalledProcessError as e:
        logger.error(f"Build failed: {e}")
        logger.error(e.stderr)
        return False

def test_cuda_validation():
    """Test the C++ CUDA validation executable."""
    logger.info("Testing C++ CUDA validation...")
    
    exe_path = os.path.join("build", "Release", "validate_cuda.exe")
    if not os.path.exists(exe_path):
        logger.error(f"Validation executable not found at {exe_path}")
        return False
    
    try:
        result = subprocess.run([exe_path], capture_output=True, text=True, check=True)
        logger.info("CUDA validation output:")
        for line in result.stdout.strip().split('\n'):
            logger.info(line)
        
        # Check for expected output
        if "Using GPU:" in result.stdout and "result: 3.1415" in result.stdout:
            logger.info("CUDA validation test passed!")
            return True
        else:
            logger.error("CUDA validation test failed - unexpected output")
            return False
    except subprocess.CalledProcessError as e:
        logger.error(f"CUDA validation failed: {e}")
        logger.error(e.stderr)
        return False

def main():
    """Run all verification steps."""
    logger.info("=== CUDA Verification ===")
    
    # Step 1: Check if NVIDIA GPU is available
    if not check_nvidia_gpu():
        logger.error("NVIDIA GPU not detected. Cannot proceed.")
        return False
    
    # Step 2: Check if NVIDIA CUDA compiler is available
    if not check_nvcc():
        logger.warning("NVIDIA CUDA compiler not detected. Build may fail.")
    
    # Step 3: Build CUDA validation
    if not build_cuda_validation():
        logger.error("Failed to build CUDA validation.")
        return False
    
    # Step 4: Test CUDA validation
    if not test_cuda_validation():
        logger.error("CUDA validation test failed.")
        return False
    
    logger.info("\n=== Verification Successful! ===")
    logger.info("C++ CUDA validation is working correctly.")
    logger.info("Your system is ready for GPU-accelerated training.")
    
    return True

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
