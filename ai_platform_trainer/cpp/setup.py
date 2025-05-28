#!/usr/bin/env python3
"""Setup script for GPU environment CUDA extension."""

import os
import sys
import subprocess
from pathlib import Path
from pybind11.setup_helpers import Pybind11Extension, build_ext
from pybind11 import get_cmake_dir
import pybind11
from setuptools import setup, Extension
import torch

# Get CUDA path
def find_cuda():
    """Find CUDA installation."""
    cuda_home = os.environ.get('CUDA_HOME') or os.environ.get('CUDA_PATH')
    if cuda_home is None:
        # Try common Windows CUDA locations
        common_paths = [
            'C:/Program Files/NVIDIA GPU Computing Toolkit/CUDA/v12.9',
            'C:/Program Files/NVIDIA GPU Computing Toolkit/CUDA/v12.8',
            'C:/Program Files/NVIDIA GPU Computing Toolkit/CUDA/v12.7',
        ]
        for path in common_paths:
            if os.path.exists(path):
                cuda_home = path
                break
    
    if cuda_home and os.path.exists(cuda_home):
        return cuda_home
    return None

def check_cuda_available():
    """Check if CUDA compiler is available."""
    try:
        result = subprocess.run(['nvcc', '--version'], 
                              capture_output=True, text=True, check=True)
        print(f"CUDA compiler found: {result.stdout.split()[5]}")
        return True
    except (subprocess.CalledProcessError, FileNotFoundError):
        print("CUDA compiler not found")
        return False

# Check if building with CUDA
USE_CUDA = check_cuda_available()
CUDA_HOME = find_cuda()

if USE_CUDA and CUDA_HOME:
    print(f"Building with CUDA support")
    print(f"CUDA_HOME: {CUDA_HOME}")
    
    # CUDA sources
    cuda_sources = [
        "src/physics.cu",
    ]
    
    # C++ sources
    cpp_sources = [
        "pybind/bindings.cpp",
        "src/entity.cpp", 
        "src/environment.cpp",
        "src/reward.cpp",
    ]
    
    # Include directories
    include_dirs = [
        "include",
        f"{CUDA_HOME}/include",
        pybind11.get_include(),
    ]
    
    # Library directories
    library_dirs = [
        f"{CUDA_HOME}/lib/x64",
    ]
    
    # Libraries
    libraries = [
        "cudart",
        "cublas",
        "curand",
    ]
    
    # Compile arguments
    extra_compile_args = {
        'cxx': ['/std:c++17', '/MD'],
        'nvcc': [
            '-std=c++17',
            '--compiler-options', '/MD',
            '-arch=sm_89',  # RTX 5070 (Ada Lovelace)
            '--use_fast_math',
            '-O3',
        ]
    }
    
    # Create extension
    ext_modules = [
        Extension(
            'gpu_environment',
            sources=cpp_sources + cuda_sources,
            include_dirs=include_dirs,
            library_dirs=library_dirs,
            libraries=libraries,
            extra_compile_args=extra_compile_args['cxx'],
            language='c++'
        )
    ]
    
else:
    print("Building without CUDA support (CPU only)")
    
    # CPU-only sources
    cpp_sources = [
        "pybind/bindings.cpp",
        "src/entity.cpp",
        "src/environment.cpp", 
        "src/physics_cpu.cpp",
        "src/reward.cpp",
    ]
    
    # Include directories
    include_dirs = [
        "include",
        pybind11.get_include(),
    ]
    
    # Create extension
    ext_modules = [
        Pybind11Extension(
            'gpu_environment',
            sources=cpp_sources,
            include_dirs=include_dirs,
            cxx_std=17,
        )
    ]

class BuildExt(build_ext):
    """Custom build extension to handle CUDA compilation."""
    
    def build_extensions(self):
        # Set compiler options for CUDA
        if USE_CUDA:
            print("Configuring CUDA compilation...")
            
            # Find nvcc
            nvcc_path = self.find_nvcc()
            if not nvcc_path:
                raise RuntimeError("CUDA compiler (nvcc) not found")
            
            # Compile CUDA sources separately
            for ext in self.extensions:
                self.build_cuda_extension(ext, nvcc_path)
        
        super().build_extensions()
    
    def find_nvcc(self):
        """Find nvcc compiler."""
        try:
            result = subprocess.run(['where', 'nvcc'], capture_output=True, text=True, check=True)
            return result.stdout.strip().split('\n')[0]
        except subprocess.CalledProcessError:
            return None
    
    def build_cuda_extension(self, ext, nvcc_path):
        """Build CUDA sources."""
        print(f"Building CUDA extension with {nvcc_path}")
        
        # Separate CUDA and C++ sources
        cuda_sources = [s for s in ext.sources if s.endswith('.cu')]
        cpp_sources = [s for s in ext.sources if not s.endswith('.cu')]
        
        # Compile CUDA sources to object files
        cuda_objects = []
        for cu_file in cuda_sources:
            obj_file = cu_file.replace('.cu', '.obj')
            
            cmd = [
                nvcc_path,
                '-c', cu_file,
                '-o', obj_file,
                '-std=c++17',
                '--compiler-options', '/MD',
                '-arch=sm_89',
                '--use_fast_math',
                '-O3',
            ] + [f'-I{inc}' for inc in ext.include_dirs]
            
            print(f"Compiling {cu_file}...")
            result = subprocess.run(cmd, capture_output=True, text=True)
            if result.returncode != 0:
                print(f"CUDA compilation failed: {result.stderr}")
                raise RuntimeError(f"CUDA compilation failed for {cu_file}")
            
            cuda_objects.append(obj_file)
        
        # Update extension sources to include compiled objects
        ext.sources = cpp_sources
        ext.extra_objects = cuda_objects

if __name__ == "__main__":
    setup(
        name="gpu_environment",
        ext_modules=ext_modules,
        cmdclass={'build_ext': BuildExt},
        zip_safe=False,
        python_requires=">=3.8",
    )
