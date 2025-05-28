#!/usr/bin/env python3
"""Test script for CUDA extension import with proper DLL handling."""

import os
import sys
import glob

def test_cuda_import():
    """Test CUDA extension import with proper DLL directory management."""
    print("=== CUDA Extension Import Test ===\n")
    
    # Add CUDA DLL directories
    cuda_paths = [
        r"C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.9\bin",
        r"C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.8\bin", 
        r"C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.7\bin",
        r"C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v11.8\bin",
        r"C:\Windows\System32"
    ]
    
    print("Adding CUDA DLL directories:")
    for path in cuda_paths:
        if os.path.exists(path):
            try:
                os.add_dll_directory(path)
                print(f"✓ Added DLL directory: {path}")
            except Exception as e:
                print(f"⚠ Failed to add {path}: {e}")
        else:
            print(f"✗ Directory not found: {path}")
    
    print()
    
    # Add module search paths
    module_paths = [
        'ai_platform_trainer/Release',
        'ai_platform_trainer/cpp',
        'ai_platform_trainer/cpp/build/Release'
    ]
    
    print("Adding module search paths:")
    for path in module_paths:
        if os.path.exists(path):
            sys.path.insert(0, path)
            print(f"✓ Added module path: {path}")
        else:
            print(f"✗ Path not found: {path}")
    
    print()
    
    # Find all .pyd files
    print("Searching for .pyd files:")
    pyd_files = []
    for pattern in ["ai_platform_trainer/**/*.pyd", "**/*.pyd"]:
        files = glob.glob(pattern, recursive=True)
        pyd_files.extend(files)
    
    if pyd_files:
        for pyd in pyd_files:
            print(f"✓ Found: {pyd}")
    else:
        print("✗ No .pyd files found")
    
    print()
    
    # Try importing the extension
    print("Attempting to import CUDA extension:")
    try:
        import gpu_environment
        print("🎉 SUCCESS: CUDA extension imported successfully!")
        
        # Test basic functionality if possible
        try:
            # Check if the module has expected attributes
            attrs = dir(gpu_environment)
            print(f"Module attributes: {attrs[:5]}..." if len(attrs) > 5 else f"Module attributes: {attrs}")
            
            # Try to get device info if available
            if hasattr(gpu_environment, 'get_device_count') or hasattr(gpu_environment, 'test'):
                print("✓ Module has expected functionality")
            
        except Exception as e:
            print(f"⚠ Module imported but functionality test failed: {e}")
            
    except ImportError as e:
        print(f"❌ FAILED: Import error: {e}")
        
        # Additional debugging
        print("\nDebugging information:")
        print(f"Python version: {sys.version}")
        print(f"Platform: {sys.platform}")
        print(f"Current working directory: {os.getcwd()}")
        print(f"Python path: {sys.path[:3]}...")
        
        # Check if the specific file exists and get more info
        pyd_path = "ai_platform_trainer/Release/gpu_environment.cp313-win_amd64.pyd"
        if os.path.exists(pyd_path):
            stat = os.stat(pyd_path)
            print(f"Module file exists: {pyd_path} ({stat.st_size} bytes)")
            
            # Try to get more specific error info
            try:
                import importlib.util
                spec = importlib.util.spec_from_file_location("gpu_environment", pyd_path)
                module = importlib.util.module_from_spec(spec)
                spec.loader.exec_module(module)
                print("✓ Module loaded successfully via importlib")
            except Exception as e2:
                print(f"❌ importlib also failed: {e2}")
        
    except Exception as e:
        print(f"❌ FAILED: Unexpected error: {e}")
    
    print("\n=== End Test ===")

if __name__ == "__main__":
    test_cuda_import()
