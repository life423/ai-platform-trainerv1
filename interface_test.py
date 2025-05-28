#!/usr/bin/env python
"""
Test script to diagnose interface mismatches between C++ extension and Python code
"""
import os
import sys
import platform
import logging

# Setup logging
logging.basicConfig(level=logging.DEBUG, 
                   format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def setup_environment():
    """Setup the environment for testing"""
    # Add CUDA DLLs on Windows
    if platform.system() == "Windows":
        cuda_path = os.environ.get('CUDA_PATH')
        if cuda_path:
            cuda_bin = os.path.join(cuda_path, 'bin')
            if os.path.exists(cuda_bin):
                try:
                    os.add_dll_directory(cuda_bin)
                    logger.info(f"Added CUDA DLL directory: {cuda_bin}")
                except:
                    pass
    
    # Add cpp directory to path
    script_dir = os.path.dirname(os.path.abspath(__file__))
    cpp_dir = os.path.join(script_dir, "ai_platform_trainer", "cpp")
    release_dir = os.path.join(script_dir, "ai_platform_trainer", "Release")
    
    for path in [release_dir, cpp_dir]:
        if os.path.exists(path) and path not in sys.path:
            sys.path.insert(0, path)
            logger.info(f"Added to Python path: {path}")

def test_interface():
    """Test the C++ extension interface"""
    print("="*60)
    print("Interface Mismatch Test")
    print("="*60)
    
    # Step 1: Check if module exists
    print("\n1. Checking if compiled extension exists...")
    import glob
    script_dir = os.path.dirname(os.path.abspath(__file__))
    
    # Check multiple possible locations
    search_paths = [
        os.path.join(script_dir, "ai_platform_trainer", "cpp"),
        os.path.join(script_dir, "ai_platform_trainer", "Release"),
        os.path.join(script_dir, "ai_platform_trainer", "cpp", "build", "Release")
    ]
    
    extensions = []
    for search_path in search_paths:
        extensions.extend(glob.glob(os.path.join(search_path, "gpu_environment*.pyd")))
        extensions.extend(glob.glob(os.path.join(search_path, "gpu_environment*.so")))
    
    if extensions:
        print(f"✓ Found extension: {extensions[0]}")
        print(f"  Size: {os.path.getsize(extensions[0]) / 1024:.1f} KB")
    else:
        print("✗ No compiled extension found!")
        print("  Run: python setup.py build_ext --inplace")
        return False
    
    # Step 2: Try to import the module
    print("\n2. Attempting to import gpu_environment...")
    setup_environment()
    
    try:
        import gpu_environment
        print("✓ Successfully imported gpu_environment")
    except ImportError as e:
        print(f"✗ Import failed: {e}")
        
        if "DLL load failed" in str(e):
            print("\nDLL Loading Error - Common causes:")
            print("- CUDA runtime DLLs not found")
            print("- Visual C++ Redistributables missing")
            print("- Python/CUDA version mismatch")
        return False
    
    # Step 3: Check module contents
    print("\n3. Checking module interface...")
    module_attrs = dir(gpu_environment)
    print(f"Module attributes: {sorted(module_attrs)}")
    
    # Expected interface
    expected = {
        'EnvironmentConfig': 'class',
        'Environment': 'class',
        'test_cuda_add': 'function'
    }
    
    missing = []
    found = []
    
    for attr, attr_type in expected.items():
        if hasattr(gpu_environment, attr):
            obj = getattr(gpu_environment, attr)
            found.append(f"✓ {attr} ({type(obj).__name__})")
        else:
            missing.append(f"✗ {attr} ({attr_type})")
    
    print("\nExpected interface:")
    for item in found:
        print(f"  {item}")
    for item in missing:
        print(f"  {item}")
    
    if missing:
        print(f"\n⚠️  Missing {len(missing)} expected attributes!")
        return False
    
    # Step 4: Test the interface
    print("\n4. Testing interface functionality...")
    
    try:
        # Test EnvironmentConfig
        config = gpu_environment.EnvironmentConfig()
        print(f"✓ Created EnvironmentConfig: {config}")
        
        # Check config attributes
        config_attrs = ['screen_width', 'screen_height', 'max_missiles', 
                       'enemy_speed', 'missile_speed']
        for attr in config_attrs:
            if hasattr(config, attr):
                value = getattr(config, attr)
                print(f"  - config.{attr} = {value}")
            else:
                print(f"  ✗ Missing: config.{attr}")
        
        # Test Environment
        env = gpu_environment.Environment(config)
        print(f"✓ Created Environment: {env}")
        
        # Test methods
        import numpy as np
        
        # Reset
        obs = env.reset(42)
        print(f"✓ env.reset() returned: {type(obs).__name__} with shape {obs.shape}")
        
        # Step
        action = np.array([0.5, -0.5], dtype=np.float32)
        result = env.step(action)
        if isinstance(result, tuple) and len(result) == 5:
            obs, reward, done, truncated, info = result
            print(f"✓ env.step() returned: (obs, reward={reward}, done={done}, "
                  f"truncated={truncated}, info={type(info).__name__})")
        else:
            print(f"✗ env.step() returned unexpected type: {type(result)}")
        
        # Get shapes
        obs_shape = env.get_observation_shape()
        action_shape = env.get_action_shape()
        print(f"✓ Observation shape: {obs_shape}")
        print(f"✓ Action shape: {action_shape}")
        
        print("\n✅ All interface tests passed!")
        return True
        
    except Exception as e:
        print(f"\n✗ Interface test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def suggest_fixes():
    """Suggest fixes for common issues"""
    print("\n" + "="*60)
    print("Suggested Fixes")
    print("="*60)
    
    print("\n1. If import fails with DLL error:")
    print("   - Ensure CUDA is installed and CUDA_PATH is set")
    print("   - Install Visual C++ Redistributables")
    print("   - Check PyTorch CUDA version matches system CUDA")
    
    print("\n2. If interface is missing expected attributes:")
    print("   - Check pybind/bindings.cpp exports all required classes")
    print("   - Rebuild: python setup.py build_ext --inplace")
    print("   - Ensure CMake found CUDA during build")
    
    print("\n3. If methods return wrong types:")
    print("   - Check PyBind11 return type annotations")
    print("   - Ensure numpy arrays are properly converted")
    print("   - Verify tuple unpacking in step() method")

if __name__ == "__main__":
    success = test_interface()
    
    if not success:
        suggest_fixes()
        sys.exit(1)
    else:
        print("\n🎉 GPU environment interface is working correctly!")
        sys.exit(0)
