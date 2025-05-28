#!/usr/bin/env python
"""
Direct test of the C++ extension without the Python wrapper
"""
import os
import sys
import platform

def setup_cuda_dlls():
    """Setup CUDA DLLs on Windows"""
    if platform.system() == "Windows":
        cuda_path = os.environ.get('CUDA_PATH')
        if cuda_path:
            cuda_bin = os.path.join(cuda_path, 'bin')
            if os.path.exists(cuda_bin):
                try:
                    os.add_dll_directory(cuda_bin)
                    print(f"✓ Added CUDA DLL directory: {cuda_bin}")
                except Exception as e:
                    print(f"Failed to add CUDA DLL directory: {e}")

def test_cpp_extension():
    """Test the C++ extension directly"""
    print("="*60)
    print("Direct C++ Extension Test")
    print("="*60)
    
    setup_cuda_dlls()
    
    # Add Release directory to path (where the .pyd file is)
    release_dir = os.path.join(os.path.dirname(__file__), "ai_platform_trainer", "Release")
    if release_dir not in sys.path:
        sys.path.insert(0, release_dir)
        print(f"✓ Added Release directory to path: {release_dir}")
    
    try:
        # Import the compiled extension directly from Release directory
        print("\n1. Importing C++ extension directly...")
        import gpu_environment
        print("✓ Successfully imported C++ extension")
        
        # Check attributes
        print(f"\nC++ Extension attributes: {sorted(dir(gpu_environment))}")
        
        # Test for expected classes
        expected_attrs = ['EnvironmentConfig', 'Environment']
        missing = []
        found = []
        
        for attr in expected_attrs:
            if hasattr(gpu_environment, attr):
                obj = getattr(gpu_environment, attr)
                found.append(f"✓ {attr} ({type(obj).__name__})")
            else:
                missing.append(f"✗ {attr}")
        
        print("\nExpected interface check:")
        for item in found:
            print(f"  {item}")
        for item in missing:
            print(f"  {item}")
        
        if missing:
            print(f"\n❌ Missing {len(missing)} expected attributes!")
            return False
        
        # Test functionality
        print("\n2. Testing C++ extension functionality...")
        
        # Test EnvironmentConfig
        config = gpu_environment.EnvironmentConfig()
        print(f"✓ Created EnvironmentConfig")
        
        # Check config attributes
        config_attrs = dir(config)
        print(f"Config attributes: {sorted([attr for attr in config_attrs if not attr.startswith('_')])}")
        
        # Test Environment
        env = gpu_environment.Environment(config)
        print(f"✓ Created Environment")
        
        # Test methods
        import numpy as np
        
        # Reset
        obs = env.reset(42)
        print(f"✓ env.reset() returned: {type(obs).__name__} with shape {obs.shape}")
        
        # Step
        action = np.array([0.5, -0.5], dtype=np.float32)
        result = env.step(action)
        print(f"✓ env.step() returned: {type(result)}")
        
        if isinstance(result, tuple):
            print(f"  Step result tuple length: {len(result)}")
            if len(result) >= 2:
                obs, reward = result[0], result[1]
                print(f"  Observation type: {type(obs)}, Reward: {reward}")
        
        print("\n✅ C++ extension interface is working!")
        return True
        
    except ImportError as e:
        print(f"❌ Failed to import C++ extension: {e}")
        return False
    except Exception as e:
        print(f"❌ Error testing C++ extension: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = test_cpp_extension()
    if success:
        print("\n🎉 The C++ extension has the correct interface!")
    else:
        print("\n💥 C++ extension interface test failed!")
    sys.exit(0 if success else 1)
