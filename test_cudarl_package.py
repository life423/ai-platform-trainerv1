#!/usr/bin/env python3

# Test the updated CudaRL package
import os
import sys

# Add the python package to path
sys.path.insert(0, 'python')

print("Testing CudaRL-Arena Package...")
print("=" * 50)

# Test package import
try:
    import cudarl
    print("✓ Package imported successfully")
    print(f"  CUDA Available: {cudarl.CUDA_AVAILABLE}")
    print(f"  Version: {cudarl.VERSION}")
    
    # Test device info
    device_info = cudarl.get_cuda_device_info()
    print(f"  Device Count: {device_info.device_count}")
    print(f"  Device Name: {device_info.device_name}")
    
    # Test environment creation
    print("\nTesting Environment Creation...")
    env = cudarl.create_environment(grid_width=16, grid_height=16, batch_size=64)
    print(f"✓ Environment created")
    
    if hasattr(env, 'is_cuda_enabled'):
        print(f"  CUDA Enabled: {env.is_cuda_enabled()}")
    
    # Test benchmarking
    print("\nRunning Performance Benchmark...")
    result = cudarl.benchmark_environment(batch_size=128, num_steps=1000)
    print(f"✓ Benchmark completed")
    print(f"  Steps per second: {result['steps_per_second']:,.0f}")
    print(f"  Total steps: {result['total_steps']:,}")
    print(f"  Duration: {result['duration_ms']:.1f}ms")
    print(f"  CUDA enabled: {result['cuda_enabled']}")
    
    # Test action constants
    print("\nTesting Action Constants...")
    print(f"  ACTION_UP: {cudarl.ACTION_UP}")
    print(f"  ACTION_DOWN: {cudarl.ACTION_DOWN}")
    print(f"  ACTION_LEFT: {cudarl.ACTION_LEFT}")
    print(f"  ACTION_RIGHT: {cudarl.ACTION_RIGHT}")
    
    print("\n" + "=" * 50)
    print("✓ All tests passed! CudaRL-Arena package is working correctly.")
    
except Exception as e:
    print(f"✗ Error testing package: {e}")
    import traceback
    traceback.print_exc()
