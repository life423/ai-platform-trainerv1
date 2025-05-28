#!/usr/bin/env python3

"""
CudaRL-Arena System Validation and Final Report

This script runs comprehensive tests to validate the complete system
and generates a final Phase 2 completion report.
"""

import sys

sys.path.insert(0, 'python')

import json
import time
from datetime import datetime

import cudarl


def system_info_check():
    """Gather system information and capabilities."""
    print("System Information Check")
    print("=" * 40)
    
    info = {
        'timestamp': datetime.now().isoformat(),
        'cuda_available': cudarl.CUDA_AVAILABLE,
        'version': cudarl.VERSION,
        'python_version': sys.version,
    }
    
    if cudarl.CUDA_AVAILABLE:
        device_info = cudarl.get_cuda_device_info()
        info.update({
            'device_count': device_info.device_count,
            'device_name': device_info.device_name,
        })
        print(f"✓ CUDA Available: {cudarl.CUDA_AVAILABLE}")
        print(f"✓ GPU Device: {device_info.device_name}")
        print(f"✓ Device Count: {device_info.device_count}")
    else:
        print("✓ CPU Fallback Mode")
    
    print(f"✓ CudaRL Version: {cudarl.VERSION}")
    print()
    return info

def api_validation_test():
    """Test all public API functions."""
    print("API Validation Test")
    print("=" * 40)
    
    tests_passed = 0
    total_tests = 0
    
    # Test 1: Environment creation
    total_tests += 1
    try:
        env = cudarl.create_environment(grid_width=8, grid_height=8, batch_size=4)
        print("✓ Environment creation")
        tests_passed += 1
    except Exception as e:
        print(f"✗ Environment creation failed: {e}")
    
    # Test 2: Device info
    total_tests += 1
    try:
        device_info = cudarl.get_cuda_device_info()
        assert hasattr(device_info, 'device_count')
        assert hasattr(device_info, 'device_name')
        print("✓ Device info retrieval")
        tests_passed += 1
    except Exception as e:
        print(f"✗ Device info failed: {e}")
    
    # Test 3: CUDA availability check
    total_tests += 1
    try:
        is_available = cudarl.is_cuda_available()
        assert isinstance(is_available, bool)
        print("✓ CUDA availability check")
        tests_passed += 1
    except Exception as e:
        print(f"✗ CUDA availability check failed: {e}")
    
    # Test 4: Benchmark function
    total_tests += 1
    try:
        result = cudarl.benchmark_environment(batch_size=8, num_steps=100)
        required_keys = ['steps_per_second', 'total_steps', 'duration_ms', 'cuda_enabled']
        assert all(key in result for key in required_keys)
        print("✓ Benchmark function")
        tests_passed += 1
    except Exception as e:
        print(f"✗ Benchmark function failed: {e}")
    
    # Test 5: Action constants
    total_tests += 1
    try:
        actions = [cudarl.ACTION_UP, cudarl.ACTION_DOWN, 
                  cudarl.ACTION_LEFT, cudarl.ACTION_RIGHT]
        assert actions == [0, 1, 2, 3]
        print("✓ Action constants")
        tests_passed += 1
    except Exception as e:
        print(f"✗ Action constants failed: {e}")
    
    print(f"\nAPI Tests: {tests_passed}/{total_tests} passed")
    print()
    return tests_passed, total_tests

def performance_validation():
    """Validate performance across different scenarios."""
    print("Performance Validation")
    print("=" * 40)
    
    test_configs = [
        {'batch_size': 1, 'num_steps': 100, 'name': 'Single Environment'},
        {'batch_size': 32, 'num_steps': 1000, 'name': 'Medium Batch'},
        {'batch_size': 1024, 'num_steps': 1000, 'name': 'Large Batch'},
    ]
    
    results = []
    
    for config in test_configs:
        try:
            result = cudarl.benchmark_environment(
                batch_size=config['batch_size'],
                num_steps=config['num_steps']
            )
            
            results.append({
                'name': config['name'],
                'batch_size': config['batch_size'],
                'steps_per_second': result['steps_per_second'],
                'total_steps': result['total_steps'],
                'duration_ms': result['duration_ms'],
                'cuda_enabled': result['cuda_enabled']
            })
            
            print(f"✓ {config['name']:20} | "
                  f"{result['steps_per_second']:>10,.0f} steps/sec | "
                  f"{result['duration_ms']:>6.1f}ms")
            
        except Exception as e:
            print(f"✗ {config['name']} failed: {e}")
    
    print()
    return results

def memory_stress_test():
    """Test memory handling with large batch sizes."""
    print("Memory Stress Test")
    print("=" * 40)
    
    # Test progressively larger batch sizes
    batch_sizes = [1000, 5000, 10000, 20000]
    max_successful_batch = 0
    
    for batch_size in batch_sizes:
        try:
            start_time = time.time()
            result = cudarl.benchmark_environment(
                batch_size=batch_size,
                num_steps=100
            )
            end_time = time.time()
            
            max_successful_batch = batch_size
            print(f"✓ Batch {batch_size:>6,} | "
                  f"{result['steps_per_second']:>10,.0f} steps/sec | "
                  f"OK")
            
        except Exception as e:
            print(f"✗ Batch {batch_size:>6,} | Failed: {str(e)[:50]}...")
            break
    
    print(f"\nMax successful batch size: {max_successful_batch:,}")
    print()
    return max_successful_batch

def generate_final_report(system_info, api_results, performance_results, max_batch):
    """Generate comprehensive final report."""
    print("Generating Final Report")
    print("=" * 40)
    
    report = {
        'phase': 'Phase 2 - Complete',
        'timestamp': datetime.now().isoformat(),
        'system_info': system_info,
        'api_validation': {
            'tests_passed': api_results[0],
            'total_tests': api_results[1],
            'success_rate': api_results[0] / api_results[1] * 100
        },
        'performance_results': performance_results,
        'max_batch_size': max_batch,
        'capabilities': {
            'gpu_acceleration': system_info['cuda_available'],
            'peak_performance': max([r['steps_per_second'] for r in performance_results]),
            'production_ready': api_results[0] == api_results[1],
            'stress_tested': max_batch >= 10000
        }
    }
    
    # Save report to file
    with open('phase2_completion_report.json', 'w') as f:
        json.dump(report, f, indent=2)
    
    print("✓ Report saved to 'phase2_completion_report.json'")
    
    # Print summary
    print("\n" + "=" * 60)
    print("PHASE 2 COMPLETION SUMMARY")
    print("=" * 60)
    print(f"Status: {'✅ COMPLETE' if report['capabilities']['production_ready'] else '❌ INCOMPLETE'}")
    print(f"API Tests: {api_results[0]}/{api_results[1]} passed")
    print(f"GPU Acceleration: {'✅ Active' if system_info['cuda_available'] else '❌ Unavailable'}")
    print(f"Peak Performance: {report['capabilities']['peak_performance']:,.0f} steps/sec")
    print(f"Max Batch Size: {max_batch:,} environments")
    print(f"Production Ready: {'✅ Yes' if report['capabilities']['production_ready'] else '❌ No'}")
    
    if report['capabilities']['production_ready'] and system_info['cuda_available']:
        print("\n🎉 CudaRL-Arena Phase 2 is SUCCESSFULLY COMPLETE!")
        print("   Ready to proceed to Phase 3: Advanced RL & Godot Integration")
    else:
        print("\n⚠️  Some issues detected. Review test results before proceeding.")
    
    return report

if __name__ == "__main__":
    print("CudaRL-Arena System Validation")
    print("=" * 60)
    print()
    
    # Run all validation tests
    system_info = system_info_check()
    api_results = api_validation_test()
    performance_results = performance_validation()
    max_batch = memory_stress_test()
    
    # Generate final report
    report = generate_final_report(system_info, api_results, performance_results, max_batch)
    
    print(f"\nValidation complete. Total runtime: {time.time():.2f} seconds")
    print("All test results saved to 'phase2_completion_report.json'")
    
    print(f"\nValidation complete. Total runtime: {time.time():.2f} seconds")
    print("All test results saved to 'phase2_completion_report.json'")
