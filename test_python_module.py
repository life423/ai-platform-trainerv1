#!/usr/bin/env python3

import sys

sys.path.insert(0, 'build/Release')
import cudarl_core_python as core

print('Testing Environment Creation and Operations...')

# Create environment
env = core.create_environment(grid_width=16, grid_height=16, batch_size=32)
print(f'Environment created, CUDA enabled: {env.is_cuda_enabled()}')

# Test benchmark
print('Running benchmark...')
result = core.benchmark_environment(batch_size=32, num_steps=1000)
print(f'Steps per second: {result["steps_per_second"]:,.0f}')
print(f'Total steps: {result["total_steps"]:,}')
print(f'Duration: {result["duration_ms"]}ms')
print(f'CUDA enabled: {result["cuda_enabled"]}')

# Test larger benchmark for performance comparison
print('\nRunning larger benchmark...')
result_large = core.benchmark_environment(batch_size=1024, num_steps=10000)
print(f'Large batch steps per second: {result_large["steps_per_second"]:,.0f}')
print(f'Large batch total steps: {result_large["total_steps"]:,}')
print(f'Large batch duration: {result_large["duration_ms"]}ms')

print('\nPython module test completed successfully!')
print('\nPython module test completed successfully!')
