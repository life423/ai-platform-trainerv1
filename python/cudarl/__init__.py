"""
CudaRL-Arena: CUDA-accelerated reinforcement learning environment

This package provides direct Python bindings to the C++/CUDA implementation
with smart fallback to CPU-only mode when CUDA is not available.
"""

# Smart import with fallback mechanism
CUDA_AVAILABLE = False
Environment = None
EnvironmentConfig = None

try:
    # Try to import the compiled C++/CUDA module
    from cudarl_core_python import ACTION_DOWN, ACTION_LEFT, ACTION_RIGHT, ACTION_UP
    from cudarl_core_python import CUDA_AVAILABLE as _CUDA_AVAILABLE
    from cudarl_core_python import VERSION, AgentState, CudaDeviceInfo
    from cudarl_core_python import Environment as _CudaEnvironment
    from cudarl_core_python import EnvironmentConfig as _CudaEnvironmentConfig
    from cudarl_core_python import (
        Observation,
        benchmark_environment,
        create_environment,
        get_cuda_device_info,
        is_cuda_available,
    )
    
    Environment = _CudaEnvironment
    EnvironmentConfig = _CudaEnvironmentConfig
    CUDA_AVAILABLE = _CUDA_AVAILABLE
    
    print(f"CudaRL-Arena v{VERSION} loaded with CUDA support: {CUDA_AVAILABLE}")
    
    if CUDA_AVAILABLE:
        device_info = get_cuda_device_info()
        print(f"Using GPU: {device_info.device_name}")
    
except ImportError as e:
    # Fallback to CPU-only mock implementation
    print(f"CUDA module not available ({e}), using CPU fallback")
    
    from .mock_env import MockAgentState as AgentState
    from .mock_env import MockCudaDeviceInfo as CudaDeviceInfo
    from .mock_env import MockEnvironment as Environment
    from .mock_env import MockEnvironmentConfig as EnvironmentConfig
    from .mock_env import MockObservation as Observation

    # Fallback implementations
    def is_cuda_available():
        return False
    
    def get_cuda_device_info():
        info = CudaDeviceInfo()
        info.device_count = 0
        info.device_name = "CPU (fallback)"
        return info
    
    def create_environment(grid_width=32, grid_height=32, batch_size=1):
        config = EnvironmentConfig()
        config.grid_width = grid_width
        config.grid_height = grid_height
        env = Environment(config)
        env.initialize(batch_size)
        return env
    
    def benchmark_environment(batch_size=32, num_steps=1000):
        env = create_environment(batch_size=batch_size)
        
        import time
        start = time.time()
        
        env.reset()
        actions = [0] * batch_size  # All move up
        
        for _ in range(num_steps):
            env.step(actions)
        
        end = time.time()
        duration_ms = (end - start) * 1000
        steps_per_second = (batch_size * num_steps * 1000) / duration_ms
        
        return {
            "steps_per_second": steps_per_second,
            "total_steps": batch_size * num_steps,
            "duration_ms": duration_ms,
            "cuda_enabled": False
        }
    
    # Action constants
    ACTION_UP = 0
    ACTION_DOWN = 1
    ACTION_LEFT = 2
    ACTION_RIGHT = 3
    
    VERSION = "1.0.0-cpu"
    
    print(f"CudaRL-Arena v{VERSION} loaded in CPU-only mode")

# Export public API
__all__ = [
    'Environment',
    'EnvironmentConfig', 
    'AgentState',
    'Observation',
    'CudaDeviceInfo',
    'is_cuda_available',
    'get_cuda_device_info',
    'create_environment',
    'benchmark_environment',
    'ACTION_UP',
    'ACTION_DOWN',
    'ACTION_LEFT',
    'ACTION_RIGHT',
    'CUDA_AVAILABLE',
    'VERSION'
]

__version__ = VERSION
