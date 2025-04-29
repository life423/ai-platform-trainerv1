"""
Dependency Injection Launcher Module (Adapter)

This is an adapter module that forwards to the canonical implementation
in engine/core/unified_launcher.py for backward compatibility.

DEPRECATED: Use ai_platform_trainer.engine.core.unified_launcher instead.
"""

import warnings
import os
from ai_platform_trainer.engine.core.unified_launcher import main

# Keep register_services function for backward compatibility

# Re-export the service registration functionality
# This is needed for the unified launcher to work with the DI system
from ai_platform_trainer.engine.core.launcher_di import register_services

# Add deprecation warning
warnings.warn(
    "Importing from ai_platform_trainer.core.launcher_di is deprecated. "
    "Use ai_platform_trainer.engine.core.unified_launcher instead.",
    DeprecationWarning,
    stacklevel=2,
)

# Override environment variable to use DI mode
os.environ["AI_PLATFORM_LAUNCHER_MODE"] = "DI"

# For backwards compatibility, provide the main function
__all__ = ["main", "register_services"]
