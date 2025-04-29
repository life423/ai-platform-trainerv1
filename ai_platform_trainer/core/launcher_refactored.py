"""
State Machine Launcher Module (Adapter)

This is an adapter module that forwards to the canonical implementation
in engine/core/unified_launcher.py for backward compatibility.

DEPRECATED: Use ai_platform_trainer.engine.core.unified_launcher instead.
"""

import warnings
import os
from ai_platform_trainer.engine.core.unified_launcher import main

# Add deprecation warning
warnings.warn(
    "Importing from ai_platform_trainer.core.launcher_refactored is deprecated. "
    "Use ai_platform_trainer.engine.core.unified_launcher instead.",
    DeprecationWarning,
    stacklevel=2,
)

# Override environment variable to use state machine mode
os.environ["AI_PLATFORM_LAUNCHER_MODE"] = "STATE_MACHINE"

# For backwards compatibility, provide the main function
__all__ = ["main"]
