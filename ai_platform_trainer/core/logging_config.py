"""
Logging Configuration Module (Adapter)

This is an adapter module that forwards to the canonical implementation
in engine/core/logging_config.py for backward compatibility.

DEPRECATED: Use ai_platform_trainer.engine.core.logging_config instead.
"""

import warnings
from ai_platform_trainer.core.adapter import CoreAdapter

# Import and re-export functions from engine/core
_logging_module = CoreAdapter.get_module("logging_config")
setup_logging = _logging_module.setup_logging

# Add deprecation warning
warnings.warn(
    "Importing from ai_platform_trainer.core.logging_config is deprecated. "
    "Use ai_platform_trainer.engine.core.logging_config instead.",
    DeprecationWarning,
    stacklevel=2,
)
