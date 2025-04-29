"""
Standard Launcher Module (Engine Version)

This module is kept for backward compatibility but forwards to the unified launcher.
Use ai_platform_trainer.engine.core.unified_launcher directly for new code.
"""

import warnings
from ai_platform_trainer.engine.core.unified_launcher import main

# Add deprecation warning
warnings.warn(
    "Importing from ai_platform_trainer.engine.core.launcher is deprecated. "
    "Use ai_platform_trainer.engine.core.unified_launcher instead.",
    DeprecationWarning,
    stacklevel=2,
)

# For backwards compatibility, provide the main function
__all__ = ["main"]
