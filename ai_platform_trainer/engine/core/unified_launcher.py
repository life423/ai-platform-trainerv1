"""
Unified Launcher Module for AI Platform Trainer

DEPRECATED: This module is deprecated. Use the root unified_launcher.py instead.
"""
import warnings
import sys
from unified_launcher import main

# Add deprecation warning
warnings.warn(
    "Importing from ai_platform_trainer.engine.core.unified_launcher is deprecated. "
    "Use unified_launcher instead.",
    DeprecationWarning,
    stacklevel=2
)

# For backwards compatibility, provide the main function
__all__ = ["main"]

if __name__ == "__main__":
    sys.exit(main())