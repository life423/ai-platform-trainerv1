"""
Missile AI Controller module for game missile guidance.

DEPRECATED: This module is deprecated. Use ai_platform_trainer.ai.inference.missile_controller instead.
"""
import warnings

# Add deprecation warning
warnings.warn(
    "The gameplay.missile_ai_controller module is deprecated. "
    "Use ai_platform_trainer.ai.inference.missile_controller instead.",
    DeprecationWarning,
    stacklevel=2
)

# Import from the new location for backward compatibility
from ai_platform_trainer.ai.inference.missile_controller import update_missile_ai

__all__ = [
    'update_missile_ai',
]