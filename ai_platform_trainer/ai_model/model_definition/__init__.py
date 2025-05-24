"""
Model definition module for the AI Platform Trainer.

DEPRECATED: This module is deprecated. Use ai_platform_trainer.ai.models instead.
"""
import warnings

# Add deprecation warning
warnings.warn(
    "The ai_model.model_definition module is deprecated. Use ai_platform_trainer.ai.models instead.",
    DeprecationWarning,
    stacklevel=2
)

# Import from the new location for backward compatibility
from ai_platform_trainer.ai.models.enemy_movement_model import EnemyMovementModel

__all__ = [
    'EnemyMovementModel',
]