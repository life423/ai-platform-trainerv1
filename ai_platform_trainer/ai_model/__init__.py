"""
AI model module for the AI Platform Trainer.

DEPRECATED: This module is deprecated. Use ai_platform_trainer.ai instead.
"""
import warnings

# Add deprecation warning
warnings.warn(
    "The ai_model module is deprecated. Use ai_platform_trainer.ai instead.",
    DeprecationWarning,
    stacklevel=2
)

# Import from the new location for backward compatibility
from ai_platform_trainer.ai.models.missile_model import MissileModel as SimpleMissileModel
from ai_platform_trainer.ai.models.enemy_movement_model import EnemyMovementModel
from ai_platform_trainer.ai.models.enemy_rl_agent import EnemyGameEnv

__all__ = [
    'SimpleMissileModel',
    'EnemyMovementModel',
    'EnemyGameEnv',
]