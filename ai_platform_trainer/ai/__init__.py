"""
AI module for the AI Platform Trainer.

This package contains all AI-related components including models,
training pipelines, and inference logic.
"""

# Import key classes for easier access
from ai_platform_trainer.ai.models.missile_model import MissileModel, SimpleMissileModel
from ai_platform_trainer.ai.models.enemy_movement_model import EnemyMovementModel
from ai_platform_trainer.ai.models.enemy_rl_agent import EnemyGameEnv

__all__ = [
    'MissileModel',
    'SimpleMissileModel',  # For backward compatibility
    'EnemyMovementModel',
    'EnemyGameEnv',
]