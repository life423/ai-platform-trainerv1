"""
AI models for the AI Platform Trainer.

This package contains neural network models and reinforcement learning agents.
"""

from ai_platform_trainer.ai.models.missile_model import MissileModel, SimpleMissileModel
from ai_platform_trainer.ai.models.enemy_movement_model import EnemyMovementModel
from ai_platform_trainer.ai.models.enemy_rl_agent import EnemyGameEnv

__all__ = [
    'MissileModel',
    'SimpleMissileModel',  # For backward compatibility
    'EnemyMovementModel',
    'EnemyGameEnv',
]