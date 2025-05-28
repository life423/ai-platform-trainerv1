"""
Game Environment Module for AI Platform Trainer.

This module contains the core game environment, entity management,
and reward computation systems for both supervised and reinforcement learning.
"""

from .environment import GameEnvironment
from .rewards import RewardSystem

__all__ = ['GameEnvironment', 'RewardSystem']
