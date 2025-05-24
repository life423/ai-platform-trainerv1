"""
State Machine Game class for AI Platform Trainer.

DEPRECATED: This module is deprecated. Use ai_platform_trainer.gameplay.game instead.
"""
import warnings
import logging
from typing import Optional

from ai_platform_trainer.gameplay.game_core import GameCore

# Add deprecation warning
warnings.warn(
    "The game_refactored module is deprecated. Use ai_platform_trainer.gameplay.game instead.",
    DeprecationWarning,
    stacklevel=2
)


class Game(GameCore):
    """
    State Machine game implementation that extends the core game logic.
    
    DEPRECATED: This class is maintained for backward compatibility.
    """
    
    def __init__(self) -> None:
        """Initialize the state machine game implementation."""
        super().__init__(use_state_machine=True)
        logging.info("Using state machine game implementation (deprecated)")