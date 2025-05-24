"""
Standard Game class for AI Platform Trainer.

This module provides the standard game implementation that uses the core game logic.
"""
import logging
from typing import Optional

from ai_platform_trainer.gameplay.game_core import GameCore


class Game(GameCore):
    """
    Standard game implementation that extends the core game logic.
    
    This class provides backward compatibility with the original game implementation.
    """
    
    def __init__(self) -> None:
        """Initialize the standard game implementation."""
        super().__init__(use_state_machine=False)
        logging.info("Using standard game implementation")
        
    # All functionality is inherited from GameCore