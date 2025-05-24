"""
Dependency Injection Game class for AI Platform Trainer.

DEPRECATED: This module is deprecated. Use ai_platform_trainer.gameplay.game instead.
"""
import warnings
import logging
from typing import Optional

from ai_platform_trainer.gameplay.game_core import GameCore
from ai_platform_trainer.core.service_locator import ServiceLocator

# Add deprecation warning
warnings.warn(
    "The game_di module is deprecated. Use ai_platform_trainer.gameplay.game instead.",
    DeprecationWarning,
    stacklevel=2
)


class Game(GameCore):
    """
    Dependency Injection game implementation that extends the core game logic.
    
    DEPRECATED: This class is maintained for backward compatibility.
    """
    
    def __init__(self) -> None:
        """Initialize the DI game implementation."""
        # Initialize services if they don't exist
        self._register_services()
        
        super().__init__(use_state_machine=True)
        logging.info("Using DI game implementation (deprecated)")
        
    def _register_services(self) -> None:
        """Register required services if they don't exist."""
        from ai_platform_trainer.core.launcher_di import register_services
        
        # Only register if services aren't already registered
        try:
            ServiceLocator.get("config_manager")
        except KeyError:
            register_services()