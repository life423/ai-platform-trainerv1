# file: ai_platform_trainer/core/launcher_refactored.py
"""
Launcher module for the refactored AI Platform Trainer game.
Uses the state machine architecture for game flow management.
"""

from ai_platform_trainer.core.logging_config import setup_logging
from ai_platform_trainer.gameplay.game_refactored import Game


def main():
    """
    Main entry point for the AI Platform Trainer application.
    Initializes and runs the game with state machine architecture.
    """
    setup_logging()
    game = Game()
    game.run()
