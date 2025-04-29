#!/usr/bin/env python
"""
Direct entry point for AI Platform Trainer.

This script provides a simplified entry point that bypasses the unified launcher
and directly runs the game in standard mode.
"""

import os
import sys


def run_game():
    try:
        # Try to run using the standard game mode
        from ai_platform_trainer.engine.core.game import Game

        print("Starting AI Platform Trainer in standard mode...")
        game = Game()
        game.run()
    except ImportError as e:
        print(f"Error importing standard game: {e}")
        # Try to run using gameplay.game directly
        try:
            from ai_platform_trainer.gameplay.game import Game

            print("Starting AI Platform Trainer using legacy game...")
            game = Game()
            game.run()
        except ImportError as e2:
            print(f"Error importing legacy game: {e2}")
            print("Unable to start the game.")
            sys.exit(1)
    except Exception as e:
        print(f"Error running game: {e}")
        sys.exit(1)


if __name__ == "__main__":
    # Add the project root to sys.path
    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    if project_root not in sys.path:
        sys.path.insert(0, project_root)

    run_game()
    run_game()
