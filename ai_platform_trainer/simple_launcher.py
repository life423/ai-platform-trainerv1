#!/usr/bin/env python
"""
AI Platform Trainer - Simple Launcher

Launches the main game using the reorganized module structure.
"""

import sys
from pathlib import Path


def main():
    """Run the game."""
    # Add the project root to path (if needed)
    project_root = Path(__file__).parent.absolute()
    sys.path.insert(0, str(project_root))

    print("Starting AI Platform Trainer...")

    try:
        from ai_platform_trainer.gameplay.game import Game
        print("Launching game...")
        game = Game()
        game.run()
        return True
    except ImportError as e:
        print(f"Error importing Game: {e}")
        print("\nCould not start the game. Please try:")
        print("1. Install dependencies: pip install -r requirements.txt")
        print("2. Reinstall the package: pip install -e .")
        return False


if __name__ == "__main__":
    main()
