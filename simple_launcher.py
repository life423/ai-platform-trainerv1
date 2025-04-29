#!/usr/bin/env python
"""
AI Platform Trainer - Simple Launcher

Based on the detected module structure.
"""

import sys
from pathlib import Path


def main():
    """Run the game."""
    # Add the project root to path
    project_root = Path(__file__).parent.absolute()
    sys.path.insert(0, str(project_root))

    print("Starting AI Platform Trainer...")

    try:
        # This path is correct based on the modules we found
        from src.ai_platform_trainer.core.game import Game

        print("Using core.game...")
        game = Game()
        game.run()
        return True
    except ImportError as e:
        print(f"Error importing core.game: {e}")
        try:
            # Look for a Game class in core module
            import src.ai_platform_trainer.core as core

            possible_games = [attr for attr in dir(core) if "Game" in attr]
            if possible_games:
                print(f"Found possible game classes: {possible_games}")
                for game_class in possible_games:
                    try:
                        game = getattr(core, game_class)()
                        if hasattr(game, "run"):
                            print(f"Running with {game_class}...")
                            game.run()
                            return True
                    except Exception as game_error:
                        print(f"Error with {game_class}: {game_error}")
        except ImportError:
            print("Could not import core module")

        # Try running a task from tasks.json if available
        try:
            print("Attempting to run VS Code task...")
            import subprocess

            result = subprocess.run(
                [
                    "python",
                    "-c",
                    "import runpy; runpy.run_module('src.ai_platform_trainer.main', run_name='__main__')",
                ],
                cwd=project_root,
            )
            if result.returncode == 0:
                return True
        except Exception as task_error:
            print(f"Error running VS Code task: {task_error}")

    print("\nCould not start the game. Please try:")
    print("1. Install dependencies: pip install -r requirements.txt")
    print("2. Reinstall the package: pip install -e .")

    return False


if __name__ == "__main__":
    main()
    main()
