#!/usr/bin/env python
"""
AI Platform Trainer - Universal Launcher

This script provides a robust entry point for the AI Platform Trainer,
handling different module structures and providing helpful error messages.
"""

import importlib.util
import os
import sys
import traceback
from pathlib import Path


def print_header(text):
    """Print a header with the text."""
    print("\n" + "=" * 80)
    print(f"  {text}")
    print("=" * 80)


def find_module_path(module_name, search_paths=None):
    """Find the path to a module by searching in different locations."""
    if search_paths is None:
        search_paths = sys.path

    for path in search_paths:
        path = Path(path)
        # Check for direct module
        module_path = path / f"{module_name}.py"
        if module_path.exists():
            return str(module_path)
        # Check for package
        init_path = path / module_name / "__init__.py"
        if init_path.exists():
            return str(init_path.parent)
    return None


def try_import(name):
    """Try to import a module and return None if it fails."""
    try:
        return importlib.import_module(name)
    except ImportError:
        return None


def run_game():
    """Attempt to run the game using various approaches."""
    # Add the project root and src directories to path if needed
    project_root = Path(__file__).parent.absolute()
    sys.path.insert(0, str(project_root))

    src_dir = project_root / "src"
    if src_dir.exists():
        sys.path.insert(0, str(src_dir))

    print_header("AI Platform Trainer")
    print(f"Project root: {project_root}")

    # Try all possible entry points
    entry_points = [
        # Standard module paths
        "ai_platform_trainer.main",
        "ai_platform_trainer.gameplay.game",
        "ai_platform_trainer.engine.core.game",
        # Legacy/restructured paths
        "src.ai_platform_trainer.main",
        "src.ai_platform_trainer.gameplay.game",
    ]

    for entry_point in entry_points:
        parts = entry_point.split(".")
        module_name = ".".join(parts[:-1])
        attr_name = parts[-1]

        try:
            print(f"Attempting to load {entry_point}...")
            module = importlib.import_module(module_name)

            if hasattr(module, attr_name):
                attr = getattr(module, attr_name)
                if callable(attr):
                    print(f"Starting game with {entry_point}...")
                    attr()
                    return True
                else:
                    # Try to create an instance and call run()
                    if attr_name[0].isupper():  # Likely a class
                        instance = attr()
                        if hasattr(instance, "run") and callable(instance.run):
                            print(f"Starting game with {entry_point}.run()...")
                            instance.run()
                            return True
            else:
                # Last resort - import as module and try to run
                try:
                    full_module = importlib.import_module(entry_point)
                    if hasattr(full_module, "main"):
                        print(f"Starting game with {entry_point}.main()...")
                        full_module.main()
                        return True
                    elif hasattr(full_module, "run"):
                        print(f"Starting game with {entry_point}.run()...")
                        full_module.run()
                        return True
                except ImportError:
                    pass

        except ImportError as e:
            print(f"  Failed to import {entry_point}: {e}")
        except Exception as e:
            print(f"  Error running {entry_point}: {e}")
            print(traceback.format_exc())

    print_header("ERROR: Failed to start the game")
    print("Could not find a valid entry point. Please check your installation.")
    print("\nSuggestions for troubleshooting:")
    print("1. Install required dependencies: pip install -r requirements.txt")
    print("2. Install the package in development mode: pip install -e .")
    print("3. Check for import errors in your code")

    # Print available Python path
    print("\nPython path:")
    for p in sys.path:
        print(f"  {p}")

    # Print available modules
    print("\nAvailable ai_platform_trainer modules:")
    try:
        import ai_platform_trainer

        print(f"  Package found at: {ai_platform_trainer.__file__}")

        for root, dirs, files in os.walk(os.path.dirname(ai_platform_trainer.__file__)):
            rel_path = os.path.relpath(
                root, os.path.dirname(ai_platform_trainer.__file__)
            )
            if rel_path == ".":
                rel_path = ""
            else:
                rel_path = rel_path.replace(os.sep, ".")

            for f in files:
                if f.endswith(".py") and f != "__pycache__":
                    if rel_path:
                        print(f"  ai_platform_trainer.{rel_path}.{f[:-3]}")
                    else:
                        print(f"  ai_platform_trainer.{f[:-3]}")
    except ImportError:
        print("  Could not import ai_platform_trainer package")

    return False


if __name__ == "__main__":
    run_game()
    run_game()
