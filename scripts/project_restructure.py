"""
Project Restructuring Tool for AI Platform Trainer.

This script reorganizes the project structure to improve modularity
and maintainability. It creates necessary directories and moves files
to appropriate locations.
"""

import os
import shutil


# Define the new project structure
NEW_STRUCTURE = {
    "ai_platform_trainer": {
        "ai": {
            "models": {},  # Neural network model definitions
            "training": {},  # Training logic
            "inference": {},  # Inference logic
            "visualization": {},  # Training visualization
        },
        "engine": {
            "core": {},  # Core engine components
            "rendering": {},  # Rendering systems
            "input": {},  # Input handling
            "physics": {},  # Physics and collision systems
        },
        "entities": {
            "components": {},  # Entity components
            "systems": {},  # Entity systems
            "behaviors": {},  # Entity behaviors
        },
        "gameplay": {
            "modes": {},  # Game modes
            "mechanics": {},  # Game mechanics
            "ui": {},  # User interface components
        },
        "utils": {},  # Utility functions
    },
    "assets": {
        "sprites": {"player": {}, "enemy": {}, "missile": {}, "effects": {}, "ui": {}},
        "audio": {"sfx": {}, "music": {}},
        "fonts": {},
    },
    "scripts": {  # Command-line scripts
        "training": {},  # Training scripts
        "tools": {},  # Development tools
        "analysis": {},  # Analysis tools
    },
    "logs": {  # Log files
        "training": {},  # Training logs
        "gameplay": {},  # Gameplay logs
    },
    "tests": {  # Tests
        "unit": {},  # Unit tests
        "integration": {},  # Integration tests
        "performance": {},  # Performance tests
    },
    "docs": {  # Documentation
        "api": {},  # API documentation
        "guides": {},  # User guides
        "design": {},  # Design documentation
    },
}


# File mapping: old path -> new path
FILE_MAPPINGS = {
    # AI modules
    "ai_platform_trainer/ai_model/enemy_movement_model.py": "ai_platform_trainer/ai/models/enemy_movement_model.py",
    "ai_platform_trainer/ai_model/simple_missile_model.py": "ai_platform_trainer/ai/models/simple_missile_model.py",
    "ai_platform_trainer/ai_model/train_missile_model.py": "ai_platform_trainer/ai/training/train_missile_model.py",
    "ai_platform_trainer/ai_model/missile_dataset.py": "ai_platform_trainer/ai/training/missile_dataset.py",
    "ai_platform_trainer/ai_model/enemy_rl_agent.py": "ai_platform_trainer/ai/models/enemy_rl_agent.py",
    "ai_platform_trainer/ai_model/train_enemy_rl.py": "ai_platform_trainer/ai/training/train_enemy_rl.py",
    "ai_platform_trainer/ai_model/training_monitor.py": "ai_platform_trainer/ai/visualization/training_monitor.py",
    # Core engine components
    "ai_platform_trainer/core/config_manager.py": "ai_platform_trainer/engine/core/config_manager.py",
    "ai_platform_trainer/core/data_logger.py": "ai_platform_trainer/engine/core/data_logger.py",
    "ai_platform_trainer/core/interfaces.py": "ai_platform_trainer/engine/core/interfaces.py",
    "ai_platform_trainer/core/launcher.py": "ai_platform_trainer/engine/core/launcher.py",
    "ai_platform_trainer/core/launcher_di.py": "ai_platform_trainer/engine/core/launcher_di.py",
    "ai_platform_trainer/core/launcher_refactored.py": "ai_platform_trainer/engine/core/launcher_refactored.py",
    "ai_platform_trainer/core/logging_config.py": "ai_platform_trainer/engine/core/logging_config.py",
    "ai_platform_trainer/core/service_locator.py": "ai_platform_trainer/engine/core/service_locator.py",
    # Entity components
    "ai_platform_trainer/entities/enemy.py": "ai_platform_trainer/entities/components/enemy.py",
    "ai_platform_trainer/entities/enemy_play.py": "ai_platform_trainer/entities/components/enemy_play.py",
    "ai_platform_trainer/entities/enemy_training.py": "ai_platform_trainer/entities/components/enemy_training.py",
    "ai_platform_trainer/entities/player.py": "ai_platform_trainer/entities/components/player.py",
    "ai_platform_trainer/entities/player_play.py": "ai_platform_trainer/entities/components/player_play.py",
    "ai_platform_trainer/entities/player_training.py": "ai_platform_trainer/entities/components/player_training.py",
    "ai_platform_trainer/entities/missile.py": "ai_platform_trainer/entities/components/missile.py",
    "ai_platform_trainer/entities/entity_factory.py": "ai_platform_trainer/entities/systems/entity_factory.py",
    # Gameplay
    "ai_platform_trainer/gameplay/ai/enemy_ai_controller.py": "ai_platform_trainer/entities/behaviors/enemy_ai_controller.py",
    "ai_platform_trainer/gameplay/missile_ai_controller.py": "ai_platform_trainer/entities/behaviors/missile_ai_controller.py",
    "ai_platform_trainer/gameplay/state_machine.py": "ai_platform_trainer/gameplay/mechanics/state_machine.py",
    "ai_platform_trainer/gameplay/collisions.py": "ai_platform_trainer/engine/physics/collisions.py",
    "ai_platform_trainer/gameplay/common_utils.py": "ai_platform_trainer/utils/common_utils.py",
    "ai_platform_trainer/gameplay/config.py": "ai_platform_trainer/engine/core/game_config.py",
    "ai_platform_trainer/gameplay/display_manager.py": "ai_platform_trainer/engine/rendering/display_manager.py",
    "ai_platform_trainer/gameplay/game.py": "ai_platform_trainer/engine/core/game.py",
    "ai_platform_trainer/gameplay/game_di.py": "ai_platform_trainer/engine/core/game_di.py",
    "ai_platform_trainer/gameplay/game_refactored.py": "ai_platform_trainer/engine/core/game_refactored.py",
    "ai_platform_trainer/gameplay/input_handler.py": "ai_platform_trainer/engine/input/input_handler.py",
    "ai_platform_trainer/gameplay/menu.py": "ai_platform_trainer/gameplay/ui/menu.py",
    "ai_platform_trainer/gameplay/renderer.py": "ai_platform_trainer/engine/rendering/renderer.py",
    "ai_platform_trainer/gameplay/renderer_di.py": "ai_platform_trainer/engine/rendering/renderer_di.py",
    "ai_platform_trainer/gameplay/spawn_utils.py": "ai_platform_trainer/engine/physics/spawn_utils.py",
    "ai_platform_trainer/gameplay/spawner.py": "ai_platform_trainer/engine/physics/spawner.py",
    # These files remain in their current location since they're already in the right structure
    # "ai_platform_trainer/gameplay/modes/play_mode.py": "ai_platform_trainer/gameplay/modes/play_mode.py",
    # "ai_platform_trainer/gameplay/modes/training_mode.py": "ai_platform_trainer/gameplay/modes/training_mode.py",
    # Utilities
    "ai_platform_trainer/utils/helpers.py": "ai_platform_trainer/utils/helpers.py",
    "ai_platform_trainer/utils/sprite_manager.py": "ai_platform_trainer/engine/rendering/sprite_manager.py",
    # Scripts
    "train_enemy_rl_model.py": "scripts/training/train_enemy_rl.py",
    "generate_sprites.py": "scripts/tools/generate_sprites.py",
    "run_game.bat": "scripts/run_game.bat",
    "run_tests.bat": "scripts/run_tests.bat",
}


# Entry point files that need to be updated with new imports
ENTRY_POINTS = [
    "ai_platform_trainer/main.py",
]


# Function to create directories based on structure dictionary


def create_directory_structure(structure, base_path=""):
    for name, contents in structure.items():
        path = os.path.join(base_path, name)
        os.makedirs(path, exist_ok=True)

        # Create __init__.py files
        if name.startswith("ai_platform_trainer") or (
            base_path and "ai_platform_trainer" in base_path
        ):
            init_file = os.path.join(path, "__init__.py")
            if not os.path.exists(init_file):
                with open(init_file, "w") as f:
                    f.write('"""Package initialization file."""\n')

        # Recursively create subdirectories
        if contents:  # If not empty
            create_directory_structure(contents, path)


# Function to move files according to mapping


def move_files(file_mappings):
    for old_path, new_path in file_mappings.items():
        if os.path.exists(old_path):
            # Create target directory if it doesn't exist
            os.makedirs(os.path.dirname(new_path), exist_ok=True)

            # Copy file to new location
            shutil.copy2(old_path, new_path)
            print(f"Moved: {old_path} -> {new_path}")
        else:
            print(f"Warning: {old_path} does not exist, skipping")


# Update imports in moved files


def update_imports(file_mappings):
    # Build a mapping of old module paths to new ones
    module_mappings = {}
    for old_path, new_path in file_mappings.items():
        if old_path.endswith(".py"):
            old_module = old_path.replace("/", ".").replace(".py", "")
            new_module = new_path.replace("/", ".").replace(".py", "")
            module_mappings[old_module] = new_module

    # Update imports in all moved files
    for _, new_path in file_mappings.items():
        if os.path.exists(new_path) and new_path.endswith(".py"):
            with open(new_path, "r") as f:
                content = f.read()

            # Replace imports
            for old_module, new_module in module_mappings.items():
                # Match various import patterns
                content = content.replace(
                    f"from {old_module} import", f"from {new_module} import"
                )
                content = content.replace(
                    f"import {old_module}", f"import {new_module}"
                )
                content = content.replace(
                    f"import {old_module} as", f"import {new_module} as"
                )

            # Write updated content
            with open(new_path, "w") as f:
                f.write(content)

            print(f"Updated imports in: {new_path}")


# Copy sprite files to the new assets structure


def reorganize_assets():
    # Create assets directories if they don't exist
    os.makedirs("assets/sprites/player", exist_ok=True)
    os.makedirs("assets/sprites/enemy", exist_ok=True)
    os.makedirs("assets/sprites/missile", exist_ok=True)
    os.makedirs("assets/sprites/effects", exist_ok=True)
    os.makedirs("assets/sprites/ui", exist_ok=True)

    # Move sprite files to categorized directories
    if os.path.exists("assets/sprites"):
        for file in os.listdir("assets/sprites"):
            src = os.path.join("assets/sprites", file)
            if not os.path.isfile(src):
                continue

            # Determine destination based on filename
            if file.startswith("player"):
                dst = os.path.join("assets/sprites/player", file)
            elif file.startswith("enemy"):
                dst = os.path.join("assets/sprites/enemy", file)
            elif file.startswith("missile"):
                dst = os.path.join("assets/sprites/missile", file)
            elif file.startswith("explosion"):
                dst = os.path.join("assets/sprites/effects", file)
            else:
                dst = os.path.join("assets/sprites/ui", file)

            # Move the file
            if src != dst:
                shutil.copy2(src, dst)
                print(f"Reorganized asset: {src} -> {dst}")


# Create a new main entry point


def create_new_main():
    """Create a new main.py file with updated imports."""
    content = '''"""
Main entry point for AI Platform Trainer.

This module initializes and runs the game.
"""
from ai_platform_trainer.engine.core.launcher import main

if __name__ == "__main__":
    main()
'''

    with open("ai_platform_trainer/main.py", "w") as f:
        f.write(content)

    print("Created new main.py with updated imports")


def main():
    """Execute the project restructuring."""
    print("Starting project restructuring...")

    # Create the new directory structure
    print("Creating directory structure...")
    create_directory_structure(NEW_STRUCTURE)

    # Move files to their new locations
    print("Moving files...")
    move_files(FILE_MAPPINGS)

    # Update imports in moved files
    print("Updating imports...")
    update_imports(FILE_MAPPINGS)

    # Reorganize assets
    print("Reorganizing assets...")
    reorganize_assets()

    # Create new main entry point
    print("Creating new main entry point...")
    create_new_main()

    print("Project restructuring completed!")
    print(
        "Note: The original files have not been deleted. After verifying that the restructured project works correctly, you may want to delete the old files."
    )


if __name__ == "__main__":
    main()
