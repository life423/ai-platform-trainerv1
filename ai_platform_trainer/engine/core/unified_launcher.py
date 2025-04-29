"""
Unified Launcher Module for AI Platform Trainer

This module provides a consolidated entry point for the game with support for
different game initialization methods (dependency injection, standard, or state machine).
It selects the appropriate launcher based on settings and provides fallback mechanisms.
"""
import logging
import os
import sys
import traceback
from enum import Enum, auto
from typing import Any, Dict, Optional

# Import all launcher options
try:
    from ai_platform_trainer.core.logging_config import setup_logging
    from ai_platform_trainer.core.service_locator import ServiceLocator

    # Import game modes
    from ai_platform_trainer.engine.core.game import Game as StandardGame
<<<<<<< HEAD
    from ai_platform_trainer.gameplay.game_di import Game as DIGame
    from ai_platform_trainer.gameplay.game_refactored import Game as StateMachineGame
=======
    from ai_platform_trainer.engine.core.game_di import Game as DIGame
    from ai_platform_trainer.engine.core.game_refactored import Game as StateMachineGame
>>>>>>> d209db6 (refactor: update all game implementations to use engine/core versions)
except ImportError as e:
    print(f"Critical import error: {e}")
    print("Cannot initialize launcher system.")
    sys.exit(1)


class LauncherMode(Enum):
    """Enum for different launcher modes."""
    STANDARD = auto()
    DEPENDENCY_INJECTION = auto()
    STATE_MACHINE = auto()


def get_launcher_mode_from_settings() -> LauncherMode:
    """
    Determine which launcher mode to use based on settings.
    
    Returns:
        LauncherMode: The launcher mode to use
    """
    try:
        # First check environment variable
        env_mode = os.environ.get("AI_PLATFORM_LAUNCHER_MODE", "").upper()
        if env_mode == "STANDARD":
            return LauncherMode.STANDARD
        elif env_mode == "DI":
            return LauncherMode.DEPENDENCY_INJECTION
        elif env_mode == "STATE_MACHINE":
            return LauncherMode.STATE_MACHINE
        
        # Then check settings file
        try:
            from config_manager import load_settings
            settings = load_settings("settings.json")
            if settings and "launcher_mode" in settings:
                mode = settings["launcher_mode"].upper()
                if mode == "STANDARD":
                    return LauncherMode.STANDARD
                elif mode == "DI":
                    return LauncherMode.DEPENDENCY_INJECTION
                elif mode == "STATE_MACHINE":
                    return LauncherMode.STATE_MACHINE
        except Exception as settings_error:
            logging.warning(f"Could not load settings: {settings_error}")
            
        # Default to DI mode
        return LauncherMode.DEPENDENCY_INJECTION
    except Exception as e:
        logging.error(f"Error determining launcher mode: {e}")
        # Fallback to standard mode if something goes wrong
        return LauncherMode.STANDARD


def launch_standard() -> None:
    """
    Launch the game using the standard launcher.
    """
    try:
        # Initialize logging
        setup_logging()
        game = StandardGame()
        game.run()
    except Exception as e:
        logging.error(f"Error in standard launcher: {e}")
        logging.debug(traceback.format_exc())
        raise


def launch_dependency_injection(config: Optional[Dict[str, Any]] = None) -> None:
    """
    Launch the game using dependency injection.
    
    Args:
        config: Optional configuration dictionary
    """
    try:
        # Import DI specific components
        from ai_platform_trainer.core.launcher_di import register_services

        # Register all the services
        register_services()
        
        # Create and run the game
        game = DIGame()
        game.run()
    except Exception as e:
        logging.error(f"Error in DI launcher: {e}")
        logging.debug(traceback.format_exc())
        # If DI fails, try to fall back to standard mode
        logging.warning("Falling back to standard launcher...")
        launch_standard()


def launch_state_machine() -> None:
    """
    Launch the game using the state machine implementation.
    """
    try:
        # Initialize logging
        setup_logging()
        game = StateMachineGame()
        game.run()
    except Exception as e:
        logging.error(f"Error in state machine launcher: {e}")
        logging.debug(traceback.format_exc())
        # If state machine fails, try to fall back to DI mode
        logging.warning("Falling back to DI launcher...")
        try:
            launch_dependency_injection()
        except Exception:
            # If DI also fails, try standard mode
            logging.warning("Falling back to standard launcher...")
            launch_standard()


def main() -> None:
    """
    Main entry point that selects and runs the appropriate launcher.
    """
    # Setup logging first
    setup_logging()
    
    logging.info("Starting AI Platform Trainer")
    
    try:
        # Determine which launcher to use
        mode = get_launcher_mode_from_settings()
        
        logging.info(f"Using launcher mode: {mode.name}")
        
        # Launch with the selected mode
        if mode == LauncherMode.STANDARD:
            launch_standard()
        elif mode == LauncherMode.DEPENDENCY_INJECTION:
            launch_dependency_injection()
        elif mode == LauncherMode.STATE_MACHINE:
            launch_state_machine()
        else:
            logging.error(f"Unknown launcher mode: {mode}")
            raise ValueError(f"Unknown launcher mode: {mode}")
            
    except Exception as e:
        logging.critical(f"Fatal error in game launcher: {e}")
        logging.debug(traceback.format_exc())
        raise
    finally:
        logging.info("Exiting AI Platform Trainer")


if __name__ == "__main__":
    main()
    main()
