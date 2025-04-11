# file: ai_platform_trainer/engine/core/launcher_di.py
"""
Launcher module for the AI Platform Trainer with dependency injection.
Registers all services with the ServiceLocator before starting the game.
"""
import pygame

from ai_platform_trainer.core.config_manager import get_config_manager
from ai_platform_trainer.core.logging_config import setup_logging
from ai_platform_trainer.core.service_locator import ServiceLocator
from ai_platform_trainer.engine.core.game_di import Game
from ai_platform_trainer.engine.input.input_handler import InputHandler
from ai_platform_trainer.engine.physics.collisions import handle_missile_collisions
from ai_platform_trainer.engine.rendering.display_manager import init_pygame_display
from ai_platform_trainer.engine.rendering.renderer_di import Renderer
from ai_platform_trainer.entities.entity_factory import PlayEntityFactory, TrainingEntityFactory

# Menu not yet refactored to engine directory
from ai_platform_trainer.gameplay.menu import Menu
from config_manager import load_settings, save_settings


class DisplayService:
    """Service for managing the display."""

    def __init__(self, fullscreen=False):
        """Initialize the display service."""
        self.screen, self.width, self.height = init_pygame_display(fullscreen)

    def get_dimensions(self):
        """Get the screen dimensions."""
        return self.width, self.height

    def set_screen(self, screen, width, height):
        """Set the screen and dimensions."""
        self.screen = screen
        self.width = width
        self.height = height


class SettingsService:
    """Service for managing game settings."""

    def __init__(self, settings_file="settings.json"):
        """Initialize the settings service."""
        self.settings_file = settings_file
        self.settings = load_settings(settings_file)

    def get_settings(self):
        """Get the current settings."""
        return self.settings

    def save_settings(self, settings=None):
        """Save the settings."""
        if settings:
            self.settings = settings
        save_settings(self.settings, self.settings_file)


class CollisionService:
    """Service for handling collisions."""

    def check_missile_collisions(self, player, enemy, respawn_callback):
        """Check for collisions between missiles and enemy."""
        handle_missile_collisions(player, enemy, respawn_callback)


class MenuFactory:
    """Factory for creating menus."""

    def create_menu(self, width, height):
        """Create a new menu with the given dimensions."""
        return Menu(width, height)


def register_services():
    """Register all services with the ServiceLocator."""
    # Create and register configuration manager
    config_manager = get_config_manager("config.json")
    ServiceLocator.register("config_manager", config_manager)

    # Load settings (legacy settings service)
    settings_service = SettingsService()
    ServiceLocator.register("settings", settings_service)
    settings = settings_service.get_settings()

    # Initialize pygame
    pygame.init()

    # Create and register display service
    display_service = DisplayService(settings.get("fullscreen", False))
    ServiceLocator.register("display", display_service)

    # Register screen
    ServiceLocator.register("screen", display_service.screen)

    # Create and register menu
    menu_factory = MenuFactory()
    ServiceLocator.register("menu_factory", menu_factory)

    menu = menu_factory.create_menu(display_service.width, display_service.height)
    ServiceLocator.register("menu", menu)

    # Create and register renderer
    renderer = Renderer(display_service.screen)
    ServiceLocator.register("renderer", renderer)

    # Create and register input handler
    input_handler = InputHandler()
    ServiceLocator.register("input_handler", input_handler)

    # Create and register entity factories
    play_entity_factory = PlayEntityFactory()
    ServiceLocator.register("play_entity_factory", play_entity_factory)

    training_entity_factory = TrainingEntityFactory()
    ServiceLocator.register("training_entity_factory", training_entity_factory)

    # Create and register collision service
    collision_service = CollisionService()
    ServiceLocator.register("collision_service", collision_service)

    # Create and register clock
    clock = pygame.time.Clock()
    ServiceLocator.register("clock", clock)

    # Register configuration (legacy config)
    from ai_platform_trainer.gameplay.config import config
    ServiceLocator.register("config", config)

    # States will be registered by the game instance itself


def main():
    """
    Main entry point for the AI Platform Trainer application.
    Sets up dependency injection and runs the game.
    """
    setup_logging()
    register_services()

    game = Game()
    game.run()


if __name__ == "__main__":
    main()
