"""
Game Loop Coordinator for AI Platform Trainer.

This module handles the main game loop coordination,
separating UI logic from core game mechanics.
"""
import logging
import pygame
from typing import Optional, Dict, Any

from ai_platform_trainer.core.logging_config import setup_logging
from ai_platform_trainer.core.config_manager import get_config_manager
from ai_platform_trainer.gameplay.display_manager import (
    init_pygame_display,
    toggle_fullscreen_display,
)


class GameLoop:
    """
    Coordinates the main game loop and UI interactions.
    
    This class manages the high-level game flow, menu interactions,
    and coordination between different game modes.
    """
    
    def __init__(self, render_mode=None):
        """
        Initialize the game loop coordinator.
        
        Args:
            render_mode: Rendering mode (FULL or HEADLESS)
        """
        setup_logging()
        self.running = True
        self.render_mode = render_mode
        
        # Get configuration manager
        self.config_manager = get_config_manager()
        
        # Initialize display
        self._init_display()
        
        # Create clock and menu
        self.clock = pygame.time.Clock()
        
        # Import here to avoid circular imports
        from .menus import Menu
        self.menu = Menu(self.screen_width, self.screen_height)
        
        # Game state
        self.current_mode = None
        self.game_instance = None
        
        logging.info("Game loop coordinator initialized.")
    
    def _init_display(self) -> None:
        """Initialize the display system."""
        pygame.init()
        
        # Always start in fullscreen mode
        self.config_manager.set("display.fullscreen", True)
        self.config_manager.save()
        
        if (self.render_mode and 
            hasattr(self.render_mode, "HEADLESS") and 
            self.render_mode == self.render_mode.HEADLESS):
            # Headless mode
            import os
            os.environ['SDL_VIDEODRIVER'] = 'dummy'
            self.screen = pygame.Surface((1280, 720))
            self.screen_width = 1280
            self.screen_height = 720
        else:
            # Normal mode with display
            (self.screen, 
             self.screen_width, 
             self.screen_height) = init_pygame_display(fullscreen=True)
            pygame.display.set_caption("AI Platform Trainer")
    
    def run(self) -> None:
        """Run the main game loop."""
        from ai_platform_trainer.gameplay.config import config
        
        while self.running:
            self._handle_events()
            self._update()
            self._render()
            self.clock.tick(config.FRAME_RATE)
        
        # Cleanup
        if self.game_instance and hasattr(self.game_instance, 'data_logger'):
            if self.game_instance.data_logger:
                self.game_instance.data_logger.save()
        
        pygame.quit()
        logging.info("Game loop exited and Pygame quit.")
    
    def _handle_events(self) -> None:
        """Handle pygame events."""
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                logging.info("Quit event detected. Exiting game.")
                self.running = False
                return
            
            elif event.type == pygame.KEYDOWN:
                # Global hotkeys
                if event.key == pygame.K_f:
                    self._toggle_fullscreen()
                elif event.key == pygame.K_ESCAPE:
                    if self.current_mode:
                        # Return to menu from game
                        self._return_to_menu()
                    else:
                        # Exit from menu
                        self.running = False
                elif event.key == pygame.K_m and self.current_mode:
                    # Return to menu hotkey
                    self._return_to_menu()
            
            # Handle menu events when in menu mode
            if not self.current_mode:
                action = self.menu.handle_menu_events(event)
                if action:
                    self._process_menu_action(action)
            else:
                # Forward events to game instance
                if self.game_instance:
                    self.game_instance.handle_events([event])
    
    def _update(self) -> None:
        """Update game state."""
        if self.current_mode and self.game_instance:
            # Update the active game instance
            current_time = pygame.time.get_ticks()
            if hasattr(self.game_instance, 'update'):
                self.game_instance.update(current_time)
            elif hasattr(self.game_instance, 'update_once'):
                self.game_instance.update_once()
    
    def _render(self) -> None:
        """Render the current screen."""
        if (self.render_mode and 
            hasattr(self.render_mode, "HEADLESS") and 
            self.render_mode == self.render_mode.HEADLESS):
            return  # Skip rendering in headless mode
        
        if not self.current_mode:
            # Render menu
            self.menu.draw(self.screen)
        else:
            # Render game
            if self.game_instance and hasattr(self.game_instance, 'renderer'):
                if self.game_instance.renderer:
                    self.game_instance.renderer.render(
                        self.menu,
                        self.game_instance.player,
                        self.game_instance.enemy,
                        False  # menu_active = False
                    )
        
        pygame.display.flip()
    
    def _process_menu_action(self, action: str) -> None:
        """
        Process menu action and start appropriate game mode.
        
        Args:
            action: The menu action to process
        """
        logging.info(f"Processing menu action: {action}")
        
        if action == "exit":
            self.running = False
        elif action == "play_sl":
            self._start_supervised_game()
        elif action == "train":
            self._start_training_mode()
        elif action == "play_rl_pretrained_easy":
            self._start_rl_game("easy")
        elif action == "play_rl_pretrained_medium":
            self._start_rl_game("medium")
        elif action == "play_rl_pretrained_hard":
            self._start_rl_game("hard")
        elif action == "play_rl_learning":
            self._start_rl_learning_game()
    
    def _start_supervised_game(self) -> None:
        """Start the supervised learning game mode."""
        logging.info("Starting supervised learning game mode.")
        
        # Import here to avoid circular imports
        from ai_platform_trainer.gameplay.game_core import GameCore
        
        self.game_instance = GameCore(render_mode=self.render_mode)
        self.game_instance.start_game("play")
        self.current_mode = "play_sl"
    
    def _start_training_mode(self) -> None:
        """Start the training mode."""
        logging.info("Starting training mode.")
        
        from ai_platform_trainer.gameplay.game_core import GameCore
        
        self.game_instance = GameCore(render_mode=self.render_mode)
        self.game_instance.start_game("train")
        self.current_mode = "train"
    
    def _start_rl_game(self, difficulty: str) -> None:
        """
        Start RL game with pretrained agent.
        
        Args:
            difficulty: Difficulty level (easy, medium, hard)
        """
        logging.info(f"Starting RL game with {difficulty} difficulty.")
        
        # This will be implemented in Phase 2 with C++/CUDA RL
        # For now, fallback to supervised learning
        logging.warning(f"RL {difficulty} mode not yet implemented, "
                       "falling back to supervised learning.")
        self._start_supervised_game()
        self.current_mode = f"play_rl_{difficulty}"
    
    def _start_rl_learning_game(self) -> None:
        """Start RL game with live learning agent."""
        logging.info("Starting RL learning game mode.")
        
        # This will be implemented in Phase 2 with C++/CUDA RL
        # For now, fallback to supervised learning
        logging.warning("RL learning mode not yet implemented, "
                       "falling back to supervised learning.")
        self._start_supervised_game()
        self.current_mode = "play_rl_learning"
    
    def _return_to_menu(self) -> None:
        """Return to the main menu."""
        logging.info("Returning to main menu.")
        
        # Save any data if needed
        if (self.game_instance and 
            hasattr(self.game_instance, 'data_logger') and
            self.game_instance.data_logger):
            self.game_instance.data_logger.save()
        
        # Reset game state
        self.game_instance = None
        self.current_mode = None
        
        # Reset menu state
        self.menu.current_menu = "main"
        self.menu.show_help = False
    
    def _toggle_fullscreen(self) -> None:
        """Toggle between windowed and fullscreen modes."""
        if (self.render_mode and 
            hasattr(self.render_mode, "HEADLESS") and 
            self.render_mode == self.render_mode.HEADLESS):
            return  # Skip fullscreen toggle in headless mode
        
        from ai_platform_trainer.gameplay.config import config
        
        was_fullscreen = self.config_manager.get("display.fullscreen", False)
        new_display, w, h = toggle_fullscreen_display(
            not was_fullscreen,
            config.SCREEN_SIZE
        )
        
        self.config_manager.set("display.fullscreen", not was_fullscreen)
        self.config_manager.save()
        
        self.screen = new_display
        self.screen_width, self.screen_height = w, h
        pygame.display.set_caption("AI Platform Trainer")
        
        # Recreate menu with new dimensions
        from .menus import Menu
        self.menu = Menu(self.screen_width, self.screen_height)
        
        # Restart current game mode if active
        if self.current_mode and self.game_instance:
            current_game_mode = self.current_mode
            self._return_to_menu()
            # Restart the same mode
            if current_game_mode == "play_sl":
                self._start_supervised_game()
            elif current_game_mode == "train":
                self._start_training_mode()
            # RL modes will be implemented in Phase 2
