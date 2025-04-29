# file: ai_platform_trainer/gameplay/game_refactored.py
import logging
import os
from typing import Dict, Optional, Tuple

import pygame
import torch
from config_manager import load_settings, save_settings

# AI and model imports
from ai_platform_trainer.ai_model.model_definition.enemy_movement_model import \
    EnemyMovementModel
from ai_platform_trainer.ai_model.simple_missile_model import \
    SimpleMissileModel
# Data logger and entity imports
from ai_platform_trainer.core.data_logger import DataLogger
# Logging setup
from ai_platform_trainer.core.logging_config import setup_logging
from ai_platform_trainer.engine.core.game_config import config
from ai_platform_trainer.entities.enemy_play import EnemyPlay
from ai_platform_trainer.entities.enemy_training import EnemyTrain
from ai_platform_trainer.entities.player_play import PlayerPlay
from ai_platform_trainer.entities.player_training import PlayerTraining
# Gameplay imports
from ai_platform_trainer.gameplay.collisions import handle_missile_collisions
from ai_platform_trainer.gameplay.display_manager import (
    init_pygame_display, toggle_fullscreen_display)
from ai_platform_trainer.gameplay.menu import Menu
# Used in state_machine.py - PlayState.update
from ai_platform_trainer.gameplay.missile_ai_controller import \
    update_missile_ai  # noqa: F401
from ai_platform_trainer.gameplay.modes.training_mode import TrainingMode
from ai_platform_trainer.gameplay.renderer import Renderer
# These imports are used by Game class methods and by GameState subclasses
from ai_platform_trainer.gameplay.spawner import \
    spawn_entities  # Used in state_machine.py - PlayState.enter
from ai_platform_trainer.gameplay.spawner import \
    respawn_enemy_with_fade_in  # noqa: F401
# Import state machine components
from ai_platform_trainer.gameplay.state_machine import (GameOverState,
                                                        GameState, MenuState,
                                                        PausedState, PlayState,
                                                        TrainingState)


class Game:
    """
    Main class to run the Pixel Pursuit game.
    Implements a state machine to manage different game states.
    """

    def __init__(self) -> None:
        setup_logging()
        self.running: bool = True
        self.menu_active: bool = True
        self.mode: Optional[str] = None
        self.paused: bool = False  # Track pause state

        # 1) Load user settings
        self.settings = load_settings("settings.json")

        # 2) Initialize Pygame and the display
        (self.screen, self.screen_width, self.screen_height) = init_pygame_display(
            fullscreen=self.settings.get("fullscreen", False)
        )

        # 3) Create clock, menu, and renderer
        pygame.display.set_caption(config.WINDOW_TITLE)
        self.clock = pygame.time.Clock()
        self.menu = Menu(self.screen_width, self.screen_height)
        self.renderer = Renderer(self.screen)

        # 4) Entities and managers
        self.player: Optional[PlayerPlay] = None
        self.enemy: Optional[EnemyPlay] = None
        self.data_logger: Optional[DataLogger] = None
        self.training_mode_manager: Optional[TrainingMode] = None  # For train mode

        # 5) Load missile model once
        self.missile_model: Optional[SimpleMissileModel] = None
        self._load_missile_model_once()

        # 6) Additional logic
        self.respawn_delay = 1000
        self.respawn_timer = 0
        self.is_respawning = False

        # Reusable tensor for missile AI input
        self._missile_input = torch.zeros((1, 9), dtype=torch.float32)

        # 7) Initialize state machine
        self.states: Dict[str, GameState] = {
            "menu": MenuState(self),
            "play": PlayState(self),
            "train": TrainingState(self),
            "paused": PausedState(self),
            "game_over": GameOverState(self),
        }
        self.current_state = self.states["menu"]
        self.current_state.enter()

        # Make these classes accessible to the state machine
        self.DataLogger = DataLogger
        self.PlayerTraining = PlayerTraining
        self.EnemyTrain = EnemyTrain
        self.TrainingMode = TrainingMode

        logging.info("Game initialized with state machine architecture.")

    def _load_missile_model_once(self) -> None:
        """
        Load the missile AI model once during initialization.
        """
        missile_model_path = "models/missile_model.pth"
        if os.path.isfile(missile_model_path):
            logging.info(
                f"Found missile model at '{missile_model_path}'. Loading once..."
            )
            try:
                model = SimpleMissileModel()
                model.load_state_dict(
                    torch.load(missile_model_path, map_location="cpu")
                )
                model.eval()
                self.missile_model = model
            except Exception as e:
                logging.error(f"Failed to load missile model: {e}")
                self.missile_model = None
        else:
            logging.warning(
                f"No missile model found at '{missile_model_path}'. "
                "Skipping missile AI."
            )

    def run(self) -> None:
        """Main game loop using state machine architecture."""
        while self.running:
            delta_time = self.clock.tick(config.FRAME_RATE) / 1000.0

            # Handle events
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    logging.info("Quit event detected. Exiting game.")
                    self.running = False
                else:
                    next_state = self.current_state.handle_event(event)
                    if next_state:
                        self.transition_to(next_state)

            # Update and render
            next_state = self.current_state.update(delta_time)
            if next_state:
                self.transition_to(next_state)

            self.current_state.render(self.renderer)
            pygame.display.flip()

        # Save data if we were training
        if self.mode == "train" and self.data_logger:
            self.data_logger.save()

        pygame.quit()
        logging.info("Game loop exited and Pygame quit.")

    def transition_to(self, state_name: str) -> None:
        """
        Transition from the current state to a new state.

        Args:
            state_name: The name of the state to transition to
        """
        if state_name in self.states:
            logging.info(
                f"Transitioning from {type(self.current_state).__name__} to {state_name}"
            )
            self.current_state.exit()
            self.current_state = self.states[state_name]
            self.current_state.enter()
        else:
            logging.error(f"Attempted to transition to unknown state: {state_name}")

    def _init_play_mode(self) -> Tuple[PlayerPlay, EnemyPlay]:
        """Initialize entities for play mode."""
        model = EnemyMovementModel(input_size=5, hidden_size=64, output_size=2)
        try:
            model.load_state_dict(torch.load(config.MODEL_PATH, map_location="cpu"))
            model.eval()
            logging.info("Enemy AI model loaded for play mode.")
        except Exception as e:
            logging.error(f"Failed to load enemy model: {e}")
            raise e

        player = PlayerPlay(self.screen_width, self.screen_height)
        enemy = EnemyPlay(self.screen_width, self.screen_height, model)
        logging.info("Initialized PlayerPlay and EnemyPlay for play mode.")
        return player, enemy

    def _toggle_fullscreen(self) -> None:
        """
        Helper that toggles between windowed and fullscreen,
        updating self.screen, self.screen_width, self.screen_height.
        """
        was_fullscreen = self.settings["fullscreen"]
        new_display, w, h = toggle_fullscreen_display(
            not was_fullscreen, config.SCREEN_SIZE
        )
        self.settings["fullscreen"] = not was_fullscreen
        save_settings(self.settings, "settings.json")

        self.screen = new_display
        self.screen_width, self.screen_height = w, h
        pygame.display.set_caption(config.WINDOW_TITLE)
        self.menu = Menu(self.screen_width, self.screen_height)

        # If we're in a game, restart the current state
        if self.mode and self.mode != "menu":
            current_mode = self.mode
            self.reset_game_state()
            self.transition_to(current_mode)

    def check_collision(self) -> bool:
        """Check for collision between player and enemy."""
        if not (self.player and self.enemy):
            return False
        player_rect = pygame.Rect(
            self.player.position["x"],
            self.player.position["y"],
            self.player.size,
            self.player.size,
        )
        enemy_rect = pygame.Rect(
            self.enemy.pos["x"], self.enemy.pos["y"], self.enemy.size, self.enemy.size
        )
        return player_rect.colliderect(enemy_rect)

    def check_missile_collisions(self) -> None:
        """Check for collisions between missiles and enemy."""
        if not self.enemy or not self.player:
            return

        def respawn_callback() -> None:
            self.is_respawning = True
            self.respawn_timer = pygame.time.get_ticks() + self.respawn_delay
            logging.info("Missile-Enemy collision in play mode, enemy will respawn.")

        handle_missile_collisions(self.player, self.enemy, respawn_callback)

    def handle_respawn(self, current_time: int) -> None:
        """Handle respawning the enemy after a delay."""
        if (
            self.is_respawning
            and current_time >= self.respawn_timer
            and self.enemy
            and self.player
        ):
            respawn_enemy_with_fade_in(self, current_time)

    def reset_game_state(self) -> None:
        """Reset game state, typically when returning to menu."""
        self.player = None
        self.enemy = None
        self.data_logger = None
        self.is_respawning = False
        self.respawn_timer = 0
        logging.info("Game state reset.")
