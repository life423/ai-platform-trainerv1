# file: ai_platform_trainer/gameplay/game.py

import logging
import os
from typing import Optional, Tuple

import pygame
import torch

from ai_platform_trainer.core.logging_config import setup_logging
from ai_platform_trainer.entities.player_play import PlayerPlay
from ai_platform_trainer.entities.enemy_play import EnemyPlay
from ai_platform_trainer.entities.player_training import PlayerTraining
from ai_platform_trainer.entities.enemy_training import EnemyTrain
from ai_platform_trainer.ai_model.train_missile_model import SimpleMissileModel

from ai_platform_trainer.gameplay.env.pixel_pursuit_env import PixelPursuitEnv
from ai_platform_trainer.gameplay.config import config
from ai_platform_trainer.gameplay.menu import Menu
from ai_platform_trainer.gameplay.renderer import Renderer
from ai_platform_trainer.gameplay.spawner import spawn_entities
from config_manager import load_settings, save_settings
from ai_platform_trainer.core.data_logger import DataLogger
from ai_platform_trainer.gameplay.modes.training_mode import TrainingMode


class Game:
    """
    Main class to run the Pixel Pursuit game.
    Works in either 'train' or 'play' modes.
    In 'play', we handle user input & Pygame rendering.
    In 'train', we can run headless or minimal.
    """

    def __init__(self) -> None:
        setup_logging()
        pygame.init()

        self.running: bool = True
        self.menu_active: bool = True
        self.mode: Optional[str] = None
        self.settings = load_settings("settings.json")

        # Initialize display
        if self.settings.get("fullscreen", False):
            self.screen = pygame.display.set_mode((0, 0), pygame.FULLSCREEN)
        else:
            self.screen = pygame.display.set_mode(config.SCREEN_SIZE)
        pygame.display.set_caption(config.WINDOW_TITLE)

        self.screen_width, self.screen_height = self.screen.get_size()
        self.clock = pygame.time.Clock()

        # Menu & Renderer
        self.menu = Menu(self.screen_width, self.screen_height)
        self.renderer = Renderer(self.screen)

        # Entities (player, enemy)
        self.player = None
        self.enemy = None
        self.data_logger: Optional[DataLogger] = None

        # Missile model
        self.missile_model = None
        self._load_missile_model_once()

        # The "Env" for advanced AI/training logic
        self.env = None

        logging.info("Game initialized.")

    def _load_missile_model_once(self) -> None:
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
            logging.warning(f"No missile model found at '{missile_model_path}'.")

    def run(self) -> None:
        """Main game loop. If 'train', we might do minimal rendering."""
        while self.running:
            self.handle_events()

            if self.menu_active:
                self.menu.draw(self.screen)
            else:
                # 'train' or 'play' update
                if self.mode == "train":
                    # Example: step environment once if you want each frame
                    # (Or do it in a loop if headless.)
                    if self.env:
                        # No user input for training, so pass None or AI action
                        observation, reward, done, info = self.env.step(action=None)
                        # If done, reset environment
                        if done:
                            self.env.reset(self.player, self.enemy)
                else:
                    # Normal "play" updates:
                    self._play_update()
                self.renderer.render(
                    self.menu, self.player, self.enemy, self.menu_active
                )

            pygame.display.flip()
            self.clock.tick(config.FRAME_RATE)

        # If in training mode, save data if needed
        if self.mode == "train" and self.data_logger:
            self.data_logger.save()

        pygame.quit()
        logging.info("Game loop exited.")

    def start_game(self, mode: str) -> None:
        """Initialize game in 'train' or 'play' mode."""
        self.mode = mode
        logging.info(f"Starting game in '{mode}' mode.")

        if mode == "train":
            self.data_logger = DataLogger(config.DATA_PATH)
            self.player = PlayerTraining(self.screen_width, self.screen_height)
            self.enemy = EnemyTrain(self.screen_width, self.screen_height)
            spawn_entities(self)
            self.player.reset()
            self.env = PixelPursuitEnv(self.screen_width, self.screen_height)
            self.env.reset(self.player, self.enemy)
            self.training_mode_manager = TrainingMode(self)

        else:
            # 'play' mode
            from ai_platform_trainer.ai_model.model_definition.enemy_movement_model import (
                EnemyMovementModel,
            )

            self.player, self.enemy = self._init_play_mode()
            spawn_entities(self)
            self.player.reset()
            # Possibly no self.env if you keep old logic.
            # But you *could* unify it here too.

    def _init_play_mode(self) -> Tuple[PlayerPlay, EnemyPlay]:
        # Load an EnemyMovementModel for "play" AI
        from ai_platform_trainer.ai_model.model_definition.enemy_movement_model import (
            EnemyMovementModel,
        )

        model = EnemyMovementModel(input_size=5, hidden_size=64, output_size=2)
        model_path = config.MODEL_PATH
        if os.path.isfile(model_path):
            try:
                model.load_state_dict(torch.load(model_path, map_location="cpu"))
                model.eval()
                logging.info("Enemy AI model loaded for play mode.")
            except Exception as e:
                logging.error(f"Failed to load enemy model: {e}")
        else:
            logging.warning(f"No enemy model found at '{model_path}'. Using default.")

        player = PlayerPlay(self.screen_width, self.screen_height)
        enemy = EnemyPlay(self.screen_width, self.screen_height, model)
        return player, enemy

    def _play_update(self) -> None:
        """
        The old 'play' logic.
        Optionally, you can move collisions, missile updates, etc. into PixelPursuitEnv later.
        """
        # E.g. handle collisions, update enemy movement, etc.
        pass

    def handle_events(self) -> None:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                logging.info("Quit event detected.")
                self.running = False
            elif event.type == pygame.KEYDOWN:
                if event.key == pygame.K_ESCAPE:
                    self.running = False
                if event.key == pygame.K_f:
                    self.toggle_fullscreen()

                if self.menu_active:
                    selected_action = self.menu.handle_menu_events(event)
                    if selected_action:
                        self.check_menu_selection(selected_action)
                else:
                    # In-game key logic
                    if event.key == pygame.K_m:
                        # Return to menu
                        self.menu_active = True
                        self.reset_game_state()

    def check_menu_selection(self, selected_action: str) -> None:
        if selected_action == "exit":
            logging.info("Exit selected.")
            self.running = False
        elif selected_action in ["train", "play"]:
            logging.info(f"Selected: {selected_action}")
            self.menu_active = False
            self.start_game(selected_action)

    def toggle_fullscreen(self) -> None:
        self.settings["fullscreen"] = not self.settings.get("fullscreen", False)
        save_settings(self.settings, "settings.json")
        flags = pygame.FULLSCREEN if self.settings["fullscreen"] else 0
        if self.settings["fullscreen"]:
            self.screen = pygame.display.set_mode((0, 0), flags)
        else:
            self.screen = pygame.display.set_mode(config.SCREEN_SIZE, flags)
        self.screen_width, self.screen_height = self.screen.get_size()
        logging.info(
            f"Fullscreen toggled. Now {self.screen_width}x{self.screen_height}."
        )
        self.menu = Menu(self.screen_width, self.screen_height)

    def reset_game_state(self) -> None:
        """Clear entities and data logger if returning to menu or toggling resolution."""
        self.player = None
        self.enemy = None
        self.data_logger = None
        logging.info("Game state reset, back to menu.")
