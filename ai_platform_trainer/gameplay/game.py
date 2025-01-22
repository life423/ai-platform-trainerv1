# file: ai_platform_trainer/gameplay/game.py
import logging
import math
import os
from typing import Optional, Tuple

import pygame
import torch

# Logging setup (optional)
from ai_platform_trainer.core.logging_config import setup_logging

# Suppose you have this new environment:
from ai_platform_trainer.gameplay.pixel_pursuit_env import PixelPursuitEnv

# AI and model imports
from ai_platform_trainer.ai_model.model_definition.enemy_movement_model import (
    EnemyMovementModel,
)
from ai_platform_trainer.ai_model.train_missile_model import SimpleMissileModel

# Data logger and entity imports
from ai_platform_trainer.core.data_logger import DataLogger
from ai_platform_trainer.entities.enemy_play import EnemyPlay
from ai_platform_trainer.entities.enemy_training import EnemyTrain
from ai_platform_trainer.entities.player_play import PlayerPlay
from ai_platform_trainer.entities.player_training import PlayerTraining

# Gameplay imports
from ai_platform_trainer.gameplay.collisions import handle_missile_collisions
from ai_platform_trainer.gameplay.config import config
from ai_platform_trainer.gameplay.menu import Menu
from ai_platform_trainer.gameplay.modes.training_mode import TrainingMode
from ai_platform_trainer.gameplay.renderer import Renderer
from ai_platform_trainer.gameplay.spawner import (
    spawn_entities,
    respawn_enemy_with_fade_in,
)
# Configuration manager imports for fullscreen settings
from config_manager import load_settings, save_settings


class Game:
    """
    A "launcher / UI" class to run the Pixel Pursuit game with a menu, 
    handle training or play mode, and manage event loops.
    Eventually, the real 'step' logic will move to `PixelPursuitEnv`.
    """

    def __init__(self) -> None:
        """
        Initialize pygame, set up logging, load settings, and configure display.
        Load the missile model exactly once for the entire session.
        """
        setup_logging()
        pygame.init()

        self.running: bool = True
        self.menu_active: bool = True
        self.mode: Optional[str] = None

        # 1) Load user settings (fullscreen, etc.)
        self.settings = load_settings("settings.json")

        # 2) Initialize display based on settings
        if self.settings.get("fullscreen", False):
            self.screen = pygame.display.set_mode((0, 0), pygame.FULLSCREEN)
        else:
            self.screen = pygame.display.set_mode(config.SCREEN_SIZE)

        # Optional: set window caption
        pygame.display.set_caption(config.WINDOW_TITLE)

        # 3) Store screen dimensions, create clock
        self.screen_width, self.screen_height = self.screen.get_size()
        self.clock = pygame.time.Clock()

        # 4) Create menu and renderer
        self.menu = Menu(self.screen_width, self.screen_height)
        self.renderer = Renderer(self.screen)

        # 5) Entities and managers (we may phase these out to `pixel_pursuit_env`)
        self.player: Optional[PlayerPlay] = None
        self.enemy: Optional[EnemyPlay] = None
        self.data_logger: Optional[DataLogger] = None

        # Load missile model once
        self.missile_model: Optional[SimpleMissileModel] = None
        self._load_missile_model_once()

        # Respawn logic
        self.respawn_delay = 1000
        self.respawn_timer = 0
        self.is_respawning = False

        # Reusable tensor for missile AI input
        self._missile_input = torch.zeros((1, 9), dtype=torch.float32)

        # NEW: A reference to the environment (for training or advanced usage)
        # We could pass screen_width/height if needed
        self.env = PixelPursuitEnv(width=self.screen_width, height=self.screen_height)

        logging.info("Game initialized.")

    def _load_missile_model_once(self) -> None:
        """
        Attempt to load the missile model at initialization, so it's only done once.
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
                f"No missile model found at '{missile_model_path}'. Skipping missile AI."
            )

    def run(self) -> None:
        """
        Main game loop. Processes events, updates logic, and renders frames.
        If in training mode, also saves data upon exit.
        """
        while self.running:
            current_time = pygame.time.get_ticks()
            self.handle_events()

            if self.menu_active:
                self.menu.draw(self.screen)
            else:
                self.update(current_time)     # We'll eventually reduce direct updates here
                self.renderer.render(
                    self.menu, self.player, self.enemy, self.menu_active
                )

            pygame.display.flip()
            self.clock.tick(config.FRAME_RATE)

        # Save data if we were in training mode
        if self.mode == "train" and self.data_logger:
            self.data_logger.save()

        pygame.quit()
        logging.info("Game loop exited and Pygame quit.")

    def start_game(self, mode: str) -> None:
        """
        Start the game in 'train' or 'play' mode.
        Initializes relevant entities, data logging, and spawns as needed.
        """
        self.mode = mode
        logging.info(f"Starting game in '{mode}' mode.")

        if mode == "train":
            self.data_logger = DataLogger(config.DATA_PATH)
            self.player = PlayerTraining(self.screen_width, self.screen_height)
            self.enemy = EnemyTrain(self.screen_width, self.screen_height)

            spawn_entities(self)
            self.player.reset()
            self.training_mode_manager = TrainingMode(self)

        else:
            self.player, self.enemy = self._init_play_mode()
            self.player.reset()
            spawn_entities(self)

    def _init_play_mode(self) -> Tuple[PlayerPlay, EnemyPlay]:
        """
        Initialize player and enemy in 'play' mode.
        Tries to load an EnemyMovementModel for AI-based enemy movement.
        """
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

    def handle_events(self) -> None:
        """
        Process Pygame events (key presses, mouse) and handle menu selections.
        """
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                logging.info("Quit event detected. Exiting game.")
                self.running = False

            elif event.type == pygame.KEYDOWN:
                if event.key == pygame.K_f:
                    logging.debug("F pressed - toggling fullscreen.")
                    self.toggle_fullscreen()

                if self.menu_active:
                    selected_action = self.menu.handle_menu_events(event)
                    if selected_action:
                        self.check_menu_selection(selected_action)
                else:
                    # In-game key logic
                    if event.key == pygame.K_ESCAPE:
                        logging.info("Escape key pressed. Exiting game.")
                        self.running = False
                    elif event.key == pygame.K_f:
                        self.toggle_fullscreen()
                    elif event.key == pygame.K_SPACE and self.player:
                        self.player.shoot_missile()
                    elif event.key == pygame.K_m:
                        logging.info("M key pressed. Returning to menu.")
                        self.menu_active = True
                        self.reset_game_state()

            elif event.type == pygame.MOUSEBUTTONDOWN and event.button == 1:
                mouse_x, mouse_y = pygame.mouse.get_pos()
                if self.menu_active:
                    selected_action = self.menu.handle_menu_events(event)
                    if selected_action:
                        self.check_menu_selection(selected_action)
                else:
                    logging.debug(f"In-game left-click at ({mouse_x}, {mouse_y})")

    def check_menu_selection(self, selected_action: str) -> None:
        """
        Handle menu choices: 'exit', 'train', 'play'
        """
        if selected_action == "exit":
            logging.info("Exit action selected from menu.")
            self.running = False
        elif selected_action in ["train", "play"]:
            logging.info(f"'{selected_action}' selected from menu.")
            self.menu_active = False
            self.start_game(selected_action)

    def toggle_fullscreen(self) -> None:
        """
        Toggle fullscreen by flipping the 'fullscreen' setting,
        creating the display surface exactly once, and re-initializing if necessary.
        """
        # Toggle setting
        self.settings["fullscreen"] = not self.settings.get("fullscreen", False)
        save_settings(self.settings, "settings.json")

        flags = pygame.FULLSCREEN if self.settings["fullscreen"] else 0
        if self.settings["fullscreen"]:
            self.screen = pygame.display.set_mode((0, 0), flags)
        else:
            self.screen = pygame.display.set_mode(config.SCREEN_SIZE, flags)

        # Re-query the new display size
        self.screen_width, self.screen_height = self.screen.get_size()
        logging.info(
            f"Fullscreen toggled to {self.settings['fullscreen']}. "
            f"New display resolution: {self.screen_width}x{self.screen_height}."
        )

        self.menu = Menu(self.screen_width, self.screen_height)

        if not self.menu_active:
            current_mode = self.mode
            self.reset_game_state()
            self.start_game(current_mode)

    def update(self, current_time: int) -> None:
        """
        Update the game each frame. If 'train', calls training_mode_manager, 
        else does 'play' update. We'll gradually move more logic into the environment.
        """
        if self.mode == "train":
            self.training_mode_manager.update()
        elif self.mode == "play":
            self.play_update(current_time)
            self.check_missile_collisions()
            self.handle_respawn(current_time)

            if self.enemy and self.enemy.fading_in:
                self.enemy.update_fade_in(current_time)
            if self.player:
                self.player.update_missiles()

    def play_update(self, current_time: int) -> None:
        """
        Main update logic for 'play' mode:
         - Handle player input
         - Possibly call environment step in the future
         - For now, update enemy/player directly as before.
        """
        if self.player and not self.player.handle_input():
            logging.info("Player requested to quit.")
            self.running = False
            return

        if self.enemy:
            try:
                # For now, keep direct update. Later, we might do:
                # next_state, reward, done, info = self.env.step(action)
                self.enemy.update_movement(
                    self.player.position["x"],
                    self.player.position["y"],
                    self.player.step,
                    current_time,
                )
            except Exception as e:
                logging.error(f"Error updating enemy movement: {e}")
                self.running = False
                return

        if self.check_collision():
            logging.info("Collision detected between player and enemy.")
            if self.enemy:
                self.enemy.hide()
            self.is_respawning = True
            self.respawn_timer = current_time + self.respawn_delay
            logging.info("Player-Enemy collision in play mode.")

        # We'll keep missile logic here for the moment
        if self.missile_model and self.player and self.player.missiles:
            for missile in self.player.missiles:
                current_angle = math.atan2(missile.vy, missile.vx)
                px, py = self.player.position["x"], self.player.position["y"]
                ex, ey = self.enemy.pos["x"], self.enemy.pos["y"]
                dist_val = math.hypot(px - ex, py - ey)

                self._missile_input[0, 0] = px
                self._missile_input[0, 1] = py
                self._missile_input[0, 2] = ex
                self._missile_input[0, 3] = ey
                self._missile_input[0, 4] = missile.pos["x"]
                self._missile_input[0, 5] = missile.pos["y"]
                self._missile_input[0, 6] = current_angle
                self._missile_input[0, 7] = dist_val
                self._missile_input[0, 8] = 0.0

            with torch.no_grad():
                angle_prediction = self.missile_model(self._missile_input)
                logging.debug(f"Missile model output: {angle_prediction}")
                angle_delta = angle_prediction.item()

                new_angle = current_angle + angle_delta
                speed = 5.0
                missile.vx = math.cos(new_angle) * speed
                missile.vy = math.sin(new_angle) * speed

    def check_collision(self) -> bool:
        """
        Return True if the player and enemy overlap, else False.
        """
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
        """
        Check if any player missiles collide with the enemy. If so, trigger respawn.
        """
        if not self.enemy or not self.player:
            return

        def respawn_callback() -> None:
            self.is_respawning = True
            self.respawn_timer = pygame.time.get_ticks() + self.respawn_delay
            logging.info("Missile-Enemy collision in play mode, enemy will respawn.")

        handle_missile_collisions(self.player, self.enemy, respawn_callback)

    def handle_respawn(self, current_time: int) -> None:
        """
        After a collision and respawn delay, fade in a new enemy if needed.
        """
        if (
            self.is_respawning
            and current_time >= self.respawn_timer
            and self.enemy
            and self.player
        ):
            respawn_enemy_with_fade_in(self, current_time)

    def reset_game_state(self) -> None:
        """
        Reset the in-game state when returning to the menu
        or re-initializing for fullscreen toggles.
        Clears entities, data loggers, etc.
        """
        self.player = None
        self.enemy = None
        self.data_logger = None
        self.is_respawning = False
        self.respawn_timer = 0
        logging.info("Game state reset, returning to menu.")