# file: ai_platform_trainer/gameplay/game.py
import logging
import pygame
import torch
from typing import Optional, Tuple

# Logging setup
from ai_platform_trainer.core.logging_config import setup_logging
from config_manager import load_settings, save_settings

# Gameplay imports
from ai_platform_trainer.gameplay.config import config as game_config
from ai_platform_trainer.gameplay.menu import Menu
from ai_platform_trainer.gameplay.renderer import Renderer
from ai_platform_trainer.gameplay.spawner import (
    spawn_entities,
    respawn_enemy_with_fade_in,
)
from ai_platform_trainer.gameplay.display_manager import (
    init_pygame_display,
    toggle_fullscreen_display,
)
from ai_platform_trainer.gameplay.missile_manager import MissileManager

# AI model imports
from ai_platform_trainer.ai_model.model_definition.enemy_movement_model import (
    EnemyMovementModel
)

# Data logger and entity imports
from ai_platform_trainer.core.data_logger import DataLogger
from ai_platform_trainer.entities.enemy_play import EnemyPlay
from ai_platform_trainer.entities.enemy_training import EnemyTrain
from ai_platform_trainer.entities.player_play import PlayerPlay
from ai_platform_trainer.entities.player_training import PlayerTraining
from ai_platform_trainer.gameplay.modes.training_mode import TrainingMode


class Game:
    """
    Main class to run the Pixel Pursuit game.
    Manages both training ('train') and play ('play') modes,
    as well as the main loop, event handling, and initialization.
    """

    def __init__(self, config_params: dict = None) -> None:
        setup_logging()
        self.running: bool = True
        self.menu_active: bool = True
        self.mode: Optional[str] = None
        self.config_params = config_params or {}

        # 1) Load user settings
        self.settings = load_settings("settings.json")
        
        # 2) Apply config to settings
        self.settings.update(self.config_params)
        
        # 3) Set mode directly if specified in config
        if ("mode" in self.config_params and 
                self.config_params["mode"] in ["train", "play"]):
            self.menu_active = False
            self.mode = self.config_params["mode"]

        # Check if we're in headless mode before initializing display
        self.headless_mode = self.config_params.get("headless", False)

        # 2) Initialize Pygame and the display (dummy in headless mode)
        (self.screen, self.screen_width, self.screen_height) = init_pygame_display(
            fullscreen=self.settings.get("fullscreen", False)
        )

        # 3) Create clock, menu, and renderer
        # Only set caption in non-headless mode
        if not self.headless_mode:
            pygame.display.set_caption(game_config.WINDOW_TITLE)
            
        self.clock = pygame.time.Clock()
        self.menu = Menu(self.screen_width, self.screen_height)
        self.renderer = Renderer(self.screen)

        # 4) Entities and managers
        self.player: Optional[PlayerPlay] = None
        self.enemy: Optional[EnemyPlay] = None
        self.data_logger: Optional[DataLogger] = None
        self.training_mode_manager: Optional[TrainingMode] = None  # For train mode

        # 5) Initialize missile manager and collision manager
        self.missile_manager = MissileManager(
            self.screen_width, self.screen_height, self.data_logger
        )
        
        # 6) Initialize collision manager
        from ai_platform_trainer.gameplay.collision_manager import CollisionManager
        self.collision_manager = CollisionManager(
            data_logger=self.data_logger,
            missile_manager=self.missile_manager
        )
        
        # 7) Initialize model manager and load necessary models
        from ai_platform_trainer.utils.model_manager import ModelManager
        self.model_manager = ModelManager()
        self.missile_model = self.model_manager.get_model("missile")
        
        # 8) Additional logic
        self.respawn_delay = 1000
        self.respawn_timer = 0
        self.is_respawning = False

        # Reusable tensor for missile AI input
        self._missile_input = torch.zeros((1, 9), dtype=torch.float32)

        logging.info("Game initialized.")

    def run(self) -> None:
        # If we have a direct mode, start it
        if self.mode and not self.player:
            self.start_game(self.mode)

        # read config for headless & training speed
        headless_mode = self.config_params.get("headless", False)
        training_speed = self.config_params.get("training_speed", 1.0)
        adjusted_fps = int(game_config.FRAME_RATE * training_speed)

        while self.running:
            current_time = pygame.time.get_ticks()
            self.handle_events()

            if self.menu_active:
                if not headless_mode:
                    self.menu.draw(self.screen)
            else:
                self.update(current_time)
                if not headless_mode:
                    self.renderer.render(
                        self.menu, self.player, self.enemy, self.menu_active,
                        self.missile_manager
                    )

            # Flip or skip
            if not headless_mode:
                pygame.display.flip()

            if headless_mode and self.mode == "train":
                self.clock.tick(adjusted_fps)
            else:
                self.clock.tick(game_config.FRAME_RATE)

        if self.mode == "train" and self.data_logger:
            self.data_logger.save()

        pygame.quit()
        logging.info("Game loop exited and Pygame quit.")

    def start_game(self, mode: str) -> None:
        self.mode = mode
        logging.info(f"Starting game in '{mode}' mode.")

        if mode == "train":
            # Create a unique filename for this training session with timestamp
            import time
            timestamp = int(time.time())
            unique_filename = (
                f"data/raw/training_data_{timestamp}.json"
            )
            
            # Create data logger with the unique filename
            self.data_logger = DataLogger(filename=unique_filename)
            
            # Update collision manager and missile manager with the data logger
            self.collision_manager.set_data_logger(self.data_logger)
            self.missile_manager.data_logger = self.data_logger
            
            # Log the filename being used
            logging.info(
                f"Training data will be saved to: {unique_filename}"
            )
            self.player = PlayerTraining(self.screen_width, self.screen_height)
            self.enemy = EnemyTrain(self.screen_width, self.screen_height)

            spawn_entities(self)
            self.player.reset()
            self.training_mode_manager = TrainingMode(self)

        else:  # "play"
            self.player, self.enemy = self._init_play_mode()
            self.player.reset()
            spawn_entities(self)

    def _init_play_mode(self) -> Tuple[PlayerPlay, EnemyPlay]:
        # Get the enemy model from the model manager
        model = self.model_manager.get_model("enemy")
        
        if model is None:
            logging.error("Failed to load enemy model from model manager")
            # Fallback to direct loading for backward compatibility
            model = EnemyMovementModel(input_size=5, hidden_size=64, output_size=2)
            try:
                model.load_state_dict(
                    torch.load(game_config.MODEL_PATH, map_location="cpu")
                )
                model.eval()
                logging.info("Enemy AI model loaded directly (fallback) for play mode.")
            except Exception as e:
                logging.error(f"Failed to load enemy model: {e}")
                raise e

        player = PlayerPlay(self.screen_width, self.screen_height)
        enemy = EnemyPlay(self.screen_width, self.screen_height, model)
        logging.info("Initialized PlayerPlay and EnemyPlay for play mode.")
        return player, enemy

    def handle_events(self) -> None:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                logging.info("Quit event detected. Exiting game.")
                self.running = False

            elif event.type == pygame.KEYDOWN:
                # Fullscreen toggling
                if event.key == pygame.K_f:
                    logging.debug("F pressed - toggling fullscreen.")
                    self._toggle_fullscreen()

                if self.menu_active:
                    selected_action = self.menu.handle_menu_events(event)
                    if selected_action:
                        self.check_menu_selection(selected_action)
                else:
                    if event.key == pygame.K_ESCAPE:
                        logging.info("Escape key pressed. Exiting game.")
                        self.running = False
                    elif event.key == pygame.K_SPACE and self.player:
                        self.player.shoot_missile(
                            self.enemy.pos, self.missile_manager
                        )
                    elif event.key == pygame.K_m:
                        logging.info("M key pressed. Returning to menu.")
                        self.menu_active = True
                        self.reset_game_state()

            elif event.type == pygame.MOUSEBUTTONDOWN and event.button == 1:
                if self.menu_active:
                    selected_action = self.menu.handle_menu_events(event)
                    if selected_action:
                        self.check_menu_selection(selected_action)

    def check_menu_selection(self, selected_action: str) -> None:
        if selected_action == "exit":
            logging.info("Exit action selected from menu.")
            self.running = False
        elif selected_action in ["train", "play"]:
            logging.info(f"'{selected_action}' selected from menu.")
            self.menu_active = False
            self.start_game(selected_action)

    def _toggle_fullscreen(self) -> None:
        """
        Helper that toggles between windowed and fullscreen, 
        updating self.screen, self.screen_width, self.screen_height.
        """
        was_fullscreen = self.settings["fullscreen"]
        new_display, w, h = toggle_fullscreen_display(
            not was_fullscreen,
            game_config.SCREEN_SIZE
        )
        self.settings["fullscreen"] = not was_fullscreen
        save_settings(self.settings, "settings.json")

        self.screen = new_display
        self.screen_width, self.screen_height = w, h
        pygame.display.set_caption(game_config.WINDOW_TITLE)
        self.menu = Menu(self.screen_width, self.screen_height)

        if not self.menu_active:
            current_mode = self.mode
            self.reset_game_state()
            self.start_game(current_mode)

    def update(self, current_time: int) -> None:
        if self.mode == "train" and self.training_mode_manager:
            self.training_mode_manager.update()
        elif self.mode == "play":
            # If we haven't created a play_mode_manager yet, do so now
            if (not hasattr(self, 'play_mode_manager') or 
                    self.play_mode_manager is None):
                from ai_platform_trainer.gameplay.modes.play_mode import PlayMode
                self.play_mode_manager = PlayMode(self)

            self.play_mode_manager.update(current_time)

    def check_collision(self) -> bool:
        """
        Use the collision manager to check for player-enemy collisions.
        This is a thin wrapper around the collision manager for backward compatibility.
        
        Returns:
            bool: True if player and enemy are colliding, False otherwise
        """
        return self.collision_manager.check_player_enemy_collision(
            self.player, self.enemy
        )

    def check_missile_collisions(self) -> None:
        """
        Use the collision manager to check for missile-enemy collisions.
        This is a thin wrapper around the collision manager for backward compatibility.
        """
        if not self.enemy:
            return

        def respawn_callback() -> None:
            self.is_respawning = True
            self.respawn_timer = pygame.time.get_ticks() + self.respawn_delay
            logging.info("Missile-Enemy collision in play mode, enemy will respawn.")

        # Use collision manager to check for missile-enemy collisions
        self.collision_manager.check_missile_enemy_collisions(
            self.missile_manager.missiles,
            self.enemy,
            pygame.time.get_ticks(),
            respawn_callback
        )

    def handle_respawn(self, current_time: int) -> None:
        if (
            self.is_respawning
            and current_time >= self.respawn_timer
            and self.enemy
            and self.player
        ):
            respawn_enemy_with_fade_in(self, current_time)

    def reset_game_state(self) -> None:
        self.player = None
        self.enemy = None
        self.data_logger = None
        self.is_respawning = False
        self.respawn_timer = 0
        self.missile_manager.clear_all()
        logging.info("Game state reset, returning to menu.")
