import logging
import math
import os
import random
from typing import Optional, Tuple, List

import pygame
import torch

from ai_platform_trainer.ai_model.model_definition.enemy_movement_model import EnemyMovementModel
from ai_platform_trainer.ai_model.simple_missile_model import SimpleMissileModel
from ai_platform_trainer.core.data_logger import DataLogger
from ai_platform_trainer.core.logging_config import setup_logging
from ai_platform_trainer.entities.enemy_play import EnemyPlay
from ai_platform_trainer.entities.components.enemy_variants import create_enemy_by_type
from ai_platform_trainer.entities.enemy_training import EnemyTrain
from ai_platform_trainer.entities.player_play import PlayerPlay
from ai_platform_trainer.entities.player_training import PlayerTraining
from ai_platform_trainer.gameplay.collisions import handle_missile_collisions
from ai_platform_trainer.gameplay.difficulty_manager import DifficultyManager
from ai_platform_trainer.gameplay.config import config
from ai_platform_trainer.gameplay.display_manager import (
    init_pygame_display,
    toggle_fullscreen_display,
)
from ai_platform_trainer.gameplay.menu import Menu
from ai_platform_trainer.gameplay.missile_ai_controller import update_missile_ai
from ai_platform_trainer.gameplay.modes.training_mode import TrainingMode
from ai_platform_trainer.gameplay.renderer import Renderer
from ai_platform_trainer.gameplay.spawner import respawn_enemy_with_fade_in, spawn_entities
from config_manager import load_settings, save_settings


class Game:
    """
    Main class to run the Pixel Pursuit game.
    Manages both training ('train') and play ('play') modes,
    as well as the main loop, event handling, and initialization.
    """

    def __init__(self) -> None:
        setup_logging()
        self.running: bool = True
        self.menu_active: bool = True
        self.mode: Optional[str] = None

        self.settings = load_settings("settings.json")

        (self.screen, self.screen_width, self.screen_height) = init_pygame_display(
            fullscreen=self.settings.get("fullscreen", False)
        )

        pygame.display.set_caption(config.WINDOW_TITLE)
        self.clock = pygame.time.Clock()
        self.menu = Menu(self.screen_width, self.screen_height)
        self.renderer = Renderer(self.screen)

        self.player: Optional[PlayerPlay] = None
        self.enemy: Optional[EnemyPlay] = None
        self.enemies: List[EnemyPlay] = []
        self.num_enemies: int = 3  # Default starting number of enemies
        self.enemy_types = ["standard", "fast", "tank"]  # Available enemy types
        
        self.data_logger: Optional[DataLogger] = None
        self.training_mode_manager: Optional[TrainingMode] = None

        self.missile_model: Optional[SimpleMissileModel] = None
        self._load_missile_model_once()

        self.respawn_delay = 1000
        self.respawn_timer = 0
        self.is_respawning = False
        
        # Score tracking
        self.score = 0
        self.last_score_time = 0
        self.survival_score_interval = 1000  # 1 point per second
        self.paused = False

        self._missile_input = torch.zeros((1, 9), dtype=torch.float32)

        logging.info("Game initialized.")

    def _load_missile_model_once(self) -> None:
        missile_model_path = "models/missile_model.pth"
        if os.path.isfile(missile_model_path):
            logging.info(f"Found missile model at '{missile_model_path}'.")
            logging.info("Loading missile model once...")
            try:
                model = SimpleMissileModel()
                model.load_state_dict(torch.load(missile_model_path, map_location="cpu"))
                model.eval()
                self.missile_model = model
            except Exception as e:
                logging.error(f"Failed to load missile model: {e}")
                self.missile_model = None
        else:
            logging.warning(f"No missile model found at '{missile_model_path}'.")
            logging.warning("Skipping missile AI.")

    def run(self) -> None:
        while self.running:
            current_time = pygame.time.get_ticks()
            self.handle_events()

            if self.menu_active:
                self.menu.draw(self.screen)
            else:
                self.update(current_time)
                self.renderer.render(self.menu, self.player, self.enemy, self.menu_active)

            pygame.display.flip()
            self.clock.tick(config.FRAME_RATE)

        if self.mode == "train" and self.data_logger:
            self.data_logger.save()

        pygame.quit()
        logging.info("Game loop exited and Pygame quit.")

    def start_game(self, mode: str) -> None:
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
            self.player, self.enemy, self.enemies = self._init_play_mode()
            
            # Initialize score tracking
            self.score = 0
            self.last_score_time = pygame.time.get_ticks()
            self.player.reset()
            spawn_entities(self)

    def _init_play_mode(self) -> Tuple[PlayerPlay, EnemyPlay, List[EnemyPlay]]:
        model = EnemyMovementModel(input_size=5, hidden_size=64, output_size=2)
        try:
            model.load_state_dict(torch.load(config.MODEL_PATH, map_location="cpu"))
            model.eval()
            logging.info("Enemy AI model loaded for play mode.")
        except Exception as e:
            logging.error(f"Failed to load enemy model: {e}")
            raise e

        player = PlayerPlay(self.screen_width, self.screen_height)
        # Create single enemy for backward compatibility
        enemy = EnemyPlay(self.screen_width, self.screen_height, model)
        
        # Get initial enemy count from difficulty manager
        difficulty_manager = DifficultyManager()
        params = difficulty_manager.get_current_parameters()
        self.num_enemies = params['max_enemies']
        
        # Create multiple enemies of different types
        enemies = []
        for i in range(self.num_enemies):
            # Randomly select an enemy type with weighted probability
            # Standard: 50%, Fast: 25%, Tank: 25%
            weights = [0.5, 0.25, 0.25]
            enemy_type = random.choices(self.enemy_types, weights=weights)[0]
            
            new_enemy = create_enemy_by_type(
                enemy_type, 
                self.screen_width, 
                self.screen_height, 
                model
            )
            
            # Add respawn time attribute for individual enemy respawn management
            new_enemy.respawn_time = 0
            enemies.append(new_enemy)
            
            logging.info(f"Created {enemy_type} enemy ({i+1} of {self.num_enemies})")

        # Load RL model for all enemies if available
        rl_model_path = "models/enemy_rl/final_model.zip"
        if os.path.exists(rl_model_path):
            try:
                success = enemy.load_rl_model(rl_model_path)
                if success:
                    logging.info("Using reinforcement learning model for enemy behavior")
                    # Apply RL model to all enemies as well
                    for e in enemies:
                        e.load_rl_model(rl_model_path)
                else:
                    logging.warning("RL model exists but couldn't be loaded.")
                    logging.warning("Falling back to neural network.")
            except Exception as e:
                logging.error(f"Error loading RL model: {e}.")
                logging.error("Using neural network instead.")
        else:
            logging.info("No RL model found, using traditional neural network")

        logging.info(f"Initialized PlayerPlay and {len(enemies)} enemies for play mode.")
        return player, enemy, enemies
    
    def spawn_random_enemy(self) -> EnemyPlay:
        """
        Create a new enemy of a random type.
        
        Returns:
            EnemyPlay: A new enemy instance
        """
        # Load the model once if it doesn't exist
        model = None
        if hasattr(self, 'enemy') and self.enemy and hasattr(self.enemy, 'model'):
            model = self.enemy.model
            
        # Randomly select an enemy type with weighted probability
        # Standard: 50%, Fast: 25%, Tank: 25%
        weights = [0.5, 0.25, 0.25]
        enemy_type = random.choices(self.enemy_types, weights=weights)[0]
        
        new_enemy = create_enemy_by_type(
            enemy_type, 
            self.screen_width, 
            self.screen_height, 
            model
        )
        
        # Add respawn time attribute for individual enemy respawn management
        new_enemy.respawn_time = 0
        
        # Load RL model if available
        rl_model_path = "models/enemy_rl/final_model.zip"
        if os.path.exists(rl_model_path):
            try:
                new_enemy.load_rl_model(rl_model_path)
            except Exception as e:
                logging.error(f"Error loading RL model for new enemy: {e}")
        
        logging.info(f"Spawned new {enemy_type} enemy")
        return new_enemy

    def handle_events(self) -> None:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                logging.info("Quit event detected. Exiting game.")
                self.running = False

            elif event.type == pygame.KEYDOWN:
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
                        # Target the nearest visible enemy for missile
                        target_pos = self._get_nearest_enemy_position()
                        if target_pos:
                            self.player.shoot_missile(target_pos)
                    elif event.key == pygame.K_m:
                        logging.info("M key pressed. Returning to menu.")
                        self.menu_active = True
                        self.reset_game_state()

            elif event.type == pygame.MOUSEBUTTONDOWN and event.button == 1:
                if self.menu_active:
                    selected_action = self.menu.handle_menu_events(event)
                    if selected_action:
                        self.check_menu_selection(selected_action)
                        
    def _get_nearest_enemy_position(self) -> Optional[dict]:
        """
        Find the nearest visible enemy to the player.
        
        Returns:
            Optional[dict]: Position of the nearest enemy or None if no visible enemies
        """
        if not self.player:
            return None
            
        # Check multiple enemies first
        if hasattr(self, 'enemies') and self.enemies:
            nearest_enemy = None
            min_distance = float('inf')
            
            for enemy in self.enemies:
                if not enemy.visible:
                    continue
                    
                # Calculate distance to player
                dx = enemy.pos["x"] - self.player.position["x"]
                dy = enemy.pos["y"] - self.player.position["y"]
                distance = (dx**2 + dy**2)**0.5
                
                if distance < min_distance:
                    min_distance = distance
                    nearest_enemy = enemy
                    
            if nearest_enemy:
                return nearest_enemy.pos
                
        # Fall back to single enemy
        if self.enemy and self.enemy.visible:
            return self.enemy.pos
            
        return None

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
            config.SCREEN_SIZE
        )
        self.settings["fullscreen"] = not was_fullscreen
        save_settings(self.settings, "settings.json")

        self.screen = new_display
        self.screen_width, self.screen_height = w, h
        pygame.display.set_caption(config.WINDOW_TITLE)
        self.menu = Menu(self.screen_width, self.screen_height)

        if not self.menu_active:
            current_mode = self.mode
            self.reset_game_state()
            self.start_game(current_mode)

    def update(self, current_time: int) -> None:
        if self.mode == "train" and self.training_mode_manager:
            self.training_mode_manager.update()
        elif self.mode == "play":
            if not hasattr(self, 'play_mode_manager') or self.play_mode_manager is None:
                from ai_platform_trainer.gameplay.modes.play_mode import PlayMode
                self.play_mode_manager = PlayMode(self)

            self.play_mode_manager.update(current_time)

    def play_update(self, current_time: int) -> None:
        """
        Main update logic for 'play' mode.
        """
        if self.player and not self.player.handle_input():
            logging.info("Player requested to quit.")
            self.running = False
            return

        if self.enemy:
            try:
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

        if self.missile_model and self.player and self.player.missiles:
            update_missile_ai(
                self.player.missiles,
                self.player.position,
                self.enemy.pos if self.enemy else None,
                self._missile_input,
                self.missile_model
            )

    def check_collision(self) -> bool:
        if not (self.player and self.enemy):
            return False
        player_rect = pygame.Rect(
            self.player.position["x"],
            self.player.position["y"],
            self.player.size,
            self.player.size,
        )
        enemy_rect = pygame.Rect(
            self.enemy.pos["x"],
            self.enemy.pos["y"],
            self.enemy.size,
            self.enemy.size
        )
        return player_rect.colliderect(enemy_rect)

    def check_missile_collisions(self) -> None:
        if not self.enemy or not self.player:
            return

        def respawn_callback() -> None:
            self.is_respawning = True
            self.respawn_timer = pygame.time.get_ticks() + self.respawn_delay
            logging.info("Missile-Enemy collision in play mode, enemy will respawn.")

        handle_missile_collisions(self.player, self.enemy, respawn_callback)

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
        self.enemies = []  # Reset the enemies list too
        self.data_logger = None
        self.is_respawning = False
        self.respawn_timer = 0
        logging.info("Game state reset, returning to menu.")

    def reset_enemy(self) -> None:
        """Reset the enemy's position but keep it in the game.

        This is primarily used during RL training to reset the
        environment without disturbing other game elements.
        """
        if self.enemy:
            import random
            if self.player:
                while True:
                    x = random.randint(0, self.screen_width - self.enemy.size)
                    y = random.randint(0, self.screen_height - self.enemy.size)

                    distance = math.sqrt(
                        (x - self.player.position["x"])**2 +
                        (y - self.player.position["y"])**2
                    )

                    min_distance = max(self.screen_width, self.screen_height) * 0.3
                    if distance >= min_distance:
                        break
            else:
                x = random.randint(0, self.screen_width - self.enemy.size)
                y = random.randint(0, self.screen_height - self.enemy.size)

            self.enemy.set_position(x, y)
            self.enemy.visible = True
            logging.debug(f"Enemy reset to position ({x}, {y})")

    def update_once(self) -> None:
        """Process a single update frame for the game.

        This is used during RL training to advance the game state
        without relying on the main game loop.
        """
        current_time = pygame.time.get_ticks()

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                self.running = False

        if self.mode == "play" and not self.menu_active:
            if hasattr(self, 'play_mode_manager') and self.play_mode_manager:
                self.play_mode_manager.update(current_time)
            else:
                self.play_update(current_time)