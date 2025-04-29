# ai_platform_trainer/entities/enemy_play.py

import logging
import math
from typing import Tuple

import numpy as np
import pygame

from ai_platform_trainer.entities.behaviors.enemy_ai_controller import (
    update_enemy_movement,
)

# Import optionally - will be None if not available
try:
    from stable_baselines3 import PPO as SB3PPO

    RL_AVAILABLE = True
except ImportError:
    RL_AVAILABLE = False
    SB3PPO = None
    logging.info("Stable Baselines 3 not available - RL features disabled")


class EnemyPlay:
    def __init__(self, screen_width: int, screen_height: int, model=None):
        self.screen_width = screen_width
        self.screen_height = screen_height
        self.size = 50
        self.color = (255, 215, 0)  # Gold
        self.pos = {"x": self.screen_width // 2, "y": self.screen_height // 2}
        self.model = model
        self.base_speed = max(2, screen_width // 400)
        self.visible = True

        # Fade-in attributes
        self.fading_in = False
        self.fade_alpha = 255
        self.fade_duration = 300  # milliseconds
        self.fade_start_time = 0

        # Create a Surface for the enemy with per-pixel alpha
        self.surface = pygame.Surface((self.size, self.size), pygame.SRCALPHA)
        self.surface.fill((*self.color, 255))

    def wrap_position(self, x: float, y: float) -> Tuple[float, float]:
        """
        Example wrap-around logic.
        You might unify this in a separate utils file if used by multiple entities.
        """

        def wrap(val: float, lower: float, upper: float) -> float:
            if val < lower:
                return upper
            elif val > upper:
                return lower
            return val

        new_x = wrap(x, -self.size, self.screen_width)
        new_y = wrap(y, -self.size, self.screen_height)
        return new_x, new_y

    def load_rl_model(
        self, model_path: str = "models/enemy_rl/final_model.zip"
    ) -> bool:
        """
        Load a trained reinforcement learning model for enemy behavior.

        Args:
            model_path: Path to the saved PPO model

        Returns:
            True if successful, False otherwise
        """
        if not RL_AVAILABLE or SB3PPO is None:
            logging.warning("Cannot load RL model - Stable Baselines 3 not available")
            return False

        try:
            self.rl_model = SB3PPO.load(model_path)
            self.using_rl = True
            logging.info(f"Loaded RL model from {model_path}")
            return True
        except Exception as e:
            logging.error(f"Failed to load RL model: {e}")
            self.using_rl = False
            return False

    def apply_rl_action(self, action: np.ndarray) -> None:
        """
        Apply an action from the reinforcement learning model.

        This method is called by the EnemyGameEnv during training and evaluation.

        Args:
            action: Normalized action vector from the RL model (-1 to 1 range)
        """
        if not self.visible:
            return

        # Scale action to actual speed
        speed = self.base_speed
        self.pos["x"] += action[0] * speed
        self.pos["y"] += action[1] * speed

        # Apply wrap-around
        self.pos["x"], self.pos["y"] = self.wrap_position(self.pos["x"], self.pos["y"])

    def update_movement(
        self, player_x: float, player_y: float, player_speed: int, current_time: int
    ):
        """
        Update enemy movement based on AI model or reinforcement learning.

        This method checks if an RL model is available and uses it if so.
        Otherwise, it falls back to the traditional neural network approach.

        Args:
            player_x: Player's x position
            player_y: Player's y position
            player_speed: Player's movement speed
            current_time: Current game time in milliseconds
        """
        if not self.visible:
            return

        # Use RL model if available
        if hasattr(self, "using_rl") and self.using_rl and hasattr(self, "rl_model"):
            # Create observation for the model
            screen_width, screen_height = self.screen_width, self.screen_height

            # Normalize values to help with training stability
            px = player_x / screen_width
            py = player_y / screen_height
            ex = self.pos["x"] / screen_width
            ey = self.pos["y"] / screen_height

            # Calculate distance
            dist = math.sqrt(
                (player_x - self.pos["x"]) ** 2 + (player_y - self.pos["y"]) ** 2
            ) / max(screen_width, screen_height)

            player_speed_norm = player_speed / 10.0
            time_factor = 0.5  # Placeholder for time since last hit

            # Create observation array
            obs = np.array(
                [px, py, ex, ey, dist, player_speed_norm, time_factor], dtype=np.float32
            )

            # Get action from model
            try:
                action, _ = self.rl_model.predict(obs, deterministic=False)
                self.apply_rl_action(action)
            except Exception as e:
                # Fallback to traditional approach on error
                logging.error(
                    f"RL model inference failed: {e}. Falling back to neural network."
                )
                self.using_rl = False
                update_enemy_movement(
                    self, player_x, player_y, player_speed, current_time
                )
        else:
            # Use traditional neural network approach
            update_enemy_movement(
                self,
                player_x=player_x,
                player_y=player_y,
                player_speed=player_speed,
                current_time=current_time,
            )

    def draw(self, screen: pygame.Surface) -> None:
        if self.visible:
            self.surface.set_alpha(self.fade_alpha)
            screen.blit(self.surface, (self.pos["x"], self.pos["y"]))

    def hide(self) -> None:
        self.visible = False
        logging.info("Enemy hidden due to collision.")

    def show(self, current_time: int) -> None:
        self.visible = True
        self.fading_in = True
        self.fade_alpha = 0
        self.fade_start_time = current_time
        logging.info("Enemy set to fade in.")

    def update_fade_in(self, current_time: int) -> None:
        if self.fading_in:
            elapsed = current_time - self.fade_start_time
            if elapsed >= self.fade_duration:
                self.fade_alpha = 255
                self.fading_in = False
                logging.info("Enemy fade-in completed.")
            else:
                self.fade_alpha = int((elapsed / self.fade_duration) * 255)
                logging.debug(f"Enemy fade-in alpha: {self.fade_alpha}")

    def set_position(self, x: int, y: int) -> None:
        self.pos["x"], self.pos["y"] = x, y
