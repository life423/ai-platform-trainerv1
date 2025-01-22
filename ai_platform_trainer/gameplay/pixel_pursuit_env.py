# file: ai_platform_trainer/gameplay/pixel_pursuit_env.py

import math
import random
import pygame
import logging


class PixelPursuitEnv:
    """
    A standalone environment class for Pixel Pursuit, with a 'step()' method
    returning (observation, reward, done, info). This allows integration with
    RL or advanced AI logic without forcing Pygame user input.
    """

    def __init__(self, width: int, height: int):
        """
        Initialize environment variables, world dimensions, etc.
        """
        self.width = width
        self.height = height

        # Example for random seeding, or placeholders for your entities
        self.random_seed = 42
        random.seed(self.random_seed)

        # Track time or step count if needed
        self.current_time = 0
        self.done = False

        # Entities
        self.player = None
        self.enemy = None
        self.missiles = []

        logging.info(
            "PixelPursuitEnv initialized with width=%d, height=%d", width, height
        )

    def reset(self, player, enemy, missiles=None):
        """
        Reset the environment to an initial state, providing references to
        existing player/enemy objects if you want to keep them consistent
        across resets. Otherwise, you can instantiate them here.

        Returns an initial observation (dict) if you want to do RL.
        """
        self.current_time = 0
        self.done = False
        self.player = player
        self.enemy = enemy
        self.missiles = missiles if missiles else []

        logging.info("Environment reset. Entities assigned.")

        return self._get_observation()

    def step(self, action=None):
        """
        Advance the environment by one tick.

        1) Interpret 'action' (if you're controlling the player or enemy).
        2) Update positions, collisions, etc.
        3) Compute reward, check if done.
        4) Return (observation, reward, done, info).
        """
        if self.done:
            # If it's already done, might auto-reset or skip
            return self._get_observation(), 0.0, True, {}

        self.current_time += 1

        # Example: If 'action' is (dx, dy) for the player:
        if action and self.player:
            dx, dy = action
            self.player.position["x"] += dx
            self.player.position["y"] += dy

        # TODO: Update enemy, missiles, collisions, etc.
        # e.g. self._update_enemy() or self._update_missiles()

        # Check collisions
        reward = 0.0
        done = False
        if self._check_collision():
            reward = -1.0
            done = True
            self.done = True
            logging.info("Collision detected -> environment done.")

        # Example: survival reward
        reward += 0.01

        observation = self._get_observation()
        info = {}

        return observation, reward, done, info

    def render(self, surface: pygame.Surface):
        """
        OPTIONAL: If you want the environment itself to handle drawing logic,
        pass in the Pygame surface from 'Game'.
        """
        # Example:
        # if self.player: self.player.draw(surface)
        # if self.enemy: self.enemy.draw(surface)
        pass

    def _check_collision(self) -> bool:
        """
        Check collision between self.player and self.enemy as an example.
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

    def _get_observation(self) -> dict:
        """
        Return any info about the current environment state:
        e.g. positions, velocities, time, etc.
        """
        obs = {
            "time": self.current_time,
            "player_x": self.player.position["x"] if self.player else None,
            "player_y": self.player.position["y"] if self.player else None,
            "enemy_x": self.enemy.pos["x"] if self.enemy else None,
            "enemy_y": self.enemy.pos["y"] if self.enemy else None,
        }
        return obs
