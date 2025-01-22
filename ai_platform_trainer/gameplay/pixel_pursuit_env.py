import math
import random
import pygame


class PixelPursuitEnv:
    """
    A simple, standalone environment class to manage the 'world state' of Pixel Pursuit.
    This file should NOT import or reference the 'Game' class to avoid circular imports.
    """

    def __init__(self, width: int, height: int):
        """
        Initialize environment variables, screen dimensions, and any basic config.
        Note: Do NOT call pygame.display.set_mode here, because that's a 'Game' concern.
        """
        self.width = width
        self.height = height

        # Example for random seeding or placeholders for entities
        self.random_seed = 42
        random.seed(self.random_seed)

        self.current_time = 0
        self.done = False

        # Example placeholders for player, enemies, or missiles
        self.player = None
        self.enemy = None
        self.missiles = []

        # Initialize your environment
        self.reset()

    def reset(self):
        """
        Reset the environment to an initial state.
        Called at the beginning of a game or an episode for training.
        """
        self.current_time = 0
        self.done = False

        # Clear or re-initialize your entities
        self.player = None
        self.enemy = None
        self.missiles.clear()
        # If you have a Player/Enemy class, you'd instantiate them here
        # e.g. self.player = Player(x=50, y=50) etc.

    def step(self, action=None):
        """
        Advance the environment by one 'tick' or 'frame'.
        'action' can be a command for an AI agent, or you can ignore it for now.

        Returns:
          observation (dict): Info about the current game state (positions, etc.)
          reward (float): For RL or advanced AI
          done (bool): Indicates if game/episode is over
          info (dict): Any debugging or extra info
        """
        if self.done:
            # If it's already done, you might auto-reset or just skip updates
            return {}, 0.0, True, {}

        self.current_time += 1

        # 1. Process action if you have an AI controlling the player or enemy
        # 2. Update player/enemy/missile positions
        # 3. Check collisions or end conditions
        # 4. Build an observation
        observation = {
            "time": self.current_time,
            # "player_x": self.player.x if self.player else None,
            # "player_y": self.player.y if self.player else None,
            # ...
        }
        reward = 0.0  # e.g. +1 for survival, -1 for collision, etc.
        done = False  # Set True if collision or victory occurs
        info = {}

        # Suppose we detect a collision => done = True
        # self.done = done

        return observation, reward, done, info

    def render(self, surface):
        """
        OPTIONAL: If you want the environment to handle its own drawing, do it here.
        The 'surface' is typically your main Pygame screen passed in by 'Game'.
        """
        # e.g. if self.player: self.player.draw(surface)
        #      if self.enemy: self.enemy.draw(surface)
        pass

    def check_collisions(self):
        """
        OPTIONAL: Check collisions between player, enemy, missiles, etc.
        Return True if collisions occur, or more details if needed.
        """
        collision_detected = False
        return collision_detected
