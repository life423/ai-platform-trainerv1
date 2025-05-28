import logging
import math
import random
from typing import Optional

from ai_platform_trainer.entities.enemy import Enemy
from ai_platform_trainer.utils.helpers import wrap_position


class EnemyTrain(Enemy):
    """
    EnemyTrain is a subclass of Enemy that does not rely on a model.
    It uses pattern-based movement to generate training data or mimic
    certain AI behaviors.

    The available patterns are:
    - random_walk
    - circle_move
    - diagonal_move
    - pursue (optional direct pursuit of the player)
    """

    DEFAULT_SIZE = 50
    DEFAULT_COLOR = (173, 153, 228)
    PATTERNS = [
        "random_walk",
        "circle_move",
        "diagonal_move",
        "pursue",
    ]  # Added "pursue"
    WALL_MARGIN = 20
    PURSUIT_SPEED_FACTOR = 0.8  # Relative speed factor when pursuing the player

    def __init__(self, screen_width: int, screen_height: int):
        """
        Initialize the EnemyTrain with no model. Movement is purely
        pattern-based.
        """
        super().__init__(screen_width, screen_height, model=None)
        self.size = self.DEFAULT_SIZE
        self.color = self.DEFAULT_COLOR
        self.pos = {
            "x": float(self.screen_width // 2),
            "y": float(self.screen_height // 2),
        }
        self.base_speed = max(2, screen_width // 400)
        self.visible = True

        # Timers and states controlling pattern selection
        self.state_timer = 0
        self.current_pattern = None

        # Wall escape logic
        self.wall_stall_counter = 0
        self.wall_stall_threshold = 10
        self.forced_escape_timer = 0
        self.forced_angle: float = 0.0  # Initialize as float
        self.forced_speed: float = 0.0  # Initialize as float

        # Circle-move parameters
        self.circle_center = (self.pos["x"], self.pos["y"])
        self.circle_angle = 0.0
        self.circle_radius = 100

        # Diagonal-move parameters
        self.diagonal_direction = (1, 1)

        # Random-walk parameters
        self.random_walk_timer = 0
        self.random_walk_angle = 0.0
        self.random_walk_speed = self.base_speed

        # Pick an initial pattern
        self.switch_pattern()

    def switch_pattern(self):
        """
        Switch to a new movement pattern from PATTERNS, ensuring it's
        different from the current pattern. Also resets state_timer
        for the new pattern.
        """
        if self.forced_escape_timer > 0:
            return  # If we're forcing escape from a wall, don't switch

        new_pattern = self.current_pattern
        while new_pattern == self.current_pattern:
            new_pattern = random.choice(self.PATTERNS)

        self.current_pattern = new_pattern
        self.state_timer = random.randint(120, 300)

        if self.current_pattern == "circle_move":
            self.circle_center = (self.pos["x"], self.pos["y"])
            self.circle_angle = random.uniform(0, 2 * math.pi)
            self.circle_radius = random.randint(50, 150)

        elif self.current_pattern == "diagonal_move":
            dx = random.choice([-1, 1])
            dy = random.choice([-1, 1])
            self.diagonal_direction = (dx, dy)

    def update_movement(self, player_x, player_y, player_speed):
        """
        Primary movement logic, choosing from the current pattern:
        random_walk, circle_move, diagonal_move, or pursue.
        """
        prev_x, prev_y = self.pos["x"], self.pos["y"]
        pattern = self.current_pattern
        # Log before movement
        logging.debug(
            f"EnemyTrain pattern: {pattern}, prev_pos: (%.2f, %.2f), "
            f"player_pos: (%.2f, %.2f), player_speed: %.2f",
            prev_x, prev_y, player_x, player_y, player_speed
        )

        if self.forced_escape_timer > 0:
            self.forced_escape_timer -= 1
            self.apply_forced_escape_movement()
        else:
            self.state_timer -= 1
            if self.state_timer <= 0:
                self.switch_pattern()

            if pattern == "random_walk":
                self.random_walk_pattern()
            elif pattern == "circle_move":
                self.circle_pattern()
            elif pattern == "diagonal_move":
                self.diagonal_pattern()
            elif pattern == "pursue":
                self.pursue_pattern(player_x, player_y, player_speed)
                # Log extra info for pursue mode
                logging.debug(
                    "PURSUING: Enemy (%.2f, %.2f) -> Player (%.2f, %.2f)",
                    self.pos["x"], self.pos["y"], player_x, player_y
                )

        # Wrap around screen edges
        self.pos["x"], self.pos["y"] = wrap_position(
            self.pos["x"],
            self.pos["y"],
            self.screen_width,
            self.screen_height,
            self.size,
        )
        # Log after movement
        logging.debug(
            "EnemyTrain new_pos: (%.2f, %.2f) [pattern: %s]",
            self.pos["x"], self.pos["y"], pattern
        )

    def pursue_pattern(self, player_x: float, player_y: float, player_speed: float):
        """
        Move enemy towards the player, adjusting speed to avoid immediate
        collisions.
        """
        # Move at a fraction of the player's speed to avoid collisions
        dx = player_x - self.pos["x"]
        dy = player_y - self.pos["y"]
        dist = math.hypot(dx, dy)
        if dist > 0:
            move_dist = min(self.speed, dist)
            self.pos["x"] += (dx / dist) * move_dist
            self.pos["y"] += (dy / dist) * move_dist

    def initiate_forced_escape(self):
        """
        Determine shortest distance to a wall and move away from it
        to prevent stalling near edges. Overrides normal patterns.
        """
        dist_left = self.pos["x"]
        dist_right = (self.screen_width - self.size) - self.pos["x"]
        dist_top = self.pos["y"]
        dist_bottom = (self.screen_height - self.size) - self.pos["y"]

        min_dist = min(dist_left, dist_right, dist_top, dist_bottom)

        if min_dist == dist_left:
            base_angle = 0.0
        elif min_dist == dist_right:
            base_angle = math.pi
        elif min_dist == dist_top:
            base_angle = math.pi / 2
        else:
            base_angle = math.pi * 3 / 2

        angle_variation = math.radians(30)
        self.forced_angle = base_angle + random.uniform(
            -angle_variation, angle_variation
        )
        self.forced_speed = self.base_speed * 1.0
        self.forced_escape_timer = random.randint(1, 30)
        self.wall_stall_counter = 0
        self.state_timer = self.forced_escape_timer * 2

    def apply_forced_escape_movement(self):
        """
        Execute forced escape movement to unstick from a wall.
        """
        dx = math.cos(self.forced_angle) * self.forced_speed
        dy = math.sin(self.forced_angle) * self.forced_speed
        self.pos["x"] += dx
        self.pos["y"] += dy

    def is_hugging_wall(self) -> bool:
        """
        Check if the enemy is near the wall boundary within WALL_MARGIN.
        """
        return (
            self.pos["x"] < self.WALL_MARGIN
            or self.pos["x"] > (self.screen_width - self.size - self.WALL_MARGIN)
            or self.pos["y"] < self.WALL_MARGIN
            or self.pos["y"] > (self.screen_height - self.size - self.WALL_MARGIN)
        )

    def random_walk_pattern(self):
        """
        Move in a random direction for a random duration.
        Then pick another angle.
        """
        if self.pattern_timer <= 0:
            self.pattern_timer = random.randint(30, 90)
            self.angle = random.uniform(0, 2 * math.pi)
        self.pattern_timer -= 1
        self.pos["x"] += math.cos(self.angle) * self.speed
        self.pos["y"] += math.sin(self.angle) * self.speed

    def circle_pattern(self):
        """
        Move in a circular pattern around a center point.
        """
        if not hasattr(self, "circle_angle"):
            self.circle_angle = 0
        self.circle_angle += 0.05
        radius = 50
        self.pos["x"] += math.cos(self.circle_angle) * self.speed
        self.pos["y"] += math.sin(self.circle_angle) * self.speed

    def diagonal_pattern(self):
        """
        Move diagonally across the screen, bouncing off edges.
        """
        if not hasattr(self, "diagonal_direction"):
            self.diagonal_direction = [1, 1]
        self.pos["x"] += self.diagonal_direction[0] * self.speed
        self.pos["y"] += self.diagonal_direction[1] * self.speed
        if (
            self.pos["x"] < self.size + self.WALL_MARGIN or
            self.pos["x"] > (
                self.screen_width - self.size - self.WALL_MARGIN
            )
        ):
            self.diagonal_direction[0] *= -1
        if (
            self.pos["y"] < self.size + self.WALL_MARGIN or
            self.pos["y"] > (
                self.screen_height - self.size - self.WALL_MARGIN
            )
        ):
            self.diagonal_direction[1] *= -1
        angle = math.atan2(
            self.diagonal_direction[1],
            self.diagonal_direction[0]
        )

    def hide(self):
        """
        Immediately hide EnemyTrain (no fade). Typically called upon
        collision.
        """
        self.visible = False
        logging.info("EnemyTrain hidden due to collision.")

    def show(self, current_time: Optional[int] = None):
        """
        Make EnemyTrain visible again (no fade) after a collision.
        The current_time argument is optional and unused here.
        """
        self.visible = True
        logging.info("EnemyTrain made visible again.")
        logging.info("EnemyTrain made visible again.")
