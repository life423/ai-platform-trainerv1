import logging
import math
import random

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
    # Replace patterns list with weighted dictionary for more realistic behavior
    PATTERN_WEIGHTS = {
        "pursue": 0.7,  # 70% probability for pursuit
        "random_walk": 0.1,
        "circle_move": 0.1,
        "diagonal_move": 0.1
    }
    WALL_MARGIN = 20
    PURSUIT_SPEED_FACTOR = 0.8  # Relative speed factor when pursuing the player

    def __init__(self, screen_width: int, screen_height: int):
        """
        Initialize the EnemyTrain with no model. Movement is purely pattern-based.
        """
        super().__init__(screen_width, screen_height, model=None)
        self.size = self.DEFAULT_SIZE
        self.color = self.DEFAULT_COLOR
        self.pos = {
            "x": self.screen_width // 2,
            "y": self.screen_height // 2,
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
        self.forced_angle = None
        self.forced_speed = None

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
        
        # Hit reaction system
        self.hit_flash_timer = 0
        self.hit_flash_duration = 15  # Frames
        self.hit_color = (255, 100, 100)  # Reddish tint
        self.original_color = self.color
        
        # Recoil effect
        self.recoil_vector = {"x": 0, "y": 0}
        self.recoil_duration = 0
        
        # Tracking recent movement for missile avoidance
        self.recent_vx = 0
        self.recent_vy = 0
        self.game = None  # Will be set by the game instance
        
        # Pick an initial pattern
        self.switch_pattern()

    def switch_pattern(self):
        """
        Switch to a new movement pattern based on weighted probabilities,
        ensuring it's different from the current pattern. Also resets state_timer.
        """
        if self.forced_escape_timer > 0:
            return  # If we're forcing escape from a wall, don't switch

        # Choose pattern based on weights
        patterns = list(self.PATTERN_WEIGHTS.keys())
        weights = list(self.PATTERN_WEIGHTS.values())
        
        new_pattern = self.current_pattern
        while new_pattern == self.current_pattern:
            new_pattern = random.choices(patterns, weights=weights, k=1)[0]

        self.current_pattern = new_pattern
        
        # Different durations based on pattern type
        if self.current_pattern == "pursue":
            # Longer pursuit periods
            self.state_timer = random.randint(180, 240)
        else:
            # Shorter non-pursuit periods
            self.state_timer = random.randint(60, 120)

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
        Primary movement logic, with missile avoidance and hit reactions.
        """
        # Handle hit flash visual effect
        if self.hit_flash_timer > 0:
            # Alternate between normal and hit color
            if self.hit_flash_timer % 2 == 0:
                self.color = self.hit_color
            else:
                self.color = self.original_color
            self.hit_flash_timer -= 1
            if self.hit_flash_timer <= 0:
                self.color = self.original_color
        
        # Apply recoil if active
        if self.recoil_duration > 0:
            self.apply_recoil()
            # Skip other movement if strong recoil is active
            if self.recoil_duration > 5:
                return
        
        # First check for incoming missiles and dodge if needed
        missile_threat = self.check_missile_threat()
        
        if missile_threat:
            # Override pattern with dodge behavior
            self.dodge_missile(missile_threat)
        elif self.forced_escape_timer > 0:
            # Forced escape mode takes precedence over pattern-based movement
            self.forced_escape_timer -= 1
            self.apply_forced_escape_movement()
        else:
            # Decrement timer and possibly switch patterns
            self.state_timer -= 1
            if self.state_timer <= 0:
                self.switch_pattern()

            if self.current_pattern == "random_walk":
                self.random_walk_pattern()
            elif self.current_pattern == "circle_move":
                self.circle_pattern()
            elif self.current_pattern == "diagonal_move":
                self.diagonal_pattern()
            elif self.current_pattern == "pursue":
                self.pursue_pattern(player_x, player_y, player_speed)

        # Wrap around screen edges
        self.pos["x"], self.pos["y"] = wrap_position(
            self.pos["x"],
            self.pos["y"],
            self.screen_width,
            self.screen_height,
            self.size,
        )

    def pursue_pattern(self, player_x: float, player_y: float, player_speed: float):
        """
        Pursue the player by moving directly toward the player's position.
        The movement speed is based on the player's speed and a pursuit factor.
        """
        dx = player_x - self.pos["x"]
        dy = player_y - self.pos["y"]
        dist = math.hypot(dx, dy)

        if dist > 0:
            # Normalize dx, dy
            dx /= dist
            dy /= dist

            # Move at a fraction of the player's speed to avoid immediate collisions
            speed = player_speed * self.PURSUIT_SPEED_FACTOR
            self.pos["x"] += dx * speed
            self.pos["y"] += dy * speed

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
        Move in a random direction for a random duration, then pick another angle.
        """
        if self.random_walk_timer <= 0:
            self.random_walk_angle = random.uniform(0, 2 * math.pi)
            self.random_walk_speed = self.base_speed * random.uniform(0.5, 2.0)
            self.random_walk_timer = random.randint(30, 90)
        else:
            self.random_walk_timer -= 1

        dx = math.cos(self.random_walk_angle) * self.random_walk_speed
        dy = math.sin(self.random_walk_angle) * self.random_walk_speed
        self.pos["x"] += dx
        self.pos["y"] += dy

    def circle_pattern(self):
        """
        Move in a circular trajectory around a center point.
        """
        speed = self.base_speed
        angle_increment = 0.02 * (speed / self.base_speed)
        self.circle_angle += angle_increment

        dx = math.cos(self.circle_angle) * self.circle_radius
        dy = math.sin(self.circle_angle) * self.circle_radius
        self.pos["x"] = self.circle_center[0] + dx
        self.pos["y"] = self.circle_center[1] + dy

        # Occasionally alter circle radius
        if random.random() < 0.01:
            self.circle_radius += random.randint(-5, 5)
            self.circle_radius = max(20, min(200, self.circle_radius))

    def diagonal_pattern(self):
        """
        Move diagonally; occasionally randomize the diagonal angle.
        """
        if random.random() < 0.05:
            angle = math.atan2(self.diagonal_direction[1], self.diagonal_direction[0])
            angle += random.uniform(-0.3, 0.3)
            self.diagonal_direction = (math.cos(angle), math.sin(angle))

        speed = self.base_speed
        self.pos["x"] += self.diagonal_direction[0] * speed
        self.pos["y"] += self.diagonal_direction[1] * speed

    def register_hit(self):
        """
        Visual and movement reaction to being hit, without despawning.
        Replacing the hide/respawn mechanism with a more realistic hit effect.
        """
        # Ensure enemy remains completely visible during hit reaction
        self.visible = True
        
        # Flash effect
        self.hit_flash_timer = self.hit_flash_duration
        self.original_color = self.color
        
        # Calculate recoil direction (away from the closest player missile)
        if hasattr(self, "game") and self.game and self.game.player and self.game.player.missiles:
            # Get the closest missile
            missile = self.game.player.missiles[0]
            dx = self.pos["x"] - missile.pos["x"]
            dy = self.pos["y"] - missile.pos["y"]
            dist = math.hypot(dx, dy) or 1.0
            
            # Set recoil vector (away from missile)
            self.recoil_vector = {
                "x": (dx / dist) * 5.0,  # Recoil strength
                "y": (dy / dist) * 5.0
            }
            self.recoil_duration = 10  # Frames
        else:
            # Default recoil in random direction
            angle = random.uniform(0, 2 * math.pi)
            self.recoil_vector = {
                "x": math.cos(angle) * 5.0,
                "y": math.sin(angle) * 5.0
            }
            self.recoil_duration = 10
            
        logging.info("Enemy registered hit reaction (visual effect without despawning)")

    def apply_recoil(self):
        """Apply recoil movement from being hit"""
        if self.recoil_duration > 0:
            self.pos["x"] += self.recoil_vector["x"]
            self.pos["y"] += self.recoil_vector["y"]
            self.recoil_duration -= 1
            
            # Gradually reduce recoil strength
            self.recoil_vector["x"] *= 0.8
            self.recoil_vector["y"] *= 0.8

    def check_missile_threat(self):
        """Check if there's an incoming missile that poses a threat"""
        if not hasattr(self, "game") or not self.game or not hasattr(self.game, "player") or not self.game.player.missiles:
            return None
           
        for missile in self.game.player.missiles:
            # Calculate distance to missile
            dx = missile.pos["x"] - self.pos["x"]
            dy = missile.pos["y"] - self.pos["y"]
            dist = math.hypot(dx, dy)
            
            # Project missile trajectory
            future_x = missile.pos["x"] + missile.vx * 10  # Look ahead 10 frames
            future_y = missile.pos["y"] + missile.vy * 10
            
            future_dx = future_x - self.pos["x"]
            future_dy = future_y - self.pos["y"]
            future_dist = math.hypot(future_dx, future_dy)
            
            # If missile is approaching and getting close
            if future_dist < dist and dist < 150:
                return missile
               
        return None
       
    def dodge_missile(self, missile):
        """Execute a dodge maneuver away from missile path"""
        # Calculate perpendicular direction to missile velocity
        perp_x = -missile.vy
        perp_y = missile.vx
        
        # Normalize
        mag = math.hypot(perp_x, perp_y) or 1.0
        perp_x = perp_x / mag
        perp_y = perp_y / mag
        
        # Choose dodge direction (randomly pick one side)
        if random.random() < 0.5:
            perp_x = -perp_x
            perp_y = -perp_y
            
        # Apply dodge impulse
        dodge_speed = self.base_speed * 2.0
        self.pos["x"] += perp_x * dodge_speed
        self.pos["y"] += perp_y * dodge_speed
        
        # Reset pattern timer to allow more natural movement after dodge
        self.state_timer = min(self.state_timer, 30)
        
    # Keeping the original hide/show methods for backward compatibility
    def hide(self):
        """
        Immediately hide EnemyTrain (no fade). Typically called upon collision.
        Now just delegates to register_hit for a visual effect without despawning.
        """
        # Instead of hiding, register a hit reaction
        self.register_hit()
        logging.info("EnemyTrain hit reaction - not hiding for continuity")

    def show(self, current_time: int = None):
        """
        Make EnemyTrain visible again (no fade) after a collision.
        The current_time argument is optional and unused here.
        """
        self.visible = True
        logging.info("EnemyTrain made visible again.")
