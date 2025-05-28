"""
Enemy AI Controller for AI Platform Trainer.

This module handles the enemy's AI-driven movement using neural network models.
"""
import math
import random
import logging
import time
import torch
import numpy as np # Added for NumPy model
from typing import Tuple, Dict, List, Optional # Union removed as it's unused

# Assuming NumpyEnemyModel will be in this path
from ai_platform_trainer.ai.models.numpy_enemy_model import NumpyEnemyModel

logger = logging.getLogger(__name__) # Added logger definition

class EnemyAIController:
    """
    Controller for enemy AI behavior using neural networks only.

    This class manages the enemy movement logic using neural network inference
    and fallback behaviors to prevent freezing.
    """

    def __init__(self):
        """Initialize the enemy AI controller."""
        self.last_action_time = time.time()
        # Reduced from 0.05 to 0.01 (10ms between actions)
        self.action_interval = 0.01
        # Reduced from 0.7 to 0.4 for more responsive movement
        self.smoothing_factor = 0.4
        self.prev_dx = 0
        self.prev_dy = 0
        self.stuck_counter = 0
        self.prev_positions: List[Dict[str, float]] = []
        self.max_positions = 10  # Store last 10 positions to detect being stuck
        
        # Missile avoidance parameters
        self.missile_detection_radius = 150.0  # How far to detect missiles
        self.missile_danger_radius = 80.0      # When to start emergency evasion
        self.evasion_strength = 1.5            # How strongly to evade (multiplier)
        self.prediction_time = 10              # How many frames to predict missile movement
        self.numpy_model: Optional[NumpyEnemyModel] = None # For the NumPy model instance
        self.use_numpy_model_if_available = True # Flag to switch between torch and numpy

    def load_numpy_model(self, model_path: str = "models/numpy_enemy_model.npz"):
        """Attempts to load the NumPy-based model."""
        try:
            self.numpy_model = NumpyEnemyModel(model_path)
            logger.info(f"Successfully loaded NumPy enemy model from {model_path}")
        except Exception as e:
            logger.error(f"Failed to load NumPy enemy model from {model_path}: {e}")
            self.numpy_model = None # Ensure it's None if loading fails

    def _detect_missiles(self, enemy, player) -> List[Dict]:
        """
        Detect player missiles in the vicinity of the enemy.
        
        Args:
            enemy: Enemy instance
            player: Player instance with missiles attribute
            
        Returns:
            List of missile information dicts with positions and velocities
        """
        nearby_missiles = []
        
        # Check if player has missiles attribute and it's not empty
        if not hasattr(player, 'missiles') or not player.missiles:
            return nearby_missiles
            
        enemy_x, enemy_y = enemy.pos["x"], enemy.pos["y"]
        
        for missile in player.missiles:
            missile_x, missile_y = missile.pos["x"], missile.pos["y"]
            
            # Calculate distance to missile
            dx = missile_x - enemy_x
            dy = missile_y - enemy_y
            distance = math.sqrt(dx*dx + dy*dy)
            
            # Check if the missile is within detection range
            if distance <= self.missile_detection_radius:
                # Predict future position based on velocity
                future_x = missile_x + (self.prediction_time * missile.vx)
                future_y = missile_y + (self.prediction_time * missile.vy)
                
                # Calculate future distance
                future_dx = future_x - enemy_x
                future_dy = future_y - enemy_y
                # Calculate the squared distance
                # Split calculation to avoid long line
                future_dx_squared = future_dx * future_dx
                future_dy_squared = future_dy * future_dy
                future_dist_squared = future_dx_squared + future_dy_squared
                # Get actual distance
                # by taking the square root
                future_distance = math.sqrt(future_dist_squared)
                
                # Create the missile info dictionary
                missile_info = {
                    "missile": missile,
                    "distance": distance,
                    "future_distance": future_distance,
                    "dx": dx,
                    "dy": dy,
                    "future_dx": future_dx,
                    "future_dy": future_dy,
                    "vx": missile.vx,
                    "vy": missile.vy
                }
                nearby_missiles.append(missile_info)
                
        return nearby_missiles
    
    def _calculate_evasion_vector(
        self,
        enemy_pos: Dict[str, float],
        missiles: List[Dict]
    ) -> Tuple[float, float]:
        """
        Calculate optimal evasion vector based on nearby missiles.
        
        Args:
            enemy_pos: Enemy position dictionary
            missiles: List of detected missile information
            
        Returns:
            (dx, dy) evasion direction tuple
        """
        if not missiles:
            return 0, 0
            
        evasion_x, evasion_y = 0, 0
        
        for missile_info in missiles:
            # Calculate danger level - higher for closer missiles
            danger = 1.0
            if missile_info["distance"] < self.missile_danger_radius:
                # Exponential increase in danger as missiles get very close
                max_dist = max(missile_info["distance"], 10)
                danger = self.evasion_strength * (
                    self.missile_danger_radius / max_dist
                )
            
            # Strongest evasion from predicted future position
            evade_dx = -missile_info["future_dx"]  # Move away from missile's future position
            evade_dy = -missile_info["future_dy"]
            
            # Normalize evasion vector
            evade_magnitude = math.sqrt(evade_dx**2 + evade_dy**2)
            if evade_magnitude > 0:
                evade_dx /= evade_magnitude
                evade_dy /= evade_magnitude
                
                # Weight by danger level and add to cumulative evasion
                evasion_x += evade_dx * danger
                evasion_y += evade_dy * danger
        
        # Normalize the final evasion vector
        return self._normalize_vector(evasion_x, evasion_y)
    
    def _blend_behaviors(self, chase_vector: Tuple[float, float], 
                         evasion_vector: Tuple[float, float], 
                         evasion_weight: float) -> Tuple[float, float]:
        """
        Blend chasing behavior with evasion behavior.
        
        Args:
            chase_vector: Original movement vector towards player
            evasion_vector: Vector for evading missiles
            evasion_weight: How strongly to weight evasion (0-1)
            
        Returns:
            Blended direction vector
        """
        # No evasion means no change
        if evasion_weight == 0 or (evasion_vector[0] == 0 and evasion_vector[1] == 0):
            return chase_vector
            
        # Full evasion means use only evasion vector
        if evasion_weight >= 1.0:
            return evasion_vector
        
        # Blend the two behaviors
        blend_x = chase_vector[0] * (1 - evasion_weight) + evasion_vector[0] * evasion_weight
        blend_y = chase_vector[1] * (1 - evasion_weight) + evasion_vector[1] * evasion_weight
        
        # Normalize the result
        return self._normalize_vector(blend_x, blend_y)
    
    def update_enemy_movement(
        self,
        enemy, # EnemyPlay instance
        player_x: float,
        player_y: float,
        player_speed: float,
        current_time: int,
        # Allow passing a pre-loaded numpy_model, or it will use its own instance
        numpy_model_instance: Optional[NumpyEnemyModel] = None 
    ) -> None:
        """
        Handle the enemy's AI-driven movement using neural networks.
        Can use either PyTorch model from enemy instance or a NumPy model.

        Args:
            enemy: EnemyPlay instance (expected to have a 'model' attribute for PyTorch)
            player_x: Player's x position
            player_y: Player's y position
            player_speed: Player's movement speed
            current_time: Current game time
            numpy_model_instance: Optional pre-loaded NumpyEnemyModel instance.
        """
        if not enemy.visible:
            return

        current_time_sec = time.time()
        if current_time_sec - self.last_action_time < self.action_interval:
            return
        self.last_action_time = current_time_sec

        self._update_position_history(enemy.pos)
        
        # Determine which model to use
        active_numpy_model = numpy_model_instance if numpy_model_instance else self.numpy_model

        if self.use_numpy_model_if_available and active_numpy_model:
            action_dx, action_dy = self._get_numpy_nn_action(active_numpy_model, enemy, player_x, player_y)
        elif hasattr(enemy, 'model') and enemy.model: # Fallback to PyTorch model if available
            action_dx, action_dy = self._get_pytorch_nn_action(enemy, player_x, player_y)
        else: # No model available, use random or simple chase
            logger.warning("No suitable AI model found for enemy. Using random movement.")
            action_dx, action_dy = self._get_random_direction()


        if self._is_enemy_stuck():
            action_dx, action_dy = self._handle_stuck_enemy(player_x, player_y, enemy.pos)
        
        player = None
        if hasattr(enemy, 'game') and hasattr(enemy.game, 'player'):
            player = enemy.game.player
            
        evasion_dx, evasion_dy = 0, 0
        evasion_weight = 0
        
        if player:
            nearby_missiles = self._detect_missiles(enemy, player)
            if nearby_missiles:
                evasion_dx, evasion_dy = self._calculate_evasion_vector(enemy.pos, nearby_missiles)
                closest_missile = min(nearby_missiles, key=lambda m: m["distance"])
                closest_distance = closest_missile["distance"]
                if closest_distance < self.missile_danger_radius:
                    evasion_weight = min(1.0, self.missile_danger_radius / closest_distance) * 0.8
                else:
                    ratio = closest_distance / self.missile_detection_radius
                    evasion_weight = max(0, 0.5 * (1 - ratio))
                logging.debug(
                    f"Missile detected! Distance: {closest_distance:.1f}, "
                    f"Evasion weight: {evasion_weight:.2f}"
                )
            action_dx, action_dy = self._blend_behaviors(
                (action_dx, action_dy), 
                (evasion_dx, evasion_dy), 
                evasion_weight
            )

        action_dx = self.smoothing_factor * action_dx + (1 - self.smoothing_factor) * self.prev_dx
        action_dy = self.smoothing_factor * action_dy + (1 - self.smoothing_factor) * self.prev_dy

        action_dx, action_dy = self._normalize_vector(action_dx, action_dy)
        self.prev_dx, self.prev_dy = action_dx, action_dy

        speed = player_speed * 1.0
        enemy.pos["x"] += action_dx * speed
        enemy.pos["y"] += action_dy * speed
        enemy.pos["x"], enemy.pos["y"] = enemy.wrap_position(enemy.pos["x"], enemy.pos["y"])

    def _get_numpy_nn_action(
        self,
        model: NumpyEnemyModel, # Expects a NumpyEnemyModel instance
        enemy,
        player_x: float,
        player_y: float
    ) -> Tuple[float, float]:
        """Get action from the NumPy-based neural network model."""
        dist = math.sqrt((player_x - enemy.pos["x"])**2 + (player_y - enemy.pos["y"])**2)
        # State for NumPy model: [enemy_x, enemy_y, player_x, player_y, distance]
        # Assuming positions are already normalized if required by the model, or handle normalization here.
        # For now, using raw positions as per the dummy data generator.
        state_np = np.array(
            [enemy.pos["x"], enemy.pos["y"], player_x, player_y, dist],
            dtype=np.float32
        )
        try:
            action_id = model.predict(state_np)
            # Convert action ID to dx, dy. This mapping depends on your NumpyEnemyTeacher's action definitions.
            # ACTION_LEFT = 0, ACTION_RIGHT = 1, ACTION_UP = 2, ACTION_DOWN = 3
            if action_id == 0: return -1.0, 0.0  # Left
            elif action_id == 1: return 1.0, 0.0   # Right
            elif action_id == 2: return 0.0, -1.0  # Up (assuming Y decreases upwards)
            elif action_id == 3: return 0.0, 1.0   # Down
            else: return self._get_random_direction() # Fallback for unknown action
        except Exception as e:
            logging.error(f"NumPy model inference error: {e}")
            return self._get_random_direction()

    def _get_pytorch_nn_action(
        self,
        enemy, # Enemy instance with a PyTorch model
        player_x: float,
        player_y: float
    ) -> Tuple[float, float]:
        """Get action from the PyTorch-based neural network model."""
        dist = math.sqrt((player_x - enemy.pos["x"])**2 + (player_y - enemy.pos["y"])**2)
        # State for PyTorch model (as per original _get_nn_action)
        state_torch = torch.tensor(
            [[player_x, player_y, enemy.pos["x"], enemy.pos["y"], dist]], # Note: order might differ from NumPy
            dtype=torch.float32
        )
        try:
            with torch.no_grad():
                action_torch = enemy.model(state_torch)  # Expected shape: [1, 2] for dx, dy
            action_dx, action_dy = action_torch[0].tolist()
            if abs(action_dx) < 1e-6 and abs(action_dy) < 1e-6:
                return self._get_random_direction()
            return action_dx, action_dy
        except Exception as e:
            logging.error(f"PyTorch model inference error: {e}")
            return self._get_random_direction()


    def _normalize_vector(self, dx: float, dy: float) -> Tuple[float, float]:
        """
        Normalize a direction vector to unit length.

        Args:
            dx: X component of direction
            dy: Y component of direction

        Returns:
            Normalized (dx, dy) tuple
        """
        length = math.sqrt(dx**2 + dy**2)
        if length > 0:
            return dx / length, dy / length
        else:
            # Apply a random direction if vector is zero length
            return self._get_random_direction()

    def _get_random_direction(self) -> Tuple[float, float]:
        """
        Get a random unit direction vector.

        Returns:
            Random (dx, dy) direction
        """
        angle = random.uniform(0, 2 * math.pi)
        return math.cos(angle), math.sin(angle)

    def _update_position_history(self, position: Dict[str, float]) -> None:
        """
        Update the history of enemy positions to detect if stuck.

        Args:
            position: Current enemy position
        """
        # Add current position to history
        self.prev_positions.append({"x": position["x"], "y": position["y"]})

        # Limit history size
        if len(self.prev_positions) > self.max_positions:
            self.prev_positions.pop(0)

    def _is_enemy_stuck(self) -> bool:
        """
        Check if the enemy appears to be stuck based on position history.

        Returns:
            True if enemy seems stuck, False otherwise
        """
        if len(self.prev_positions) < self.max_positions:
            return False

        # Calculate variance in positions
        x_positions = [pos["x"] for pos in self.prev_positions]
        y_positions = [pos["y"] for pos in self.prev_positions]

        x_var = max(x_positions) - min(x_positions)
        y_var = max(y_positions) - min(y_positions)

        # If the enemy hasn't moved much, it might be stuck
        if x_var < 10 and y_var < 10:
            self.stuck_counter += 1
            if self.stuck_counter > 3:  # Stuck for several frames
                return True
        else:
            self.stuck_counter = 0

        return False

    def _handle_stuck_enemy(
        self,
        player_x: float,
        player_y: float,
        enemy_pos: Dict[str, float]
    ) -> Tuple[float, float]:
        """
        Special behavior for when the enemy is detected as stuck.

        Args:
            player_x: Player's x position
            player_y: Player's y position
            enemy_pos: Enemy's current position

        Returns:
            Direction vector to move the enemy
        """
        logging.debug(f"Enemy detected as stuck at {enemy_pos}, applying escape behavior")

        # Option 1: Move away from player (reversed chase)
        dx = enemy_pos["x"] - player_x
        dy = enemy_pos["y"] - player_y

        # Option 2: Sometimes use completely random movement to break patterns
        if random.random() < 0.3:
            return self._get_random_direction()

        # Normalize the escape vector
        return self._normalize_vector(dx, dy)


        # Initialize the controller as a singleton
enemy_controller = EnemyAIController()
# Attempt to load the numpy model when the controller is initialized
# This path should ideally come from a config file or be passed in.
enemy_controller.load_numpy_model("models/numpy_enemy_model.npz") 


# This is the function provided in your snippet for gameplay integration.
# It's slightly different from the class method, so I'm adding it here.
# It assumes `model` is an instance of `NumpyEnemyModel`.
def update_enemy_movement_numpy(enemy, player, model: Optional[NumpyEnemyModel]):
    """
    Updates enemy movement based on the NumPy model prediction.
    This function is based on the snippet provided for direct integration.
    """
    if not model:
        # Fallback logic: if no model, enemy does nothing or uses simple rules.
        # For brevity, as in snippet, just return.
        # A more robust fallback would be to call a rule-based teacher.
        # logger.debug("NumPy model not available for enemy movement.")
        return

    # Create state vector for the NumPy model
    # [enemy_x, enemy_y, player_x, player_y, distance]
    state = np.array([
        enemy.pos["x"], enemy.pos["y"],
        player.pos["x"], player.pos["y"],
        math.sqrt((player.pos["x"] - enemy.pos["x"])**2 + (player.pos["y"] - enemy.pos["y"])**2)
    ], dtype=np.float32)

    action_id = model.predict(state)

    # Apply action based on ID
    # This mapping should match NumpyEnemyTeacher.ACTION_...
    if action_id == 0: # ACTION_LEFT
        enemy.pos["x"] -= enemy.speed # Example: direct position update
    elif action_id == 1: # ACTION_RIGHT
        enemy.pos["x"] += enemy.speed
    elif action_id == 2: # ACTION_UP
        enemy.pos["y"] -= enemy.speed # Screen coordinates: Y decreases upwards
    elif action_id == 3: # ACTION_DOWN
        enemy.pos["y"] += enemy.speed
    
    # Placeholder for actual enemy movement methods like enemy.move_left()
    # The above directly modifies pos for simplicity matching the snippet's style.
    # In a real game, you'd call methods on the enemy object that handle physics, speed, etc.
    # e.g., if action_id == 0: enemy.move_left(enemy.speed)

    # enemy.wrap_position() # Assuming this method exists on enemy to handle screen wrap


# The original update_enemy_movement function that delegates to the class instance
def update_enemy_movement(
    enemy, # EnemyPlay instance
    player_x: float,
    player_y: float,
    player_speed: float,
    current_time: int,
    numpy_model_instance: Optional[NumpyEnemyModel] = None # Allow passing a model
) -> None:
    """
    Main entry point for updating enemy movement.
    Delegates to the EnemyAIController instance.
    Can use either PyTorch model from enemy or a passed/loaded NumPy model.
    """
    # If a specific numpy_model_instance is passed, use it. Otherwise, the controller
    # will use its own self.numpy_model if loaded.
    active_numpy_model = numpy_model_instance if numpy_model_instance else enemy_controller.numpy_model

    enemy_controller.update_enemy_movement(
        enemy, player_x, player_y, player_speed, current_time,
        numpy_model_instance=active_numpy_model # Pass it to the class method
    )
