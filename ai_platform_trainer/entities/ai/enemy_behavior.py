import math
from typing import Tuple
import torch


class EnemyBehavior:
    """
    Base behavior interface that defines how an enemy entity should move.
    Different implementations can provide different movement strategies.
    """
    def decide_movement(self, enemy, player_pos, missiles) -> Tuple[float, float]:
        """
        Returns (dx, dy) normalized direction vector for the enemy to move.
        
        Args:
            enemy: The enemy entity (containing enemy.pos, etc.)
            player_pos: Dict with 'x' and 'y' keys for player position
            missiles: List of missile objects
            
        Returns:
            Tuple[float, float]: Normalized direction vector (dx, dy)
        """
        raise NotImplementedError("Subclasses must implement decide_movement")


class ChaseBehavior(EnemyBehavior):
    """
    Behavior that makes the enemy chase the player.
    This can use either ML model or direct calculation.
    """
    def __init__(self, use_model: bool = True):
        self.use_model = use_model
    
    def decide_movement(self, enemy, player_pos, missiles) -> Tuple[float, float]:
        if self.use_model and enemy.model is not None:
            return self._model_based_movement(enemy, player_pos)
        else:
            return self._direct_chase_movement(enemy, player_pos)
    
    def _model_based_movement(self, enemy, player_pos) -> Tuple[float, float]:
        """Use the neural network model to determine movement direction"""
        dist = math.sqrt(
            (player_pos["x"] - enemy.pos["x"]) ** 2 +
            (player_pos["y"] - enemy.pos["y"]) ** 2
        )
        state = torch.tensor(
            [
                [player_pos["x"], player_pos["y"], enemy.pos["x"], enemy.pos["y"], dist]
            ],
            dtype=torch.float32
        )
        
        with torch.no_grad():
            action = enemy.model(state)
        
        action_dx, action_dy = action[0].tolist()
        
        # Normalize
        action_len = math.sqrt(action_dx**2 + action_dy**2)
        if action_len > 0:
            action_dx /= action_len
            action_dy /= action_len
        else:
            action_dx, action_dy = 0.0, 0.0
            
        return action_dx, action_dy
    
    def _direct_chase_movement(self, enemy, player_pos) -> Tuple[float, float]:
        """Simple direct chase algorithm (for use when no model is available)"""
        dx = player_pos["x"] - enemy.pos["x"]
        dy = player_pos["y"] - enemy.pos["y"]
        
        # Normalize
        dist = math.sqrt(dx**2 + dy**2)
        if dist > 0:
            dx /= dist
            dy /= dist
        else:
            dx, dy = 0.0, 0.0
            
        return dx, dy


class EvadeBehavior(EnemyBehavior):
    """
    Behavior that makes the enemy evade missiles.
    """
    def __init__(self, detection_radius: float = 200.0):
        self.detection_radius = detection_radius
    
    def decide_movement(self, enemy, player_pos, missiles) -> Tuple[float, float]:
        if not missiles:
            return 0.0, 0.0  # No missiles to evade
        
        # Find the closest missile within detection radius
        closest_missile = None
        closest_dist = float('inf')
        
        for missile in missiles:
            dx = missile.pos["x"] - enemy.pos["x"]
            dy = missile.pos["y"] - enemy.pos["y"]
            dist = math.sqrt(dx**2 + dy**2)
            
            # Consider direction of missile (only evade if it's coming toward us)
            missile_direction = math.atan2(missile.vy, missile.vx)
            missile_to_enemy = math.atan2(-dy, -dx)
            angle_diff = abs(
                (missile_direction - missile_to_enemy + math.pi) % (2 * math.pi) - math.pi
            )
            
            # If missile is heading toward the enemy (within 90 degrees)
            if (dist < self.detection_radius and angle_diff < math.pi/2 and 
                    dist < closest_dist):
                closest_missile = missile
                closest_dist = dist
        
        if closest_missile is None:
            return 0.0, 0.0  # No threats detected
        
        # Calculate evade direction (perpendicular to missile trajectory)
        missile_dx = closest_missile.vx
        missile_dy = closest_missile.vy
        
        # Normalize missile direction
        missile_len = math.sqrt(missile_dx**2 + missile_dy**2)
        if missile_len > 0:
            missile_dx /= missile_len
            missile_dy /= missile_len
        
        # Get perpendicular vector (two options)
        perp1_dx, perp1_dy = -missile_dy, missile_dx
        perp2_dx, perp2_dy = missile_dy, -missile_dx
        
        # Determine which perpendicular direction is better (away from the wall, toward more open space)
        # Simple approach: choose the one that keeps enemy more centered in the screen
        center_x = enemy.screen_width / 2
        center_y = enemy.screen_height / 2
        
        # Project position if moving in direction 1
        pos1_x = enemy.pos["x"] + perp1_dx * 100
        pos1_y = enemy.pos["y"] + perp1_dy * 100
        dist1 = math.sqrt((pos1_x - center_x)**2 + (pos1_y - center_y)**2)
        
        # Project position if moving in direction 2
        pos2_x = enemy.pos["x"] + perp2_dx * 100
        pos2_y = enemy.pos["y"] + perp2_dy * 100
        dist2 = math.sqrt((pos2_x - center_x)**2 + (pos2_y - center_y)**2)
        
        # Choose the direction that keeps us more centered
        if dist1 < dist2:
            return perp1_dx, perp1_dy
        else:
            return perp2_dx, perp2_dy


class CompositeBehavior(EnemyBehavior):
    """
    Combines multiple behaviors with weights that can be dynamically adjusted.
    """
    def __init__(self, behaviors=None, weights=None):
        """
        Initialize with list of behaviors and their weights.
        
        Args:
            behaviors: List of EnemyBehavior objects
            weights: List of initial weights for each behavior
        """
        self.behaviors = behaviors or []
        self.weights = weights or [1.0] * len(self.behaviors)
        
        if len(self.behaviors) != len(self.weights):
            raise ValueError("Number of behaviors must match number of weights")
    
    def decide_movement(self, enemy, player_pos, missiles) -> Tuple[float, float]:
        if not self.behaviors:
            return 0.0, 0.0
        
        # Dynamically adjust weights based on missile proximity
        self._adjust_weights(enemy, missiles)
        
        # Get movement vectors from each behavior
        dx_total, dy_total = 0.0, 0.0
        
        for i, behavior in enumerate(self.behaviors):
            dx, dy = behavior.decide_movement(enemy, player_pos, missiles)
            dx_total += dx * self.weights[i]
            dy_total += dy * self.weights[i]
        
        # Normalize the final vector
        total_len = math.sqrt(dx_total**2 + dy_total**2)
        if total_len > 0:
            dx_total /= total_len
            dy_total /= total_len
        
        return dx_total, dy_total
    
    def _adjust_weights(self, enemy, missiles):
        """
        Adjust weights based on game state, such as missile proximity.
        """
        # Only adjust if we have both chase and evade behaviors (indices 0 and 1)
        if len(self.behaviors) < 2:
            return
        
        # Check if any missiles are close
        missile_danger = 0.0
        if missiles:
            for missile in missiles:
                dx = missile.pos["x"] - enemy.pos["x"]
                dy = missile.pos["y"] - enemy.pos["y"]
                dist = math.sqrt(dx**2 + dy**2)
                
                # Calculate missile-to-enemy direction to see if missile is approaching
                missile_direction = math.atan2(missile.vy, missile.vx)
                missile_to_enemy = math.atan2(-dy, -dx)
                angle_diff = abs(
                    (missile_direction - missile_to_enemy + math.pi) % (2 * math.pi) - math.pi
                )
                
                # If missile is heading toward the enemy (within 90 degrees)
                if angle_diff < math.pi/2:
                    # Higher danger for closer missiles
                    danger_factor = 1.0 - min(1.0, dist / 250.0)
                    missile_danger = max(missile_danger, danger_factor)
        
        # Adjust weights based on danger
        # Assume index 0 is chase, index 1 is evade
        chase_weight = 1.0 - missile_danger
        evade_weight = missile_danger
        
        # Ensure a minimum weight for chase behavior
        chase_weight = max(0.1, chase_weight)
        
        self.weights[0] = chase_weight  # Chase weight
        self.weights[1] = evade_weight  # Evade weight
