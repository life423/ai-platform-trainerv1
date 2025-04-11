"""
Enemy variants for AI Platform Trainer.

This module defines different types of enemies with varying behaviors, 
speeds, and health values.
"""
import logging
import math
import random

from ai_platform_trainer.entities.components.enemy_play import EnemyPlay


class FastEnemy(EnemyPlay):
    """
    A faster enemy that moves quicker but has less health.
    
    This enemy is harder to hit but easier to destroy once hit.
    """
    
    def __init__(self, screen_width: int, screen_height: int, model=None):
        """Initialize a fast enemy variant."""
        super().__init__(screen_width, screen_height, model)
        
        # Define specific attributes for this enemy type
        self.max_health = 1  # Less health than standard enemy
        self.health = self.max_health
        self.base_speed = 5.0  # Faster than standard enemy
        self.enemy_type = "fast"
        
        # Set a slightly smaller size for faster enemy
        self.size = int(self.size * 0.85)  # 15% smaller
        
        # Add zigzag movement pattern
        self.zigzag_counter = 0
        self.zigzag_amplitude = 20  # Pixels of zigzag movement
        self.zigzag_period = 20  # Frames for one complete zigzag
        
        logging.info(f"Created fast enemy, health={self.health}, speed={self.base_speed}")
    
    def update_movement(self, player_x: float, player_y: float, player_step: float, 
                        current_time: int) -> None:
        """
        Update the fast enemy movement with a zigzag pattern.
        
        Args:
            player_x: Player's x position
            player_y: Player's y position
            player_step: Player's movement speed
            current_time: Current game time in milliseconds
        """
        if not self.visible or self.fading_in:
            return  # Skip movement if not visible or currently fading in
            
        # Calculate basic direction to player as in parent class
        dx = player_x - self.pos["x"]
        dy = player_y - self.pos["y"]
        dist = math.sqrt(dx * dx + dy * dy)
        
        # Normalize direction
        if dist > 0:
            dx /= dist
            dy /= dist
        
        # Apply zigzag pattern perpendicular to movement direction
        if dist > 0:
            # Calculate perpendicular direction
            perp_dx = -dy
            perp_dy = dx
            
            # Calculate zigzag factor (between -1 and 1)
            self.zigzag_counter = (self.zigzag_counter + 1) % self.zigzag_period
            zigzag_factor = math.sin(2 * math.pi * self.zigzag_counter / self.zigzag_period)
            
            # Apply zigzag to movement direction
            dx += perp_dx * zigzag_factor * 0.5  # Scale down to not be too extreme
            dy += perp_dy * zigzag_factor * 0.5
            
            # Renormalize direction after adding zigzag
            new_dist = math.sqrt(dx * dx + dy * dy)
            if new_dist > 0:
                dx /= new_dist
                dy /= new_dist
        
        # Apply faster speed
        if hasattr(self, 'speed_multiplier'):
            speed = self.base_speed * self.speed_multiplier
        else:
            speed = self.base_speed
            
        # Use RL model for movement if available
        if hasattr(self, 'rl_model') and self.rl_model:
            # Get action from RL model
            action = self._get_rl_action(player_x, player_y)
            dx, dy = self._action_to_direction(action)
        
        # Move the enemy
        self.pos["x"] += dx * speed
        self.pos["y"] += dy * speed
        
        # Constrain to screen
        self._constrain_to_screen()


class TankEnemy(EnemyPlay):
    """
    A tank-like enemy that moves slower but has more health.
    
    This enemy is easier to hit but harder to destroy, taking multiple missile hits.
    """
    
    def __init__(self, screen_width: int, screen_height: int, model=None):
        """Initialize a tank enemy variant."""
        super().__init__(screen_width, screen_height, model)
        
        # Define specific attributes for this enemy type
        self.max_health = 3  # More health than standard enemy
        self.health = self.max_health
        self.base_speed = 1.5  # Slower than standard enemy
        self.enemy_type = "tank"
        
        # Set a slightly larger size for tank enemy
        self.size = int(self.size * 1.3)  # 30% larger
        
        # Charge attack parameters
        self.is_charging = False
        self.charge_cooldown = 5000  # ms between charges
        self.last_charge_time = 0
        self.charge_duration = 1000  # ms for charge duration
        self.charge_speed = 7.0  # Speed during charge
        self.charge_target_x = 0
        self.charge_target_y = 0
        
        # Current damage state (visual indicator)
        self.damage_state = 0  # 0 = full health, increases as health decreases
        
        logging.info(f"Created tank enemy, health={self.health}, speed={self.base_speed}")
    
    def update_movement(self, player_x: float, player_y: float, player_step: float,
                       current_time: int) -> None:
        """
        Update the tank enemy movement with occasional charging attacks.
        
        Args:
            player_x: Player's x position
            player_y: Player's y position
            player_step: Player's movement speed
            current_time: Current game time in milliseconds
        """
        if not self.visible or self.fading_in:
            return  # Skip movement if not visible or currently fading in
            
        # Check for charge attack
        if not self.is_charging:
            # Maybe start a charge
            if (current_time - self.last_charge_time >= self.charge_cooldown and
                random.random() < 0.02):  # 2% chance per frame when off cooldown
                self.is_charging = True
                self.charge_end_time = current_time + self.charge_duration
                # Set charge target as player's current position
                self.charge_target_x = player_x
                self.charge_target_y = player_y
                logging.debug("Tank enemy starting charge attack")
        
        # Handle charging state
        if self.is_charging:
            # Check if charge is done
            if current_time >= self.charge_end_time:
                self.is_charging = False
                self.last_charge_time = current_time
                logging.debug("Tank enemy charge attack finished")
            
            # Move towards the charge target
            dx = self.charge_target_x - self.pos["x"]
            dy = self.charge_target_y - self.pos["y"]
            dist = math.sqrt(dx * dx + dy * dy)
            
            # Normalize direction
            if dist > 0:
                dx /= dist
                dy /= dist
                
            # Move at charge speed
            self.pos["x"] += dx * self.charge_speed
            self.pos["y"] += dy * self.charge_speed
        else:
            # Normal movement when not charging
            # Calculate basic direction to player as in parent class
            dx = player_x - self.pos["x"]
            dy = player_y - self.pos["y"]
            dist = math.sqrt(dx * dx + dy * dy)
            
            # Normalize direction
            if dist > 0:
                dx /= dist
                dy /= dist
            
            # Apply slower base speed
            if hasattr(self, 'speed_multiplier'):
                speed = self.base_speed * self.speed_multiplier
            else:
                speed = self.base_speed
                
            # Use RL model for movement if available
            if hasattr(self, 'rl_model') and self.rl_model:
                # Get action from RL model
                action = self._get_rl_action(player_x, player_y)
                dx, dy = self._action_to_direction(action)
            
            # Move the enemy
            self.pos["x"] += dx * speed
            self.pos["y"] += dy * speed
        
        # Constrain to screen
        self._constrain_to_screen()
    
    def take_damage(self) -> bool:
        """
        Reduce health when hit by missile.
        
        Returns:
            bool: True if destroyed, False otherwise
        """
        self.health -= 1
        self.damage_state = self.max_health - self.health
        
        if self.health <= 0:
            self.visible = False
            return True
            
        logging.info(f"Tank enemy took damage, health={self.health}/{self.max_health}")
        return False
    
    def hide(self) -> None:
        """Override hide to reset health when hidden."""
        super().hide()
        self.health = self.max_health
        self.damage_state = 0


def create_enemy_by_type(enemy_type: str, screen_width: int, screen_height: int, 
                        model=None) -> EnemyPlay:
    """
    Factory function to create an enemy of the specified type.
    
    Args:
        enemy_type: Type of enemy to create ("standard", "fast", or "tank")
        screen_width: Width of game screen
        screen_height: Height of game screen
        model: Neural network model for enemy movement
        
    Returns:
        EnemyPlay: An instance of the appropriate enemy class
    """
    if enemy_type == "fast":
        return FastEnemy(screen_width, screen_height, model)
    elif enemy_type == "tank":
        return TankEnemy(screen_width, screen_height, model)
    else:  # default to standard
        return EnemyPlay(screen_width, screen_height, model)
