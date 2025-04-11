"""
PowerUp entities for AI Platform Trainer.

This module defines the PowerUp class and its variants that provide temporary
special abilities to the player.
"""
import logging
import os
from enum import Enum, auto
import pygame


class PowerUpType(Enum):
    """Types of power-ups available in the game."""
    SPEED = auto()
    SHIELD = auto()
    RAPID_FIRE = auto()


class PowerUp:
    """
    Base class for all power-ups that can be collected by the player.
    
    Attributes:
        pos (dict): The x, y coordinates as a dictionary
        size (int): The width/height of the power-up
        type (PowerUpType): The specific type of power-up
        visible (bool): Whether the power-up is currently visible/active
        duration (int): How long the power-up effect lasts in milliseconds
        effect_value (float): The magnitude of the power-up effect
        sprite (pygame.Surface): The loaded sprite for this power-up
        collect_time (int): When the power-up was collected (0 if not collected)
    """
    
    def __init__(self, x: int, y: int, size: int, power_type: PowerUpType):
        """
        Initialize a power-up at the specified position.
        
        Args:
            x: X-coordinate
            y: Y-coordinate
            size: Width/height of the power-up (assumed square)
            power_type: The type of power-up
        """
        self.pos = {"x": x, "y": y}
        self.size = size
        self.type = power_type
        self.visible = True
        self.duration = 5000  # Default 5 seconds duration
        self.effect_value = 1.5  # Default 1.5x effect
        self.sprite = None
        self.collect_time = 0  # Not collected yet
        self.is_active = False
        
        # Load appropriate sprite based on type
        self._load_sprite()
        
        logging.debug(f"PowerUp created: type={power_type.name} at ({x}, {y}), size={size}")
    
    def _load_sprite(self) -> None:
        """Load the power-up sprite based on its type."""
        try:
            # Determine sprite path based on type
            sprite_name = self.type.name.lower()
            sprite_path = os.path.join("assets", "sprites", "power_ups", f"{sprite_name}.png")
            
            # Fallback to a colored rectangle if sprite doesn't exist
            if os.path.exists(sprite_path):
                self.sprite = pygame.image.load(sprite_path)
                self.sprite = pygame.transform.scale(self.sprite, (self.size, self.size))
                logging.debug(f"Loaded {sprite_name} sprite for power-up")
            else:
                logging.warning(f"Power-up sprite not found at {sprite_path}, using fallback")
                self.sprite = None
        except Exception as e:
            logging.error(f"Error loading power-up sprite: {e}")
            self.sprite = None
    
    def get_rect(self) -> pygame.Rect:
        """Get the power-up's collision rectangle."""
        return pygame.Rect(
            self.pos["x"], 
            self.pos["y"], 
            self.size, 
            self.size
        )
    
    def collect(self, current_time: int) -> None:
        """Mark the power-up as collected at the current time."""
        self.visible = False
        self.collect_time = current_time
        self.is_active = True
        logging.info(f"PowerUp collected: {self.type.name} at time {current_time}")
    
    def apply(self, player) -> None:
        """
        Apply the power-up effect to the player.
        This should be overridden by specific power-up classes.
        """
        logging.debug(f"Applying {self.type.name} power-up to player")
    
    def update(self, current_time: int) -> bool:
        """
        Update the power-up state based on current time.
        
        Args:
            current_time: Current game time in milliseconds
            
        Returns:
            bool: True if the power-up is still active, False if expired
        """
        if not self.is_active:
            return False
            
        # Check if power-up effect has expired
        if current_time - self.collect_time >= self.duration:
            self.is_active = False
            logging.info(f"PowerUp expired: {self.type.name}")
            return False
            
        return True
    
    def revert(self, player) -> None:
        """
        Remove the power-up effect from the player.
        This should be overridden by specific power-up classes.
        """
        logging.debug(f"Reverting {self.type.name} power-up effect")


class SpeedBoost(PowerUp):
    """
    Speed boost power-up that increases player movement speed.
    """
    
    def __init__(self, x: int, y: int, size: int):
        super().__init__(x, y, size, PowerUpType.SPEED)
        self.duration = 8000  # 8 seconds
        self.effect_value = 1.8  # 1.8x speed
        self.original_step = None
    
    def apply(self, player) -> None:
        """Increase player movement speed."""
        if not hasattr(player, 'step'):
            logging.warning("Player doesn't have 'step' attribute, can't apply speed boost")
            return
            
        self.original_step = player.step
        player.step = player.step * self.effect_value
        logging.info(f"Applied speed boost: {self.original_step} -> {player.step}")
    
    def revert(self, player) -> None:
        """Reset player movement speed to original value."""
        if self.original_step is not None and hasattr(player, 'step'):
            player.step = self.original_step
            logging.info(f"Reverted speed boost, speed back to {player.step}")


class Shield(PowerUp):
    """
    Shield power-up that provides temporary invulnerability.
    """
    
    def __init__(self, x: int, y: int, size: int):
        super().__init__(x, y, size, PowerUpType.SHIELD)
        self.duration = 6000  # 6 seconds
        self.shield_active = False
    
    def apply(self, player) -> None:
        """Make player invulnerable."""
        if not hasattr(player, 'invulnerable'):
            # Add invulnerability attribute if it doesn't exist
            player.invulnerable = True
        else:
            player.invulnerable = True
            
        self.shield_active = True
        logging.info("Applied shield power-up, player is now invulnerable")
    
    def revert(self, player) -> None:
        """Remove player invulnerability."""
        if hasattr(player, 'invulnerable'):
            player.invulnerable = False
            self.shield_active = False
            logging.info("Reverted shield power-up, player is no longer invulnerable")
    
    def get_shield_alpha(self, current_time: int) -> int:
        """Get shield visual effect opacity (pulsing effect)."""
        if not self.shield_active:
            return 0
            
        # Calculate pulse pattern (0-255) based on time
        time_factor = (current_time % 1000) / 1000.0  # 0.0 to 1.0 over 1 second
        pulse = int(155 + 100 * abs(time_factor - 0.5) * 2)  # Pulse between 155-255
        return pulse


class RapidFire(PowerUp):
    """
    Rapid fire power-up that increases missile firing rate.
    """
    
    def __init__(self, x: int, y: int, size: int):
        super().__init__(x, y, size, PowerUpType.RAPID_FIRE)
        self.duration = 7000  # 7 seconds
        self.original_cooldown = None
    
    def apply(self, player) -> None:
        """Decrease missile cooldown time."""
        if not hasattr(player, 'missile_cooldown'):
            logging.warning("Player doesn't have 'missile_cooldown' attribute")
            return
            
        self.original_cooldown = player.missile_cooldown
        player.missile_cooldown = int(player.missile_cooldown / self.effect_value)
        logging.info(
            f"Applied rapid fire: cooldown {self.original_cooldown} -> {player.missile_cooldown}"
        )
    
    def revert(self, player) -> None:
        """Reset missile cooldown to original value."""
        if self.original_cooldown is not None and hasattr(player, 'missile_cooldown'):
            player.missile_cooldown = self.original_cooldown
            logging.info(f"Reverted rapid fire, cooldown back to {player.missile_cooldown}")
