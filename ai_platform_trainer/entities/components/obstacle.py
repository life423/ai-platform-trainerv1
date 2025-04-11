"""
Obstacle entity for AI Platform Trainer.

This module defines the Obstacle class that represents barriers or hazards
in the game environment.
"""
import logging
import os
import random

import pygame


class Obstacle:
    """
    A static obstacle in the game environment that blocks or impedes movement.
    
    Attributes:
        pos (dict): The x, y coordinates as a dictionary
        size (int): The width/height of the obstacle (assumed square)
        visible (bool): Whether the obstacle is currently visible
        destructible (bool): Whether the obstacle can be destroyed by missiles
        color (tuple): RGB color tuple used as a fallback if no sprite is available
        sprite (pygame.Surface): The loaded sprite for this obstacle
    """
    
    def __init__(self, x: int, y: int, size: int, destructible: bool = False):
        """
        Initialize an obstacle at the specified position.
        
        Args:
            x: X-coordinate
            y: Y-coordinate
            size: Width/height of the obstacle (assumed square)
            destructible: Whether missiles can destroy this obstacle
        """
        self.pos = {"x": x, "y": y}
        self.size = size
        self.visible = True
        self.destructible = destructible
        
        # Visual properties
        self.color = (100, 100, 100)  # Gray fallback color
        self._load_sprite()
        
        # Health for destructible obstacles
        self.max_health = 2 if destructible else 1
        self.health = self.max_health
        
        logging.debug(f"Obstacle created at ({x}, {y}), size={size}, destructible={destructible}")
    
    def _load_sprite(self) -> None:
        """Load the obstacle sprite from assets."""
        try:
            # Try to load either a rock or wall sprite
            sprite_types = ["rock", "wall", "obstacle"]
            sprite_type = random.choice(sprite_types)
            
            sprite_path = os.path.join("assets", "sprites", "obstacles", f"{sprite_type}.png")
            if os.path.exists(sprite_path):
                self.sprite = pygame.image.load(sprite_path)
                self.sprite = pygame.transform.scale(self.sprite, (self.size, self.size))
                logging.debug(f"Loaded {sprite_type} sprite for obstacle")
            else:
                logging.warning(f"Obstacle sprite not found at {sprite_path}, using fallback")
                self.sprite = None
        except Exception as e:
            logging.error(f"Error loading obstacle sprite: {e}")
            self.sprite = None
    
    def get_rect(self) -> pygame.Rect:
        """Get the obstacle's collision rectangle."""
        return pygame.Rect(
            self.pos["x"], 
            self.pos["y"], 
            self.size, 
            self.size
        )
    
    def take_damage(self) -> bool:
        """
        Reduce the obstacle's health when hit by a missile.
        
        Returns:
            bool: True if the obstacle was destroyed, False otherwise
        """
        if not self.destructible:
            return False
            
        self.health -= 1
        
        if self.health <= 0:
            self.visible = False
            logging.info("Destructible obstacle destroyed")
            return True
            
        # Visual feedback for damage
        if self.sprite:
            # Darken the sprite to show damage
            dark_sprite = self.sprite.copy()
            dark_surface = pygame.Surface(dark_sprite.get_size(), pygame.SRCALPHA)
            dark_surface.fill((0, 0, 0, 100))  # Semi-transparent black
            dark_sprite.blit(dark_surface, (0, 0))
            self.sprite = dark_sprite
            
        return False
