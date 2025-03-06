"""
Powerup Manager to handle spawning and managing power-ups in the game.

This module provides functionality to:
1. Spawn power-ups at random locations based on configurable chances
2. Manage power-up lifecycles (creation, collection, expiration)
3. Apply power-up effects to the player
"""

import logging
import random
import pygame
from typing import List, Any

from ai_platform_trainer.gameplay.config import config as game_config
from ai_platform_trainer.gameplay.spawn_utils import find_valid_spawn_position


class Powerup:
    """
    Represents a power-up item that can be collected by the player.
    """
    
    def __init__(self, x: int, y: int, type_name: str, birth_time: int):
        """
        Initialize a power-up.
        
        Args:
            x: X-coordinate position
            y: Y-coordinate position
            type_name: Type of power-up ("shield", "rapid_fire", "speed_boost")
            birth_time: Creation time in milliseconds
        """
        self.pos = {"x": x, "y": y}
        self.type = type_name
        self.birth_time = birth_time
        self.lifespan = 10000  # 10 seconds until it disappears
        self.active = True
        self.size = 15
        self.flash_interval = 100  # Flash every 100ms
        self.flash_state = False
        self.last_flash_time = birth_time
        
        # Different colors based on type
        self.colors = {
            "shield": (0, 100, 255),      # Blue
            "rapid_fire": (255, 165, 0),  # Orange
            "speed_boost": (0, 255, 0)    # Green
        }
        
        self.color = self.colors.get(type_name, (255, 255, 255))  # Default white
        
    def update(self, current_time: int) -> None:
        """
        Update power-up state.
        
        Args:
            current_time: Current game time in milliseconds
        """
        # Check if powerup should expire
        if current_time > self.birth_time + self.lifespan:
            self.active = False
            return
            
        # Make power-up flash faster as it approaches expiration
        remaining_time = (self.birth_time + self.lifespan) - current_time
        if remaining_time < 3000:  # Last 3 seconds
            self.flash_interval = 50  # Flash faster
            
        # Update flash state
        if current_time - self.last_flash_time >= self.flash_interval:
            self.flash_state = not self.flash_state
            self.last_flash_time = current_time
    
    def draw(self, screen: pygame.Surface) -> None:
        """
        Draw the power-up on the screen.
        
        Args:
            screen: Pygame surface to draw on
        """
        if not self.active:
            return
            
        # Skip drawing if in "off" flash state and close to expiration
        remaining_time = ((self.birth_time + self.lifespan) - 
                          pygame.time.get_ticks())
        if remaining_time < 3000 and not self.flash_state:
            return
            
        # Draw power-up as a filled circle with outline
        pygame.draw.circle(
            screen,
            self.color,
            (int(self.pos["x"]), int(self.pos["y"])),
            self.size
        )
        
        # Draw white outline
        pygame.draw.circle(
            screen,
            (255, 255, 255),
            (int(self.pos["x"]), int(self.pos["y"])),
            self.size,
            2  # Line width
        )
        
        # Draw a symbol inside based on type
        inner_color = (255, 255, 255)  # White
        if self.type == "shield":
            # Draw shield symbol (circle)
            pygame.draw.circle(
                screen,
                inner_color,
                (int(self.pos["x"]), int(self.pos["y"])),
                self.size // 2,
                2
            )
        elif self.type == "rapid_fire":
            # Draw rapid fire symbol (cross/X)
            size = self.size // 2
            pygame.draw.line(
                screen, 
                inner_color,
                (self.pos["x"] - size, self.pos["y"] - size),
                (self.pos["x"] + size, self.pos["y"] + size),
                2
            )
            pygame.draw.line(
                screen, 
                inner_color,
                (self.pos["x"] - size, self.pos["y"] + size),
                (self.pos["x"] + size, self.pos["y"] - size),
                2
            )
        elif self.type == "speed_boost":
            # Draw speed boost symbol (arrow)
            size = self.size // 2
            pygame.draw.line(
                screen, 
                inner_color,
                (self.pos["x"] - size, self.pos["y"]),
                (self.pos["x"] + size, self.pos["y"]),
                2
            )
            pygame.draw.line(
                screen, 
                inner_color,
                (self.pos["x"] + size - 4, self.pos["y"] - 4),
                (self.pos["x"] + size, self.pos["y"]),
                2
            )
            pygame.draw.line(
                screen, 
                inner_color,
                (self.pos["x"] + size - 4, self.pos["y"] + 4),
                (self.pos["x"] + size, self.pos["y"]),
                2
            )


class PowerupManager:
    """
    Manages power-ups in the game, including spawning, expiration, and collection.
    """
    
    def __init__(self, screen_width: int, screen_height: int):
        """
        Initialize the power-up manager.
        
        Args:
            screen_width: Width of the game screen
            screen_height: Height of the game screen
        """
        self.screen_width = screen_width
        self.screen_height = screen_height
        self.powerups: List[Powerup] = []
        
        self.spawn_chance = game_config.POWERUP_SPAWN_CHANCE
        self.next_natural_spawn = 0
        self.natural_spawn_interval = 15000  # 15 seconds between possible natural spawns
    
    def update(self, current_time: int, player=None) -> None:
        """
        Update all power-ups and check for collisions with player.
        
        Args:
            current_time: Current game time in milliseconds
            player: Player object to check for collisions
        """
        # Check for natural spawn
        if current_time >= self.next_natural_spawn:
            self.next_natural_spawn = current_time + self.natural_spawn_interval
            if random.random() < self.spawn_chance:
                self.spawn_random_powerup(current_time)
        
        # Update all powerups
        for powerup in self.powerups[:]:
            powerup.update(current_time)
            
            # Remove expired powerups
            if not powerup.active:
                self.powerups.remove(powerup)
                continue
                
            # Check for collision with player
            if player and self._check_collision(powerup, player):
                # Add powerup to player
                player.add_powerup(powerup.type, current_time)
                
                # Remove collected powerup
                powerup.active = False
                self.powerups.remove(powerup)
                logging.info(f"Player collected {powerup.type} power-up")
    
    def draw(self, screen: pygame.Surface) -> None:
        """
        Draw all active power-ups.
        
        Args:
            screen: Pygame surface to draw on
        """
        for powerup in self.powerups:
            powerup.draw(screen)
    
    def spawn_random_powerup(self, current_time: int) -> None:
        """
        Spawn a random power-up.
        
        Args:
            current_time: Current game time in milliseconds
        """
        # Select random powerup type
        powerup_type = random.choice(game_config.POWERUP_TYPES)
        
        # Find valid spawn position
        spawn_pos = find_valid_spawn_position(
            self.screen_width,
            self.screen_height,
            30,  # Size with some margin
            margin=game_config.WALL_MARGIN,
            min_dist=0,
            other_pos=None
        )
        
        # Create new powerup
        powerup = Powerup(
            spawn_pos[0], spawn_pos[1],
            powerup_type, current_time
        )
        
        # Add to list
        self.powerups.append(powerup)
        logging.info(f"Spawned {powerup_type} power-up at {spawn_pos}")
    
    def spawn_powerup_at_position(self, x: int, y: int, 
                                  current_time: int) -> None:
        """
        Spawn a random power-up at the specified position.
        
        Args:
            x: X-coordinate for the power-up
            y: Y-coordinate for the power-up
            current_time: Current game time in milliseconds
        """
        # Only spawn if random chance succeeds
        if random.random() > self.spawn_chance:
            return
            
        # Select random powerup type
        powerup_type = random.choice(game_config.POWERUP_TYPES)
        
        # Create new powerup
        powerup = Powerup(x, y, powerup_type, current_time)
        
        # Add to list
        self.powerups.append(powerup)
        logging.info(f"Spawned {powerup_type} power-up at ({x}, {y})")
    
    def clear_all(self) -> None:
        """
        Clear all power-ups.
        """
        self.powerups.clear()
        self.next_natural_spawn = 0
        logging.info("All power-ups cleared")
    
    def _check_collision(self, powerup: Powerup, player: Any) -> bool:
        """
        Check if a power-up collides with the player.
        
        Args:
            powerup: Power-up object
            player: Player object
            
        Returns:
            True if collision detected
        """
        # Create rectangles for collision detection
        powerup_rect = pygame.Rect(
            powerup.pos["x"] - powerup.size,
            powerup.pos["y"] - powerup.size,
            powerup.size * 2,
            powerup.size * 2
        )
        
        player_rect = pygame.Rect(
            player.position["x"],
            player.position["y"],
            player.size,
            player.size
        )
        
        return powerup_rect.colliderect(player_rect)
