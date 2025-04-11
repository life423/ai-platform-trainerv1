"""
PowerUpManager for AI Platform Trainer.

This module defines the PowerUpManager class that handles spawning,
updating, and checking collisions with power-ups.
"""
import logging
import random
import pygame

from ai_platform_trainer.entities.components.power_up import (
    PowerUp, PowerUpType, SpeedBoost, Shield, RapidFire
)
from ai_platform_trainer.gameplay.spawn_utils import find_valid_spawn_position
from ai_platform_trainer.gameplay.config import config


class PowerUpManager:
    """
    Manages the spawning, updating, and collision detection for power-ups.
    
    Attributes:
        powerups (list): List of active power-ups in the game
        active_powerups (list): List of collected power-ups that are in effect
        spawn_interval (int): Milliseconds between power-up spawns
        last_spawn_time (int): Time of the last power-up spawn
        max_powerups (int): Maximum number of power-ups allowed at once
        difficulty_manager (DifficultyManager): Optional reference to control spawn rates
    """
    
    def __init__(self, difficulty_manager=None):
        """
        Initialize the PowerUpManager.
        
        Args:
            difficulty_manager: Optional difficulty manager for scaling spawn rates
        """
        self.powerups = []  # Uncollected power-ups
        self.active_powerups = []  # Collected, active power-ups
        self.spawn_interval = 15000  # 15 seconds between spawns by default
        self.last_spawn_time = 0
        self.max_powerups = 3  # Maximum 3 power-ups on screen at once
        self.difficulty_manager = difficulty_manager
        
        # Size of power-ups
        self.powerup_size = 30
        
        logging.info("PowerUpManager initialized")
    
    def spawn_powerup(self, screen_width, screen_height, avoid_positions=None):
        """
        Spawn a new power-up at a random position.
        
        Args:
            screen_width: Width of the game screen
            screen_height: Height of the game screen
            avoid_positions: List of (x,y) positions to avoid when spawning
            
        Returns:
            PowerUp: The spawned power-up, or None if spawn failed
        """
        # Don't spawn if maximum reached
        if len(self.powerups) >= self.max_powerups:
            return None
            
        # Find a valid position for the power-up
        new_pos = find_valid_spawn_position(
            screen_width=screen_width,
            screen_height=screen_height,
            entity_size=self.powerup_size,
            margin=config.WALL_MARGIN,
            min_dist=config.MIN_DISTANCE,
            other_pos=avoid_positions[0] if avoid_positions else None,
        )
        
        # Choose a random power-up type
        powerup_type = random.choice(list(PowerUpType))
        
        # Create appropriate power-up based on type
        if powerup_type == PowerUpType.SPEED:
            powerup = SpeedBoost(new_pos[0], new_pos[1], self.powerup_size)
        elif powerup_type == PowerUpType.SHIELD:
            powerup = Shield(new_pos[0], new_pos[1], self.powerup_size)
        elif powerup_type == PowerUpType.RAPID_FIRE:
            powerup = RapidFire(new_pos[0], new_pos[1], self.powerup_size)
        else:
            # Fallback
            powerup = PowerUp(new_pos[0], new_pos[1], self.powerup_size, powerup_type)
        
        # Add to list of active power-ups
        self.powerups.append(powerup)
        logging.info(f"Spawned {powerup_type.name} power-up at {new_pos}")
        
        return powerup
    
    def update(self, current_time, screen_width, screen_height, player_pos):
        """
        Update all power-ups and spawn new ones as needed.
        
        Args:
            current_time: Current game time in milliseconds
            screen_width: Width of the game screen
            screen_height: Height of the game screen
            player_pos: Player position to avoid when spawning power-ups
            
        Returns:
            None
        """
        # Check if we should spawn a new power-up based on time
        if (current_time - self.last_spawn_time >= self.spawn_interval and 
                len(self.powerups) < self.max_powerups):
            self.spawn_powerup(
                screen_width, 
                screen_height, 
                avoid_positions=[(player_pos["x"], player_pos["y"])]
            )
            self.last_spawn_time = current_time
        
        # Update active (collected) power-ups and remove expired ones
        expired_powerups = []
        for powerup in self.active_powerups:
            if not powerup.update(current_time):
                # Power-up has expired, add to removal list
                expired_powerups.append(powerup)
        
        # Remove expired power-ups and revert their effects
        for powerup in expired_powerups:
            if hasattr(self, 'player') and self.player:
                powerup.revert(self.player)
            self.active_powerups.remove(powerup)
            logging.info(f"Removed expired {powerup.type.name} power-up")
    
    def check_collisions(self, player, current_time):
        """
        Check for collisions between the player and power-ups.
        
        Args:
            player: The player entity
            current_time: Current game time in milliseconds
            
        Returns:
            List of collected power-ups
        """
        if not player:
            return []
            
        # Store a reference to player for reverting effects later
        self.player = player
        
        # Create player collision rect
        player_rect = pygame.Rect(
            player.position["x"],
            player.position["y"],
            player.size,
            player.size,
        )
        
        # Check collision with each power-up
        collected = []
        for powerup in list(self.powerups):  # Use list() to avoid modification during iteration
            if not powerup.visible:
                continue
                
            if player_rect.colliderect(powerup.get_rect()):
                # Collision detected, collect the power-up
                powerup.collect(current_time)
                self.powerups.remove(powerup)
                
                # Apply the power-up effect
                powerup.apply(player)
                
                # Add to active power-ups
                self.active_powerups.append(powerup)
                
                # Add to collection list for return
                collected.append(powerup)
                
                logging.info(f"Player collected {powerup.type.name} power-up")
        
        return collected
    
    def render(self, screen):
        """
        Render all visible power-ups.
        
        Args:
            screen: The pygame surface to render on
            
        Returns:
            None
        """
        # Render uncollected power-ups
        for powerup in self.powerups:
            if not powerup.visible:
                continue
                
            # If power-up has a sprite, draw it
            if powerup.sprite:
                screen.blit(powerup.sprite, (powerup.pos["x"], powerup.pos["y"]))
            else:
                # Draw a colored rectangle as fallback
                if powerup.type == PowerUpType.SPEED:
                    color = (0, 255, 0)  # Green for speed
                elif powerup.type == PowerUpType.SHIELD:
                    color = (0, 0, 255)  # Blue for shield
                elif powerup.type == PowerUpType.RAPID_FIRE:
                    color = (255, 0, 0)  # Red for rapid fire
                else:
                    color = (255, 255, 0)  # Yellow for unknown
                
                pygame.draw.rect(
                    screen,
                    color,
                    (powerup.pos["x"], powerup.pos["y"], powerup.size, powerup.size)
                )
        
        # Render shield effect if active
        for powerup in self.active_powerups:
            is_shield = powerup.type == PowerUpType.SHIELD
            has_shield_active = hasattr(powerup, 'shield_active')
            
            if is_shield and has_shield_active and powerup.shield_active:
                # Only render the shield for active shield power-ups
                self._render_shield_effect(screen, powerup)
    
    def _render_shield_effect(self, screen, shield_powerup):
        """
        Render a shield effect around the player.
        
        Args:
            screen: The pygame surface to render on
            shield_powerup: The shield power-up to render
            
        Returns:
            None
        """
        if not hasattr(self, 'player') or not self.player:
            return
            
        current_time = pygame.time.get_ticks()
        shield_alpha = shield_powerup.get_shield_alpha(current_time)
        
        if shield_alpha <= 0:
            return
            
        # Create a slightly larger circle around the player
        center_x = self.player.position["x"] + self.player.size // 2
        center_y = self.player.position["y"] + self.player.size // 2
        shield_radius = int(self.player.size * 0.75)  # Shield is 1.5x player size
        
        # Create a surface for the shield with alpha channel
        shield_surface = pygame.Surface((shield_radius * 2, shield_radius * 2), pygame.SRCALPHA)
        
        # Draw the shield circle with appropriate alpha
        pygame.draw.circle(
            shield_surface,
            (0, 100, 255, shield_alpha),  # Blue with variable alpha
            (shield_radius, shield_radius),
            shield_radius,
            width=3  # Just a 3px border
        )
        
        # Draw the shield to the screen centered on the player
        screen.blit(
            shield_surface,
            (center_x - shield_radius, center_y - shield_radius)
        )
