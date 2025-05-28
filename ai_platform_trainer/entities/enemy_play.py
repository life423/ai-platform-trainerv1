"""
Enemy entity for play mode.

This module defines the enemy entity used in play mode, which uses AI models
for movement decisions.
"""
import logging
import math

import pygame
import torch

from ai_platform_trainer.ai.models.enemy_movement_model import EnemyMovementModel


class EnemyPlay:
    """
    Enemy entity for play mode with AI-controlled movement.
    
    This class represents the enemy in play mode, controlled by a neural network.
    """
    
    def __init__(
        self, 
        screen_width: int, 
        screen_height: int, 
        model: EnemyMovementModel
    ) -> None:
        """
        Initialize the enemy entity.
        
        Args:
            screen_width: Width of the game screen
            screen_height: Height of the game screen
            model: Neural network model for enemy movement
        """
        self.screen_width = screen_width
        self.screen_height = screen_height
        self.size = 50
        self.color = (139, 0, 0)  # Dark red
        self.pos = {"x": screen_width // 2, "y": screen_height // 2}
        self.speed = 5.0
        self.model = model
        self.visible = True
        self.fading_in = False
        self.fade_alpha = 255
        self.fade_start_time = 0
        self.fade_duration = 1000  # 1 second fade-in

    def update_movement(
        self, 
        player_x: float, 
        player_y: float, 
        player_speed: float, 
        current_time: int
    ) -> None:
        """
        Update the enemy's position based on neural network predictions.
        
        Args:
            player_x: Player's x position
            player_y: Player's y position
            player_speed: Player's movement speed
            current_time: Current game time in milliseconds
        """
        if not self.visible:
            return

        # Use neural network model
        self._update_with_nn(player_x, player_y, player_speed)

        # Update fade-in effect if active
        if self.fading_in:
            self.update_fade_in(current_time)

    def _update_with_nn(
        self, 
        player_x: float, 
        player_y: float, 
        player_speed: float
    ) -> None:
        """
        Update enemy position using the neural network model.
        
        Args:
            player_x: Player's x position
            player_y: Player's y position
            player_speed: Player's movement speed
        """
        # Calculate distance to player
        dx = player_x - self.pos["x"]
        dy = player_y - self.pos["y"]
        distance = math.sqrt(dx * dx + dy * dy)
        
        # Normalize inputs for the model
        normalized_px = player_x / self.screen_width
        normalized_py = player_y / self.screen_height
        normalized_ex = self.pos["x"] / self.screen_width
        normalized_ey = self.pos["y"] / self.screen_height
        normalized_dist = distance / max(self.screen_width, self.screen_height)
        
        # Prepare input tensor for the model
        model_input = torch.tensor([
            normalized_px, normalized_py, 
            normalized_ex, normalized_ey, 
            normalized_dist
        ], dtype=torch.float32).unsqueeze(0)
        
        # Get model prediction
        with torch.no_grad():
            movement = self.model(model_input).squeeze(0)
            
        # Apply movement (scale from [-1,1] to actual pixels)
        move_x = movement[0].item() * self.speed
        move_y = movement[1].item() * self.speed
        
        # Update position
        self.pos["x"] += move_x
        self.pos["y"] += move_y
        
        # Wrap around screen edges
        self._wrap_position()

    def _wrap_position(self) -> None:
        """Wrap the enemy position around screen edges."""
        if self.pos["x"] < -self.size:
            self.pos["x"] = self.screen_width
        elif self.pos["x"] > self.screen_width:
            self.pos["x"] = -self.size
            
        if self.pos["y"] < -self.size:
            self.pos["y"] = self.screen_height
        elif self.pos["y"] > self.screen_height:
            self.pos["y"] = -self.size

    def set_position(self, x: float, y: float) -> None:
        """
        Set the enemy position.
        
        Args:
            x: New x position
            y: New y position
        """
        self.pos["x"] = x
        self.pos["y"] = y

    def hide(self) -> None:
        """Hide the enemy (e.g., after being hit)."""
        self.visible = False

    def show(self, current_time: int) -> None:
        """
        Show the enemy with a fade-in effect.
        
        Args:
            current_time: Current game time in milliseconds
        """
        self.visible = True
        self.fading_in = True
        self.fade_alpha = 0
        self.fade_start_time = current_time

    def update_fade_in(self, current_time: int) -> None:
        """
        Update the fade-in effect.
        
        Args:
            current_time: Current game time in milliseconds
        """
        if not self.fading_in:
            return
            
        elapsed = current_time - self.fade_start_time
        progress = min(1.0, elapsed / self.fade_duration)
        
        self.fade_alpha = int(255 * progress)
        
        if progress >= 1.0:
            self.fading_in = False
            self.fade_alpha = 255

    def draw(self, screen: pygame.Surface) -> None:
        """
        Draw the enemy on the screen.
        
        Args:
            screen: Pygame surface to draw on
        """
        if not self.visible:
            return
            
        if self.fading_in:
            # Create a surface with per-pixel alpha
            s = pygame.Surface((self.size, self.size), pygame.SRCALPHA)
            color_with_alpha = (*self.color, self.fade_alpha)
            pygame.draw.rect(s, color_with_alpha, (0, 0, self.size, self.size))
            screen.blit(s, (self.pos["x"], self.pos["y"]))
        else:
            pygame.draw.rect(
                screen,
                self.color,
                (self.pos["x"], self.pos["y"], self.size, self.size)
            )
