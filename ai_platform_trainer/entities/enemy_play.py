"""
Enemy entity for play mode.

This module defines the enemy entity used in play mode, which uses AI models
for movement decisions.
"""
import logging
import math
from typing import Optional, Tuple # Added Tuple for type hinting

import pygame
import torch

from ai_platform_trainer.ai.models.enemy_movement_model import EnemyMovementModel
# Import the global enemy AI controller
from ai_platform_trainer.gameplay.ai.enemy_ai_controller import enemy_controller, NumpyEnemyModel


class EnemyPlay:
    """
    Enemy entity for play mode with AI-controlled movement.
    
    This class represents the enemy in play mode, controlled by a neural network.
    """
    
    def __init__(
        self, 
        screen_width: int, 
        screen_height: int, 
        model: Optional[EnemyMovementModel] = None # PyTorch model is now optional
    ) -> None:
        """
        Initialize the enemy entity.
        
        Args:
            screen_width: Width of the game screen
            screen_height: Height of the game screen
            model: Optional PyTorch neural network model for enemy movement (used as fallback).
        """
        self.screen_width = screen_width
        self.screen_height = screen_height
        self.size = 50
        self.color = (139, 0, 0)  # Dark red
        self.pos = {"x": screen_width // 2, "y": screen_height // 2}
        self.speed = 5.0 # Default speed, can be overridden by controller
        self.pytorch_model = model # Store the PyTorch model if provided
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
            current_time: Current game time in milliseconds (used by controller for timing)
        """
        if not self.visible:
            return

        # Delegate movement to the global enemy_controller
        # The controller will decide whether to use NumPy, PyTorch (via self.pytorch_model), or random
        enemy_controller.update_enemy_movement(
            self, # Pass self (EnemyPlay instance)
            player_x, 
            player_y, 
            player_speed, 
            current_time,
            # Pass the numpy_model from the controller itself, or allow override if GameCore provides one
            numpy_model_instance=enemy_controller.numpy_model 
        )

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
        # This PyTorch specific logic is now primarily handled within 
        # EnemyAIController._get_pytorch_nn_action if self.pytorch_model is used.
        if not self.pytorch_model: # Should not happen if controller logic is correct
            logging.warning("EnemyPlay._update_with_nn called without a PyTorch model.")
            return

        model_input = torch.tensor([
            normalized_px, normalized_py, 
            normalized_ex, normalized_ey, 
            normalized_dist
        ], dtype=torch.float32).unsqueeze(0)
        
        # Get model prediction
        with torch.no_grad():
            movement = self.pytorch_model(model_input).squeeze(0) # Use self.pytorch_model
            
        # Apply movement (scale from [-1,1] to actual pixels)
        move_x = movement[0].item() * self.speed
        move_y = movement[1].item() * self.speed
        
        # Update position - This will be handled by EnemyAIController now.
        # self.pos["x"] += move_x
        # self.pos["y"] += move_y
        # self._wrap_position() # Also handled by EnemyAIController
        # For now, this method might not be directly called if EnemyAIController handles all.
        # However, keeping its structure for potential direct PyTorch use if needed.
        # The actual position update is done in EnemyAIController.update_enemy_movement
        pass # Movement is now applied by the controller

    def wrap_position(self, x: float, y: float) -> Tuple[float, float]: # Made public for controller
        """Wrap the enemy position around screen edges."""
        if x < -self.size:
            x = self.screen_width
        elif x > self.screen_width:
            x = -self.size
            
        if y < -self.size:
            y = self.screen_height
        elif y > self.screen_height:
            y = -self.size
        return x, y

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
