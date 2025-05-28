"""
Supervised Learning Agent Implementation.

This module wraps the existing supervised learning enemy AI functionality
into the new agent architecture.
"""
import logging
from typing import Dict, Any, Tuple, Optional

from ai_platform_trainer.agents.base_agent import BaseAgent
from ai_platform_trainer.entities.behaviors.enemy_ai_controller import (
    update_enemy_movement
)


class SupervisedAgent(BaseAgent):
    """
    Supervised learning agent that wraps existing neural network functionality.
    
    This agent uses the existing trained neural network to control enemy 
    behavior, preserving all current functionality while fitting the new 
    architecture.
    """
    
    def __init__(self, screen_width: int, screen_height: int, 
                 model_path: Optional[str] = None):
        """
        Initialize the supervised learning agent.
        
        Args:
            screen_width: Game screen width
            screen_height: Game screen height  
            model_path: Path to trained model (optional)
        """
        self.screen_width = screen_width
        self.screen_height = screen_height
        self.model_path = model_path
        
        # Create a mock enemy entity for compatibility with existing code
        self.enemy = self._create_enemy_entity()
        
        # Agent state
        self.last_observation = None
        self.model_loaded = False
        self.last_enemy_pos = {"x": 0, "y": 0}
        
        # Try to load model if path provided
        if model_path:
            self.load_model(model_path)
    
    def _create_enemy_entity(self):
        """Create a mock enemy entity for compatibility."""
        # Import here to avoid circular imports
        from ai_platform_trainer.entities.enemy_play import EnemyPlay
        
        enemy = EnemyPlay(self.screen_width, self.screen_height)
        return enemy
    
    def select_action(self, observation: Dict[str, Any]) -> Tuple[float, float]:
        """
        Select action using the supervised learning model.
        
        Args:
            observation: Current game state observation
            
        Returns:
            Action tuple (dx, dy) representing movement
        """
        # Store current enemy position
        prev_x = self.enemy.pos["x"]
        prev_y = self.enemy.pos["y"]
        
        # Convert observation to format expected by existing code
        player_x = observation.get("player_x", 0.5) * self.screen_width
        player_y = observation.get("player_y", 0.5) * self.screen_height
        player_speed = 5  # Default player speed
        current_time = 0  # Placeholder
        
        # Update enemy position from observation
        enemy_x = observation.get("enemy_x", 0.5) * self.screen_width
        enemy_y = observation.get("enemy_y", 0.5) * self.screen_height
        self.enemy.set_position(int(enemy_x), int(enemy_y))
        
        # Use existing AI controller to update movement
        try:
            update_enemy_movement(
                self.enemy,
                player_x=player_x,
                player_y=player_y,
                player_speed=player_speed,
                current_time=current_time,
            )
        except Exception as e:
            logging.warning(f"Error in supervised agent: {e}")
            # Fallback to simple movement toward player
            dx = player_x - enemy_x
            dy = player_y - enemy_y
            # Normalize and scale
            length = (dx * dx + dy * dy) ** 0.5
            if length > 0:
                dx = (dx / length) * 2
                dy = (dy / length) * 2
            else:
                dx, dy = 0, 0
            return (dx, dy)
        
        # Calculate action from position change
        new_x = self.enemy.pos["x"]
        new_y = self.enemy.pos["y"]
        
        # Return the movement delta
        dx = new_x - prev_x
        dy = new_y - prev_y
        
        return (float(dx), float(dy))
    
    def reset(self) -> None:
        """Reset the agent state for a new episode."""
        self.last_observation = None
        self.last_enemy_pos = {"x": 0, "y": 0}
        # Reset enemy to center position
        center_x = self.screen_width // 2
        center_y = self.screen_height // 2
        self.enemy.set_position(center_x, center_y)
    
    def load_model(self, path: str) -> bool:
        """
        Load a trained model from file.
        
        Args:
            path: Path to the model file
            
        Returns:
            True if successful, False otherwise
        """
        try:
            # The existing enemy AI controller handles model loading internally
            # We just need to ensure the enemy entity has a model
            if hasattr(self.enemy, 'model') and self.enemy.model is not None:
                self.model_loaded = True
                logging.info(f"Supervised model loaded from {path}")
                return True
        except Exception as e:
            logging.error(f"Failed to load supervised model: {e}")
        
        self.model_loaded = False
        return False
