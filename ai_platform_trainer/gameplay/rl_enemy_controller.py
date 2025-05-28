"""
RL Enemy Controller for Game Integration

Uses trained NumPy neural network for enemy AI in play mode.
No PyTorch dependency - pure NumPy implementation.
"""
import numpy as np
import logging
import os
from typing import Tuple, Optional

logger = logging.getLogger(__name__)


class RLEnemyController:
    """Controller for RL-trained enemy in play mode."""
    
    def __init__(self, model_path: str = None):
        """Initialize RL enemy controller.
        
        Args:
            model_path: Path to trained model file
        """
        self.agent = None
        self.is_loaded = False
        
        # Default model path
        if model_path is None:
            model_path = 'ai_platform_trainer/models/enemy_rl_gpu.npz'
        
        self.model_path = model_path
        
        # Try to load trained model
        self._load_model()
    
    def _load_model(self) -> None:
        """Load the trained RL model."""
        try:
            if not os.path.exists(self.model_path):
                logger.warning(f"Model file not found: {self.model_path}")
                logger.info("Using random controller until model is trained")
                return
            
            # Import agent class
            from ai_platform_trainer.ai.models.train_enemy_rl_gpu import SimpleEnemyAgent
            
            # Load model data
            data = np.load(self.model_path)
            
            # Get model dimensions
            obs_dim = int(data['obs_dim']) if 'obs_dim' in data else 4
            action_dim = int(data['action_dim']) if 'action_dim' in data else 2
            hidden_dim = int(data['hidden_dim']) if 'hidden_dim' in data else 64
            
            # Create and load agent
            self.agent = SimpleEnemyAgent(obs_dim=obs_dim, 
                                        action_dim=action_dim,
                                        hidden_dim=hidden_dim)
            self.agent.load(self.model_path)
            
            self.is_loaded = True
            logger.info(f"Loaded RL enemy model from {self.model_path}")
            
        except Exception as e:
            logger.error(f"Failed to load RL model: {e}")
            logger.info("Using random controller as fallback")
            self.agent = None
            self.is_loaded = False
    
    def get_action(self, enemy_pos: Tuple[float, float], 
                  player_pos: Tuple[float, float],
                  enemy_vel: Tuple[float, float] = (0.0, 0.0)) -> Tuple[float, float]:
        """Get action for a single enemy in play mode.
        
        Args:
            enemy_pos: Enemy position (x, y)
            player_pos: Player position (x, y)
            enemy_vel: Enemy velocity (vx, vy) - optional
            
        Returns:
            Action as (vx, vy) velocity
        """
        if not self.is_loaded or self.agent is None:
            # Random fallback behavior
            else:
                logger.warning(f"Model file not found: {model_path}")
                logger.warning("Using random behavior until model is trained")
                return False
        except Exception as e:
            logger.error(f"Failed to load model: {e}")
            logger.warning("Using random behavior instead")
            return False
    
    def get_action(self, 
                   enemy_pos: Tuple[float, float], 
                   player_pos: Tuple[float, float],
                   screen_width: int = 800,
                   screen_height: int = 600) -> Tuple[float, float]:
        """Get action for enemy based on current game state.
        
        Args:
            enemy_pos: Current enemy position (x, y)
            player_pos: Current player position (x, y)
            screen_width: Game screen width
            screen_height: Game screen height
            
        Returns:
            Action tuple (dx, dy) - movement direction
        """
        if not self.model_loaded:
            # Fallback to simple behavior if model not loaded
            return self._simple_chase_behavior(enemy_pos, player_pos)
        
        try:
            # Prepare observation (same format as training)
            relative_pos = [
                player_pos[0] - enemy_pos[0],  # rel_x
                player_pos[1] - enemy_pos[1]   # rel_y
            ]
            
            normalized_enemy = [
                enemy_pos[0] / screen_width,   # norm_enemy_x
                enemy_pos[1] / screen_height   # norm_enemy_y
            ]
            
            # Combine into observation vector
            obs = relative_pos + normalized_enemy
            obs_tensor = torch.tensor(obs, dtype=torch.float32).to(self.device)
            
            # Get action from neural network
            with torch.no_grad():
                action_tensor = self.agent(obs_tensor.unsqueeze(0)).squeeze(0)
                action = action_tensor.cpu().numpy()
            
            # Store for smoothing/debugging
            self.last_action = action.copy()
            self.last_enemy_pos = enemy_pos
            
            return tuple(action)
            
        except Exception as e:
            logger.error(f"Error in RL action computation: {e}")
            # Fallback to simple behavior
            return self._simple_chase_behavior(enemy_pos, player_pos)
    
    def _simple_chase_behavior(self, 
                              enemy_pos: Tuple[float, float], 
                              player_pos: Tuple[float, float]) -> Tuple[float, float]:
        """Simple chase behavior as fallback."""
        dx = player_pos[0] - enemy_pos[0]
        dy = player_pos[1] - enemy_pos[1]
        
        # Normalize to unit vector
        distance = np.sqrt(dx*dx + dy*dy)
        if distance > 0:
            dx /= distance
            dy /= distance
        
        # Add some noise for variation
        dx += np.random.normal(0, 0.1)
        dy += np.random.normal(0, 0.1)
        
        # Clamp to [-1, 1] range
        dx = np.clip(dx, -1.0, 1.0)
        dy = np.clip(dy, -1.0, 1.0)
        
        return (dx, dy)
    
    def update_enemy_position(self, 
                             enemy_pos: Tuple[float, float],
                             action: Tuple[float, float],
                             speed: float = 200.0,
                             dt: float = 0.016) -> Tuple[float, float]:
        """Update enemy position based on RL action.
        
        Args:
            enemy_pos: Current enemy position
            action: Action from get_action()
            speed: Enemy movement speed (pixels/second)
            dt: Time delta (seconds)
            
        Returns:
            New enemy position
        """
        # Apply action as velocity
        new_x = enemy_pos[0] + action[0] * speed * dt
        new_y = enemy_pos[1] + action[1] * speed * dt
        
        return (new_x, new_y)
    
    def get_debug_info(self) -> dict:
        """Get debug information about the controller."""
        return {
            'model_loaded': self.model_loaded,
            'device': self.device,
            'last_action': self.last_action.tolist() if self.last_action is not None else None,
            'last_enemy_pos': self.last_enemy_pos
        }


class EnemyRLIntegration:
    """Helper class for integrating RL enemy into existing game loop."""
    
    def __init__(self, model_path: str = "ai_platform_trainer/models/enemy_rl_gpu.pth"):
        """Initialize RL integration.
        
        Args:
            model_path: Path to trained model
        """
        self.controller = RLEnemyController(model_path)
        self.enabled = True
        
    def control_enemy(self, enemy, player, dt: float = 0.016):
        """Control enemy using RL agent.
        
        Args:
            enemy: Enemy entity with pos attribute
            player: Player entity with pos attribute  
            dt: Time delta
        """
        if not self.enabled:
            return
        
        # Get action from RL controller
        action = self.controller.get_action(
            enemy_pos=(enemy.pos[0], enemy.pos[1]),
            player_pos=(player.pos[0], player.pos[1])
        )
        
        # Update enemy position
        speed = getattr(enemy, 'speed', 200.0)
        new_pos = self.controller.update_enemy_position(
            enemy_pos=(enemy.pos[0], enemy.pos[1]),
            action=action,
            speed=speed,
            dt=dt
        )
        
        # Apply new position
        enemy.pos[0] = new_pos[0]
        enemy.pos[1] = new_pos[1]
        
        # Keep in bounds (if enemy has bounds checking, this might be redundant)
        enemy.pos[0] = max(25, min(775, enemy.pos[0]))
        enemy.pos[1] = max(25, min(575, enemy.pos[1]))
    
    def toggle_rl_control(self):
        """Toggle RL control on/off."""
        self.enabled = not self.enabled
        logger.info(f"RL enemy control: {'enabled' if self.enabled else 'disabled'}")
    
    def get_status(self) -> str:
        """Get status string for UI display."""
        if not self.enabled:
            return "RL Control: OFF"
        
        if self.controller.model_loaded:
            return f"RL Control: ON ({self.controller.device})"
        else:
            return "RL Control: ON (fallback mode)"


def test_rl_controller():
    """Test the RL controller."""
    logger.info("=== Testing RL Enemy Controller ===")
    
    # Create controller
    controller = RLEnemyController()
    
    # Test with some positions
    enemy_pos = (400, 300)
    player_pos = (500, 400)
    
    logger.info(f"Enemy at: {enemy_pos}")
    logger.info(f"Player at: {player_pos}")
    
    # Get action
    action = controller.get_action(enemy_pos, player_pos)
    logger.info(f"RL Action: {action}")
    
    # Update position
    new_pos = controller.update_enemy_position(enemy_pos, action)
    logger.info(f"New enemy pos: {new_pos}")
    
    # Debug info
    debug_info = controller.get_debug_info()
    logger.info(f"Debug info: {debug_info}")
    
    logger.info("✓ RL controller test completed!")


if __name__ == "__main__":
    test_rl_controller()
