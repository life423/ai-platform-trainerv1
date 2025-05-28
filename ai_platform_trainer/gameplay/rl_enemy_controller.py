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
            return self._random_action(enemy_pos, player_pos)
        
        try:
            # Prepare observation
            relative_pos = np.array([
                player_pos[0] - enemy_pos[0],
                player_pos[1] - enemy_pos[1]
            ])
            
            # Normalized enemy position
            normalized_enemy = np.array([
                enemy_pos[0] / 800.0,
                enemy_pos[1] / 600.0
            ])
            
            # Combine observations
            obs = np.concatenate([relative_pos, normalized_enemy])
            
            # Get action from neural network
            action = self.agent.forward(obs.reshape(1, -1))[0]
            
            return float(action[0]), float(action[1])
            
        except Exception as e:
            logger.error(f"Error in RL controller: {e}")
            return self._random_action(enemy_pos, player_pos)
    
    def _random_action(self, enemy_pos: Tuple[float, float], 
                      player_pos: Tuple[float, float]) -> Tuple[float, float]:
        """Fallback random action with some intelligence.
        
        Args:
            enemy_pos: Enemy position
            player_pos: Player position
            
        Returns:
            Random action biased toward player
        """
        # Calculate direction to player
        dx = player_pos[0] - enemy_pos[0]
        dy = player_pos[1] - enemy_pos[1]
        
        # Normalize
        distance = np.sqrt(dx*dx + dy*dy)
        if distance > 0:
            dx /= distance
            dy /= distance
        
        # Add some randomness
        noise_x = np.random.randn() * 0.3
        noise_y = np.random.randn() * 0.3
        
        # Combine directed movement with noise
        action_x = np.clip(dx + noise_x, -1.0, 1.0)
        action_y = np.clip(dy + noise_y, -1.0, 1.0)
        
        return float(action_x), float(action_y)
    
    def reload_model(self) -> bool:
        """Reload the model from disk.
        
        Returns:
            True if model was successfully reloaded
        """
        logger.info("Reloading RL enemy model...")
        self._load_model()
        return self.is_loaded
    
    def get_status(self) -> dict:
        """Get controller status information.
        
        Returns:
            Status dictionary
        """
        return {
            'model_loaded': self.is_loaded,
            'model_path': self.model_path,
            'agent_available': self.agent is not None
        }


class BatchRLEnemyController:
    """Batch controller for multiple enemies using vectorized operations."""
    
    def __init__(self, model_path: str = None, max_enemies: int = 10):
        """Initialize batch RL enemy controller.
        
        Args:
            model_path: Path to trained model file
            max_enemies: Maximum number of enemies to handle
        """
        self.max_enemies = max_enemies
        self.single_controller = RLEnemyController(model_path)
    
    def get_actions(self, enemy_positions: np.ndarray, 
                   player_positions: np.ndarray) -> np.ndarray:
        """Get actions for multiple enemies.
        
        Args:
            enemy_positions: Array of enemy positions [n_enemies, 2]
            player_positions: Array of player positions [n_enemies, 2] 
                            (can be same player for all enemies)
            
        Returns:
            Actions as [n_enemies, 2] array
        """
        if not self.single_controller.is_loaded:
            # Random fallback for all enemies
            n_enemies = enemy_positions.shape[0]
            actions = np.random.randn(n_enemies, 2)
            return np.clip(actions, -1.0, 1.0)
        
        try:
            # Prepare batch observations
            relative_pos = player_positions - enemy_positions
            normalized_enemies = enemy_positions.copy()
            normalized_enemies[:, 0] /= 800.0
            normalized_enemies[:, 1] /= 600.0
            
            # Combine observations
            obs_batch = np.concatenate([relative_pos, normalized_enemies], axis=1)
            
            # Get batch actions
            actions = self.single_controller.agent.forward(obs_batch)
            
            return actions
            
        except Exception as e:
            logger.error(f"Error in batch RL controller: {e}")
            n_enemies = enemy_positions.shape[0]
            actions = np.random.randn(n_enemies, 2)
            return np.clip(actions, -1.0, 1.0)


def test_rl_controller():
    """Test the RL enemy controller."""
    logger.info("=== Testing RL Enemy Controller ===")
    
    # Test single controller
    controller = RLEnemyController()
    status = controller.get_status()
    
    logger.info(f"Controller status: {status}")
    
    # Test single action
    enemy_pos = (400.0, 300.0)
    player_pos = (200.0, 150.0)
    
    action = controller.get_action(enemy_pos, player_pos)
    logger.info(f"Single action: {action}")
    
    # Test batch controller
    batch_controller = BatchRLEnemyController()
    
    # Test batch actions
    enemy_positions = np.array([
        [400.0, 300.0],
        [600.0, 400.0],
        [100.0, 200.0]
    ])
    
    player_positions = np.array([
        [200.0, 150.0],
        [200.0, 150.0],
        [200.0, 150.0]
    ])
    
    batch_actions = batch_controller.get_actions(enemy_positions, player_positions)
    logger.info(f"Batch actions shape: {batch_actions.shape}")
    logger.info(f"Batch actions: {batch_actions}")
    
    logger.info("RL controller test completed!")
    return controller


if __name__ == "__main__":
    # Configure logging
    logging.basicConfig(level=logging.INFO)
    
    # Run test
    test_rl_controller()
