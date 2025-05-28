"""
Game Environment for AI Platform Trainer.

Unified game environment abstraction for both SL and RL agents.
"""
import pygame
from typing import Dict, Any, Tuple, Optional
import numpy as np
import math


class GameEnvironment:
    """Unified game environment interface for all AI agents."""
    
    def __init__(self, screen_width: int, screen_height: int):
        """
        Initialize the game environment.
        
        Args:
            screen_width: Screen width in pixels
            screen_height: Screen height in pixels
        """
        self.screen_width = screen_width
        self.screen_height = screen_height
        
        # Game state
        self.player_pos = {"x": screen_width // 4, "y": screen_height // 2}
        self.enemy_pos = {"x": 3 * screen_width // 4, "y": screen_height // 2}
        self.missiles = []
        
        # Game metrics
        self.score = 0
        self.episode_step = 0
        self.max_episode_steps = 1000
        
        # RL-specific state
        self.last_distance = self._calculate_distance()
        self.hits_dealt = 0
        self.hits_taken = 0
        
    def reset(self) -> Dict[str, Any]:
        """Reset environment and return initial observation."""
        # Reset positions
        self.player_pos = {
            "x": self.screen_width // 4, 
            "y": self.screen_height // 2
        }
        self.enemy_pos = {
            "x": 3 * self.screen_width // 4, 
            "y": self.screen_height // 2
        }
        self.missiles = []
        
        # Reset metrics
        self.score = 0
        self.episode_step = 0
        self.hits_dealt = 0
        self.hits_taken = 0
        self.last_distance = self._calculate_distance()
        
        return self.get_observation()
    
    def step(self, action: Tuple[float, float]) -> Dict[str, Any]:
        """
        Execute one step in the environment.
        
        Args:
            action: Enemy action (dx, dy)
            
        Returns:
            Dictionary with observation, reward, done, info
        """
        # Update episode step
        self.episode_step += 1
        
        # Apply enemy action
        self.enemy_pos["x"] += action[0]
        self.enemy_pos["y"] += action[1]
        
        # Keep enemy in bounds with wrap-around
        self.enemy_pos["x"] = self.enemy_pos["x"] % self.screen_width
        self.enemy_pos["y"] = self.enemy_pos["y"] % self.screen_height
        
        # Update missiles (simplified)
        self._update_missiles()
        
        # Check for collisions and compute reward
        reward = self._compute_reward()
        
        # Check if episode is done
        done = (self.episode_step >= self.max_episode_steps or 
                self.hits_taken >= 3 or 
                self.hits_dealt >= 3)
        
        # Update last distance for next reward calculation
        self.last_distance = self._calculate_distance()
        
        return {
            "observation": self.get_observation(),
            "reward": reward,
            "done": done,
            "info": {
                "hits_dealt": self.hits_dealt,
                "hits_taken": self.hits_taken,
                "episode_step": self.episode_step
            }
        }
    
    def get_observation(self) -> Dict[str, Any]:
        """Get current game state observation for agents."""
        # Normalize positions
        px = self.player_pos["x"] / self.screen_width
        py = self.player_pos["y"] / self.screen_height
        ex = self.enemy_pos["x"] / self.screen_width
        ey = self.enemy_pos["y"] / self.screen_height
        
        # Calculate distance
        distance = self._calculate_distance() / max(self.screen_width, 
                                                   self.screen_height)
        
        return {
            "player_x": px,
            "player_y": py,
            "enemy_x": ex,
            "enemy_y": ey,
            "distance": distance,
            "hits_dealt": self.hits_dealt,
            "hits_taken": self.hits_taken,
            "episode_step": self.episode_step / self.max_episode_steps
        }
    
    def _calculate_distance(self) -> float:
        """Calculate distance between player and enemy."""
        dx = self.player_pos["x"] - self.enemy_pos["x"]
        dy = self.player_pos["y"] - self.enemy_pos["y"]
        return math.sqrt(dx * dx + dy * dy)
    
    def _update_missiles(self) -> None:
        """Update missile positions (simplified)."""
        # This is a placeholder for missile update logic
        # In the full implementation, this would update all missiles
        # and check for collisions
        pass
    
    def _compute_reward(self) -> float:
        """Compute reward for RL training."""
        reward = 0.0
        
        # Distance-based reward (encourage getting closer to player)
        current_distance = self._calculate_distance()
        distance_change = self.last_distance - current_distance
        reward += distance_change * 0.01  # Small reward for getting closer
        
        # Large positive reward for hitting player
        # (This would be detected by collision system)
        # reward += self.hits_dealt * 10.0
        
        # Large negative reward for getting hit
        # reward -= self.hits_taken * 10.0
        
        # Small negative reward for each step (encourage efficiency)
        reward -= 0.01
        
        return reward
    
    def set_player_position(self, x: float, y: float) -> None:
        """Update player position (called by main game loop)."""
        self.player_pos["x"] = x
        self.player_pos["y"] = y
    
    def set_enemy_position(self, x: float, y: float) -> None:
        """Update enemy position (called by main game loop)."""
        self.enemy_pos["x"] = x
        self.enemy_pos["y"] = y
    
    def add_missile(self, x: float, y: float, vx: float, vy: float) -> None:
        """Add a new missile to the environment."""
        self.missiles.append({
            "x": x, "y": y, "vx": vx, "vy": vy
        })
    
    def register_hit(self, enemy_hit: bool) -> None:
        """Register a hit event."""
        if enemy_hit:
            self.hits_dealt += 1
        else:
            self.hits_taken += 1
