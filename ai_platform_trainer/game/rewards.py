"""
Reward System for Reinforcement Learning.

This module computes rewards for RL agent training based on game events.
"""
from typing import Dict, Any
import math


class RewardSystem:
    """Reward computation system for RL training."""
    
    def __init__(self):
        """Initialize reward system with default parameters."""
        # Reward parameters
        self.hit_player_reward = 10.0
        self.get_hit_penalty = -10.0
        self.distance_reward_scale = 0.01
        self.step_penalty = -0.01
        self.survival_bonus = 0.1
        
    def compute_reward(self, events: Dict[str, Any]) -> float:
        """
        Compute reward based on game events.
        
        Args:
            events: Dictionary containing game events and state
            
        Returns:
            Computed reward value
        """
        reward = 0.0
        
        # Reward for hitting player
        if events.get("enemy_hit_player", False):
            reward += self.hit_player_reward
            
        # Penalty for getting hit
        if events.get("player_hit_enemy", False):
            reward += self.get_hit_penalty
            
        # Distance-based reward (encourage approaching player)
        distance_change = events.get("distance_change", 0.0)
        reward += distance_change * self.distance_reward_scale
        
        # Small penalty for each step (encourage efficiency)
        reward += self.step_penalty
        
        # Small bonus for surviving (staying in game)
        reward += self.survival_bonus
        
        return reward
    
    def set_reward_parameters(self, **kwargs) -> None:
        """
        Update reward parameters.
        
        Args:
            **kwargs: Reward parameter updates
        """
        for key, value in kwargs.items():
            if hasattr(self, key):
                setattr(self, key, value)
