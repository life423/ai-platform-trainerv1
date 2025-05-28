"""
Base Agent Interface for AI Platform Trainer.

This module defines the abstract base class that all AI agents must implement.
"""
from abc import ABC, abstractmethod
from typing import Dict, Any, Tuple


class BaseAgent(ABC):
    """
    Abstract base class for all AI agents.
    
    This interface ensures consistent behavior between supervised learning
    and reinforcement learning agents.
    """
    
    @abstractmethod
    def select_action(self, observation: Dict[str, Any]) -> Tuple[float, float]:
        """
        Select an action given the current observation.
        
        Args:
            observation: Current game state observation
            
        Returns:
            Action tuple (dx, dy) representing movement
        """
        pass
    
    @abstractmethod
    def reset(self) -> None:
        """Reset the agent state for a new episode."""
        pass
    
    @abstractmethod
    def load_model(self, path: str) -> bool:
        """
        Load a trained model from file.
        
        Args:
            path: Path to the model file
            
        Returns:
            True if successful, False otherwise
        """
        pass
    
    def save_model(self, path: str) -> bool:
        """
        Save the current model to file.
        
        Args:
            path: Path to save the model
            
        Returns:
            True if successful, False otherwise
        """
        # Default implementation - not all agents need to save
        return False
    
    def update(self, observation: Dict[str, Any], action: Tuple[float, float], 
               reward: float, next_observation: Dict[str, Any], done: bool) -> None:
        """
        Update the agent with experience (for learning agents).
        
        Args:
            observation: Previous observation
            action: Action taken
            reward: Reward received
            next_observation: New observation
            done: Whether episode ended
        """
        # Default implementation - not all agents learn during play
        pass
