# API Reference

This document provides a comprehensive reference for the AI Platform Trainer APIs and interfaces.

## Core Interfaces

### BaseAgent Interface

The foundation interface for all AI agents in the system.

```python
from abc import ABC, abstractmethod
from typing import Dict, Any, Tuple

class BaseAgent(ABC):
    """Base class for all AI agents."""
    
    @abstractmethod
    def select_action(self, observation: Dict[str, Any]) -> Tuple[float, float]:
        """
        Select an action based on the current game state observation.
        
        Args:
            observation: Dictionary containing normalized game state:
                - player_x: Player X position (0.0-1.0)
                - player_y: Player Y position (0.0-1.0)
                - enemy_x: Enemy X position (0.0-1.0)
                - enemy_y: Enemy Y position (0.0-1.0)
                - player_velocity_x: Player X velocity (-1.0-1.0)
                - player_velocity_y: Player Y velocity (-1.0-1.0)
                - time_elapsed: Time since episode start (normalized)
                - episode_step: Current step in episode
                
        Returns:
            Tuple of (dx, dy) movement values in pixels per frame
        """
        pass
    
    @abstractmethod
    def reset(self) -> None:
        """Reset the agent state for a new episode."""
        pass
    
    def load_model(self, path: str) -> bool:
        """
        Load a trained model from file.
        
        Args:
            path: Path to the model file
            
        Returns:
            True if successful, False otherwise
        """
        return False
    
    def save_model(self, path: str) -> bool:
        """
        Save the current model to file.
        
        Args:
            path: Path where to save the model
            
        Returns:
            True if successful, False otherwise
        """
        return False
```

### GameEnvironment Interface

Provides a unified interface for agents to interact with the game world.

```python
class GameEnvironment:
    """Game environment interface for AI agents."""
    
    def __init__(self, screen_width: int, screen_height: int):
        """
        Initialize the game environment.
        
        Args:
            screen_width: Game screen width in pixels
            screen_height: Game screen height in pixels
        """
        pass
    
    def get_observation(self) -> Dict[str, Any]:
        """
        Get the current game state as a normalized observation.
        
        Returns:
            Dictionary with normalized game state values (0.0-1.0 range)
        """
        pass
    
    def step(self, action: Tuple[float, float]) -> Tuple[Dict[str, Any], float, bool]:
        """
        Execute an action in the environment.
        
        Args:
            action: Tuple of (dx, dy) movement values
            
        Returns:
            Tuple of (new_observation, reward, done)
            - new_observation: Updated game state
            - reward: Reward value for the action
            - done: Whether the episode has ended
        """
        pass
    
    def reset(self) -> Dict[str, Any]:
        """
        Reset the environment for a new episode.
        
        Returns:
            Initial observation for the new episode
        """
        pass
    
    def render(self, mode: str = "human") -> None:
        """
        Render the current game state.
        
        Args:
            mode: Rendering mode ("human" or "rgb_array")
        """
        pass
```

### RewardSystem Interface

Computes rewards for reinforcement learning agents.

```python
class RewardSystem:
    """Reward computation system for RL training."""
    
    def __init__(self, config: Dict[str, float]):
        """
        Initialize the reward system.
        
        Args:
            config: Reward configuration parameters:
                - hit_player_reward: Reward for hitting the player
                - get_hit_penalty: Penalty for getting hit
                - distance_reward_scale: Scale for distance-based rewards
                - time_penalty: Penalty per time step
        """
        pass
    
    def compute_reward(self, 
                      prev_state: Dict[str, Any], 
                      action: Tuple[float, float],
                      new_state: Dict[str, Any],
                      events: List[str]) -> float:
        """
        Compute reward for a state transition.
        
        Args:
            prev_state: Previous game state
            action: Action that was taken
            new_state: Resulting game state
            events: List of events that occurred ("hit_player", "got_hit", etc.)
            
        Returns:
            Reward value (positive for good outcomes, negative for bad)
        """
        pass
```

## Menu System API

### MenuManager

Handles the enhanced menu system with multiple levels.

```python
from enum import Enum

class MenuState(Enum):
    MAIN = "main"
    RL_OPTIONS = "rl_options"
    RL_DIFFICULTY = "rl_difficulty"
    HELP = "help"
    GAME = "game"

class MenuManager:
    """Enhanced menu system manager."""
    
    def __init__(self, screen_width: int, screen_height: int):
        """Initialize the menu system."""
        pass
    
    def handle_event(self, event) -> bool:
        """
        Handle a pygame event.
        
        Args:
            event: Pygame event object
            
        Returns:
            True if event was handled, False otherwise
        """
        pass
    
    def update(self, dt: float) -> None:
        """
        Update menu animations and state.
        
        Args:
            dt: Delta time since last update
        """
        pass
    
    def render(self, screen) -> None:
        """
        Render the current menu.
        
        Args:
            screen: Pygame screen surface
        """
        pass
    
    def get_selected_mode(self) -> Dict[str, Any]:
        """
        Get the currently selected game mode.
        
        Returns:
            Dictionary with mode configuration:
                - agent_type: "supervised" or "reinforcement"
                - difficulty: "easy", "medium", "hard" (for RL)
                - live_learning: True/False (for RL)
        """
        pass
```

## Agent Implementations

### SupervisedAgent

Wrapper for the existing supervised learning enemy AI.

```python
class SupervisedAgent(BaseAgent):
    """Supervised learning agent implementation."""
    
    def __init__(self, screen_width: int, screen_height: int, 
                 model_path: Optional[str] = None):
        """
        Initialize the supervised agent.
        
        Args:
            screen_width: Game screen width
            screen_height: Game screen height
            model_path: Path to trained model file
        """
        pass
    
    def select_action(self, observation: Dict[str, Any]) -> Tuple[float, float]:
        """Use existing neural network to select actions."""
        pass
    
    def reset(self) -> None:
        """Reset agent state."""
        pass
```

### RLAgent (Phase 2)

High-performance reinforcement learning agent with C++/CUDA backend.

```python
class RLAgent(BaseAgent):
    """Reinforcement learning agent with C++/CUDA backend."""
    
    def __init__(self, config: RLConfig):
        """
        Initialize the RL agent.
        
        Args:
            config: RL configuration object with:
                - state_size: Dimension of state vector
                - action_size: Number of discrete actions
                - learning_rate: Learning rate for training
                - buffer_size: Experience replay buffer size
                - batch_size: Training batch size
                - gamma: Discount factor
                - epsilon: Exploration rate
                - live_learning: Enable training during play
        """
        pass
    
    def select_action(self, observation: Dict[str, Any]) -> Tuple[float, float]:
        """Select action using DQN policy."""
        pass
    
    def record_outcome(self, reward: float, new_state: Dict[str, Any], done: bool):
        """Record experience for training."""
        pass
    
    def reset(self) -> None:
        """Reset agent state."""
        pass
    
    def get_training_stats(self) -> Dict[str, float]:
        """
        Get current training statistics.
        
        Returns:
            Dictionary with training metrics:
                - episodes_trained: Number of episodes completed
                - average_reward: Moving average reward
                - epsilon: Current exploration rate
                - loss: Recent training loss
        """
        pass
```

## Configuration Classes

### GameConfig

Configuration for game environment settings.

```python
@dataclass
class GameConfig:
    """Game environment configuration."""
    screen_width: int = 800
    screen_height: int = 600
    fps: int = 60
    player_speed: float = 5.0
    enemy_speed: float = 3.0
    episode_max_steps: int = 1000
    
    @classmethod
    def from_file(cls, path: str) -> 'GameConfig':
        """Load configuration from JSON file."""
        pass
    
    def to_file(self, path: str) -> None:
        """Save configuration to
