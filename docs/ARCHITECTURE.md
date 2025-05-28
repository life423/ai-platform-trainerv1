# AI Platform Trainer Architecture

## Overview

The AI Platform Trainer has been refactored into a modular architecture that cleanly separates concerns and provides a foundation for both supervised learning (SL) and reinforcement learning (RL) implementations.

## High-Level Architecture

```
┌─────────────────┐    ┌─────────────────┐
│   User Input    │    │   Game Display  │
└─────────┬───────┘    └─────────┬───────┘
          │                      │
          ▼                      ▼
┌─────────────────────────────────────────┐
│              UI Layer                   │
│  ┌─────────────┐  ┌─────────────────┐  │
│  │   Menus     │  │   Game Loop     │  │
│  └─────────────┘  └─────────────────┘  │
└─────────────────┬───────────────────────┘
                  │
                  ▼
┌─────────────────────────────────────────┐
│           Game Environment              │
│  ┌─────────────┐  ┌─────────────────┐  │
│  │ Environment │  │    Rewards      │  │
│  └─────────────┘  └─────────────────┘  │
└─────────────────┬───────────────────────┘
                  │
                  ▼
┌─────────────────────────────────────────┐
│            Agent Layer                  │
│  ┌─────────────┐  ┌─────────────────┐  │
│  │ SL Agents   │  │   RL Agents     │  │
│  │             │  │   (Phase 2)     │  │
│  └─────────────┘  └─────────────────┘  │
└─────────────────────────────────────────┘
```

## Core Components

### UI Layer (`ai_platform_trainer/ui/`)

**Purpose**: Handles all user interactions and orchestrates the game flow.

**Components**:
- `menus.py`: Multi-level menu system with keyboard/mouse support
- `game_loop.py`: Main game loop coordinator and state manager

**Key Features**:
- State-based menu navigation
- Agent selection and configuration
- Render mode support (full/headless)
- Clean separation between UI and game logic

### Game Environment (`ai_platform_trainer/game/`)

**Purpose**: Provides a unified interface for agents to interact with the game world.

**Components**:
- `environment.py`: Core environment abstraction
- `rewards.py`: Reward computation for RL training

**Key Features**:
- Normalized state observations
- Consistent action processing
- Episode management
- Reward signal generation

### Agent Architecture (`ai_platform_trainer/agents/`)

**Purpose**: Defines the interface and implementations for AI agents.

**Components**:
- `base_agent.py`: Abstract base class for all agents
- `sl/agent.py`: Supervised learning agent implementation
- `rl/` (Phase 2): Reinforcement learning agents

**Key Features**:
- Consistent agent interface
- Modular agent implementations
- Easy extensibility for new agent types

## Data Flow

### Menu Navigation Flow
```
User Input → Menu System → Agent Selection → Game Configuration → Game Loop
```

### Game Loop Flow
```
Game Loop → Environment → Agent → Action → Environment → Reward → Display
```

### Agent Interaction Flow
```
Environment.get_observation() → Agent.select_action() → Environment.step() → Agent.record_outcome()
```

## Design Principles

### 1. Separation of Concerns
- **UI**: Handles presentation and user interaction
- **Game**: Manages game state and rules
- **Agents**: Implement AI decision-making logic

### 2. Modularity
- Each component has a clear, single responsibility
- Components communicate through well-defined interfaces
- Easy to test and maintain individual components

### 3. Extensibility
- New agent types can be added without modifying existing code
- New game modes can be integrated through the environment interface
- Plugin-style architecture for future enhancements

### 4. Backwards Compatibility
- All existing functionality is preserved
- Existing code paths continue to work unchanged
- Gradual migration to new architecture

## State Management

### Menu States
```python
class MenuState(Enum):
    MAIN = "main"
    RL_OPTIONS = "rl_options"
    RL_DIFFICULTY = "rl_difficulty"
    HELP = "help"
    GAME = "game"
```

### Game States
- **MENU**: Menu system is active
- **PLAYING**: Game is running with active agent
- **PAUSED**: Game is paused
- **GAME_OVER**: Episode has ended

## Agent Interface

All agents implement the `BaseAgent` interface:

```python
class BaseAgent(ABC):
    @abstractmethod
    def select_action(self, observation: Dict[str, Any]) -> Tuple[float, float]:
        """Select an action based on the current observation."""
        pass
    
    @abstractmethod
    def reset(self) -> None:
        """Reset the agent state for a new episode."""
        pass
```

## Environment Interface

The environment provides a consistent interface for all agents:

```python
class GameEnvironment:
    def get_observation(self) -> Dict[str, Any]:
        """Get normalized game state observation."""
        pass
    
    def step(self, action: Tuple[float, float]) -> Tuple[Dict[str, Any], float, bool]:
        """Execute action and return new state, reward, and done flag."""
        pass
    
    def reset(self) -> Dict[str, Any]:
        """Reset environment for new episode."""
        pass
```

## Configuration Management

Configuration is handled through:
- Environment variables
- Configuration files
- Command-line arguments
- Runtime parameters

## Error Handling

The architecture includes comprehensive error handling:
- Graceful degradation when components fail
- Logging of errors and warnings
- Fallback mechanisms for critical failures
- User-friendly error messages

## Performance Considerations

- Efficient state representation
- Minimal overhead for agent interface
- Optimized rendering pipeline
- Memory management for long-running sessions

## Security Considerations

- Input validation for user interactions
- Safe handling of model files
- Proper resource cleanup
- Protection against malicious inputs

## Testing Strategy

- Unit tests for individual components
- Integration tests for component interactions
- End-to-end tests for complete workflows
- Performance benchmarks for critical paths

## Future Extensibility

The architecture is designed to support:
- Multiple RL algorithms (DQN, PPO, etc.)
- Different game environments
- Multi-agent scenarios
- Distributed training
- Real-time visualization tools
