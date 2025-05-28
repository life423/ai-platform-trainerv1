"""Mock environment implementation for CPU fallback."""
from typing import List, Optional
import numpy as np


class MockEnvironmentConfig:
    """Configuration class that matches the C++/CUDA interface."""

    def __init__(self):
        self.grid_width = 32
        self.grid_height = 32
        self.max_episode_length = 200
        self.wall_density = 0.1


class MockAgentState:
    """Agent state that matches the C++/CUDA interface."""

    def __init__(self, x: int = 0, y: int = 0):
        self.x = x
        self.y = y
        self.episode_length = 0
        self.total_reward = 0.0


class MockObservation:
    """Observation that matches the C++/CUDA interface."""

    def __init__(self, x: int = 0, y: int = 0, episode_length: int = 0):
        self.x = x
        self.y = y
        self.episode_length = episode_length


class MockCudaDeviceInfo:
    """Device info that matches the C++/CUDA interface."""

    def __init__(self):
        self.device_count = 0
        self.device_name = "CPU (fallback)"
        self.memory_total = 0
        self.memory_free = 0


class MockEnvironment:
    """CPU-only grid-world environment that matches C++/CUDA interface."""

    def __init__(self, config: Optional[MockEnvironmentConfig] = None):
        """Initialize the mock environment."""
        if config is None:
            config = MockEnvironmentConfig()

        self.width = config.grid_width
        self.height = config.grid_height
        self.agents: List[MockAgentState] = []

    def initialize(self, batch_size: int):
        """Initialize for batch processing."""
        self.agents = []
        for _ in range(batch_size):
            state = MockAgentState(1, 1)
            self.agents.append(state)

    def reset(self) -> List[MockObservation]:
        """Reset the environment to initial state."""
        observations = []
        for agent in self.agents:
            agent.x = 1
            agent.y = 1
            agent.episode_length = 0
            agent.total_reward = 0.0
            observations.append(MockObservation(agent.x, agent.y, 0))
        return observations

    def step(self, actions: List[int]):
        """Execute one environment step."""
        # Map action to movement
        moves = [(0, -1), (1, 0), (0, 1), (-1, 0)]

        observations = []
        rewards = []
        dones = []

        for i, action in enumerate(actions):
            agent = self.agents[i]
            dx, dy = moves[action % 4]

            # Calculate new position
            new_x = max(0, min(self.width - 1, agent.x + dx))
            new_y = max(0, min(self.height - 1, agent.y + dy))

            agent.x = new_x
            agent.y = new_y

            # Simple reward calculation
            goal_x, goal_y = self.width - 2, self.height - 2
            distance = abs(agent.x - goal_x) + abs(agent.y - goal_y)
            reward = -0.01 - 0.1 * (distance / (self.width + self.height))

            # Check if reached goal
            if agent.x == goal_x and agent.y == goal_y:
                reward = 10.0

            agent.episode_length += 1
            done = ((agent.x == goal_x and agent.y == goal_y) or
                    agent.episode_length >= 200)

            observations.append(MockObservation(agent.x, agent.y,
                                                agent.episode_length))
            rewards.append(reward)
            dones.append(done)

            # Reset if done
            if done:
                agent.x = 1
                agent.y = 1
                agent.episode_length = 0

        return observations, rewards, dones


if __name__ == "__main__":
    print("Testing Mock Environment...")
    config = MockEnvironmentConfig()
    env = MockEnvironment(config)
    env.initialize(1)

    observations = env.reset()
    print(f"Initial: ({observations[0].x}, {observations[0].y})")

    for i in range(5):
        actions = [1]  # Move right
        observations, rewards, dones = env.step(actions)
        obs = observations[0]
        print(f"Step {i+1}: ({obs.x},{obs.y}), "
              f"R={rewards[0]:.3f}, Done={dones[0]}")

    print("Test completed!")
