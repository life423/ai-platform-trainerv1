
"""
Pure NumPy Enemy RL Training with GPU Acceleration

No PyTorch dependency - uses Evolution Strategy optimization
with your existing CUDA C++ backend for physics simulation.
"""
import logging
import os
import time
from typing import Dict, Optional, Tuple

import numpy as np

from ai_platform_trainer.ai.models.gpu_rl_environment import GPURLEnvironment

logger = logging.getLogger(__name__)


class SimpleEnemyAgent:
    """Simple neural network using pure NumPy."""
    
    def __init__(self, obs_dim: int = 4, action_dim: int = 2, hidden_dim: int = 64):
        """Initialize neural network weights.
        
        Args:
            obs_dim: Observation dimension
            action_dim: Action dimension  
            hidden_dim: Hidden layer dimension
        """
        self.obs_dim = obs_dim
        self.action_dim = action_dim
        self.hidden_dim = hidden_dim
        
        # Initialize weights with Xavier initialization
        self.w1 = np.random.randn(obs_dim, hidden_dim) * np.sqrt(2.0 / obs_dim)
        self.b1 = np.zeros(hidden_dim)
        self.w2 = np.random.randn(hidden_dim, hidden_dim) * np.sqrt(2.0 / hidden_dim)
        self.b2 = np.zeros(hidden_dim)
        self.w3 = np.random.randn(hidden_dim, action_dim) * np.sqrt(2.0 / hidden_dim)
        self.b3 = np.zeros(action_dim)
        
        logger.info(f"Initialized agent: {obs_dim}->{hidden_dim}->{hidden_dim}->{action_dim}")
    
    def forward(self, x: np.ndarray) -> np.ndarray:
        """Forward pass through the network.
        
        Args:
            x: Input observations [batch_size, obs_dim]
            
        Returns:
            Actions [batch_size, action_dim]
        """
        # Ensure input is 2D
        if x.ndim == 1:
            x = x.reshape(1, -1)
        
        # First layer with ReLU
        h1 = np.maximum(0, x @ self.w1 + self.b1)
        
        # Second layer with ReLU  
        h2 = np.maximum(0, h1 @ self.w2 + self.b2)
        
        # Output layer with tanh (bounded actions)
        actions = np.tanh(h2 @ self.w3 + self.b3)
        
        return actions
    
    def get_weights(self) -> np.ndarray:
        """Get flattened weights for evolution strategy."""
        weights = []
        for w in [self.w1, self.b1, self.w2, self.b2, self.w3, self.b3]:
            weights.append(w.flatten())
        return np.concatenate(weights)
    
    def set_weights(self, weights: np.ndarray) -> None:
        """Set weights from flattened array."""
        idx = 0
        
        # w1
        size = self.obs_dim * self.hidden_dim
        self.w1 = weights[idx:idx+size].reshape(self.obs_dim, self.hidden_dim)
        idx += size
        
        # b1
        self.b1 = weights[idx:idx+self.hidden_dim]
        idx += self.hidden_dim
        
        # w2
        size = self.hidden_dim * self.hidden_dim
        self.w2 = weights[idx:idx+size].reshape(self.hidden_dim, self.hidden_dim)
        idx += size
        
        # b2
        self.b2 = weights[idx:idx+self.hidden_dim]
        idx += self.hidden_dim
        
        # w3
        size = self.hidden_dim * self.action_dim
        self.w3 = weights[idx:idx+size].reshape(self.hidden_dim, self.action_dim)
        idx += size
        
        # b3
        self.b3 = weights[idx:idx+self.action_dim]
    
    def save(self, path: str) -> None:
        """Save model to file."""
        os.makedirs(os.path.dirname(path), exist_ok=True)
        np.savez(path, 
                 w1=self.w1, b1=self.b1,
                 w2=self.w2, b2=self.b2, 
                 w3=self.w3, b3=self.b3,
                 obs_dim=self.obs_dim,
                 action_dim=self.action_dim,
                 hidden_dim=self.hidden_dim)
        logger.info(f"Model saved to {path}")
    
    def load(self, path: str) -> None:
        """Load model from file."""
        data = np.load(path)
        self.w1 = data['w1']
        self.b1 = data['b1']
        self.w2 = data['w2']
        self.b2 = data['b2']
        self.w3 = data['w3']
        self.b3 = data['b3']
        
        # Load dimensions if available
        if 'obs_dim' in data:
            self.obs_dim = int(data['obs_dim'])
            self.action_dim = int(data['action_dim'])
            self.hidden_dim = int(data['hidden_dim'])
        
        logger.info(f"Model loaded from {path}")


class EvolutionStrategy:
    """Evolution Strategy optimizer for neural network."""
    
    def __init__(self, num_weights: int, population_size: int = 50, 
                 learning_rate: float = 0.01, noise_std: float = 0.1):
        """Initialize ES optimizer.
        
        Args:
            num_weights: Number of parameters to optimize
            population_size: Size of the population
            learning_rate: Learning rate for weight updates
            noise_std: Standard deviation of noise for mutations
        """
        self.num_weights = num_weights
        self.population_size = population_size
        self.learning_rate = learning_rate
        self.noise_std = noise_std
        
        logger.info(f"ES Optimizer: pop_size={population_size}, lr={learning_rate}")
    
    def optimize(self, weights: np.ndarray, fitness_func) -> Tuple[np.ndarray, float]:
        """Run one step of evolution strategy.
        
        Args:
            weights: Current weights
            fitness_func: Function that evaluates fitness given weights
            
        Returns:
            Tuple of (new_weights, best_fitness)
        """
        # Generate noise for each individual
        noise = np.random.randn(self.population_size, self.num_weights) * self.noise_std
        
        # Evaluate population
        fitness_scores = []
        for i in range(self.population_size):
            candidate_weights = weights + noise[i]
            fitness = fitness_func(candidate_weights)
            fitness_scores.append(fitness)
        
        fitness_scores = np.array(fitness_scores)
        
        # Rank individuals
        ranks = np.argsort(fitness_scores)[::-1]  # Descending order
        
        # Update weights using top performers
        top_k = self.population_size // 4  # Use top 25%
        weight_update = np.zeros_like(weights)
        
        for i in range(top_k):
            idx = ranks[i]
            weight_update += noise[idx] * fitness_scores[idx]
        
        weight_update /= top_k
        new_weights = weights + self.learning_rate * weight_update
        
        best_fitness = fitness_scores[ranks[0]]
        
        return new_weights, best_fitness


def evaluate_agent(agent: SimpleEnemyAgent, env: GPURLEnvironment, 
                  num_episodes: int = 5, max_steps: int = 100) -> float:
    """Evaluate agent performance.
    
    Args:
        agent: Agent to evaluate
        env: Environment to test in
        num_episodes: Number of episodes to run
        max_steps: Maximum steps per episode
        
    Returns:
        Average reward over episodes
    """
    total_reward = 0.0
    
    for _ in range(num_episodes):
        obs = env.reset()
        episode_reward = 0.0
        
        for _ in range(max_steps):
            actions = agent.forward(obs)
            obs, rewards, dones = env.step(actions)
            episode_reward += rewards.mean()
            
            if dones.any():
                break
        
        total_reward += episode_reward
    
    return total_reward / num_episodes


def train_enemy_rl_gpu(num_generations: int = 200, batch_size: int = 128, 
                      population_size: int = 50, learning_rate: float = 0.01,
                      model_save_path: str = None) -> SimpleEnemyAgent:
    """Train enemy using Evolution Strategy with GPU acceleration.
    
    Args:
        num_generations: Number of ES generations
        batch_size: Environment batch size
        population_size: ES population size
        learning_rate: Learning rate for ES
        model_save_path: Path to save the trained model
        
    Returns:
        Trained agent
    """
    logger.info("Starting GPU RL training with Evolution Strategy")
    logger.info(f"Generations: {num_generations}, Batch size: {batch_size}")
    logger.info(f"Population: {population_size}, Learning rate: {learning_rate}")
    
    # Create environment
    env = GPURLEnvironment(batch_size=batch_size)
    
    # Get environment dimensions
    obs_shape = env.get_observation_shape()
    action_shape = env.get_action_shape()
    obs_dim = obs_shape[0] if isinstance(obs_shape, tuple) else obs_shape
    action_dim = action_shape[0] if isinstance(action_shape, tuple) else action_shape
    
    logger.info(f"Environment: obs_dim={obs_dim}, action_dim={action_dim}")
    
    # Create agent
    agent = SimpleEnemyAgent(obs_dim=obs_dim, action_dim=action_dim)
    
    # Create ES optimizer
    num_weights = len(agent.get_weights())
    es = EvolutionStrategy(num_weights, population_size, learning_rate)
    
    # Training loop
    best_fitness = float('-inf')
    weights = agent.get_weights()
    
    # Fitness evaluation function
    def fitness_func(candidate_weights):
        agent.set_weights(candidate_weights)
        return evaluate_agent(agent, env, num_episodes=3, max_steps=50)
    
    logger.info("Starting training loop...")
    
    for generation in range(num_generations):
        start_time = time.time()
        
        # Run ES optimization
        weights, fitness = es.optimize(weights, fitness_func)
        
        # Update best fitness
        if fitness > best_fitness:
            best_fitness = fitness
            # Update agent with best weights
            agent.set_weights(weights)
        
        generation_time = time.time() - start_time
        
        # Log progress
        if generation % 10 == 0:
            logger.info(f"Generation {generation:3d}: "
                       f"fitness={fitness:7.3f}, "
                       f"best={best_fitness:7.3f}, "
                       f"time={generation_time:.2f}s")
    
    # Final evaluation
    agent.set_weights(weights)
    final_fitness = evaluate_agent(agent, env, num_episodes=10, max_steps=200)
    
    logger.info(f"Training completed!")
    logger.info(f"Final fitness: {final_fitness:.3f}")
    
    # Save model
    if model_save_path is None:
        model_save_path = 'ai_platform_trainer/models/enemy_rl_gpu.npz'
    
    agent.save(model_save_path)
    
    return agent


def test_trained_agent(model_path: str = 'ai_platform_trainer/models/enemy_rl_gpu.npz'):
    """Test a trained agent."""
    logger.info("Testing trained agent...")
    
    try:
        # Load agent
        env = GPURLEnvironment(batch_size=4)
        obs_dim = env.get_observation_shape()[0]
        action_dim = env.get_action_shape()[0]
        
        agent = SimpleEnemyAgent(obs_dim=obs_dim, action_dim=action_dim)
        agent.load(model_path)
        
        # Test performance
        fitness = evaluate_agent(agent, env, num_episodes=5, max_steps=200)
        logger.info(f"Trained agent fitness: {fitness:.3f}")
        
        # Test single step
        obs = env.reset()
        actions = agent.forward(obs)
        logger.info(f"Sample action: {actions[0]}")
        
        return agent
        
    except Exception as e:
        logger.error(f"Failed to test trained agent: {e}")
        return None


if __name__ == "__main__":
    # Configure logging
    logging.basicConfig(level=logging.INFO)
    
    # Train agent
    agent = train_enemy_rl_gpu(
        num_generations=100,
        batch_size=64,
        population_size=30,
        learning_rate=0.02
    )
    
    # Test trained agent
    test_trained_agent()
    test_trained_agent()
