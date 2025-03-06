"""
Enemy Manager to handle multiple enemies in the game.

This module provides functionality to:
1. Manage a collection of enemies
2. Handle enemy spawning based on configurable intervals
3. Support different enemy types
4. Track enemy-related statistics
"""

import logging
import random
import pygame
from typing import List, Dict, Any, Optional

from ai_platform_trainer.gameplay.config import config as game_config
from ai_platform_trainer.entities.enemy_play import EnemyPlay
from ai_platform_trainer.gameplay.spawn_utils import find_valid_spawn_position


class EnemyManager:
    """
    Manages multiple enemies in the game, including their spawning,
    lifecycle, and related game mechanics.
    """
    
    def __init__(self, screen_width: int, screen_height: int, 
                 player_pos: Dict[str, float]) -> None:
        """
        Initialize the enemy manager.
        
        Args:
            screen_width: Width of the game screen
            screen_height: Height of the game screen
            player_pos: Player's current position
        """
        self.screen_width = screen_width
        self.screen_height = screen_height
        self.player_pos = player_pos
        
        # List to store active enemies
        self.enemies: List[EnemyPlay] = []
        
        # Enemy spawning settings
        self.next_spawn_time = 0
        self.spawn_interval = game_config.ENEMY_SPAWN_INTERVAL
        
        # Enemy types with their properties
        self.enemy_types = {
            "basic": {
                "size": 20, 
                "color": (255, 0, 0),  # Red
                "speed_factor": 1.0,
                "health": 1,
                "points": game_config.POINTS_PER_ENEMY
            },
            "fast": {
                "size": 15, 
                "color": (0, 255, 0),  # Green
                "speed_factor": 1.5,
                "health": 1,
                "points": game_config.POINTS_PER_ENEMY * 1.5
            },
            "tank": {
                "size": 25, 
                "color": (0, 0, 255),  # Blue
                "speed_factor": 0.7,
                "health": 3,
                "points": game_config.POINTS_PER_ENEMY * 2
            }
        }
        
        # Game statistics
        self.enemies_spawned = 0
        self.enemies_destroyed = 0
        self.score = 0
        
        logging.info("EnemyManager initialized")
    
    def update(self, current_time: int, player_pos: Dict[str, float], 
               player_step: float, missiles: List) -> None:
        """
        Update all enemies and handle spawning of new enemies.
        
        Args:
            current_time: Current game time in milliseconds
            player_pos: Current player position
            player_step: Player movement step size
            missiles: List of active missiles for avoidance
        """
        # Update player position reference
        self.player_pos = player_pos
        
        # Check if we need to spawn a new enemy
        if (current_time >= self.next_spawn_time and 
                len(self.enemies) < game_config.MAX_ENEMIES):
            self._spawn_enemy(current_time)
        
        # Update all active enemies
        for enemy in self.enemies:
            if enemy.active:
                enemy.update_movement(
                    player_pos["x"], 
                    player_pos["y"],
                    player_step,
                    current_time,
                    missiles
                )
                
                # Update fade-in if active
                if enemy.fading_in:
                    enemy.update_fade_in(current_time)
    
    def draw(self, screen: pygame.Surface) -> None:
        """
        Draw all active enemies on the screen.
        
        Args:
            screen: Pygame surface to draw on
        """
        for enemy in self.enemies:
            if enemy.active:
                enemy.draw(screen)
    
    def _spawn_enemy(self, current_time: int) -> None:
        """
        Spawn a new enemy.
        
        Args:
            current_time: Current game time in milliseconds
        """
        # Choose a random enemy type
        enemy_type = random.choice(game_config.ENEMY_TYPES)
        enemy_props = self.enemy_types[enemy_type]
        
        # Create a new enemy instance
        enemy = self._create_enemy(enemy_type, enemy_props)
        
        # Find a valid spawn position
        spawn_pos = find_valid_spawn_position(
            self.screen_width,
            self.screen_height,
            enemy.size,
            margin=game_config.WALL_MARGIN,
            min_dist=game_config.MIN_DISTANCE,
            other_pos=(self.player_pos["x"], self.player_pos["y"])
        )
        
        enemy.set_position(spawn_pos[0], spawn_pos[1])
        enemy.show(current_time)  # Show with fade-in effect
        
        # Add to enemies list
        self.enemies.append(enemy)
        self.enemies_spawned += 1
        
        # Set next spawn time
        self.next_spawn_time = current_time + self.spawn_interval
        
        logging.info(f"Spawned {enemy_type} enemy at {spawn_pos}")
    
    def _create_enemy(self, enemy_type: str, props: Dict[str, Any]) -> EnemyPlay:
        """
        Create a new enemy of the specified type.
        
        Args:
            enemy_type: Type of enemy to create
            props: Properties for the enemy
            
        Returns:
            A new enemy instance
        """
        from ai_platform_trainer.utils.model_manager import ModelManager
        
        # Get AI model for enemy
        model_manager = ModelManager()
        model = model_manager.get_model("enemy")
        
        if model is None:
            logging.error("Failed to load enemy model from model manager")
            raise RuntimeError("Could not load enemy model through ModelManager")
        
        # Create enemy with properties based on its type
        enemy = EnemyPlay(
            self.screen_width, 
            self.screen_height, 
            model
        )
        
        # Apply type-specific properties
        enemy.size = props["size"]
        enemy.color = props["color"]
        enemy.speed_multiplier = props["speed_factor"]
        enemy.health = props["health"]
        enemy.max_health = props["health"]
        enemy.points = props["points"]
        enemy.type = enemy_type  # Store the type for reference
        
        return enemy
    
    def handle_collision(self, enemy_index: int, missile: bool = False) -> int:
        """
        Handle collision with an enemy.
        
        Args:
            enemy_index: Index of the enemy in the enemies list
            missile: Whether the collision was with a missile
            
        Returns:
            Points earned from this collision
        """
        if enemy_index < 0 or enemy_index >= len(self.enemies):
            return 0
        
        enemy = self.enemies[enemy_index]
        points = 0
        
        if missile:
            # Reduce enemy health
            enemy.health -= 1
            
            if enemy.health <= 0:
                # Enemy destroyed
                enemy.active = False
                points = int(enemy.points)
                self.enemies_destroyed += 1
                self.score += points
                
                # Remove from list
                self.enemies.pop(enemy_index)
                
                logging.info(
                    f"Enemy destroyed, +{points} points. Total: {self.score}"
                )
        
        return points
    
    def clear_all(self) -> None:
        """
        Clear all enemies.
        """
        self.enemies.clear()
        self.next_spawn_time = 0
        logging.info("All enemies cleared")
    
    def get_closest_enemy_pos(self) -> Optional[Dict[str, float]]:
        """
        Get the position of the closest enemy to the player.
        
        Returns:
            Position of the closest enemy or None if no enemies
        """
        if not self.enemies:
            return None
        
        # Find closest enemy
        closest_enemy = min(
            self.enemies, 
            key=lambda e: self._calc_distance(e.pos, self.player_pos)
        )
        
        return closest_enemy.pos
    
    def _calc_distance(self, pos1: Dict[str, float], pos2: Dict[str, float]) -> float:
        """
        Calculate the distance between two positions.
        
        Args:
            pos1: First position
            pos2: Second position
            
        Returns:
            Distance between the positions
        """
        dx = pos1["x"] - pos2["x"]
        dy = pos1["y"] - pos2["y"]
        return (dx*dx + dy*dy) ** 0.5
