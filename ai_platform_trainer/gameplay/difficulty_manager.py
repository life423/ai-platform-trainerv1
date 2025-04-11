"""
Difficulty management for AI Platform Trainer.

This module defines the DifficultyManager class that controls game difficulty scaling
over time, affecting enemy speed, spawn rates, and other parameters.
"""
import logging


class DifficultyManager:
    """
    Manages game difficulty scaling over time.
    
    Attributes:
        current_level (int): Current difficulty level
        elapsed_time (int): Total elapsed game time in milliseconds
        enemy_speed_multiplier (float): Factor to multiply enemy speed by
        spawn_rate_multiplier (float): Factor to multiply spawn rates by
        difficulty_thresholds (list): Time thresholds for difficulty increases
    """
    
    def __init__(self):
        """Initialize the difficulty manager with default settings."""
        self.current_level = 1
        self.elapsed_time = 0
        self.enemy_speed_multiplier = 1.0
        self.spawn_rate_multiplier = 1.0
        
        # Thresholds for increasing difficulty (in milliseconds)
        # Every minute, increase difficulty
        self.difficulty_thresholds = [
            60000,    # 1 minute
            120000,   # 2 minutes
            180000,   # 3 minutes
            240000,   # 4 minutes
            300000,   # 5 minutes
        ]
        
        # Maximum enemy count at each difficulty level
        self.max_enemies_per_level = [
            3,  # Level 1: 3 enemies
            4,  # Level 2: 4 enemies
            5,  # Level 3: 5 enemies
            6,  # Level 4: 6 enemies
            8,  # Level 5: 8 enemies
            10,  # Level 6: 10 enemies
        ]
        
        # PowerUp spawn interval at each difficulty level (ms)
        self.powerup_interval_per_level = [
            15000,  # Level 1: 15 seconds
            13000,  # Level 2: 13 seconds
            11000,  # Level 3: 11 seconds
            10000,  # Level 4: 10 seconds
            8000,   # Level 5: 8 seconds
            7000,   # Level 6: 7 seconds
        ]
        
        logging.info("DifficultyManager initialized at level 1")
    
    def update(self, current_time: int, frame_time: int) -> bool:
        """
        Update difficulty based on elapsed time.
        
        Args:
            current_time: Current game time in milliseconds
            frame_time: Time elapsed since last frame in milliseconds
            
        Returns:
            bool: True if difficulty level changed, False otherwise
        """
        self.elapsed_time += frame_time
        
        # Check if we should increase difficulty
        if self.current_level <= len(self.difficulty_thresholds):
            next_threshold = self.difficulty_thresholds[self.current_level - 1]
            
            if self.elapsed_time >= next_threshold:
                self.increase_difficulty()
                return True
                
        return False
    
    def increase_difficulty(self) -> None:
        """Increase the difficulty level and update all multipliers."""
        self.current_level += 1
        
        # Increase enemy speed with each level
        self.enemy_speed_multiplier = 1.0 + (self.current_level - 1) * 0.15
        
        # Increase spawn rate with each level
        self.spawn_rate_multiplier = 1.0 + (self.current_level - 1) * 0.2
        
        logging.info(
            f"Difficulty increased to level {self.current_level}. "
            f"Speed: {self.enemy_speed_multiplier:.2f}x, "
            f"Spawn rate: {self.spawn_rate_multiplier:.2f}x"
        )
    
    def get_current_parameters(self) -> dict:
        """
        Get the current difficulty parameters.
        
        Returns:
            dict: Dictionary of all current difficulty parameters
        """
        return {
            'level': self.current_level,
            'enemy_speed_multiplier': self.enemy_speed_multiplier,
            'spawn_rate_multiplier': self.spawn_rate_multiplier,
            'max_enemies': self.get_max_enemies(),
            'powerup_interval': self.get_powerup_interval()
        }
    
    def get_max_enemies(self) -> int:
        """
        Get the maximum number of enemies for the current difficulty level.
        
        Returns:
            int: Maximum number of enemies
        """
        level_index = min(self.current_level - 1, len(self.max_enemies_per_level) - 1)
        return self.max_enemies_per_level[level_index]
    
    def get_powerup_interval(self) -> int:
        """
        Get the powerup spawn interval for the current difficulty level.
        
        Returns:
            int: Powerup spawn interval in milliseconds
        """
        level_index = min(self.current_level - 1, len(self.powerup_interval_per_level) - 1)
        return self.powerup_interval_per_level[level_index]
    
    def apply_enemy_speed(self, base_speed: float) -> float:
        """
        Apply the current enemy speed multiplier to a base speed.
        
        Args:
            base_speed: The base enemy speed
            
        Returns:
            float: The adjusted enemy speed
        """
        return base_speed * self.enemy_speed_multiplier
    
    def apply_spawn_rate(self, base_interval: int) -> int:
        """
        Apply the current spawn rate multiplier to a base interval.
        The higher the spawn rate multiplier, the lower the interval.
        
        Args:
            base_interval: The base interval in milliseconds
            
        Returns:
            int: The adjusted interval in milliseconds
        """
        # Note: Higher spawn rate = lower interval
        return int(base_interval / self.spawn_rate_multiplier)
