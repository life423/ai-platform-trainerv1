import pygame
import logging
import random
import math
from typing import Optional, Dict, List

from ai_platform_trainer.entities.missile import Missile
from ai_platform_trainer.gameplay.config import config as game_config


class PlayerPlay:
    def __init__(self, screen_width: int, screen_height: int):
        self.screen_width = screen_width
        self.screen_height = screen_height
        self.size = 50
        self.color = (0, 0, 139)  # Dark Blue
        self.position = {"x": screen_width // 4, "y": screen_height // 2}
        self.step = 5
        
        # Health and invincibility
        self.max_health = game_config.PLAYER_MAX_HEALTH
        self.health = self.max_health
        self.is_invincible = False
        self.invincible_until = 0
        self.invincible_flash = False  # For visual effect when invincible
        self.invincible_flash_interval = 100  # Flash every 100ms
        self.last_flash_time = 0
        
        # Power-ups
        self.active_powerups = []  # List of active power-ups
        self.powerup_end_times = {}  # When each power-up expires
        
        # Scoring
        self.score = 0

    def reset(self) -> None:
        """Reset player to initial state."""
        self.position = {"x": self.screen_width // 4, "y": self.screen_height // 2}
        self.health = self.max_health
        self.is_invincible = False
        self.invincible_until = 0
        self.active_powerups = []
        self.powerup_end_times = {}
        self.score = 0
        logging.info("Player has been reset to initial state.")

    def handle_input(self) -> bool:
        keys = pygame.key.get_pressed()

        # Calculate step with potential speed boost
        current_step = self.step
        if "speed_boost" in self.active_powerups:
            current_step *= 1.5  # 50% speed boost
            
        # WASD / Arrow key movement
        if keys[pygame.K_LEFT] or keys[pygame.K_a]:
            self.position["x"] -= current_step
        if keys[pygame.K_RIGHT] or keys[pygame.K_d]:
            self.position["x"] += current_step
        if keys[pygame.K_UP] or keys[pygame.K_w]:
            self.position["y"] -= current_step
        if keys[pygame.K_DOWN] or keys[pygame.K_s]:
            self.position["y"] += current_step

        # Wrap-around logic
        if self.position["x"] < -self.size:
            self.position["x"] = self.screen_width
        elif self.position["x"] > self.screen_width:
            self.position["x"] = -self.size
        if self.position["y"] < -self.size:
            self.position["y"] = self.screen_height
        elif self.position["y"] > self.screen_height:
            self.position["y"] = -self.size

        return True

    def shoot_missile(self, enemy_pos: Optional[Dict[str, float]] = None, 
                     missile_manager=None) -> None:
        """
        Create a new missile and add it to the missile manager.
        Now takes an optional missile_manager parameter.
        """
        if missile_manager is None:
            logging.error("No missile manager provided to shoot_missile")
            return
            
        # Previously we checked if there's already a missile
        # Now the MissileManager handles multiple missiles
        missile_start_x = self.position["x"] + self.size // 2
        missile_start_y = self.position["y"] + self.size // 2

        birth_time = pygame.time.get_ticks()
        # Use the default 7 seconds (7000ms) lifespan 
        missile_lifespan = 7000  # 7 seconds
        missile_speed = 5.0

        # Determine initial velocity based on enemy position if available
        if enemy_pos is not None:
            # Calculate the angle toward the enemy's position
            angle = math.atan2(
                enemy_pos["y"] - missile_start_y, 
                enemy_pos["x"] - missile_start_x
            )
            # Add a small random deviation to simulate inaccuracy
            angle += random.uniform(-0.1, 0.1)  # deviation in radians
            vx = missile_speed * math.cos(angle)
            vy = missile_speed * math.sin(angle)
        else:
            vx = missile_speed
            vy = 0.0

        # Create a new missile object with calculated initial velocity and specified lifespan
        missile = Missile(
            x=missile_start_x,
            y=missile_start_y,
            speed=missile_speed,
            vx=vx,
            vy=vy,
            birth_time=birth_time,
            lifespan=missile_lifespan,
        )
        
        # Add missile to manager instead of self.missiles
        missile_manager.spawn_missile(missile)
        logging.info("Play mode: Shot a missile with 7-second lifespan and dynamic initial direction.")

    # The update_missiles and draw_missiles methods are now handled by MissileManager

    def update(self, current_time: int) -> None:
        """
        Update player state.
        
        Args:
            current_time: Current game time in milliseconds
        """
        # Check for expired power-ups
        expired_powerups = []
        for powerup, end_time in self.powerup_end_times.items():
            if current_time >= end_time:
                expired_powerups.append(powerup)
                
        # Remove expired power-ups
        for powerup in expired_powerups:
            self.active_powerups.remove(powerup)
            del self.powerup_end_times[powerup]
            logging.info(f"Power-up {powerup} has expired")
            
        # Update invincibility status
        if self.is_invincible and current_time >= self.invincible_until:
            self.is_invincible = False
            logging.debug("Player is no longer invincible")
            
        # Update flash effect for invincibility
        if self.is_invincible:
            if current_time - self.last_flash_time >= self.invincible_flash_interval:
                self.invincible_flash = not self.invincible_flash
                self.last_flash_time = current_time
    
    def take_damage(self, current_time: int) -> bool:
        """
        Attempt to damage the player.
        
        Args:
            current_time: Current game time in milliseconds
            
        Returns:
            True if damage was taken, False if invincible or shield active
        """
        # Check if player has a shield or is invincible
        if "shield" in self.active_powerups:
            # Shield absorbs one hit and is then removed
            self.active_powerups.remove("shield")
            if "shield" in self.powerup_end_times:
                del self.powerup_end_times["shield"]
            logging.info("Shield absorbed a hit!")
            return False
            
        if self.is_invincible:
            return False
            
        # Take damage
        self.health -= 1
        logging.info(f"Player took damage! Health: {self.health}/{self.max_health}")
        
        # Check if player is still alive
        if self.health <= 0:
            return True  # Damage was dealt, player may be dead
            
        # Apply invincibility period
        self.is_invincible = True
        self.invincible_until = current_time + game_config.PLAYER_INVINCIBLE_TIME
        
        return True  # Damage was dealt
    
    def add_score(self, points: int) -> None:
        """
        Add points to the player's score.
        
        Args:
            points: Points to add
        """
        self.score += points
        logging.info(f"Player score increased by {points}. New score: {self.score}")
    
    def add_powerup(self, powerup_type: str, current_time: int) -> None:
        """
        Add a power-up to the player.
        
        Args:
            powerup_type: Type of power-up to add
            current_time: Current game time in milliseconds
        """
        if powerup_type not in game_config.POWERUP_TYPES:
            logging.warning(f"Unknown power-up type: {powerup_type}")
            return
            
        # Add to active power-ups
        if powerup_type not in self.active_powerups:
            self.active_powerups.append(powerup_type)
        
        # Set expiration time
        self.powerup_end_times[powerup_type] = current_time + game_config.POWERUP_DURATION
        
        logging.info(f"Player acquired power-up: {powerup_type}")
        
    def has_powerup(self, powerup_type: str) -> bool:
        """
        Check if player has a specific power-up active.
        
        Args:
            powerup_type: Type of power-up to check
            
        Returns:
            True if the power-up is active
        """
        return powerup_type in self.active_powerups
        
    def draw(self, screen: pygame.Surface) -> None:
        # Skip drawing if invincible and in a "flash off" frame
        if self.is_invincible and self.invincible_flash:
            # Draw at half transparency when invincible
            s = pygame.Surface((self.size, self.size), pygame.SRCALPHA)
            s.fill((self.color[0], self.color[1], self.color[2], 128))  # 50% transparent
            screen.blit(s, (self.position["x"], self.position["y"]))
        else:
            pygame.draw.rect(
                screen,
                self.color,
                (self.position["x"], self.position["y"], self.size, self.size),
            )
            
        # Draw power-up indicators
        if "shield" in self.active_powerups:
            # Draw a circle around the player
            pygame.draw.circle(
                screen,
                (100, 100, 255),  # Light blue
                (
                    int(self.position["x"] + self.size // 2),
                    int(self.position["y"] + self.size // 2)
                ),
                int(self.size * 0.75),  # Slightly larger than player
                3  # Line width
            )
            
        # Missiles are now drawn by the MissileManager
