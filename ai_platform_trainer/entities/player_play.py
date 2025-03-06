import pygame
import logging
import random
import math
from typing import Optional, Dict
from ai_platform_trainer.entities.missile import Missile


class PlayerPlay:
    def __init__(self, screen_width: int, screen_height: int):
        self.screen_width = screen_width
        self.screen_height = screen_height
        self.size = 50
        self.color = (0, 0, 139)  # Dark Blue
        self.position = {"x": screen_width // 4, "y": screen_height // 2}
        self.step = 5
        # Missiles are now managed by MissileManager

    def reset(self) -> None:
        self.position = {"x": self.screen_width // 4, "y": self.screen_height // 2}
        logging.info("Player has been reset to the initial position.")

    def handle_input(self) -> bool:
        keys = pygame.key.get_pressed()

        # WASD / Arrow key movement
        if keys[pygame.K_LEFT] or keys[pygame.K_a]:
            self.position["x"] -= self.step
        if keys[pygame.K_RIGHT] or keys[pygame.K_d]:
            self.position["x"] += self.step
        if keys[pygame.K_UP] or keys[pygame.K_w]:
            self.position["y"] -= self.step
        if keys[pygame.K_DOWN] or keys[pygame.K_s]:
            self.position["y"] += self.step

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

    def draw(self, screen: pygame.Surface) -> None:
        pygame.draw.rect(
            screen,
            self.color,
            (self.position["x"], self.position["y"], self.size, self.size),
        )
        # Missiles are now drawn by the MissileManager
