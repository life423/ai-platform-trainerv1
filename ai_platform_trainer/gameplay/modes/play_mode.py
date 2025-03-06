# file: ai_platform_trainer/gameplay/modes/play_mode.py

import logging
import random
import pygame
from typing import Any

from ai_platform_trainer.gameplay.missile_ai_controller import update_missile_ai
from ai_platform_trainer.gameplay.enemy_manager import EnemyManager
from ai_platform_trainer.gameplay.powerup_manager import PowerupManager
from ai_platform_trainer.gameplay.config import config as game_config


class PlayMode:
    def __init__(self, game):
        """
        Holds 'play' mode logic for the game.
        """
        self.game = game
        
        # Ensure collision manager has dependencies
        if self.game.collision_manager.missile_manager is None:
            self.game.collision_manager.set_missile_manager(
                self.game.missile_manager
            )
            
        # Initialize enemy manager (if not already done)
        self.enemy_manager = EnemyManager(
            self.game.screen_width,
            self.game.screen_height,
            self.game.player.position
        )
        
        # Initialize powerup manager
        self.powerup_manager = PowerupManager(
            self.game.screen_width,
            self.game.screen_height
        )
        
        # Track last enemy spawn time
        self.last_enemy_spawn = 0
        
        # Game state tracking
        self.level = 1
        self.game_over = False

    def update(self, current_time: int) -> None:
        """
        The main update loop for 'play' mode, handling multiple enemies,
        scoring, power-ups, and game progression.
        """
        # Check for game over
        if self.game_over:
            # Handle game over state
            keys = pygame.key.get_pressed()
            if keys[pygame.K_r]:  # Press R to restart
                self._restart_game()
            return
            
        # 1) Update player
        if self.game.player and not self.game.player.handle_input():
            logging.info("Player requested to quit. Exiting game.")
            self.game.running = False
            return
            
        # Update player state (power-ups, invincibility)
        self.game.player.update(current_time)
        
        # 2) Update all enemies through the enemy manager
        self.enemy_manager.update(
            current_time,
            self.game.player.position,
            self.game.player.step,
            self.game.missile_manager.missiles
        )
        
        # 3) Check for collisions
        self._handle_collisions(current_time)
        
        # 4) Missile AI - use closest enemy position for guidance
        closest_enemy_pos = self.enemy_manager.get_closest_enemy_pos()
        if (
            self.game.missile_model
            and self.game.player
            and self.game.missile_manager.missiles
        ):
            update_missile_ai(
                self.game.missile_manager.missiles,
                self.game.player.position,
                closest_enemy_pos,
                self.game._missile_input,
                self.game.missile_model
            )
            
        # 5) Update powerups
        self.powerup_manager.update(current_time, self.game.player)
        
        # 6) Check for level advancement
        self._check_level_advancement()
        
        # 7) Update missiles
        self.game.missile_manager.update(current_time)
    
    def _handle_collisions(self, current_time: int) -> None:
        """
        Handle collisions between player, enemies, and missiles.
        
        Args:
            current_time: Current game time in milliseconds
        """
        # Process missile-enemy collisions
        for missile_idx, missile in enumerate(self.game.missile_manager.missiles):
            # Check collision with each enemy
            for enemy_idx, enemy in enumerate(self.enemy_manager.enemies):
                if self._check_collision(missile, enemy):
                    # Handle enemy taking damage/destruction
                    points = self.enemy_manager.handle_collision(enemy_idx, missile=True)
                    
                    # Award points to player
                    if points > 0:
                        self.game.player.add_score(points)
                        
                        # Try to spawn power-up at enemy position
                        self.powerup_manager.spawn_powerup_at_position(
                            enemy.pos["x"], enemy.pos["y"], current_time
                        )
                    
                    # Remove the missile
                    self.game.missile_manager.missiles[missile_idx].active = False
                    break  # Missile can only hit one enemy
                    
        # Process player-enemy collisions
        for enemy_idx, enemy in enumerate(self.enemy_manager.enemies):
            if self._check_collision_with_player(enemy):
                # Player takes damage
                damage_taken = self.game.player.take_damage(current_time)
                
                if damage_taken:
                    # Check if player has run out of health
                    if self.game.player.health <= 0:
                        self._handle_game_over()
                    
                    # Enemy that hits player is destroyed
                    self.enemy_manager.handle_collision(enemy_idx, missile=False)
                    break  # Player can only collide with one enemy at a time
    
    def _check_collision(self, missile: Any, enemy: Any) -> bool:
        """
        Check if a missile and enemy are colliding.
        
        Args:
            missile: Missile object
            enemy: Enemy object
            
        Returns:
            True if collision detected
        """
        # Basic rectangle collision
        missile_rect = pygame.Rect(
            missile.pos["x"] - missile.size // 2,
            missile.pos["y"] - missile.size // 2,
            missile.size,
            missile.size
        )
        
        enemy_rect = pygame.Rect(
            enemy.pos["x"],
            enemy.pos["y"],
            enemy.size,
            enemy.size
        )
        
        return missile_rect.colliderect(enemy_rect)
    
    def _check_collision_with_player(self, enemy: Any) -> bool:
        """
        Check if player is colliding with an enemy.
        
        Args:
            enemy: Enemy object
            
        Returns:
            True if collision detected
        """
        player_rect = pygame.Rect(
            self.game.player.position["x"],
            self.game.player.position["y"],
            self.game.player.size,
            self.game.player.size
        )
        
        enemy_rect = pygame.Rect(
            enemy.pos["x"],
            enemy.pos["y"],
            enemy.size,
            enemy.size
        )
        
        return player_rect.colliderect(enemy_rect)
    
    def _check_level_advancement(self) -> None:
        """
        Check if player has earned enough points to advance to the next level.
        """
        points_needed = self.level * game_config.POINTS_PER_LEVEL
        
        if self.game.player.score >= points_needed:
            # Advance to next level
            self.level += 1
            logging.info(f"Advanced to level {self.level}!")
            
            # Make game harder by decreasing spawn interval
            spawn_reduction = min(0.8, 0.95 - (self.level * 0.05))  # Max 80% reduction
            self.enemy_manager.spawn_interval = int(
                game_config.ENEMY_SPAWN_INTERVAL * spawn_reduction
            )
    
    def _handle_game_over(self) -> None:
        """Handle game over state."""
        self.game_over = True
        logging.info(f"Game over! Final score: {self.game.player.score}, Level: {self.level}")
        
    def _restart_game(self) -> None:
        """Restart the game after game over."""
        # Reset player
        self.game.player.reset()
        
        # Clear enemies
        self.enemy_manager.clear_all()
        
        # Clear powerups
        self.powerup_manager.clear_all()
        
        # Clear missiles
        self.game.missile_manager.clear_all()
        
        # Reset game state
        self.level = 1
        self.game_over = False
        self.enemy_manager.spawn_interval = game_config.ENEMY_SPAWN_INTERVAL
        
        logging.info("Game restarted!")
