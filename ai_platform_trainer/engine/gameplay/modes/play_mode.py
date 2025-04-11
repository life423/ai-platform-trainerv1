"""
Play mode game logic for AI Platform Trainer.

This module handles the play mode game loop and mechanics.
"""
import logging

import pygame

from ai_platform_trainer.engine.gameplay.difficulty_manager import DifficultyManager
from ai_platform_trainer.entities.behaviors.missile_ai_controller import update_missile_ai
from ai_platform_trainer.entities.components.power_up_manager import PowerUpManager

# Config and spawn utils will need to be refactored later
from ai_platform_trainer.gameplay.config import config
from ai_platform_trainer.gameplay.spawn_utils import find_valid_spawn_position


class PlayMode:
    def __init__(self, game):
        """
        Holds 'play' mode logic for the game.
        """
        self.game = game
        
        # Initialize difficulty manager
        self.difficulty_manager = DifficultyManager()
        
        # Initialize power-up manager
        self.powerup_manager = PowerUpManager(difficulty_manager=self.difficulty_manager)
        
        # Store last frame time for difficulty updates
        self.last_frame_time = pygame.time.get_ticks()

    def update(self, current_time: int) -> None:
        """
        The main update loop for 'play' mode, replacing old play_update() logic in game.py.
        """
        # Calculate frame time delta
        frame_time = current_time - self.last_frame_time
        self.last_frame_time = current_time
        
        # Update difficulty manager
        difficulty_changed = self.difficulty_manager.update(current_time, frame_time)
        if difficulty_changed:
            # Apply difficulty settings
            params = self.difficulty_manager.get_current_parameters()
            self.game.num_enemies = params['max_enemies']
            self.powerup_manager.spawn_interval = params['powerup_interval']
            
            logging.info(
                f"Difficulty level {params['level']}: "
                f"{self.game.num_enemies} enemies, "
                f"{params['powerup_interval']/1000}s powerup interval"
            )

        # 1) Player movement & input
        if self.game.player:
            self.game.player.handle_input()
            
            # Update power-ups and check for collisions
            if hasattr(self.game, 'screen_width') and hasattr(self.game, 'screen_height'):
                self.powerup_manager.update(
                    current_time,
                    self.game.screen_width,
                    self.game.screen_height,
                    self.game.player.position
                )
                
                # Check if player has collected any power-ups
                self.powerup_manager.check_collisions(self.game.player, current_time)

        # 2) Update all enemies in the list
        if hasattr(self.game, 'enemies') and self.game.enemies:
            for enemy in self.game.enemies:
                if enemy.visible:
                    try:
                        enemy.update_movement(
                            self.game.player.position["x"],
                            self.game.player.position["y"],
                            self.game.player.step,
                            current_time,
                        )
                    except Exception as e:
                        logging.error(f"Error updating enemy movement: {e}")
                        # Don't crash the game for one enemy's error
                        continue
                
                # Update fade-in effect for each enemy
                if enemy.fading_in:
                    enemy.update_fade_in(current_time)
                    
            logging.debug(f"Updated {len(self.game.enemies)} enemies in play mode.")
        # Backward compatibility for single enemy
        elif self.game.enemy:
            try:
                self.game.enemy.update_movement(
                    self.game.player.position["x"],
                    self.game.player.position["y"],
                    self.game.player.step,
                    current_time,
                )
                logging.debug("Single enemy movement updated in play mode.")
            except Exception as e:
                logging.error(f"Error updating enemy movement: {e}")
                self.game.running = False
                return

        # 3) Player-Enemy and Player-Obstacle collisions
        self.check_player_enemy_collisions(current_time)
        self.check_player_obstacle_collisions()
        
        # 3.5) Check enemy-obstacle collisions
        self.check_enemy_obstacle_collisions(current_time)

        # 4) Missile AI
        if (
            self.game.missile_model
            and self.game.player
            and self.game.player.missiles
        ):
            # Use the nearest visible enemy for missile guidance
            nearest_enemy_pos = None
            min_distance = float('inf')
            
            if hasattr(self.game, 'enemies') and self.game.enemies:
                for enemy in self.game.enemies:
                    if enemy.visible:
                        # Calculate distance to this enemy
                        dx = enemy.pos["x"] - self.game.player.position["x"]
                        dy = enemy.pos["y"] - self.game.player.position["y"]
                        distance = (dx**2 + dy**2)**0.5
                        
                        if distance < min_distance:
                            min_distance = distance
                            nearest_enemy_pos = enemy.pos
            elif self.game.enemy and self.game.enemy.visible:
                nearest_enemy_pos = self.game.enemy.pos
                
            update_missile_ai(
                self.game.player.missiles,
                self.game.player.position,
                nearest_enemy_pos,
                self.game._missile_input,
                self.game.missile_model
            )

        # 5) Misc updates
        # Respawn logic for single enemy (backward compatibility)
        self.game.handle_respawn(current_time)

        # Handle individual enemy respawns
        self.handle_enemy_respawns(current_time)

        # Update missiles
        self.game.player.update_missiles()

        # Check for missile collisions with enemies and obstacles
        self.check_missile_enemy_collisions()
        self.check_missile_obstacle_collisions()
        
        # Update score based on survival time
        self.update_time_based_score(current_time)
        
    def check_player_enemy_collisions(self, current_time: int) -> None:
        """Check for collisions between player and all enemies."""
        if not self.game.player:
            return
            
        player_rect = pygame.Rect(
            self.game.player.position["x"],
            self.game.player.position["y"],
            self.game.player.size,
            self.game.player.size,
        )
        
        # Check collision with multiple enemies
        if hasattr(self.game, 'enemies') and self.game.enemies:
            for enemy in self.game.enemies:
                if not enemy.visible:
                    continue
                    
                enemy_rect = pygame.Rect(
                    enemy.pos["x"],
                    enemy.pos["y"],
                    enemy.size,
                    enemy.size
                )
                
                if player_rect.colliderect(enemy_rect):
                    logging.info("Collision detected between player and an enemy.")
                    enemy.hide()
                    
                    # Set enemy-specific respawn timer
                    enemy.respawn_time = current_time + self.game.respawn_delay
                    logging.info("Enemy will respawn after collision with player.")
                    
                    # For now we just hide the enemy; in the future, could affect player health
                    return True
        # Backward compatibility with single enemy
        elif self.game.check_collision():
            logging.info("Collision detected between player and enemy.")
            if self.game.enemy:
                self.game.enemy.hide()
            self.game.is_respawning = True
            self.game.respawn_timer = current_time + self.game.respawn_delay
            logging.info("Player-Enemy collision in play mode.")
            return True
            
        return False
            
    def check_missile_enemy_collisions(self) -> None:
        """Check for collisions between missiles and all enemies."""
        if not self.game.player or not self.game.player.missiles:
            return
            
        # Handle multiple enemies
        if hasattr(self.game, 'enemies') and self.game.enemies:
            for enemy in self.game.enemies:
                if not enemy.visible:
                    continue
                    
                enemy_rect = pygame.Rect(
                    enemy.pos["x"], 
                    enemy.pos["y"], 
                    enemy.size, 
                    enemy.size
                )
                
                for missile in list(self.game.player.missiles):
                    missile_rect = missile.get_rect()
                    if missile_rect.colliderect(enemy_rect):
                        logging.info("Missile hit an enemy.")
                        self.game.player.missiles.remove(missile)
                        
                        # Create explosion animation at enemy position
                        if (hasattr(self.game, 'renderer') and 
                                hasattr(self.game.renderer, 'add_explosion')):
                            # Size of explosion based on enemy type
                            explosion_size = enemy.size * 1.5
                            if hasattr(enemy, 'enemy_type'):
                                if enemy.enemy_type == "tank":
                                    explosion_size = enemy.size * 2.0  # Bigger explosion for tank
                                elif enemy.enemy_type == "fast":
                                    # Smaller explosion for fast enemy
                                    explosion_size = enemy.size * 1.2
                            
                            # Create the explosion
                            self.game.renderer.add_explosion(
                                enemy.pos["x"] + enemy.size/2,  # Center X
                                enemy.pos["y"] + enemy.size/2,  # Center Y
                                int(explosion_size)
                            )
                            enemy_type = getattr(enemy, 'enemy_type', 'enemy')
                            logging.debug(f"Created explosion for {enemy_type}")
                        
                        # Hide the enemy
                        enemy.hide()
                        
                        # Set enemy-specific respawn timer
                        now = pygame.time.get_ticks()
                        enemy.respawn_time = now + self.game.respawn_delay
                        logging.info("Enemy will respawn after being hit by missile.")
                        
                        # Add score for hitting enemy (if implemented)
                        if hasattr(self.game, 'score'):
                            # Different scores based on enemy type
                            score_value = 100  # Default score
                            if hasattr(enemy, 'enemy_type'):
                                if enemy.enemy_type == "tank":
                                    score_value = 200  # More points for tank
                                elif enemy.enemy_type == "fast":
                                    score_value = 150  # More points for fast enemy
                            
                            self.game.score += score_value
                            logging.info(f"Score: {self.game.score} "
                                         f"(+{score_value} for missile hit)")
                        
                        break  # One missile hits one enemy
        # Backward compatibility with single enemy
        else:
            # Use the single enemy approach but add explosion animation
            if self.game.enemy and hasattr(self.game, 'player') and self.game.player.missiles:
                enemy_rect = pygame.Rect(
                    self.game.enemy.pos["x"],
                    self.game.enemy.pos["y"],
                    self.game.enemy.size,
                    self.game.enemy.size
                )
                
                for missile in list(self.game.player.missiles):
                    missile_rect = missile.get_rect()
                    if missile_rect.colliderect(enemy_rect):
                        logging.info("Missile hit the enemy.")
                        self.game.player.missiles.remove(missile)
                        
                        # Create explosion animation
                        if (hasattr(self.game, 'renderer') and 
                                hasattr(self.game.renderer, 'add_explosion')):
                            self.game.renderer.add_explosion(
                                self.game.enemy.pos["x"] + self.game.enemy.size/2,
                                self.game.enemy.pos["y"] + self.game.enemy.size/2,
                                int(self.game.enemy.size * 1.5)
                            )
                        
                        # Set up respawn
                        self.game.enemy.hide()
                        self.game.is_respawning = True
                        self.game.respawn_timer = pygame.time.get_ticks() + self.game.respawn_delay
                        
                        # Add score
                        if hasattr(self.game, 'score'):
                            self.game.score += 100
                            logging.info(f"Score: {self.game.score} (+100 for missile hit)")
                        
                        break  # One missile hits the enemy
            else:
                # Fall back to original method if anything is missing
                self.game.check_missile_collisions()
            
    def handle_enemy_respawns(self, current_time: int) -> None:
        """Handle respawning of individual enemies."""
        if not hasattr(self.game, 'enemies') or not self.game.enemies:
            return
            
        for enemy in self.game.enemies:
            is_invisible = not enemy.visible
            has_respawn_time = hasattr(enemy, 'respawn_time')
            is_respawn_time = current_time >= enemy.respawn_time if has_respawn_time else False
            
            if is_invisible and has_respawn_time and is_respawn_time:
                # Find a new valid position for this enemy
                # Get player pos for spawn point calculation
                player_x = self.game.player.position["x"]
                player_y = self.game.player.position["y"]
                
                new_pos = find_valid_spawn_position(
                    screen_width=self.game.screen_width,
                    screen_height=self.game.screen_height,
                    entity_size=enemy.size,
                    margin=config.WALL_MARGIN,
                    min_dist=config.MIN_DISTANCE,
                    other_pos=(player_x, player_y),
                )
                
                # Move the enemy to the new position
                enemy.set_position(new_pos[0], new_pos[1])
                
                # Start fade-in effect
                enemy.show(current_time)
                
                logging.info(f"Enemy respawned at {new_pos} with fade-in.")
                
    def check_player_obstacle_collisions(self) -> bool:
        """Check collisions between player and obstacles."""
        if not self.game.player or not hasattr(self.game, 'obstacles'):
            return False
            
        player_rect = pygame.Rect(
            self.game.player.position["x"],
            self.game.player.position["y"],
            self.game.player.size,
            self.game.player.size,
        )
        
        for obstacle in self.game.obstacles:
            if not obstacle.visible:
                continue
                
            if player_rect.colliderect(obstacle.get_rect()):
                logging.info("Player collided with an obstacle")
                
                # For now, just prevent player from moving through obstacle
                # In the future, could implement damage or different effects
                
                # This would be where we'd handle player damage or position correction
                # but for now we're just detecting
                
                return True
                
        return False
        
    def check_enemy_obstacle_collisions(self, current_time: int) -> None:
        """Check for and handle collisions between enemies and obstacles."""
        if not hasattr(self.game, 'obstacles') or not self.game.obstacles:
            return
            
        # Check multiple enemies if available
        if hasattr(self.game, 'enemies') and self.game.enemies:
            for enemy in self.game.enemies:
                if not enemy.visible:
                    continue
                    
                enemy_rect = pygame.Rect(
                    enemy.pos["x"],
                    enemy.pos["y"],
                    enemy.size,
                    enemy.size
                )
                
                for obstacle in self.game.obstacles:
                    if not obstacle.visible:
                        continue
                        
                    if enemy_rect.colliderect(obstacle.get_rect()):
                        logging.info("Enemy collided with an obstacle")
                        
                        # Hide enemy and set respawn timer (simplified collision handling)
                        enemy.hide()
                        enemy.respawn_time = current_time + self.game.respawn_delay
                        break  # Break inner loop once collision detected
        
        # Backward compatibility with single enemy
        elif self.game.enemy and self.game.enemy.visible:
            enemy_rect = pygame.Rect(
                self.game.enemy.pos["x"],
                self.game.enemy.pos["y"],
                self.game.enemy.size,
                self.game.enemy.size
            )
            
            for obstacle in self.game.obstacles:
                if not obstacle.visible:
                    continue
                    
                if enemy_rect.colliderect(obstacle.get_rect()):
                    logging.info("Enemy collided with an obstacle")
                    
                    # Hide enemy and set respawn timer
                    self.game.enemy.hide()
                    self.game.is_respawning = True
                    self.game.respawn_timer = current_time + self.game.respawn_delay
                    break
    
    def check_missile_obstacle_collisions(self) -> None:
        """Check for collisions between missiles and obstacles."""
        if not self.game.player or not self.game.player.missiles:
            return
            
        if not hasattr(self.game, 'obstacles') or not self.game.obstacles:
            return
            
        for obstacle in self.game.obstacles:
            if not obstacle.visible:
                continue
                
            obstacle_rect = obstacle.get_rect()
            
            for missile in list(self.game.player.missiles):
                if missile.get_rect().colliderect(obstacle_rect):
                    logging.info("Missile hit an obstacle")
                    
                    # Remove the missile
                    self.game.player.missiles.remove(missile)
                    
                    # Handle destructible obstacles
                    if hasattr(obstacle, 'destructible') and obstacle.destructible:
                        if obstacle.take_damage():
                            # If obstacle was destroyed by this hit
                            if hasattr(self.game, 'score'):
                                self.game.score += 50  # Fewer points than enemy
                                logging.info(
                                    f"Score: {self.game.score} "
                                    f"(+50 for destroying obstacle)"
                                )
                    
                    # Add explosion or impact effect here if desired
                    break  # One missile per obstacle
    
    def update_time_based_score(self, current_time: int) -> None:
        """Add points for survival time."""
        # Skip if game doesn't have score attribute
        if not hasattr(self.game, 'score'):
            return
            
        # Calculate time since last score update
        interval = self.game.survival_score_interval
        if current_time - self.game.last_score_time >= interval and not self.game.paused:
            # Award 1 point per second survived
            self.game.score += 1
            self.game.last_score_time = current_time
            logging.debug(f"Score: {self.game.score} (+1 for survival)")