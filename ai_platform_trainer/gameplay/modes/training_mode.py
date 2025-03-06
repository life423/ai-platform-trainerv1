import math
import logging
import random
import pygame


class TrainingMode:
    def __init__(self, game):
        self.game = game
        self.missile_cooldown = 0
        self.missile_fire_prob = 0.1
        
        # Set data logger in collision manager if not already set
        if self.game.collision_manager.data_logger is None:
            self.game.collision_manager.set_data_logger(self.game.data_logger)
        
        # Set missile manager in collision manager if not already set
        if self.game.collision_manager.missile_manager is None:
            self.game.collision_manager.set_missile_manager(self.game.missile_manager)

    def update(self):
        current_time = pygame.time.get_ticks()

        if self.game.enemy and self.game.player:
            enemy_x = self.game.enemy.pos["x"]
            enemy_y = self.game.enemy.pos["y"]

            self.game.player.update(enemy_x, enemy_y)
            # Pass missiles to enemy movement for avoidance training
            self.game.enemy.update_movement(
                self.game.player.position["x"],
                self.game.player.position["y"],
                self.game.player.step,
                current_time,
                self.game.missile_manager.missiles,  # Pass missiles for avoidance
            )
            
            # Check for collisions using collision manager (handles logging internally)
            collision_results = self.game.collision_manager.update(
                self.game, current_time, is_training=True
            )
            
            # Handle player-enemy collision result
            if collision_results["player_enemy_collision"]:
                self.game.is_respawning = True
                self.game.respawn_timer = current_time + self.game.respawn_delay

            if self.missile_cooldown > 0:
                self.missile_cooldown -= 1

            # Randomly fire missiles in training mode with cooldown
            if random.random() < self.missile_fire_prob:
                if (self.missile_cooldown <= 0 and 
                        len(self.game.missile_manager.missiles) == 0):
                    # Use player's shoot_missile with the missile manager
                    self.game.player.shoot_missile(
                        {"x": enemy_x, "y": enemy_y}, 
                        self.game.missile_manager
                    )
                    self.missile_cooldown = 120

        # Record training data for all active missiles
        current_missiles = self.game.missile_manager.missiles[:]
        for missile in current_missiles:
            if self.game.player and self.game.enemy:
                # Record training frame data for each missile
                self.game.missile_manager.record_training_frame(
                    missile,
                    self.game.player.position,
                    self.game.enemy.pos,
                    current_time,
                    False  # Not a collision frame
                )
                
                # Check for missile-enemy collisions 
                if self.game.enemy:
                    enemy_rect = pygame.Rect(
                        self.game.enemy.pos["x"],
                        self.game.enemy.pos["y"],
                        self.game.enemy.size,
                        self.game.enemy.size,
                    )
                    if missile.get_rect().colliderect(enemy_rect):
                        logging.info("Missile hit the enemy (training mode).")
                        
                        # Handle missile collision in training mode
                        self.game.missile_manager.finalize_missile_sequence(
                            missile, True
                        )
                        self.game.missile_manager.missiles.remove(missile)
                        
                        # Hide enemy and set up respawn
                        self.game.enemy.hide()
                        self.game.is_respawning = True
                        self.game.respawn_timer = (
                            current_time + self.game.respawn_delay
                        )
                        break

        # Update missiles with the missile manager
        self.game.missile_manager.update(current_time)
        
        # Handle enemy respawn if needed
        if self.game.is_respawning and current_time >= self.game.respawn_timer:
            if self.game.enemy:
                self.game.handle_respawn(current_time)
        
    def log_collision_data(self, current_time, is_collision):
        """
        Log data about player and enemy positions with collision information.
        
        Args:
            current_time: Current game time in milliseconds
            is_collision: Boolean indicating whether a player-enemy collision occurred
        """
        if not self.game.data_logger or not self.game.player or not self.game.enemy:
            return
            
        # Calculate distance between player and enemy
        player_x = self.game.player.position["x"]
        player_y = self.game.player.position["y"]
        enemy_x = self.game.enemy.pos["x"]
        enemy_y = self.game.enemy.pos["y"]
        
        dx = player_x - enemy_x
        dy = player_y - enemy_y
        distance = math.sqrt(dx*dx + dy*dy)
        
        # Prepare position dictionaries in the format needed for validation
        player_pos = {"x": player_x, "y": player_y}
        enemy_pos = {"x": enemy_x, "y": enemy_y}
        collision_val = 1 if is_collision else 0
        
        # Use the new log_data method with validation
        self.game.data_logger.log_data(
            current_time, 
            player_pos, 
            enemy_pos, 
            distance, 
            collision_val
        )
