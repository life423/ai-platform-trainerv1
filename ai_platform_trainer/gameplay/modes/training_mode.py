import math
import logging
import random
import pygame
from ai_platform_trainer.gameplay.post_training_processor import post_training_processor


class TrainingMode:
    def __init__(self, game):
        self.game = game
        self.missile_cooldown = 0
        self.missile_lifespan = {}
        self.missile_sequences = {}
        self.frame_count = 0
        self.session_id = f"{pygame.time.get_ticks():x}"
        
        # Set game reference on enemy for missile avoidance
        if hasattr(self.game, "enemy") and hasattr(self.game.enemy, "game"):
            self.game.enemy.game = self.game

    def update(self):
        current_time = pygame.time.get_ticks()

        if self.game.enemy and self.game.player:
            enemy_x = self.game.enemy.pos["x"]
            enemy_y = self.game.enemy.pos["y"]

            self.game.player.update(enemy_x, enemy_y)
            self.game.enemy.update_movement(
                self.game.player.position["x"],
                self.game.player.position["y"],
                self.game.player.step,
            )

            if self.missile_cooldown > 0:
                self.missile_cooldown -= 1

            # Tactical missile firing instead of random
            if self.missile_cooldown <= 0 and len(self.game.player.missiles) == 0:
                can_fire = self.check_tactical_shot(enemy_x, enemy_y)
                
                if can_fire:
                    self.game.player.shoot_missile(enemy_x, enemy_y)
                    # Variable cooldown for more natural timing
                    self.missile_cooldown = random.randint(90, 150)

                    if self.game.player.missiles:
                        missile = self.game.player.missiles[0]
                        self.missile_lifespan[missile] = (
                            missile.birth_time,
                            missile.lifespan,
                        )
                        self.missile_sequences[missile] = []

        current_missiles = self.game.player.missiles[:]
        for missile in current_missiles:
            if missile in self.missile_lifespan:
                birth_time, lifespan = self.missile_lifespan[missile]
                if current_time - birth_time >= lifespan:
                    self.finalize_missile_sequence(missile, success=False)
                    self.game.player.missiles.remove(missile)
                    del self.missile_lifespan[missile]
                    logging.debug("Training mode: Missile removed (lifespan expiry).")
                else:
                    missile.update()

                    if missile in self.missile_sequences:
                        player_x = self.game.player.position["x"]
                        player_y = self.game.player.position["y"]
                        enemy_x = self.game.enemy.pos["x"]
                        enemy_y = self.game.enemy.pos["y"]

                        # Example distances

                        missile_angle = math.atan2(missile.vy, missile.vx)
                        missile_action = getattr(missile, "last_action", 0.0)

                        self.missile_sequences[missile].append(
                            {
                                "player_x": player_x,
                                "player_y": player_y,
                                "enemy_x": enemy_x,
                                "enemy_y": enemy_y,
                                "missile_x": missile.pos["x"],
                                "missile_y": missile.pos["y"],
                                "missile_angle": missile_angle,
                                "missile_collision": False,
                                "missile_action": missile_action,
                                "timestamp": current_time,
                            }
                        )

                    if self.game.enemy:
                        enemy_rect = pygame.Rect(
                            self.game.enemy.pos["x"],
                            self.game.enemy.pos["y"],
                            self.game.enemy.size,
                            self.game.enemy.size,
                        )
                        if missile.get_rect().colliderect(enemy_rect):
                            logging.info("Missile hit the enemy (training mode).")
                            self.finalize_missile_sequence(missile, success=True)
                            self.game.player.missiles.remove(missile)
                            del self.missile_lifespan[missile]

                            # Instead of hiding/respawning, register hit
                            self.game.enemy.register_hit()
                            
                            # Optional: Reposition enemy for variety
                            # But don't hide or use respawn delay
                            if getattr(self, "reposition_on_hit", True):
                                from ai_platform_trainer.gameplay.spawn_utils import find_valid_spawn_position
                                new_pos = find_valid_spawn_position(
                                    self.game.screen_width, 
                                    self.game.screen_height,
                                    self.game.enemy.size,
                                    margin=20,
                                    min_dist=100,
                                    other_pos=(self.game.player.position["x"], self.game.player.position["y"])
                                )
                                self.game.enemy.pos["x"], self.game.enemy.pos["y"] = new_pos
                            break

                    if not (
                        0 <= missile.pos["x"] <= self.game.screen_width
                        and 0 <= missile.pos["y"] <= self.game.screen_height
                    ):
                        self.finalize_missile_sequence(missile, success=False)
                        self.game.player.missiles.remove(missile)
                        del self.missile_lifespan[missile]
                        logging.debug("Training Mode: Missile left the screen.")

        # Track frames for continuous recording
        self.frame_count += 1
        
        # Periodically record state regardless of missile activity
        if self.frame_count % 10 == 0:  # Every 10th frame
            self.record_game_state(current_time)

    def finalize_missile_sequence(self, missile, success: bool) -> None:
        """
        Called when a missile's life ends or collision occurs.
        Logs each frame's data with a final 'missile_collision' outcome.
        """
        if missile not in self.missile_sequences:
            return

        outcome_val = success
        frames = self.missile_sequences[missile]

        for frame_data in frames:
            frame_data["missile_collision"] = outcome_val
            if self.game.data_logger:
                self.game.data_logger.log(frame_data)

        del self.missile_sequences[missile]
        logging.debug(
            f"Finalized missile sequence with success={success}, frames={len(frames)}"
        )

        # If this was the last active missile, process the collected data
        if not self.missile_sequences and not self.game.player.missiles:
            self.process_collected_data()

    def check_tactical_shot(self, enemy_x: float, enemy_y: float) -> bool:
        """
        Determine if current position is good for shooting based on tactical
        considerations rather than random chance.
        """
        if not self.game.player or not self.game.enemy:
            return False
            
        player_x = self.game.player.position["x"]
        player_y = self.game.player.position["y"]
        
        # Calculate distance and direction
        dx = enemy_x - player_x
        dy = enemy_y - player_y
        dist = math.hypot(dx, dy)
        
        # 1. Check if enemy is within reasonable range
        if dist > 400 or dist < 50:  # Too far or too close
            return False
            
        # 2. Check if player is "facing" the enemy
        # (Using player's recent movement as an indicator of facing direction)
        if hasattr(self.game.player, "velocity"):
            player_vx = self.game.player.velocity["x"]
            player_vy = self.game.player.velocity["y"]
            
            # Dot product - positive when moving toward enemy
            facing_score = (dx * player_vx + dy * player_vy)
            
            # Don't shoot if strongly moving away from enemy
            if facing_score < -0.5:  
                return False
        
        # 3. Introduce tactical timing
        # More likely to shoot when enemy is moving predictably
        enemy_pattern = getattr(self.game.enemy, "current_pattern", None)
        if enemy_pattern == "pursue":
            # Higher chance when enemy is pursuing (predictable path)
            return random.random() < 0.15
        else:
            # Lower chance for other patterns
            return random.random() < 0.08
    
    def record_game_state(self, current_time: int) -> None:
        """
        Record current game state regardless of missile activity.
        This ensures continuous data collection even between missile sequences.
        """
        if not self.game.data_logger:
            return
            
        if not self.game.enemy or not self.game.player:
            return
            
        # Basic state data
        state_data = {
            "timestamp": current_time,
            "session_id": self.session_id,
            "frame": self.frame_count,
            "player_x": self.game.player.position["x"],
            "player_y": self.game.player.position["y"],
            "enemy_x": self.game.enemy.pos["x"],
            "enemy_y": self.game.enemy.pos["y"],
            "player_has_missile": len(self.game.player.missiles) > 0,
            "enemy_pattern": getattr(self.game.enemy, "current_pattern", "unknown"),
            "missile_x": None,
            "missile_y": None,
            "missile_angle": None,
            "missile_collision": False,
        }
        
        # Add missile data if present
        if self.game.player.missiles:
            missile = self.game.player.missiles[0]
            state_data["missile_x"] = missile.pos["x"]
            state_data["missile_y"] = missile.pos["y"]
            state_data["missile_angle"] = math.atan2(missile.vy, missile.vx)
            
        self.game.data_logger.log(state_data)
    
    def process_collected_data(self) -> None:
        """
        Process all collected training data:
        1. Get the data from the data logger
        2. Validate and append it to the existing dataset
        3. Retrain the AI models with the combined data
        """
        if not self.game.data_logger or not hasattr(self.game.data_logger, "data"):
            logging.warning("No data logger available or no data collected")
            return

        # Get the collected data
        collected_data = self.game.data_logger.data

        if not collected_data:
            logging.warning("No training data was collected during this session")
            return

        logging.info(
            f"Processing {len(collected_data)} data points collected in training"
        )

        # Use our validator/trainer to process the data and retrain the models
        success = post_training_processor.process_training_sequence(collected_data)

        if success:
            logging.info(
                "Successfully validated data, updated dataset, and retrained models"
            )
            # Reset the data logger for the next training session
            self.game.data_logger.data = []
        else:
            logging.error("Failed to process collected data and retrain models")
