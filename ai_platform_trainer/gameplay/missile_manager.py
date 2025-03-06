import pygame
import logging
from ai_platform_trainer.entities.missile import Missile


class MissileManager:
    def __init__(self, screen_width, screen_height, data_logger=None):
        self.screen_width = screen_width
        self.screen_height = screen_height
        self.data_logger = data_logger
        self.missiles = []
        self.missile_lifespan = {}
        self.missile_sequences = {}

    def spawn_missile(self, missile: Missile):
        """Add a missile to the managed list"""
        self.missiles.append(missile)
        
        # Track missile lifespan and sequences for training mode
        self.missile_lifespan[missile] = (missile.birth_time, missile.lifespan)
        self.missile_sequences[missile] = []

    def update(self, current_time):
        """Move each missile, check collisions/lifespan"""
        for missile in self.missiles[:]:
            missile.update()
            
            # Check lifespan
            age = current_time - missile.birth_time
            if age >= missile.lifespan:
                self.missiles.remove(missile)
                if missile in self.missile_lifespan:
                    del self.missile_lifespan[missile]
                if missile in self.missile_sequences:
                    self.finalize_missile_sequence(missile, False)
                logging.debug("Missile removed (lifespan).")
                continue

            # Check screen boundaries
            if not (0 <= missile.pos["x"] <= self.screen_width and
                    0 <= missile.pos["y"] <= self.screen_height):
                self.missiles.remove(missile)
                if missile in self.missile_lifespan:
                    del self.missile_lifespan[missile]
                if missile in self.missile_sequences:
                    self.finalize_missile_sequence(missile, False)
                logging.debug("Missile off-screen, removed.")
                continue

    def draw(self, screen):
        """Draw all missiles on the screen"""
        for missile in self.missiles:
            missile.draw(screen)

    def clear_all(self):
        """Remove all missiles"""
        self.missiles.clear()
        self.missile_lifespan.clear()
        self.missile_sequences.clear()
    
    def handle_enemy_collision(self, enemy, current_time, respawn_callback=None):
        """Check if any missile collides with the enemy"""
        for missile in self.missiles[:]:
            if enemy:
                enemy_rect = pygame.Rect(
                    enemy.pos["x"],
                    enemy.pos["y"],
                    enemy.size,
                    enemy.size
                )
                if missile.get_rect().colliderect(enemy_rect):
                    logging.info("Missile hit the enemy.")
                    self.missiles.remove(missile)
                    
                    # Handle training data if this is a tracked missile
                    if missile in self.missile_sequences:
                        self.finalize_missile_sequence(missile, True)
                    if missile in self.missile_lifespan:
                        del self.missile_lifespan[missile]
                    
                    # Call respawn callback if provided
                    if respawn_callback:
                        respawn_callback()
                    
                    return True
        return False
    
    def record_training_frame(self, missile, player_pos, enemy_pos,
                              current_time, is_collision=False):
        """
        Record a frame of training data for a missile
        """
        if missile in self.missile_sequences:
            import math
            missile_angle = math.atan2(missile.vy, missile.vx)
            missile_action = getattr(missile, "last_action", 0.0)
            
            self.missile_sequences[missile].append({
                "player_x": player_pos["x"],
                "player_y": player_pos["y"],
                "enemy_x": enemy_pos["x"],
                "enemy_y": enemy_pos["y"],
                "missile_x": missile.pos["x"],
                "missile_y": missile.pos["y"],
                "missile_angle": missile_angle,
                "missile_collision": False,
                "missile_action": missile_action,
                "timestamp": current_time,
                "collision": is_collision,
            })
    
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
            if self.data_logger:
                self.data_logger.log(frame_data)

        del self.missile_sequences[missile]
        logging.debug(
            f"Finalized missile sequence with success={success}, "
            f"frames={len(frames)}"
        )
