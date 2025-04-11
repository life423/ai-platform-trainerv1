import logging
import os

import pygame

from ai_platform_trainer.engine.rendering.background import BackgroundManager

# Import sprite manager and background manager
from ai_platform_trainer.utils.sprite_manager import SpriteManager


class Renderer:
    def __init__(self, screen: pygame.Surface) -> None:
        """
        Initialize the Renderer.

        Args:
            screen: Pygame display surface
        """
        self.screen = screen
        
        # Log important initialization details
        width, height = screen.get_width(), screen.get_height()
        logging.info(f"Initializing renderer with screen size: {width}x{height}")
        logging.info(f"Current working directory: {os.getcwd()}")

        # Initialize sprite and background managers
        logging.info("Initializing sprite manager...")
        self.sprite_manager = SpriteManager()
        
        # Test load a sprite to verify sprite loading works
        try:
            test_sprite = self.sprite_manager.load_sprite("player", (50, 50))
            if test_sprite:
                logging.info(f"Test sprite loaded successfully, size: {test_sprite.get_size()}")
            else:
                logging.warning("Test sprite load returned None")
        except Exception as e:
            logging.error(f"Error during test sprite load: {e}")
            
        logging.info("Initializing background manager...")
        self.background_manager = BackgroundManager(
            screen_width=screen.get_width(),
            screen_height=screen.get_height()
        )

        # Optional effects
        self.enable_effects = True
        self.frame_count = 0
        self.particle_effects = []
        
        # For displaying pause message
        self.font = pygame.font.Font(None, 72)  # Large font for pause message
        
        logging.info("Renderer initialization complete")

    def render(self, menu, player, enemy, menu_active: bool, powerup_manager=None) -> None:
        """
        Render the game elements on the screen.

        Args:
            menu: Menu instance
            player: Player instance
            enemy: Enemy instance (for backward compatibility)
            menu_active: Boolean indicating if the menu is active
        """
        try:
            # Update frame counter for animations
            self.frame_count += 1

            # Render background
            self.background_manager.render(self.screen)

            if menu_active:
                # Draw menu
                menu.draw(self.screen)
                logging.debug("Menu rendered.")
            else:
                # Render game elements
                self._render_game(menu.game, player, enemy)
                logging.debug("Game elements rendered.")
                
                # Check if game is paused and render pause overlay
                if hasattr(menu, 'game') and hasattr(menu.game, 'paused') and menu.game.paused:
                    self._render_pause_overlay()

            # Update display
            pygame.display.flip()
            logging.debug("Frame updated on display.")

        except Exception as e:
            logging.error(f"Error during rendering: {e}")

    def _render_game(self, game, player, enemy) -> None:
        """
        Render the game elements during gameplay.

        Args:
            game: Game instance
            player: Player instance
            enemy: Enemy instance (for backward compatibility)
        """
        # Render power-ups if PlayMode has any
        has_play_mode = hasattr(game, 'play_mode_manager')
        if has_play_mode and hasattr(game.play_mode_manager, 'powerup_manager'):
            game.play_mode_manager.powerup_manager.render(self.screen)
        # Draw obstacles first (they're in the background)
        if hasattr(game, 'obstacles'):
            for obstacle in game.obstacles:
                if obstacle.visible:
                    self._render_obstacle(obstacle)
        
        # Draw player with sprite
        if hasattr(player, 'position') and hasattr(player, 'size'):
            self._render_player(player)

            # Render player missiles
            if hasattr(player, 'missiles'):
                for missile in player.missiles:
                    self._render_missile(missile)

        # Draw multiple enemies if available
        if hasattr(game, 'enemies') and game.enemies:
            for enemy_obj in game.enemies:
                if hasattr(enemy_obj, 'pos') and hasattr(enemy_obj, 'size') and enemy_obj.visible:
                    self._render_enemy(enemy_obj)
        # Fallback to single enemy for backward compatibility
        elif hasattr(enemy, 'pos') and hasattr(enemy, 'size') and enemy.visible:
            self._render_enemy(enemy)

        # Render score if available
        if hasattr(game, 'score'):
            self._render_score(game.score)
            
        # Render particle effects if enabled
        if self.enable_effects:
            self._update_and_render_effects()

    def _render_obstacle(self, obstacle) -> None:
        """
        Render an obstacle entity.
        
        Args:
            obstacle: Obstacle instance
        """
        try:
            # Position and size
            pos_x, pos_y = obstacle.pos["x"], obstacle.pos["y"]
            size = obstacle.size
            
            logging.debug(f"Rendering obstacle at ({pos_x}, {pos_y}), size={size}")
            
            # Use sprite if available, otherwise fall back to a colored rectangle
            if hasattr(obstacle, 'sprite') and obstacle.sprite:
                self.screen.blit(obstacle.sprite, (pos_x, pos_y))
                logging.debug("Obstacle sprite blitted to screen")
            else:
                # Use a fallback color rectangle
                color = obstacle.color if hasattr(obstacle, 'color') else (100, 100, 100)
                
                # Add visual hint for destructible obstacles
                if hasattr(obstacle, 'destructible') and obstacle.destructible:
                    # Draw a border for destructible obstacles
                    pygame.draw.rect(
                        self.screen,
                        (200, 0, 0),  # Red border for destructible
                        pygame.Rect(pos_x, pos_y, size, size),
                        3  # Border width
                    )
                    
                    # Fill with slightly lighter color
                    pygame.draw.rect(
                        self.screen,
                        (min(color[0] + 50, 255), min(color[1] + 20, 255), color[2]),
                        pygame.Rect(pos_x + 3, pos_y + 3, size - 6, size - 6)
                    )
                else:
                    # Solid obstacle with no border
                    pygame.draw.rect(
                        self.screen,
                        color,
                        pygame.Rect(pos_x, pos_y, size, size)
                    )
                
                logging.debug(f"Obstacle drawn as rect with color {color}")
        except Exception as e:
            logging.error(f"Error rendering obstacle: {e}")
    
    def _render_player(self, player) -> None:
        """
        Render the player entity with sprites.

        Args:
            player: Player instance
        """
        try:
            # Determine sprite size
            size = (player.size, player.size)
            pos = player.position
            
            logging.debug(f"Rendering player at position: ({pos['x']}, {pos['y']}), size: {size}")
            
            # Load sprite directly to verify it exists before rendering
            sprite = self.sprite_manager.load_sprite("player", size)
            if sprite:
                logging.debug(f"Player sprite loaded successfully, alpha: {sprite.get_alpha()}")
                
                # Render directly to verify exact position
                self.screen.blit(sprite, (pos['x'], pos['y']))
                logging.debug("Player sprite blitted directly to screen")
            else:
                logging.warning("Player sprite is None")
                
        except Exception as e:
            logging.error(f"Error rendering player: {e}")

    def _render_enemy(self, enemy) -> None:
        """
        Render the enemy entity with sprites.

        Args:
            enemy: Enemy instance
        """
        try:
            # Determine sprite size
            size = (enemy.size, enemy.size)
            pos_x, pos_y = enemy.pos["x"], enemy.pos["y"]
            
            logging.debug(f"Rendering enemy at position: ({pos_x}, {pos_y}), size: {size}")

            # Check if the enemy is fading in
            alpha = 255
            if hasattr(enemy, 'fading_in') and enemy.fading_in:
                alpha = enemy.fade_alpha
                logging.debug(f"Enemy fading in, alpha: {alpha}")

            # Render the enemy sprite
            sprite = self.sprite_manager.load_sprite("enemy", size)
            if sprite:
                logging.debug(f"Enemy sprite loaded successfully, setting alpha: {alpha}")
                sprite.set_alpha(alpha)
                self.screen.blit(sprite, (pos_x, pos_y))
                logging.debug("Enemy sprite blitted to screen")
            else:
                logging.warning("Enemy sprite is None")
        except Exception as e:
            logging.error(f"Error rendering enemy: {e}")

    def _render_missile(self, missile) -> None:
        """
        Render a missile entity with sprites.

        Args:
            missile: Missile instance
        """
        try:
            if hasattr(missile, 'position') and hasattr(missile, 'size'):
                # Determine sprite size - make it a bit more elongated
                width = missile.size
                height = int(missile.size * 1.5)
                size = (width, height)
                pos = missile.position
                
                logging.debug(f"Rendering missile at position: ({pos['x']}, {pos['y']}), " 
                              f"size: {size}")

                # Calculate rotation angle based on direction
                rotation = 0
                if hasattr(missile, 'direction'):
                    # Convert direction to angle in degrees
                    dx, dy = missile.direction
                    if dx != 0 or dy != 0:
                        import math
                        angle_rad = math.atan2(dy, dx)
                        rotation = math.degrees(angle_rad) + 90  # Adjust so 0 points up
                        logging.debug(f"Missile rotation: {rotation} degrees")

                # Load sprite directly to check
                sprite = self.sprite_manager.load_sprite("missile", size)
                if sprite:
                    logging.debug("Missile sprite loaded successfully")
                    
                    # Rotate sprite if needed
                    if rotation != 0:
                        sprite = pygame.transform.rotate(sprite, rotation)
                        logging.debug(f"Missile sprite rotated to {rotation} degrees")
                    
                    # Render directly
                    self.screen.blit(sprite, (pos['x'], pos['y']))
                    logging.debug("Missile sprite blitted to screen")
                else:
                    logging.warning("Missile sprite is None")

                # Add a trail effect if effects are enabled
                if self.enable_effects and self.frame_count % 2 == 0:
                    self._add_missile_trail(missile)
            else:
                logging.warning("Missile missing position or size attributes")
        except Exception as e:
            logging.error(f"Error rendering missile: {e}")

    def _add_missile_trail(self, missile) -> None:
        """
        Add a particle effect trail behind a missile.

        Args:
            missile: Missile instance
        """
        if not hasattr(missile, 'position'):
            return

        # Create a small particle effect behind the missile
        x = missile.position["x"] + missile.size // 2
        y = missile.position["y"] + missile.size // 2

        # Trail particles
        import random
        for _ in range(2):
            # Random offset
            offset_x = random.randint(-3, 3)
            offset_y = random.randint(-3, 3)

            # Random size
            size = random.randint(2, 5)

            # Random lifetime
            lifetime = random.randint(5, 15)

            # Create particle
            particle = {
                'x': x + offset_x,
                'y': y + offset_y,
                'size': size,
                'color': (255, 255, 200, 200),  # Yellow-ish with alpha
                'lifetime': lifetime,
                'max_lifetime': lifetime
            }

            self.particle_effects.append(particle)

    def _update_and_render_effects(self) -> None:
        """Update and render all particle effects."""
        # Update particles
        updated_particles = []
        for particle in self.particle_effects:
            # Decrease lifetime
            particle['lifetime'] -= 1

            # Skip dead particles
            if particle['lifetime'] <= 0:
                continue

            # Calculate alpha based on remaining lifetime
            alpha = int(255 * (particle['lifetime'] / particle['max_lifetime']))
            color = list(particle['color'])
            if len(color) > 3:
                color[3] = min(color[3], alpha)
            else:
                color.append(alpha)

            # Draw particle
            pygame.draw.circle(
                self.screen,
                color,
                (int(particle['x']), int(particle['y'])),
                particle['size']
            )

            # Keep particle for next frame
            updated_particles.append(particle)

        # Replace particle list with updated one
        self.particle_effects = updated_particles
        
    def _render_score(self, score: int) -> None:
        """
        Render the player's score on screen.

        Args:
            score: Current game score
        """
        try:
            # Create a font for the score display
            if not hasattr(self, 'score_font'):
                self.score_font = pygame.font.Font(None, 36)  # Use default font, size 36
            
            # Render score text
            score_text = self.score_font.render(
                f"Score: {score}", 
                True, 
                (255, 255, 255)  # White text
            )
            
            # Position in top-right corner with some padding
            padding = 10
            score_rect = score_text.get_rect(
                topright=(self.screen.get_width() - padding, padding)
            )
            
            # Draw with a dark semi-transparent background for better visibility
            bg_rect = score_rect.inflate(20, 10)  # Make bg slightly larger than text
            bg_surface = pygame.Surface((bg_rect.width, bg_rect.height), pygame.SRCALPHA)
            bg_surface.fill((0, 0, 0, 128))  # Semi-transparent black
            self.screen.blit(bg_surface, bg_rect)
            
            # Draw the score text
            self.screen.blit(score_text, score_rect)
            
        except Exception as e:
            logging.error(f"Error rendering score: {e}")

    def _render_pause_overlay(self) -> None:
        """Render a pause overlay when the game is paused."""
        # Create a semi-transparent overlay
        overlay = pygame.Surface(
            (self.screen.get_width(), self.screen.get_height()), 
            pygame.SRCALPHA
        )
        # Semi-transparent black
        overlay.fill((0, 0, 0, 128))
        self.screen.blit(overlay, (0, 0))
        
        # Render "PAUSED" text
        pause_text = self.font.render(
            "PAUSED", 
            True, 
            (255, 255, 255)
        )
        text_rect = pause_text.get_rect(
            center=(self.screen.get_width() // 2, self.screen.get_height() // 2)
        )
        self.screen.blit(pause_text, text_rect)
        
        # Render help text
        small_font = pygame.font.Font(None, 24)
        help_text = small_font.render(
            "Press 'P' to resume", 
            True, 
            (200, 200, 200)
        )
        help_rect = help_text.get_rect(
            center=(self.screen.get_width() // 2, self.screen.get_height() // 2 + 50)
        )
        self.screen.blit(help_text, help_rect)
