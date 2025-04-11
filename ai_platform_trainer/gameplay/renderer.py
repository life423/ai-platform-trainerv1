import pygame
import logging
from typing import List, Dict, Tuple, Optional

# Import sprite manager for entity rendering
from ai_platform_trainer.utils.sprite_manager import SpriteManager


class Renderer:
    def __init__(self, screen: pygame.Surface) -> None:
        """
        Initialize the Renderer.

        Args:
            screen: Pygame display surface
        """
        self.screen = screen
        self.BACKGROUND_COLOR = (135, 206, 235)  # Light blue

        # Initialize sprite manager
        self.sprite_manager = SpriteManager()

        # Optional effects
        self.enable_effects = True
        self.frame_count = 0
        self.particle_effects = []
        
        # Explosion animation tracking
        self.explosions = []
        self.explosion_frames = None  # Will be loaded on demand
        self.explosion_frame_count = 4
        
        # Enemy type color variations
        self.enemy_colors = {
            "standard": (255, 50, 50),     # Red
            "fast": (255, 180, 50),        # Orange
            "tank": (120, 50, 120)         # Purple
        }

    def render(self, menu, player, enemy, menu_active: bool) -> None:
        """
        Render the game elements on the screen.

        Args:
            menu: Menu instance
            player: Player instance
            enemy: Enemy instance
            menu_active: Boolean indicating if the menu is active
        """
        try:
            # Clear screen with background color
            self.screen.fill(self.BACKGROUND_COLOR)

            # Update frame counter for animations
            self.frame_count += 1

            if menu_active:
                # Draw menu
                menu.draw(self.screen)
                logging.debug("Menu rendered.")
            else:
                # Render game elements
                if hasattr(player, 'position') and player:
                    # In case we have a list of enemies, we'll render them all
                    enemies = []
                    if hasattr(player, 'game') and hasattr(player.game, 'enemies'):
                        enemies = player.game.enemies
                    
                    self._render_game(player, enemy, enemies)
                logging.debug("Game elements rendered.")

            # Update display
            pygame.display.flip()
            logging.debug("Frame updated on display.")

        except Exception as e:
            logging.error(f"Error during rendering: {e}")

    def _render_game(self, player, enemy, enemies=None) -> None:
            enemies: List of enemy instances (optional)
        """
        # Update and render any active explosions
        self._update_explosions()
        Render the game elements during gameplay.

        Args:
            player: Player instance
            enemy: Enemy instance
        """
        # Draw player with sprite
        if hasattr(player, 'position') and hasattr(player, 'size'):
            self._render_player(player)

            # Render player missiles
            if hasattr(player, 'missiles'):
                for missile in player.missiles:
                    self._render_missile(missile)

        # Render multiple enemies if available
        if enemies:
            for enemy_obj in enemies:
                if hasattr(enemy_obj, 'pos') and hasattr(enemy_obj, 'size') and enemy_obj.visible:
                    self._render_enemy(enemy_obj)
        
        # Fallback to single enemy for backward compatibility
        elif hasattr(enemy, 'pos') and hasattr(enemy, 'size') and enemy.visible:
            self._render_enemy(enemy)

        # Render particle effects if enabled
        if self.enable_effects:
            self._update_and_render_effects()

    def _render_player(self, player) -> None:
        """
        Render the player entity with sprites.

        Args:
            player: Player instance
        """
        # Determine sprite size
        size = (player.size, player.size)

        # Render the player sprite
        self.sprite_manager.render(
            screen=self.screen,
            entity_type="player",
            position=player.position,
            size=size
        )

    def _render_enemy(self, enemy) -> None:
        """
        Render the enemy entity with sprites.

        Args:
            enemy: Enemy instance
        """
        # Determine sprite size
        size = (enemy.size, enemy.size)

        # Check if the enemy is fading in
        alpha = 255
        if hasattr(enemy, 'fading_in') and enemy.fading_in:
            alpha = enemy.fade_alpha

        # Determine enemy type for visual differentiation
        enemy_type = "enemy"  # Default
        if hasattr(enemy, 'enemy_type'):
            enemy_type = enemy.enemy_type
            
        # Get appropriate sprite based on enemy type
        sprite = None
        
        # First try to load a type-specific sprite
        specific_sprite_name = f"enemy_{enemy_type}"
        try:
            sprite = self.sprite_manager.load_sprite(specific_sprite_name, size)
        except Exception:
            # Fall back to generic enemy sprite
            sprite = self.sprite_manager.load_sprite("enemy", size)
            
            # If we have a generic sprite but different enemy types,
            # apply color tinting to differentiate them visually
            if hasattr(enemy, 'enemy_type') and enemy.enemy_type in self.enemy_colors:
                # Create a copy of the sprite for tinting
                tinted_sprite = sprite.copy()
                color = self.enemy_colors.get(enemy.enemy_type, (255, 255, 255))
                
                # Apply the tint by creating a colored overlay
                overlay = pygame.Surface(size, pygame.SRCALPHA)
                overlay.fill((*color, 128))  # Semi-transparent color
                tinted_sprite.blit(overlay, (0, 0), special_flags=pygame.BLEND_RGBA_MULT)
                
                sprite = tinted_sprite
        sprite.set_alpha(alpha)
        self.screen.blit(sprite, (enemy.pos["x"], enemy.pos["y"]))
        # For tank enemies, show damage state if applicable
        if hasattr(enemy, 'enemy_type') and enemy.enemy_type == "tank" and hasattr(enemy, 'damage_state'):
            self._render_tank_damage_state(enemy)
    
    def _render_tank_damage_state(self, enemy) -> None:
        """
        Render visual indicators of tank enemy damage state.
        
        Args:
            enemy: Tank enemy instance
        """
        if not hasattr(enemy, 'damage_state') or enemy.damage_state == 0:
            return
            
        # Add cracks or damage indicators based on damage state
        damage = enemy.damage_state
        pos_x, pos_y = enemy.pos["x"], enemy.pos["y"]
        size = enemy.size
        
        # Draw damage indicators (simple cracks)
        if damage >= 1:
            # First damage indicator - diagonal crack
            pygame.draw.line(
                self.screen, 
                (30, 30, 30), 
                (pos_x + size * 0.2, pos_y + size * 0.2),
                (pos_x + size * 0.8, pos_y + size * 0.8),
                3
            )
            
        if damage >= 2:
            # Second damage indicator - horizontal crack
            pygame.draw.line(
                self.screen, 
                (30, 30, 30), 
                (pos_x, pos_y + size * 0.5),
                (pos_x + size, pos_y + size * 0.5),
                2
            )
    def _render_missile(self, missile) -> None:
        """
        Render a missile entity with sprites.
    def add_explosion(self, x: float, y: float, size: int = 40) -> None:
        """
        Add a new explosion animation at the specified position.
        
        Args:
            x: X position of explosion center
            y: Y position of explosion center
            size: Size of the explosion
        """
        # Adjust position to center the explosion
        x = x - size // 2
        y = y - size // 2
        
        # Create a new explosion entry
        explosion = {
            'x': x,
            'y': y,
            'size': size,
            'frame': 0,
            'max_frames': self.explosion_frame_count,
            'frame_delay': 3,  # Frames to wait before advancing to next animation frame
            'current_delay': 0
        }
        
        self.explosions.append(explosion)
        logging.debug(f"Added explosion at ({x}, {y}) with size {size}")
        
    def _update_explosions(self) -> None:
        """Update and render all active explosion animations."""
        # Initialize explosion frames if not already loaded
        if self.explosion_frames is None:
            self.explosion_frames = self.sprite_manager.load_animation(
                "effects/explosion", 
                (64, 64), 
                self.explosion_frame_count
            )
        
        # Update and render each explosion
        updated_explosions = []
        for explosion in self.explosions:
            # Increment delay counter
            explosion['current_delay'] += 1
            
            # Advance to next frame if delay reached
            if explosion['current_delay'] >= explosion['frame_delay']:
                explosion['current_delay'] = 0
                explosion['frame'] += 1
            
            # Skip explosions that have completed animation
            if explosion['frame'] >= explosion['max_frames']:
                continue
                
            # Get the current frame
            frame_idx = min(explosion['frame'], len(self.explosion_frames) - 1)
            frame = self.explosion_frames[frame_idx]
            
            # Scale frame to explosion size
            size = explosion['size']
            scaled_frame = pygame.transform.scale(frame, (size, size))
            
            # Draw explosion
            self.screen.blit(
                scaled_frame, 
                (explosion['x'], explosion['y'])
            )
            
            # Keep explosion for next frame
            updated_explosions.append(explosion)
            
        # Replace explosion list with updated one
        self.explosions = updated_explosions
        Args:
            missile: Missile instance
        """
        if hasattr(missile, 'position') and hasattr(missile, 'size'):
            # Determine sprite size - make it a bit more elongated
            width = missile.size
            height = int(missile.size * 1.5)
            size = (width, height)

            # Calculate rotation angle based on direction
            rotation = 0
            if hasattr(missile, 'direction'):
                # Convert direction to angle in degrees
                dx, dy = missile.direction
                if dx != 0 or dy != 0:
                    import math
                    angle_rad = math.atan2(dy, dx)
                    rotation = math.degrees(angle_rad) + 90  # Adjust so 0 points up

            # Render the missile sprite with rotation
            self.sprite_manager.render(
                screen=self.screen,
                entity_type="missile",
                position=missile.position,
                size=size,
                rotation=rotation
            )

            # Add a trail effect if effects are enabled
            if self.enable_effects and self.frame_count % 2 == 0:
                self._add_missile_trail(missile)

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
