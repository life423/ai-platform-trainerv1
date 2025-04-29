"""
Scaled renderer module for the AI Platform Trainer.
Handles drawing game elements with proper scaling based on the screen resolution.
"""

import logging
import pygame


class ScaledRenderer:
    """
    Renderer that handles scaling and drawing game elements.
    Uses a ScreenManager to convert between game coordinates and screen coordinates.
    """

    def __init__(self, screen_manager):
        """
        Initialize the renderer with a screen manager.
        
        Args:
            screen_manager: ScreenManager instance for coordinate transformations
        """
        self.screen_manager = screen_manager
        self.screen = screen_manager.screen
        self.BACKGROUND_COLOR = (135, 206, 235)  # Light blue
        self.BORDER_COLOR = (100, 149, 237)  # Cornflower blue

        # Optional effects
        self.enable_effects = True
        self.frame_count = 0
        self.particle_effects = []
        
        # Explosion animation tracking
        self.explosions = []
        self.explosion_frame_count = 4

        # Enemy type color variations
        self.enemy_colors = {
            "standard": (255, 50, 50),  # Red
            "fast": (255, 180, 50),  # Orange
            "tank": (120, 50, 120),  # Purple
        }

        self.explosion_frames = self._load_explosion_frames()

    def _load_explosion_frames(self):
        """Load explosion sprite frames or create a default if not available."""
        frames = []
        try:
            explosion_sheet = pygame.image.load("assets/explosion.png")
            frames = [explosion_sheet]  # Modify if you have multiple frames
        except Exception:
            logging.warning("Could not load explosion frames, using default.")
            default_frame = pygame.Surface((32, 32), pygame.SRCALPHA)
            pygame.draw.circle(default_frame, (255, 0, 0), (16, 16), 16)
            frames = [default_frame]
        return frames

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
            # Fill the entire screen with background color (no black edges)
            self.screen.fill(self.BACKGROUND_COLOR)
            
            # Draw the game area border if needed
            self._draw_game_border()

            # Update frame counter for animations
            self.frame_count += 1

            if menu_active:
                # Draw menu
                if hasattr(menu, "draw"):
                    menu.draw(self.screen)
                logging.debug("Menu rendered.")
            else:
                # Render game elements
                if hasattr(player, "position") and player:
                    # In case we have multiple enemies, check player's game attribute
                    enemies = []
                    if hasattr(player, "game") and hasattr(player.game, "enemies"):
                        enemies = player.game.enemies

                    self._render_game(player, enemy, enemies)
                logging.debug("Game elements rendered.")

            # Update display
            pygame.display.flip()

        except Exception as e:
            logging.error(f"Error during rendering: {e}")

    def _draw_game_border(self):
        """
        Draw a border around the game area if it doesn't fill the screen.
        This helps visually define the game area on widescreen displays.
        """
        game_area = self.screen_manager.get_design_area()
        screen_rect = self.screen_manager.get_screen_rect()
        
        if game_area.width < screen_rect.width or game_area.height < screen_rect.height:
            # Draw a 2px border around the game area
            pygame.draw.rect(self.screen, self.BORDER_COLOR, game_area, 2)

    def _render_game(self, player, enemy, enemies=None) -> None:
        """
        Render the game elements during gameplay.

        Args:
            player: Player instance
            enemy: Enemy instance
            enemies: List of enemy instances (optional)
        """
        # Update and render any active explosions
        self._update_explosions()

        # Draw player
        if hasattr(player, "position") and hasattr(player, "size"):
            self._render_player(player)

            # Render player missiles
            if hasattr(player, "missiles"):
                for missile in player.missiles:
                    self._render_missile(missile)

        # Render multiple enemies if available
        if enemies:
            for enemy_obj in enemies:
                if (
                    hasattr(enemy_obj, "pos")
                    and hasattr(enemy_obj, "size")
                    and enemy_obj.visible
                ):
                    self._render_enemy(enemy_obj)

        # Fallback to single enemy for backward compatibility
        elif hasattr(enemy, "pos") and hasattr(enemy, "size") and enemy.visible:
            self._render_enemy(enemy)

        # Render particle effects if enabled
        if self.enable_effects:
            self._update_and_render_effects()

    def _render_player(self, player) -> None:
        """
        Render the player entity with proper scaling.

        Args:
            player: Player instance
        """
        # Get scaled rectangle for the player
        rect = self.screen_manager.get_scaled_rect(
            player.position["x"], player.position["y"], player.size, player.size
        )

        # Draw the player rectangle
        pygame.draw.rect(self.screen, (0, 255, 0), rect)  # Green

    def _render_enemy(self, enemy) -> None:
        """
        Render the enemy entity with proper scaling.

        Args:
            enemy: Enemy instance
        """
        # Get scaled rectangle for the enemy
        rect = self.screen_manager.get_scaled_rect(
            enemy.pos["x"], enemy.pos["y"], enemy.size, enemy.size
        )

        # Check if the enemy is fading in
        alpha = 255
        if hasattr(enemy, "fading_in") and enemy.fading_in:
            alpha = enemy.fade_alpha
            
        # Determine enemy type for visual differentiation
        enemy_type = "enemy"  # Default
        if hasattr(enemy, "enemy_type"):
            enemy_type = enemy.enemy_type
        color = self.enemy_colors.get(enemy_type, (255, 0, 0))
        
        # Create a surface with transparency if needed
        surface = pygame.Surface(
            (rect.width, rect.height), 
            pygame.SRCALPHA
        )
        surface.fill((*color, alpha))
        self.screen.blit(surface, rect)

        # For tank enemies, show damage state if applicable
        if (
            hasattr(enemy, "enemy_type")
            and enemy.enemy_type == "tank"
            and hasattr(enemy, "damage_state")
        ):
            self._render_tank_damage_state(enemy, rect)

    def _render_tank_damage_state(self, enemy, rect) -> None:
        """
        Render visual indicators of tank enemy damage state.

        Args:
            enemy: Tank enemy instance
            rect: Scaled rectangle for drawing
        """
        if not hasattr(enemy, "damage_state") or enemy.damage_state == 0:
            return

        # Add cracks or damage indicators based on damage state
        damage = enemy.damage_state

        # Draw damage indicators (simple cracks)
        if damage >= 1:
            # First damage indicator - diagonal crack
            pygame.draw.line(
                self.screen,
                (30, 30, 30),
                (rect.x + rect.width * 0.2, rect.y + rect.height * 0.2),
                (rect.x + rect.width * 0.8, rect.y + rect.height * 0.8),
                max(1, int(3 * self.screen_manager.scale)),
            )

        if damage >= 2:
            # Second damage indicator - horizontal crack
            pygame.draw.line(
                self.screen,
                (30, 30, 30),
                (rect.x, rect.y + rect.height * 0.5),
                (rect.x + rect.width, rect.y + rect.height * 0.5),
                max(1, int(2 * self.screen_manager.scale)),
            )

    def _render_missile(self, missile) -> None:
        """
        Render a missile entity with proper scaling.

        Args:
            missile: Missile instance
        """
        if hasattr(missile, "position") and hasattr(missile, "size"):
            # Determine sprite size - make it a bit more elongated
            width = missile.size
            height = int(missile.size * 1.5)
            
            # Get scaled rectangle for the missile
            rect = self.screen_manager.get_scaled_rect(
                missile.position["x"], 
                missile.position["y"], 
                width, 
                height
            )

            # Calculate rotation angle based on direction
            rotation = 0
            if hasattr(missile, "direction"):
                # Convert direction to angle in degrees
                dx, dy = missile.direction
                if dx != 0 or dy != 0:
                    import math

                    angle_rad = math.atan2(dy, dx)
                    rotation = math.degrees(angle_rad) + 90  # Adjust so 0 points up

            # Simple colored rectangle for missiles
            missile_color = (255, 255, 0)  # Yellow
            
            # Create rotated surface
            surface = pygame.Surface((rect.width, rect.height), pygame.SRCALPHA)
            surface.fill(missile_color)
            
            # Apply rotation if needed
            if rotation != 0:
                surface = pygame.transform.rotate(surface, -rotation)
                
            # Draw the missile
            self.screen.blit(surface, rect)

            # Add a trail effect if effects are enabled
            if self.enable_effects and self.frame_count % 2 == 0:
                self._add_missile_trail(missile)

    def add_explosion(self, x: float, y: float, size: int = 40) -> None:
        """
        Add a new explosion animation at the specified position.

        Args:
            x: X position of explosion center in game coordinates
            y: Y position of explosion center in game coordinates
            size: Size of the explosion in game coordinates
        """
        # Convert position to screen coordinates
        screen_x, screen_y = self.screen_manager.get_scaled_position(x, y)
        scaled_size = size * self.screen_manager.scale
        
        # Adjust position to center the explosion
        screen_x = screen_x - scaled_size // 2
        screen_y = screen_y - scaled_size // 2

        # Create a new explosion entry
        explosion = {
            "x": screen_x,
            "y": screen_y,
            "size": scaled_size,
            "frame": 0,
            "max_frames": self.explosion_frame_count,
            "frame_delay": 3,
            "current_delay": 0,
        }

        self.explosions.append(explosion)
        logging.debug(f"Added explosion at ({x}, {y}) with size {size}")

    def _update_explosions(self) -> None:
        """Update and render all active explosion animations."""
        if self.explosion_frames is None:
            return

        # Update and render each explosion
        updated_explosions = []
        for explosion in self.explosions:
            # Increment delay counter
            explosion["current_delay"] += 1

            # Advance to next frame if delay reached
            if explosion["current_delay"] >= explosion["frame_delay"]:
                explosion["current_delay"] = 0
                explosion["frame"] += 1

            # Skip explosions that have completed animation
            if explosion["frame"] >= explosion["max_frames"]:
                continue

            # Get the current frame
            frame_idx = min(explosion["frame"], len(self.explosion_frames) - 1)
            frame = self.explosion_frames[frame_idx]

            # Scale frame to explosion size
            size = explosion["size"]
            scaled_frame = pygame.transform.scale(frame, (size, size))

            # Draw explosion
            self.screen.blit(scaled_frame, (explosion["x"], explosion["y"]))

            # Keep explosion for next frame
            updated_explosions.append(explosion)

        # Replace explosion list with updated one
        self.explosions = updated_explosions

    def _add_missile_trail(self, missile) -> None:
        """
        Add a particle effect trail behind a missile.

        Args:
            missile: Missile instance
        """
        if not hasattr(missile, "position"):
            return

        # Get missile position in screen coordinates
        x, y = self.screen_manager.get_scaled_position(
            missile.position["x"] + missile.size // 2,
            missile.position["y"] + missile.size // 2
        )

        # Trail particles
        import random

        for _ in range(2):
            # Random offset
            scale_factor = self.screen_manager.scale
            offset_x = random.randint(-3, 3) * scale_factor
            offset_y = random.randint(-3, 3) * scale_factor

            # Random size
            size = random.randint(2, 5) * scale_factor

            # Random lifetime
            lifetime = random.randint(5, 15)

            # Create particle
            particle = {
                "x": x + offset_x,
                "y": y + offset_y,
                "size": size,
                "color": (255, 255, 200, 200),  # Yellow-ish with alpha
                "lifetime": lifetime,
                "max_lifetime": lifetime,
            }

            self.particle_effects.append(particle)

    def _update_and_render_effects(self) -> None:
        """Update and render all particle effects."""
        # Update particles
        updated_particles = []
        for particle in self.particle_effects:
            # Decrease lifetime
            particle["lifetime"] -= 1

            # Skip dead particles
            if particle["lifetime"] <= 0:
                continue

            # Calculate alpha based on remaining lifetime
            alpha = int(255 * (particle["lifetime"] / particle["max_lifetime"]))
            color = list(particle["color"])
            if len(color) > 3:
                color[3] = min(color[3], alpha)
            else:
                color.append(alpha)

            # Draw particle
            pygame.draw.circle(
                self.screen,
                color,
                (int(particle["x"]), int(particle["y"])),
                particle["size"],
            )

            # Keep particle for next frame
            updated_particles.append(particle)

        # Replace particle list with updated one
        self.particle_effects = updated_particles

    def update_screen_reference(self, screen_manager):
        """
        Update the screen reference after a display change.
        
        Args:
            screen_manager: The updated ScreenManager instance
        """
        self.screen_manager = screen_manager
