"""
Background renderer for AI Platform Trainer.

This module provides different background styles for the game,
including starfield, grid, and gradient effects.
"""

import random
import math
import pygame


class BackgroundManager:
    """
    Manages and renders different types of backgrounds for the game.
    """

    def __init__(self, screen_width: int, screen_height: int):
        """
        Initialize the background manager.

        Args:
            screen_width: Width of the game screen
            screen_height: Height of the game screen
        """
        self.screen_width = screen_width
        self.screen_height = screen_height
        self.background_type = "starfield"  # Default background

        # Cache for rendered backgrounds
        self.background_cache = {}

        # Star field properties
        self.stars = []
        self.star_count = 100
        self._generate_stars()

        # Grid properties
        self.grid_size = 50
        self.grid_color = (50, 50, 150, 50)  # Semi-transparent blue

        # General properties
        self.frame_count = 0

    def _generate_stars(self):
        """Generate random stars for the starfield background."""
        self.stars = []
        for _ in range(self.star_count):
            x = random.randint(0, self.screen_width)
            y = random.randint(0, self.screen_height)
            size = random.choice([1, 1, 1, 2, 2, 3])  # Mostly small stars
            brightness = random.randint(150, 255)
            speed = random.uniform(0.1, 0.5)

            self.stars.append(
                {
                    "x": x,
                    "y": y,
                    "size": size,
                    "color": (brightness, brightness, brightness),
                    "speed": speed,
                }
            )

    def set_background(self, bg_type: str) -> None:
        """
        Change the current background type.

        Args:
            bg_type: Type of background ("starfield", "grid", "gradient")
        """
        if bg_type in ["starfield", "grid", "gradient"]:
            self.background_type = bg_type
        else:
            raise ValueError(f"Unknown background type: {bg_type}")

    def render(self, screen: pygame.Surface) -> None:
        """
        Render the current background type.

        Args:
            screen: Pygame surface to render on
        """
        self.frame_count += 1

        if self.background_type == "starfield":
            self._render_starfield(screen)
        elif self.background_type == "grid":
            self._render_grid(screen)
        elif self.background_type == "gradient":
            self._render_gradient(screen)

    def _render_starfield(self, screen: pygame.Surface) -> None:
        """
        Render a moving starfield background.

        Args:
            screen: Pygame surface to render on
        """
        # Fill with dark blue/black
        screen.fill((5, 10, 20))

        # Update and draw stars
        for star in self.stars:
            # Move star slightly (parallax effect)
            star["y"] += star["speed"]

            # Wrap around when reaching the bottom
            if star["y"] > self.screen_height:
                star["y"] = 0
                star["x"] = random.randint(0, self.screen_width)

            # Draw star
            pygame.draw.circle(
                screen, star["color"], (int(star["x"]), int(star["y"])), star["size"]
            )

            # Occasionally make stars twinkle
            if random.random() < 0.01:
                brightness = random.randint(150, 255)
                star["color"] = (brightness, brightness, brightness)

    def _render_grid(self, screen: pygame.Surface) -> None:
        """
        Render a grid background.

        Args:
            screen: Pygame surface to render on
        """
        # Create a new surface if not cached
        if "grid" not in self.background_cache:
            # Create a surface with alpha channel
            grid_surface = pygame.Surface(
                (self.screen_width, self.screen_height), pygame.SRCALPHA
            )

            # Fill with dark background
            grid_surface.fill((10, 20, 40, 255))

            # Draw vertical grid lines
            for x in range(0, self.screen_width, self.grid_size):
                pygame.draw.line(
                    grid_surface, self.grid_color, (x, 0), (x, self.screen_height)
                )

            # Draw horizontal grid lines
            for y in range(0, self.screen_height, self.grid_size):
                pygame.draw.line(
                    grid_surface, self.grid_color, (0, y), (self.screen_width, y)
                )

            self.background_cache["grid"] = grid_surface

        # Blit the cached grid
        screen.blit(self.background_cache["grid"], (0, 0))

    def _render_gradient(self, screen: pygame.Surface) -> None:
        """
        Render a gradient background with slowly changing colors.

        Args:
            screen: Pygame surface to render on
        """
        # Create a new gradient if not cached or if it needs refreshing
        if "gradient" not in self.background_cache or self.frame_count % 30 == 0:
            # Create a surface
            gradient = pygame.Surface((self.screen_width, self.screen_height))

            # Calculate pulsing colors based on time
            t = self.frame_count / 60  # Time in seconds
            r = int(128 + 127 * math.sin(t * 0.1))
            g = int(50 + 50 * math.sin(t * 0.05))
            b = int(128 + 127 * math.sin(t * 0.07 + 2))

            # Create gradient from one color to another
            for y in range(self.screen_height):
                # Interpolate colors from top to bottom
                color_ratio = y / self.screen_height
                color = (
                    int(10 + (r - 10) * color_ratio),
                    int(20 + (g - 20) * color_ratio),
                    int(40 + (b - 40) * color_ratio),
                )

                # Draw a colored line
                pygame.draw.line(gradient, color, (0, y), (self.screen_width, y))

            self.background_cache["gradient"] = gradient

        # Blit the cached gradient
        screen.blit(self.background_cache["gradient"], (0, 0))
