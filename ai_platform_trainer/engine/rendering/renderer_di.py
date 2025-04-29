# file: ai_platform_trainer/gameplay/renderer_di.py
"""
Concrete renderer implementation that implements the IRenderer interface.
"""

import pygame
from ai_platform_trainer.core.interfaces import IRenderer


class Renderer(IRenderer):
    """
    Renderer class for drawing game elements to the screen.
    Implements the IRenderer interface.
    """

    def __init__(self, screen):
        """
        Initialize the renderer.

        Args:
            screen: The pygame screen to render to
        """
        self.screen = screen

        # Color definitions using a harmonious color palette
        self.BACKGROUND_COLOR = (135, 206, 235)  # Sky blue
        self.PLAYER_COLOR = (46, 204, 113)  # Emerald green
        self.ENEMY_COLOR = (231, 76, 60)  # Coral red
        self.MISSILE_COLOR = (241, 196, 15)  # Gold yellow

    def render(self, menu=None, player=None, enemy=None, menu_active=False):
        """
        Render game elements to the screen.

        Args:
            menu: The menu to render (if any)
            player: The player to render (if any)
            enemy: The enemy to render (if any)
            menu_active: Whether the menu is active
        """
        # Clear the screen with sky blue background
        self.screen.fill(self.BACKGROUND_COLOR)

        if menu_active and menu:
            # If menu is active, let the menu handle rendering
            menu.draw(self.screen)
        else:
            # Render game elements
            if player:
                # Render the player
                player_color = getattr(player, "color", self.PLAYER_COLOR)
                pygame.draw.rect(
                    self.screen,
                    player_color,
                    (
                        player.position["x"],
                        player.position["y"],
                        player.size,
                        player.size,
                    ),
                )

                # Render player missiles
                for missile in player.missiles:
                    # Handle missiles with either 'position' or 'pos' attribute
                    if hasattr(missile, "position"):
                        missile_x = missile.position["x"]
                        missile_y = missile.position["y"]
                    elif hasattr(missile, "pos"):
                        missile_x = missile.pos["x"]
                        missile_y = missile.pos["y"]
                    else:
                        continue  # Skip if missile has neither attribute

                    # Use circles for missiles with enhanced size (14)
                    missile_color = getattr(missile, "color", self.MISSILE_COLOR)
                    pygame.draw.circle(
                        self.screen,
                        missile_color,
                        (
                            int(missile_x),
                            int(missile_y),
                        ),  # Use position directly as center
                        14,  # Use a larger, more visible size
                    )

            if enemy:
                # Check if enemy should be visible
                # Some enemies use 'hidden' attribute, others use 'visible'
                is_visible = True
                if hasattr(enemy, "hidden"):
                    is_visible = not enemy.hidden
                elif hasattr(enemy, "visible"):
                    is_visible = enemy.visible

                if is_visible:
                    # Render the enemy with current alpha (for fade-in effect)
                    s = pygame.Surface((enemy.size, enemy.size), pygame.SRCALPHA)
                    # Some enemies have alpha attribute, others don't
                    alpha = getattr(enemy, "alpha", 255)
                    enemy_color = getattr(enemy, "color", self.ENEMY_COLOR)
                    pygame.draw.rect(
                        s,
                        enemy_color + (alpha,),  # RGBA with current alpha
                        (0, 0, enemy.size, enemy.size),
                    )
                    self.screen.blit(s, (enemy.pos["x"], enemy.pos["y"]))
