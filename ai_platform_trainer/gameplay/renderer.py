import pygame
import logging
from typing import Optional, Any
from ai_platform_trainer.gameplay.config import config as game_config


class Renderer:
    def __init__(self, screen: pygame.Surface) -> None:
        """
        Initialize the Renderer.

        Args:
            screen: Pygame display surface
        """
        self.screen = screen
        self.BACKGROUND_COLOR = (135, 206, 235)  # Light blue
        
        # Initialize fonts for UI elements
        pygame.font.init()
        self.font = pygame.font.SysFont(None, game_config.FONT_SIZE)
        self.large_font = pygame.font.SysFont(None, game_config.FONT_SIZE * 2)
        
        # Health bar settings
        self.health_bar_width = 20
        self.health_bar_height = 150
        self.health_bar_border = 2
        self.health_bar_position = (20, 80)
        
        # Health icon is a small heart
        self.health_icon = self._create_heart_icon(20)

    def render(self, menu, player, enemy=None, menu_active: bool = False, 
               missile_manager=None, game_mode=None, enemy_manager=None,
               powerup_manager=None) -> None:
        """
        Render the game elements on the screen.

        Args:
            menu: Menu instance
            player: Player instance
            enemy: Legacy single enemy instance
            menu_active: Boolean indicating if the menu is active
            missile_manager: Missile manager to render missiles
            game_mode: Current game mode ('play' or 'train')
            enemy_manager: Enemy manager for multiple enemies in play mode
        """
        try:
            self.screen.fill(self.BACKGROUND_COLOR)
            
            if menu_active:
                # Draw the menu
                menu.draw(self.screen)
                logging.debug("Menu rendered.")
            else:
                # Draw the player
                player.draw(self.screen)
                
                # Draw enemies
                if game_mode == "play" and enemy_manager:
                    # Draw multiple enemies in play mode
                    enemy_manager.draw(self.screen)
                    logging.debug("Multiple enemies rendered via enemy_manager.")
                elif enemy:
                    # Legacy support for single enemy in training mode
                    enemy.draw(self.screen)
                    logging.debug("Single enemy rendered (legacy mode).")
                
                # Draw missiles if missile manager is available
                if missile_manager:
                    missile_manager.draw(self.screen)
                    logging.debug("Missiles rendered.")
                
                # Draw powerups if available
                if powerup_manager and game_mode == "play":
                    powerup_manager.draw(self.screen)
                
                # Draw UI elements in play mode
                if game_mode == "play":
                    self._draw_ui(player)
                    
                    # Draw game over screen if needed
                    if hasattr(player, 'health') and player.health <= 0:
                        self._draw_game_over(player.score)
            
            logging.debug("Frame rendered.")
        except Exception as e:
            logging.error(f"Error during rendering: {e}")
    
    def _draw_ui(self, player: Any) -> None:
        """
        Draw UI elements like score, health, and level.
        
        Args:
            player: Player instance with score and health attributes
        """
        # Draw score
        score_text = f"Score: {player.score}"
        score_surface = self.font.render(score_text, True, game_config.FONT_COLOR)
        self.screen.blit(score_surface, game_config.SCORE_POSITION)
        
        # Draw health
        self._draw_health(player.health, player.max_health)
        
        # Draw active power-ups
        if hasattr(player, 'active_powerups'):
            self._draw_powerups(player.active_powerups)
    
    def _draw_health(self, health: int, max_health: int) -> None:
        """
        Draw health indicator.
        
        Args:
            health: Current health value
            max_health: Maximum health value
        """
        # Draw health as a line of icons
        for i in range(max_health):
            # Full hearts for current health, empty hearts for lost health
            if i < health:
                color = (255, 0, 0)  # Red
            else:
                color = (100, 100, 100)  # Gray
                
            # Position hearts in a row
            pos = (game_config.HEALTH_POSITION[0] + i * 30, game_config.HEALTH_POSITION[1])
            
            # Draw heart outline
            pygame.draw.rect(
                self.screen, 
                (0, 0, 0), 
                (pos[0]-1, pos[1]-1, 22, 22), 
                1
            )
            
            # Draw heart
            self.screen.blit(self._create_heart_icon(20, color), pos)
    
    def _create_heart_icon(self, size: int, color=(255, 0, 0)) -> pygame.Surface:
        """
        Create a heart-shaped icon for health display.
        
        Args:
            size: Size of the heart icon
            color: Color of the heart (default: red)
            
        Returns:
            A surface with the heart icon
        """
        heart = pygame.Surface((size, size), pygame.SRCALPHA)
        
        # Draw a simple heart shape
        half_size = size // 2
        quarter_size = size // 4
        
        # Draw two circles for the top of the heart
        pygame.draw.circle(
            heart, 
            color, 
            (quarter_size, quarter_size), 
            quarter_size
        )
        pygame.draw.circle(
            heart, 
            color, 
            (half_size + quarter_size, quarter_size),
            quarter_size
        )
        
        # Draw a triangle for the bottom of the heart
        pygame.draw.polygon(
            heart,
            color,
            [
                (0, quarter_size),
                (size, quarter_size),
                (half_size, size)
            ]
        )
        
        return heart
    
    def _draw_powerups(self, active_powerups: list) -> None:
        """
        Draw indicators for active power-ups.
        
        Args:
            active_powerups: List of active power-up names
        """
        # Draw power-up icons at the top right
        base_x = self.screen.get_width() - 150
        base_y = 20
        
        # Power-up icons and colors
        powerup_info = {
            "shield": ((0, 100, 255), "Shield"),  # Blue
            "rapid_fire": ((255, 165, 0), "Rapid Fire"),  # Orange
            "speed_boost": ((0, 255, 0), "Speed")  # Green
        }
        
        # Draw each active power-up
        for i, powerup in enumerate(active_powerups):
            if powerup in powerup_info:
                color, label = powerup_info[powerup]
                
                # Draw power-up indicator
                pygame.draw.rect(
                    self.screen,
                    color,
                    (base_x, base_y + i * 30, 20, 20)
                )
                
                # Draw power-up label
                label_surface = self.font.render(label, True, game_config.FONT_COLOR)
                self.screen.blit(label_surface, (base_x + 30, base_y + i * 30))
    
    def _draw_game_over(self, score: int) -> None:
        """
        Draw game over screen.
        
        Args:
            score: Final score to display
        """
        # Dim the screen
        dim_surface = pygame.Surface(self.screen.get_size(), pygame.SRCALPHA)
        dim_surface.fill((0, 0, 0, 128))  # Semi-transparent black
        self.screen.blit(dim_surface, (0, 0))
        
        # Draw game over text
        game_over_text = self.large_font.render("GAME OVER", True, (255, 0, 0))
        text_rect = game_over_text.get_rect(center=(
            self.screen.get_width() // 2, 
            self.screen.get_height() // 2 - 50
        ))
        self.screen.blit(game_over_text, text_rect)
        
        # Draw score
        score_text = self.font.render(f"Final Score: {score}", True, (255, 255, 255))
        score_rect = score_text.get_rect(center=(
            self.screen.get_width() // 2,
            self.screen.get_height() // 2
        ))
        self.screen.blit(score_text, score_rect)
        
        # Draw restart instructions
        restart_text = self.font.render("Press R to Restart", True, (255, 255, 255))
        restart_rect = restart_text.get_rect(center=(
            self.screen.get_width() // 2,
            self.screen.get_height() // 2 + 50
        ))
        self.screen.blit(restart_text, restart_rect)
