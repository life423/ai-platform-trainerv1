"""
Enhanced Menu System for AI Platform Trainer.

This module provides an improved menu system with support for
supervised learning and reinforcement learning agent selection.
"""
import pygame
from typing import Optional, Dict, Any


class Menu:
    """Enhanced menu system with RL support."""
    
    def __init__(self, screen_width: int, screen_height: int):
        # Screen dimensions
        self.screen_width = screen_width
        self.screen_height = screen_height
        
        # Main menu options
        self.main_menu_options = ["Play (Supervised AI)", "Reinforcement Learning", "Train", "Help", "Exit"]
        self.selected_main_option = 0
        
        # RL submenu options
        self.rl_menu_options = [
            "Play Against Pretrained RL Agent",
            "Play Against Learning RL Agent", 
            "Back to Main Menu"
        ]
        self.selected_rl_option = 0
        
        # RL difficulty options (for pretrained agents)
        self.rl_difficulty_options = ["Easy", "Medium", "Hard", "Back"]
        self.selected_difficulty = 0
        
        # Menu state management
        self.current_menu = "main"  # "main", "rl", "rl_difficulty"
        self.show_help = False
        
        # Store clickable rectangles for mouse interaction
        self.option_rects: Dict[int, pygame.Rect] = {}
        
        # Fonts and colors
        self.font_title = pygame.font.Font(None, 100)
        self.font_option = pygame.font.Font(None, 60)
        self.font_subtitle = pygame.font.Font(None, 48)
        self.font_info = pygame.font.Font(None, 34)
        
        # Colors
        self.color_background = (135, 206, 235)  # Light blue
        self.color_title = (0, 51, 102)  # Dark blue
        self.color_option = (245, 245, 245)  # Light gray
        self.color_selected = (255, 223, 0)  # Yellow
        self.color_accent = (220, 20, 60)  # Crimson

    def handle_menu_events(self, event: pygame.event.Event) -> Optional[str]:
        """
        Handle menu events and return selected action.
        
        Args:
            event: Pygame event to handle
            
        Returns:
            String representing the selected action, or None
        """
        if self.show_help:
            return self._handle_help_events(event)
        elif self.current_menu == "main":
            return self._handle_main_menu_events(event)
        elif self.current_menu == "rl":
            return self._handle_rl_menu_events(event)
        elif self.current_menu == "rl_difficulty":
            return self._handle_difficulty_menu_events(event)
        
        return None
    
    def _handle_main_menu_events(self, event: pygame.event.Event) -> Optional[str]:
        """Handle main menu events."""
        if event.type == pygame.KEYDOWN:
            if event.key in [pygame.K_UP, pygame.K_w]:
                self.selected_main_option = (self.selected_main_option - 1) % len(self.main_menu_options)
            elif event.key in [pygame.K_DOWN, pygame.K_s]:
                self.selected_main_option = (self.selected_main_option + 1) % len(self.main_menu_options)
            elif event.key in [pygame.K_RETURN, pygame.K_KP_ENTER]:
                return self._process_main_menu_selection()
            elif event.key == pygame.K_ESCAPE:
                return "exit"
        
        elif event.type == pygame.MOUSEBUTTONDOWN and event.button == 1:
            mouse_x, mouse_y = pygame.mouse.get_pos()
            for index, rect in self.option_rects.items():
                if rect.collidepoint(mouse_x, mouse_y):
                    self.selected_main_option = index
                    return self._process_main_menu_selection()
        
        return None
    
    def _process_main_menu_selection(self) -> Optional[str]:
        """Process the selected main menu option."""
        selected = self.main_menu_options[self.selected_main_option]
        
        if selected == "Play (Supervised AI)":
            return "play_sl"
        elif selected == "Reinforcement Learning":
            self.current_menu = "rl"
            self.selected_rl_option = 0
            return None
        elif selected == "Train":
            return "train"
        elif selected == "Help":
            self.show_help = True
            return None
        elif selected == "Exit":
            return "exit"
        
        return None
    
    def _handle_rl_menu_events(self, event: pygame.event.Event) -> Optional[str]:
        """Handle RL submenu events."""
        if event.type == pygame.KEYDOWN:
            if event.key in [pygame.K_UP, pygame.K_w]:
                self.selected_rl_option = (self.selected_rl_option - 1) % len(self.rl_menu_options)
            elif event.key in [pygame.K_DOWN, pygame.K_s]:
                self.selected_rl_option = (self.selected_rl_option + 1) % len(self.rl_menu_options)
            elif event.key in [pygame.K_RETURN, pygame.K_KP_ENTER]:
                return self._process_rl_menu_selection()
            elif event.key == pygame.K_ESCAPE:
                self.current_menu = "main"
                return None
        
        elif event.type == pygame.MOUSEBUTTONDOWN and event.button == 1:
            mouse_x, mouse_y = pygame.mouse.get_pos()
            for index, rect in self.option_rects.items():
                if rect.collidepoint(mouse_x, mouse_y):
                    self.selected_rl_option = index
                    return self._process_rl_menu_selection()
        
        return None
    
    def _process_rl_menu_selection(self) -> Optional[str]:
        """Process the selected RL menu option."""
        selected = self.rl_menu_options[self.selected_rl_option]
        
        if selected == "Play Against Pretrained RL Agent":
            self.current_menu = "rl_difficulty"
            self.selected_difficulty = 0
            return None
        elif selected == "Play Against Learning RL Agent":
            return "play_rl_learning"
        elif selected == "Back to Main Menu":
            self.current_menu = "main"
            return None
        
        return None
    
    def _handle_difficulty_menu_events(self, event: pygame.event.Event) -> Optional[str]:
        """Handle difficulty selection menu events."""
        if event.type == pygame.KEYDOWN:
            if event.key in [pygame.K_UP, pygame.K_w]:
                self.selected_difficulty = (self.selected_difficulty - 1) % len(self.rl_difficulty_options)
            elif event.key in [pygame.K_DOWN, pygame.K_s]:
                self.selected_difficulty = (self.selected_difficulty + 1) % len(self.rl_difficulty_options)
            elif event.key in [pygame.K_RETURN, pygame.K_KP_ENTER]:
                return self._process_difficulty_selection()
            elif event.key == pygame.K_ESCAPE:
                self.current_menu = "rl"
                return None
        
        elif event.type == pygame.MOUSEBUTTONDOWN and event.button == 1:
            mouse_x, mouse_y = pygame.mouse.get_pos()
            for index, rect in self.option_rects.items():
                if rect.collidepoint(mouse_x, mouse_y):
                    self.selected_difficulty = index
                    return self._process_difficulty_selection()
        
        return None
    
    def _process_difficulty_selection(self) -> Optional[str]:
        """Process the selected difficulty option."""
        selected = self.rl_difficulty_options[self.selected_difficulty]
        
        if selected == "Easy":
            return "play_rl_pretrained_easy"
        elif selected == "Medium":
            return "play_rl_pretrained_medium"
        elif selected == "Hard":
            return "play_rl_pretrained_hard"
        elif selected == "Back":
            self.current_menu = "rl"
            return None
        
        return None
    
    def _handle_help_events(self, event: pygame.event.Event) -> Optional[str]:
        """Handle help screen events."""
        if event.type == pygame.KEYDOWN and event.key in [pygame.K_ESCAPE, pygame.K_RETURN]:
            self.show_help = False
        return None

    def draw(self, screen: pygame.Surface) -> None:
        """
        Draw the appropriate menu screen.
        
        Args:
            screen: Pygame surface to draw on
        """
        if self.show_help:
            self._draw_help_screen(screen)
        elif self.current_menu == "main":
            self._draw_main_menu(screen)
        elif self.current_menu == "rl":
            self._draw_rl_menu(screen)
        elif self.current_menu == "rl_difficulty":
            self._draw_difficulty_menu(screen)
    
    def _draw_main_menu(self, screen: pygame.Surface) -> None:
        """Draw the main menu."""
        screen.fill(self.color_background)
        
        # Title
        title_surface = self.font_title.render("AI Platform Trainer", True, self.color_title)
        title_rect = title_surface.get_rect(center=(self.screen_width // 2, self.screen_height // 5))
        screen.blit(title_surface, title_rect)
        
        # Menu options
        self.option_rects.clear()
        for index, option in enumerate(self.main_menu_options):
            color = self.color_selected if index == self.selected_main_option else self.color_option
            option_surface = self.font_option.render(option, True, color)
            option_rect = option_surface.get_rect(
                center=(self.screen_width // 2, self.screen_height // 2 + index * 80)
            )
            self.option_rects[index] = option_rect
            screen.blit(option_surface, option_rect)
    
    def _draw_rl_menu(self, screen: pygame.Surface) -> None:
        """Draw the RL submenu."""
        screen.fill(self.color_background)
        
        # Title
        title_surface = self.font_title.render("Reinforcement Learning", True, self.color_title)
        title_rect = title_surface.get_rect(center=(self.screen_width // 2, self.screen_height // 6))
        screen.blit(title_surface, title_rect)
        
        # Subtitle
        subtitle_surface = self.font_subtitle.render("Choose Your AI Opponent", True, self.color_accent)
        subtitle_rect = subtitle_surface.get_rect(center=(self.screen_width // 2, self.screen_height // 4))
        screen.blit(subtitle_surface, subtitle_rect)
        
        # Menu options
        self.option_rects.clear()
        for index, option in enumerate(self.rl_menu_options):
            color = self.color_selected if index == self.selected_rl_option else self.color_option
            option_surface = self.font_option.render(option, True, color)
            option_rect = option_surface.get_rect(
                center=(self.screen_width // 2, self.screen_height // 2 + index * 80)
            )
            self.option_rects[index] = option_rect
            screen.blit(option_surface, option_rect)
            
            # Add descriptions
            description = ""
            if index == 0:
                description = "Play against a fully trained RL agent"
            elif index == 1:
                description = "Watch AI learn and improve in real-time"
            
            if description:
                desc_surface = self.font_info.render(description, True, self.color_option)
                desc_rect = desc_surface.get_rect(
                    center=(self.screen_width // 2, option_rect.bottom + 25)
                )
                screen.blit(desc_surface, desc_rect)
    
    def _draw_difficulty_menu(self, screen: pygame.Surface) -> None:
        """Draw the difficulty selection menu."""
        screen.fill(self.color_background)
        
        # Title
        title_surface = self.font_title.render("Select Difficulty", True, self.color_title)
        title_rect = title_surface.get_rect(center=(self.screen_width // 2, self.screen_height // 6))
        screen.blit(title_surface, title_rect)
        
        # Subtitle
        subtitle_surface = self.font_subtitle.render("Pretrained RL Agent", True, self.color_accent)
        subtitle_rect = subtitle_surface.get_rect(center=(self.screen_width // 2, self.screen_height // 4))
        screen.blit(subtitle_surface, subtitle_rect)
        
        # Menu options
        self.option_rects.clear()
        for index, option in enumerate(self.rl_difficulty_options):
            color = self.color_selected if index == self.selected_difficulty else self.color_option
            option_surface = self.font_option.render(option, True, color)
            option_rect = option_surface.get_rect(
                center=(self.screen_width // 2, self.screen_height // 2 + index * 80)
            )
            self.option_rects[index] = option_rect
            screen.blit(option_surface, option_rect)
    
    def _draw_help_screen(self, screen: pygame.Surface) -> None:
        """Draw the help screen."""
        screen.fill(self.color_background)
        
        # Title
        title_surface = self.font_title.render("Help & Controls", True, self.color_title)
        title_rect = title_surface.get_rect(center=(self.screen_width // 2, self.screen_height // 10))
        screen.blit(title_surface, title_rect)
        
        # Help text
        help_text = [
            "Controls:",
            "• Arrow Keys or W/S to navigate menu",
            "• Arrow Keys or WASD to move player",
            "• Space to shoot missiles",
            "• Enter to select menu items",
            "• F to toggle fullscreen",
            "• M to return to menu",
            "• Escape to quit",
            "",
            "Game Modes:",
            "• Supervised AI: Classic neural network enemy",
            "• Pretrained RL: Fully trained reinforcement learning agent",
            "• Learning RL: Watch AI improve in real-time during gameplay",
            "",
            "The RL agent learns through trial and error, getting smarter",
            "as it plays. You'll see dramatic improvement over time!"
        ]
        
        y_offset = self.screen_height // 4
        for line in help_text:
            if line.startswith("•"):
                text_surface = self.font_info.render(line, True, self.color_option)
            elif line.endswith(":"):
                text_surface = self.font_option.render(line, True, self.color_selected)
            else:
                text_surface = self.font_info.render(line, True, self.color_option)
            
            text_rect = text_surface.get_rect(
                left=self.screen_width // 10, 
                top=y_offset
            )
            screen.blit(text_surface, text_rect)
            y_offset += 35
        
        # Instructions
        instructions = self.font_option.render("Press ESC or ENTER to return", True, self.color_title)
        instructions_rect = instructions.get_rect(center=(self.screen_width // 2, self.screen_height - 50))
        screen.blit(instructions, instructions_rect)
