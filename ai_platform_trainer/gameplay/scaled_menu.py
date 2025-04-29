"""
Scaled menu module for the AI Platform Trainer.
Handles drawing menu elements with proper scaling based on the screen resolution.
"""

import logging
import pygame


class ScaledMenu:
    """
    Menu that handles scaling and drawing menu elements.
    Uses a ScreenManager to convert between game coordinates and screen coordinates.
    """
    
    def __init__(self, screen_manager):
        """
        Initialize the menu with a screen manager.
        
        Args:
            screen_manager: ScreenManager instance for coordinate transformations
        """
        self.screen_manager = screen_manager
        
        # Menu colors
        self.TITLE_COLOR = (255, 255, 255)
        self.OPTION_COLOR = (220, 220, 220)
        self.SELECTED_COLOR = (255, 255, 0)
        self.BACKGROUND_COLOR = (0, 0, 100, 180)  # Semi-transparent blue
        
        # Menu state
        self.selected_option = 0
        self.options = ["play", "train", "exit"]
        self.option_labels = ["Play Game", "Training Mode", "Exit"]
        
        # Load fonts
        self.load_fonts()
        
    def load_fonts(self):
        """Load and create fonts for menu elements."""
        try:
            # Use scaled font sizes based on screen resolution
            title_size = int(100 * self.screen_manager.scale)
            option_size = int(50 * self.screen_manager.scale)
            
            # Ensure minimum readable size
            title_size = max(24, title_size)
            option_size = max(16, option_size)
            
            self.font_title = pygame.font.Font(None, title_size)
            self.font_option = pygame.font.Font(None, option_size)
        except Exception as e:
            logging.error(f"Error loading fonts: {e}")
            # Use default fonts as fallback
            self.font_title = pygame.font.Font(None, 72)
            self.font_option = pygame.font.Font(None, 36)
    
    def draw(self, screen):
        """
        Draw the menu on the screen.
        
        Args:
            screen: Pygame surface to draw on
        """
        try:
            # Get design area dimensions and position
            design_area = self.screen_manager.get_design_area()
            center_x = design_area.centerx
            
            # Create semi-transparent background surface for menu
            menu_bg = pygame.Surface((design_area.width, design_area.height), pygame.SRCALPHA)
            menu_bg.fill(self.BACKGROUND_COLOR)
            screen.blit(menu_bg, design_area)
            
            # Render title with scaling
            title_text = "Pixel Pursuit"
            title_surface = self.font_title.render(title_text, True, self.TITLE_COLOR)
            
            # Center the title horizontally and position it near the top
            title_rect = title_surface.get_rect()
            title_pos_y = design_area.top + design_area.height * 0.2
            screen.blit(
                title_surface, 
                (center_x - title_rect.width // 2, title_pos_y)
            )
            
            # Render menu options
            option_y = design_area.top + design_area.height * 0.4
            option_spacing = self.font_option.get_height() * 1.5
            
            for i, label in enumerate(self.option_labels):
                # Determine color based on selection
                color = self.SELECTED_COLOR if i == self.selected_option else self.OPTION_COLOR
                
                # Render option text
                option_surface = self.font_option.render(label, True, color)
                option_rect = option_surface.get_rect()
                
                # Center option horizontally and position it
                screen.blit(
                    option_surface, 
                    (center_x - option_rect.width // 2, option_y)
                )
                
                option_y += option_spacing
                
        except Exception as e:
            logging.error(f"Error drawing menu: {e}")
    
    def handle_menu_events(self, event):
        """
        Handle menu event inputs.
        
        Args:
            event: Pygame event to handle
            
        Returns:
            Selected option string or None if no option was selected
        """
        try:
            # Handle keyboard navigation
            if event.type == pygame.KEYDOWN:
                if event.key == pygame.K_UP:
                    self.selected_option = (self.selected_option - 1) % len(self.options)
                elif event.key == pygame.K_DOWN:
                    self.selected_option = (self.selected_option + 1) % len(self.options)
                elif event.key == pygame.K_RETURN or event.key == pygame.K_SPACE:
                    return self.options[self.selected_option]
            
            # Handle mouse clicks
            elif event.type == pygame.MOUSEBUTTONDOWN and event.button == 1:
                # Convert mouse position to design coordinates
                mouse_x, mouse_y = pygame.mouse.get_pos()
                design_x, design_y = self.screen_manager.convert_to_design_coordinates(mouse_x, mouse_y)
                
                # Get design area dimensions
                width, height = self.screen_manager.get_game_dimensions()
                center_x = width // 2
                
                # Calculate option positions and check if clicked
                option_y = height * 0.4
                option_spacing = 50  # Approximate spacing in design coordinates
                option_height = 40   # Approximate height in design coordinates
                
                for i in range(len(self.options)):
                    option_rect = pygame.Rect(
                        center_x - 100,  # Approximate width for click detection
                        option_y - 10,   # Padding above text
                        200,             # Width for click detection
                        option_height    # Height for click detection
                    )
                    
                    if option_rect.collidepoint(design_x, design_y):
                        return self.options[i]
                    
                    option_y += option_spacing
            
        except Exception as e:
            logging.error(f"Error handling menu event: {e}")
        
        return None
    
    def update_screen_reference(self, screen_manager):
        """
        Update the screen manager reference after a display change.
        
        Args:
            screen_manager: The updated ScreenManager instance
        """
        self.screen_manager = screen_manager
        self.load_fonts()  # Reload fonts with new scaling
