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
        
        # Menu colors - Using a color theory-based palette
        # Primary: Deep blue for background
        # Secondary: White and golden yellow for text
        # Accent: Teal for highlights
        self.BACKGROUND_COLOR = (16, 24, 52, 220)  # Deep blue with transparency
        self.TITLE_COLOR = (240, 240, 255)         # Almost white with slight blue tint
        self.OPTION_COLOR = (180, 180, 200)        # Light gray with blue tint
        self.SELECTED_COLOR = (255, 215, 90)       # Golden yellow
        self.ACCENT_COLOR = (0, 180, 180)          # Teal for accents and highlights
        
        # Menu state
        self.selected_option = 0
        self.options = ["play", "train", "exit"]
        self.option_labels = ["Play Game", "Training Mode", "Exit"]
        
        # Load fonts
        self.load_fonts()
        
    def load_fonts(self):
        """Load and create fonts for menu elements using proper typography scale."""
        try:
            # Typography scale based on 1.5 ratio for better visual hierarchy
            # Base size is scaled to screen resolution
            base_size = int(24 * self.screen_manager.scale_x)
            
            # Create a modular scale for different text elements
            title_size = int(base_size * 3.375)  # base × 1.5³ (3 steps up)
            option_size = int(base_size * 1.5)   # base × 1.5¹ (1 step up)
            info_size = int(base_size * 1.0)     # base size
            
            # Ensure minimum readable sizes
            title_size = max(36, title_size)
            option_size = max(24, option_size)
            info_size = max(18, info_size)
            
            # Create fonts
            self.font_title = pygame.font.Font(None, title_size)
            self.font_option = pygame.font.Font(None, option_size)
            self.font_info = pygame.font.Font(None, info_size)
        except Exception as e:
            logging.error(f"Error loading fonts: {e}")
            # Use default fonts as fallback
            self.font_title = pygame.font.Font(None, 72)
            self.font_option = pygame.font.Font(None, 36)
    
    def draw(self, screen):
        """
        Draw the menu on the screen with improved visual design.
        
        Args:
            screen: Pygame surface to draw on
        """
        try:
            # Fill the entire screen with the navy blue background color
            screen.fill(self.BACKGROUND_COLOR)  # Use the navy blue color for entire screen
            
            # Get design area dimensions and position
            design_area = self.screen_manager.get_design_area()
            center_x = design_area.centerx
            
            # Draw a semi-transparent panel for the menu
            menu_width = int(design_area.width * 0.6)
            menu_height = int(design_area.height * 0.8)
            menu_x = center_x - menu_width // 2
            menu_y = design_area.height * 0.1
            
            menu_bg = pygame.Surface((menu_width, menu_height), pygame.SRCALPHA)
            menu_bg.fill(self.BACKGROUND_COLOR)
            
            # No border - removed as requested
            
            screen.blit(menu_bg, (menu_x, menu_y))
            
            # Render title with scaling
            title_text = "Pixel Pursuit"
            title_surface = self.font_title.render(title_text, True, self.TITLE_COLOR)
            
            # Add title decoration
            title_rect = title_surface.get_rect()
            title_pos_y = menu_y + menu_height * 0.15
            
            # Draw a decorative line under the title
            line_y = title_pos_y + title_rect.height + 10
            line_width = title_rect.width * 1.2
            line_height = max(3, int(self.screen_manager.scale_y * 3))
            
            line_rect = pygame.Rect(
                center_x - line_width // 2, 
                line_y,
                line_width, 
                line_height
            )
            pygame.draw.rect(screen, self.ACCENT_COLOR, line_rect)
            
            # Center the title horizontally
            screen.blit(
                title_surface, 
                (center_x - title_rect.width // 2, title_pos_y)
            )
            
            # Calculate proper spacing for menu options
            options_start_y = menu_y + menu_height * 0.35
            
            # Use golden ratio (1.618) for pleasing spacing between options
            option_spacing = self.font_option.get_height() * 1.618
            
            # Render menu options with proper spacing
            option_y = options_start_y
            
            for i, label in enumerate(self.option_labels):
                # Determine color based on selection
                color = self.SELECTED_COLOR if i == self.selected_option else self.OPTION_COLOR
                
                # Render option text
                option_surface = self.font_option.render(label, True, color)
                option_rect = option_surface.get_rect()
                
                # No triangle indicator for selected item - removed as requested
                
                # Center option horizontally and position it
                screen.blit(
                    option_surface, 
                    (center_x - option_rect.width // 2, option_y)
                )
                
                option_y += option_spacing
            
            # Add instructions at the bottom
            instructions = "Press ESC to exit   |   Arrow keys to navigate   |   Enter to select"
            instructions_surface = self.font_info.render(instructions, True, (150, 150, 170))
            instructions_rect = instructions_surface.get_rect()
            instructions_y = menu_y + menu_height * 0.9
            
            screen.blit(
                instructions_surface,
                (center_x - instructions_rect.width // 2, instructions_y)
            )
                
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
