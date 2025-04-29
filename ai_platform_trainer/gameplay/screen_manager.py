"""
Screen manager for handling resolution, scaling, and display management.
Provides a unified interface for all screen-related operations with
automatic scaling to make the game adapt to any screen resolution.
"""

import logging
import pygame


class ScreenManager:
    """
    Screen manager handling resolution detection, scaling, and coordinate transformations
    to enable fullscreen without black edges on any resolution.
    """
    
    def __init__(self):
        """
        Initialize the screen manager with design resolution and auto-detected screen info.
        """
        # Design resolution (internal coordinates)
        self.design_width = 800
        self.design_height = 600
        
        # Get native screen resolution
        info = pygame.display.Info()
        self.native_width = info.current_w
        self.native_height = info.current_h
        
        # Current resolution (can change during gameplay)
        self.current_width = self.native_width
        self.current_height = self.native_height
        
        # Window mode state
        self.fullscreen = True
        
        # Pygame surface
        self.screen = None
        
        # Calculate scaling factors
        self.scale_x = 1.0
        self.scale_y = 1.0
        self.scale = 1.0
        self.offset_x = 0
        self.offset_y = 0
        
        # Initialize scaling factors
        self.update_scaling_factors()
        
        logging.info(f"Screen manager initialized. Native resolution: {self.native_width}x{self.native_height}")
        logging.info(f"Design resolution: {self.design_width}x{self.design_height}")
    
    def initialize_display(self, fullscreen=True):
        """
        Initialize the pygame display with the appropriate resolution and mode.
        
        Args:
            fullscreen: Whether to start in fullscreen mode
            
        Returns:
            The pygame surface
        """
        self.fullscreen = fullscreen
        
        if fullscreen:
            # Use native resolution for fullscreen
            self.current_width = self.native_width
            self.current_height = self.native_height
            self.screen = pygame.display.set_mode(
                (self.current_width, self.current_height),
                pygame.FULLSCREEN
            )
        else:
            # Use design resolution for windowed mode
            self.current_width = self.design_width
            self.current_height = self.design_height
            self.screen = pygame.display.set_mode(
                (self.current_width, self.current_height)
            )
        
        # Update scaling factors based on new resolution
        self.update_scaling_factors()
        
        logging.info(f"Display initialized: {self.current_width}x{self.current_height}, Fullscreen: {fullscreen}")
        return self.screen
    
    def toggle_fullscreen(self):
        """
        Toggle between fullscreen and windowed mode.
        
        Returns:
            The pygame surface
        """
        return self.initialize_display(not self.fullscreen)
    
    def update_scaling_factors(self):
        """
        Calculate scaling factors to maintain aspect ratio.
        This ensures the game fits the screen without distortion.
        """
        self.scale_x = self.current_width / self.design_width
        self.scale_y = self.current_height / self.design_height
        
        # For aspect ratio preservation, use the smaller scale factor
        self.scale = min(self.scale_x, self.scale_y)
        
        # Calculate offsets to center the game content
        self.offset_x = (self.current_width - self.design_width * self.scale) / 2
        self.offset_y = (self.current_height - self.design_height * self.scale) / 2
        
        logging.debug(f"Scaling factors updated: x={self.scale_x}, y={self.scale_y}, scale={self.scale}")
        logging.debug(f"Offsets: x={self.offset_x}, y={self.offset_y}")
    
    def get_scaled_position(self, x, y):
        """
        Convert design coordinates to screen coordinates.
        
        Args:
            x: X position in design coordinates
            y: Y position in design coordinates
            
        Returns:
            Tuple of (x, y) in screen coordinates
        """
        screen_x = x * self.scale + self.offset_x
        screen_y = y * self.scale + self.offset_y
        return (screen_x, screen_y)
    
    def get_scaled_dimensions(self, width, height):
        """
        Scale design dimensions to screen dimensions.
        
        Args:
            width: Width in design coordinates
            height: Height in design coordinates
            
        Returns:
            Tuple of (width, height) in screen coordinates
        """
        return (width * self.scale, height * self.scale)
    
    def get_design_rect(self, x, y, width, height):
        """
        Create a pygame Rect in design coordinates.
        
        Args:
            x: X position in design coordinates
            y: Y position in design coordinates
            width: Width in design coordinates
            height: Height in design coordinates
            
        Returns:
            A pygame Rect in design coordinates
        """
        return pygame.Rect(x, y, width, height)
    
    def get_scaled_rect(self, x, y, width, height):
        """
        Create a pygame Rect in screen coordinates.
        
        Args:
            x: X position in design coordinates
            y: Y position in design coordinates
            width: Width in design coordinates
            height: Height in design coordinates
            
        Returns:
            A pygame Rect in screen coordinates
        """
        screen_x, screen_y = self.get_scaled_position(x, y)
        screen_width, screen_height = self.get_scaled_dimensions(width, height)
        
        return pygame.Rect(screen_x, screen_y, screen_width, screen_height)
    
    def get_screen_rect(self):
        """
        Get a Rect representing the entire screen.
        
        Returns:
            A pygame Rect with the screen dimensions
        """
        return pygame.Rect(0, 0, self.current_width, self.current_height)
    
    def get_design_area(self):
        """
        Get a Rect representing the design area in screen coordinates.
        
        Returns:
            A pygame Rect representing the design area
        """
        return self.get_scaled_rect(0, 0, self.design_width, self.design_height)
    
    def convert_to_design_coordinates(self, screen_x, screen_y):
        """
        Convert screen coordinates to design coordinates.
        
        Args:
            screen_x: X position in screen coordinates
            screen_y: Y position in screen coordinates
            
        Returns:
            Tuple of (x, y) in design coordinates
        """
        # Reverse the scaling calculation
        design_x = (screen_x - self.offset_x) / self.scale
        design_y = (screen_y - self.offset_y) / self.scale
        
        return (design_x, design_y)
    
    def get_game_dimensions(self):
        """
        Get the design dimensions.
        
        Returns:
            Tuple of (width, height) of the design resolution
        """
        return (self.design_width, self.design_height)
    
    def get_screen_dimensions(self):
        """
        Get the current screen dimensions.
        
        Returns:
            Tuple of (width, height) of the current screen resolution
        """
        return (self.current_width, self.current_height)
