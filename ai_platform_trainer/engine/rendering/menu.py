"""
Menu System for AI Platform Trainer

This module handles the game's menu system, including the main menu,
help screen, and AI selection screen. It processes user input and
renders the appropriate menu screens.
"""
import pygame


class Menu:
    def __init__(self, screen_width, screen_height):
        # Menu options displayed in the main menu
        self.menu_options = ["Play", "AI Select", "Train", "Help", "Exit"]
        # Tracks which menu option is currently selected (for keyboard input)
        self.selected_option = 0

        # Flag to show help screen; if True, the help screen is drawn instead of the main menu
        self.show_help = False
        
        # Flag to show AI selection screen
        self.show_ai_select = False
        
        # AI selection options and currently selected AI
        self.ai_options = ["Supervised Learning", "Reinforcement Learning"]
        self.selected_ai = 0  # 0 = Supervised, 1 = RL
        self.ai_option_rects = {}

        # Store screen dimensions to position menu items correctly
        self.screen_width = screen_width
        self.screen_height = screen_height

        # Fonts and colors for text rendering and background
        self.font_title = pygame.font.Font(None, 100)  # Large font for the main title
        self.font_option = pygame.font.Font(None, 60)  # Font for menu options
        self.color_background = (135, 206, 235)  # Light blue background color
        self.color_title = (0, 51, 102)  # Dark blue for title text
        self.color_option = (245, 245, 245)  # Light gray for unselected options
        self.color_selected = (255, 223, 0)  # Yellow for the currently selected option
        self.option_rects = {}  # Store clickable rects for each menu option

    def handle_menu_events(self, event):
        """
        Handle user input for the menu.
        If self.show_help or self.show_ai_select is True, handle those screens.
        Otherwise, process keyboard/mouse events to navigate or select menu options.
        """

        # If the help screen is currently displayed
        if self.show_help:
            # Only check if user pressed ESC or ENTER to exit help
            if event.type == pygame.KEYDOWN and event.key in [
                pygame.K_ESCAPE,
                pygame.K_RETURN,
            ]:
                self.show_help = False
            return None
            
        # If the AI selection screen is displayed
        elif self.show_ai_select:
            # Handle navigation in the AI selection menu
            if event.type == pygame.KEYDOWN:
                if event.key in [pygame.K_UP, pygame.K_w]:
                    self.selected_ai = (self.selected_ai - 1) % len(self.ai_options)
                elif event.key in [pygame.K_DOWN, pygame.K_s]:
                    self.selected_ai = (self.selected_ai + 1) % len(self.ai_options)
                elif event.key in [pygame.K_RETURN, pygame.K_KP_ENTER]:
                    # Selected an AI type, return to main menu
                    self.show_ai_select = False
                    return f"ai_{self.selected_ai}"  # Return which AI was selected
                elif event.key == pygame.K_ESCAPE:
                    # Go back to main menu without changing AI
                    self.show_ai_select = False
            
            # Handle mouse clicks in AI selection
            elif event.type == pygame.MOUSEBUTTONDOWN and event.button == 1:
                mouse_x, mouse_y = pygame.mouse.get_pos()
                for index, rect in self.ai_option_rects.items():
                    if rect.collidepoint(mouse_x, mouse_y):
                        self.selected_ai = index
                        self.show_ai_select = False
                        return f"ai_{self.selected_ai}"  # Return which AI was selected
            
            return None

        # If user presses ENTER (or NUMPAD ENTER) on the main menu
        elif event.type == pygame.KEYDOWN and event.key in [
            pygame.K_RETURN,
            pygame.K_KP_ENTER,
        ]:
            chosen = self.menu_options[self.selected_option]
            if chosen == "Help":
                # Switch to the help screen
                self.show_help = True
                return None  # Don't return "help" because we handle it internally
            elif chosen == "AI Select":
                # Switch to the AI selection screen
                self.show_ai_select = True
                return None  # Don't return "ai_select" because we handle it internally
            else:
                # Return the chosen option in lowercase (e.g., "play", "train", "exit")
                return chosen.lower()

        # Handle arrow keys / WASD for navigating menu options
        if event.type == pygame.KEYDOWN:
            if event.key in [pygame.K_UP, pygame.K_w]:
                # Move selection up, wrapping around with modulo
                self.selected_option = (self.selected_option - 1) % len(
                    self.menu_options
                )
            elif event.key in [pygame.K_DOWN, pygame.K_s]:
                # Move selection down, wrapping around with modulo
                self.selected_option = (self.selected_option + 1) % len(
                    self.menu_options
                )
            # Pressing ENTER again here is a fallback check
            elif event.key in [pygame.K_RETURN, pygame.K_KP_ENTER]:
                return self.menu_options[self.selected_option].lower()
            # We let the game itself handle ESC for exiting the program

        # Mouse click detection
        elif event.type == pygame.MOUSEBUTTONDOWN and event.button == 1:
            # Left mouse click
            mouse_x, mouse_y = pygame.mouse.get_pos()
            # Check if the click is within any of the option rectangles
            for index, rect in self.option_rects.items():
                if rect.collidepoint(mouse_x, mouse_y):
                    self.selected_option = index
                    chosen = self.menu_options[self.selected_option]
                    if chosen == "Help":
                        self.show_help = True
                        return None
                    else:
                        return chosen.lower()

        # If no relevant event, return None
        return None

    def draw(self, screen):
        """
        Draw the menu on the given Pygame screen.
        If show_help is True, the help screen is drawn instead.
        If show_ai_select is True, the AI selection screen is drawn.
        Otherwise, the main menu is drawn with the title and list of options.
        """

        # If the help screen is active, draw that instead
        if self.show_help:
            self.draw_help(screen)
            return
            
        # If the AI selection screen is active, draw that
        if self.show_ai_select:
            self.draw_ai_select(screen)
            return
        
        # Draw the main menu (if no special screens are active)
        # Fill the screen background
        screen.fill(self.color_background)

        # Render the game title
        title_surface = self.font_title.render("Pixel Pursuit", True, self.color_title)
        title_rect = title_surface.get_rect(
            center=(self.screen_width // 2, self.screen_height // 5)
        )
        screen.blit(title_surface, title_rect)

        # Render menu options
        mouse_x, mouse_y = (
            pygame.mouse.get_pos()
        )  # For any mouse-based highlighting, if desired
        for index, option in enumerate(self.menu_options):
            # Choose highlight color if this option is currently selected
            color = (
                self.color_selected
                if index == self.selected_option
                else self.color_option
            )
            option_surface = self.font_option.render(option, True, color)
            option_rect = option_surface.get_rect(
                center=(self.screen_width // 2, self.screen_height // 2 + index * 80)
            )
            # Store this rect for click detection
            self.option_rects[index] = option_rect
            # Draw the menu option on screen
            screen.blit(option_surface, option_rect)

    def draw_ai_select(self, screen):
        """
        Draw the AI selection screen where users can choose between
        supervised learning and reinforcement learning for the enemy.
        """
        # Clear the screen with the background color
        screen.fill(self.color_background)

        # Title for the AI selection screen
        ai_title_surface = self.font_title.render(
            "Select Enemy AI Type", True, self.color_title
        )
        ai_title_rect = ai_title_surface.get_rect(
            center=(self.screen_width // 2, self.screen_height // 5)
        )
        screen.blit(ai_title_surface, ai_title_rect)

        # Create a smaller font for descriptions
        font_info = pygame.font.Font(None, 34)

        # Render AI options
        for index, option in enumerate(self.ai_options):
            # Choose highlight color if this option is currently selected
            color = (
                self.color_selected
                if index == self.selected_ai
                else self.color_option
            )
            option_surface = self.font_option.render(option, True, color)
            option_rect = option_surface.get_rect(
                center=(self.screen_width // 2, self.screen_height // 2 + index * 100)
            )
            # Store this rect for click detection
            self.ai_option_rects[index] = option_rect
            # Draw the option
            screen.blit(option_surface, option_rect)
            
            # Draw description below each option
            description = ""
            if index == 0:  # Supervised Learning
                description = "Classic neural network trained on demonstration data"
            else:  # Reinforcement Learning
                description = "Advanced AI that learned through trial and error"
                
            desc_surface = font_info.render(description, True, self.color_option)
            desc_rect = desc_surface.get_rect(
                center=(self.screen_width // 2, self.screen_height // 2 + index * 100 + 40)
            )
            screen.blit(desc_surface, desc_rect)

        # Instructions at bottom
        instructions = self.font_option.render(
            "Press ENTER to select or ESC to cancel", True, self.color_title
        )
        instructions_rect = instructions.get_rect(
            center=(self.screen_width // 2, self.screen_height - 50)
        )
        screen.blit(instructions, instructions_rect)

    def draw_help(self, screen):
        """
        Draw a dedicated Help / Controls screen.
        This screen instructs the user on game controls and how to return to the main menu.
        It also provides information about game modes and AI training.
        """

        # Clear the screen with the background color
        screen.fill(self.color_background)

        # Title for the Help screen
        help_title_surface = self.font_title.render(
            "Help / Controls", True, self.color_title
        )
        help_title_rect = help_title_surface.get_rect(
            center=(self.screen_width // 2, self.screen_height // 10)
        )
        screen.blit(help_title_surface, help_title_rect)

        # Create a smaller font for more detailed explanations
        font_info = pygame.font.Font(None, 34)

        # Controls section
        controls_title = self.font_option.render("Controls:", True, self.color_selected)
        controls_rect = controls_title.get_rect(
            topleft=(self.screen_width // 10, self.screen_height // 5)
        )
        screen.blit(controls_title, controls_rect)

        # Text lines describing the controls
        controls_text = [
            "• Arrow Keys or W/S to navigate menu",
            "• Control Player with Arrow Keys or WASD",
            "• Press Space to shoot missiles",
            "• Press P to pause/unpause the game",
            "• Press Enter to select menu items",
            "• Press F to toggle fullscreen",
            "• Press M to return to the menu",
            "• Press Escape to return to menu (in-game) or quit help screen"
        ]

        # Draw controls text
        for i, line in enumerate(controls_text):
            help_surface = font_info.render(line, True, self.color_option)
            help_rect = help_surface.get_rect(
                topleft=(self.screen_width // 10, self.screen_height // 5 + 40 + i * 30)
            )
            screen.blit(help_surface, help_rect)

        # Game Modes section - starts halfway down screen
        modes_title = self.font_option.render("Game Modes:", True, self.color_selected)
        modes_rect = modes_title.get_rect(
            topleft=(self.screen_width // 10, self.screen_height // 2)
        )
        screen.blit(modes_title, modes_rect)

        # Text explaining the game modes
        modes_text = [
            "• Play Mode: Regular gameplay against AI enemy",
            "• Train Mode: Collects missile data to train avoidance AI"
        ]

        # Draw modes text
        y_offset = self.screen_height // 2 + 40
        for i, line in enumerate(modes_text):
            mode_surface = font_info.render(line, True, self.color_option)
            mode_rect = mode_surface.get_rect(
                topleft=(self.screen_width // 10, y_offset + i * 30)
            )
            screen.blit(mode_surface, mode_rect)

        # AI Training section
        ai_title = self.font_option.render("AI Training:", True, self.color_selected)
        ai_rect = ai_title.get_rect(
            topleft=(self.screen_width // 10, y_offset + len(modes_text) * 30 + 20)
        )
        screen.blit(ai_title, ai_rect)

        # Text explaining the AI training
        ai_text = [
            "• Two AI types available: Supervised and Reinforcement Learning",
            "• Supervised Learning: Neural network trained on demonstrations",
            "• Reinforcement Learning: AI trained through trial and error",
            "• Use 'AI Select' from main menu to choose which AI to play against",
            "• Missile avoidance is trained using gameplay data collection",
            "• CUDA acceleration is used for physics simulation"
        ]

        # Draw AI text
        ai_y_offset = y_offset + len(modes_text) * 30 + 60
        for i, line in enumerate(ai_text):
            ai_surface = font_info.render(line, True, self.color_option)
            ai_rect = ai_surface.get_rect(
                topleft=(self.screen_width // 10, ai_y_offset + i * 30)
            )
            screen.blit(ai_surface, ai_rect)

        # Return instructions at bottom
        exit_text = self.font_option.render(
            "Press ESC or ENTER to return to menu", True, self.color_title
        )
        exit_rect = exit_text.get_rect(
            center=(self.screen_width // 2, self.screen_height - 50)
        )
        screen.blit(exit_text, exit_rect)