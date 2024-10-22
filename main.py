from gameplay.game import Game
import os
import sys

# Add the project root directory to sys.path
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

# Set Pygame to use a dummy video driver if running in an environment without a display
# os.environ["SDL_VIDEODRIlsER"] = "dummy"

if __name__ == "__main__":
    game = Game()
    game.run()  # No arguments needed

# gameplay/game.py

import pygame
from entities.player import Player
from entities.enemy import Enemy

class Game:
    def __init__(self):
        pygame.init()
        self.screen = pygame.display.set_mode((800, 600))  # Example screen size
        self.running = True
        self.menu_active = False
        self.player = Player(self.screen.get_width(), self.screen.get_height())
        self.enemy = Enemy(self.screen.get_width(), self.screen.get_height())
        # Other initializations...

    def run(self):
        while self.running:
            self.handle_events()
            if self.menu_active:
                # Menu handling code...
                pass
            else:
                self.update()
                self.render()

    def update(self):
        self.player.update()
        self.play_update()
        # Update other entities...

    def play_update(self):
        self.enemy.update(self.player.position)
        # Update other gameplay elements...

    def render(self):
        self.screen.fill((0, 0, 0))  # Clear screen
        self.player.draw(self.screen)
        self.enemy.draw(self.screen)
        # Draw other entities...
        pygame.display.flip()

    def handle_events(self):
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                self.running = False
            # Handle other events...

    def check_menu_selection(self, selected_action):
        # Check menu selection logic...
        self.start_game(selected_action)

    def start_game(self, selected_action):
        self.player.reset()
        self.enemy.reset()
        # Start game logic...
