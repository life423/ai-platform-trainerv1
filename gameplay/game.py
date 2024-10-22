import pygame
import random
from noise import pnoise1
from entities.player import Player
from entities.enemy import Enemy
from gameplay.menu import Menu
from gameplay.renderer import Renderer
from core.data_logger import DataLogger


class Game:
    def __init__(self):
        # Initialize screen and clock
        pygame.init()
        self._screen = pygame.display.set_mode((800, 600))
        pygame.display.set_caption("Pixel Pursuit")
        self._clock = pygame.time.Clock()

        # Initialize entities and managers
        self._player = Player(self._screen.get_width(),
                              self._screen.get_height())
        self._enemy = Enemy(self._screen.get_width(),
                            self._screen.get_height())
        self._menu = Menu(self._screen.get_width(), self._screen.get_height())
        self._renderer = Renderer(self._screen)
        self._data_logger = DataLogger("pixel_pursuit_db", "training_data")

        # Game states
        self._running = True
        self._menu_active = True
        self._mode = None  # "train" or "play"

        # Training mode variables
        self._player_noise_time = 0.0
        self._player_noise_offset_x = random.uniform(0, 100)
        self._player_noise_offset_y = random.uniform(0, 100)
        self._player_step = 2  # Movement speed during training

    # Properties with getters and setters
    @property
    def screen(self):
        return self._screen

    @property
    def clock(self):
        return self._clock

    @property
    def player(self):
        return self._player

    @property
    def enemy(self):
        return self._enemy

    @property
    def menu(self):
        return self._menu

    @property
    def renderer(self):
        return self._renderer

    @property
    def data_logger(self):
        return self._data_logger

    @property
    def running(self):
        return self._running

    @running.setter
    def running(self, value):
        self._running = value

    @property
    def menu_active(self):
        return self._menu_active

    @menu_active.setter
    def menu_active(self, value):
        self._menu_active = value

    @property
    def mode(self):
        return self._mode

    def run(self):
        while self._running:
            self.handle_events()
            if self._menu_active:
                self._menu.draw(self._screen)
            else:
                self.update()
                self._renderer.render(self._menu, self._player,
                                      self._enemy, self._menu_active, self._screen)

            pygame.display.flip()
            self._clock.tick(60)  # Cap the frame rate

        pygame.quit()

    def handle_events(self):
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                self._running = False
            elif self._menu_active:
                selected_action = self._menu.handle_menu_events(event)
                if selected_action:
                    self.check_menu_selection(selected_action)
            else:
                # Handle game-specific events if necessary
                pass

    def check_menu_selection(self, selected_action):
        if selected_action == "exit":
            self._running = False
        elif selected_action in ["train", "play"]:
            self._menu_active = False
            self.start_game(selected_action)

    def start_game(self, mode: str):
        self._mode = mode
        print(f"Game mode: {mode}")
        self._player.reset()
        self._enemy.reset()
        # Reset training variables if in training mode
        if mode == "train":
            self._player_noise_time = 0.0
            self._player_noise_offset_x = random.uniform(0, 100)
            self._player_noise_offset_y = random.uniform(0, 100)

    def update(self):
        if self._mode == "train":
            self.training_update()
        elif self._mode == "play":
            self.play_update()

    def check_collision(self):
        return self._player.rect.colliderect(self._enemy.rect)

    def play_update(self):
        # Update player and enemy
        self._player.update()
        self._enemy.update()

        # Check for collisions
        collision = self.check_collision()

        # Log data
        self._data_logger.log_data(
            [self._player.position.x, self._player.position.y],
            [self._enemy.position.x, self._enemy.position.y],
            collision
        )

        # Handle collision
        if collision:
            print("Collision detected!")
            # Handle collision (e.g., reset game, end game, etc.)

    def training_update(self):
        # Increment time to get new noise values for smooth movement
        self._player_noise_time += 0.01

        # Update player position using Perlin noise
        dx_player = pnoise1(self._player_noise_time +
                            self._player_noise_offset_x) * self._player_step
        dy_player = pnoise1(self._player_noise_time +
                            self._player_noise_offset_y) * self._player_step

        new_x = max(0, min(self._screen.get_width() -
                    self._player.size.x, self._player.position.x + dx_player))
        new_y = max(0, min(self._screen.get_height() -
                    self._player.size.y, self._player.position.y + dy_player))
        self._player.position = pygame.Vector2(new_x, new_y)

        # Update enemy position using combined noise and random direction movement
        self._enemy.update_combined_movement()

        # Check for collisions
        collision = self.check_collision()

        # Log data
        self._data_logger.log_data(
            [self._player.position.x, self._player.position.y],
            [self._enemy.position.x, self._enemy.position.y],
            collision
        )

        # Handle collision
        if collision:
            print("Collision detected!")
            # Handle collision (e.g., reset positions)


if __name__ == "__main__":
    game = Game()
    game.run()
