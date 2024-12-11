import pygame
import random
from entities.player import Player
from entities.enemy import Enemy
from gameplay.menu import Menu
from core.data_logger import DataLogger
from gameplay.renderer import Renderer

from noise import pnoise1


class Game:
    def __init__(self):
        # Initialize screen and clock
        pygame.init()
        WINDOW_WIDTH, WINDOW_HEIGHT = 800, 600
        self.screen = pygame.display.set_mode((WINDOW_WIDTH, WINDOW_HEIGHT), pygame.RESIZABLE)
        pygame.display.set_caption("Pixel Pursuit")
        self.clock = pygame.time.Clock()

        # Initialize entities and managers
        self.player = Player(self.screen.get_width(), self.screen.get_height())
        self.enemy = Enemy(self.screen.get_width(), self.screen.get_height())
        self.menu = Menu(self.screen.get_width(), self.screen.get_height())
        self.renderer = Renderer(self.screen)
        self.data_logger = DataLogger("data/training_data.json")

        # Game states
        self.running = True
        self.menu_active = True
        self.mode = None  # "train" or "play"

    def run(self):
        while self.running:
            self.handle_events()
            if self.menu_active:
                self.menu.draw(self.screen)
            else:
                self.update()
                self.renderer.render(self.menu, self.player, self.enemy, self.menu_active, self.screen)

            pygame.display.flip()
            self.clock.tick(60)  # Cap the frame rate

        # When the game closes, save any logged data
        self.data_logger.save()
        pygame.quit()

    def handle_events(self):
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                self.running = False
            elif event.type == pygame.KEYDOWN and event.key == pygame.K_ESCAPE:
                self.running = False
            elif self.menu_active:
                selected_action = self.menu.handle_menu_events(event)
                if selected_action:
                    self.check_menu_selection(selected_action)
        


    def check_menu_selection(self, selected_action):
        if selected_action == "exit":
            self.running = False
        elif selected_action in ["train", "play"]:
            self.menu_active = False
            self.start_game(selected_action)

    def start_game(self, mode: str):
        self.mode = mode
        print(mode)
        self.player.reset()
        self.enemy.reset()

        # Position the player and enemy in visually distinct places
        self.player.position["x"] = self.screen.get_width() // 4 - self.player.size // 2
        self.player.position["y"] = self.screen.get_height() // 2 - self.player.size // 2

        self.enemy.pos["x"] = (self.screen.get_width() * 3) // 4 - self.enemy.size // 2
        self.enemy.pos["y"] = self.screen.get_height() // 2 - self.enemy.size // 2

    def update(self):
        if self.mode == "train":
            self.training_update()
        elif self.mode == "play":
            self.play_update()

    def check_collision(self):
        # Define the player and enemy rectangles
        player_rect = pygame.Rect(
            self.player.position["x"], self.player.position["y"], self.player.size, self.player.size)
        enemy_rect = pygame.Rect(
            self.enemy.pos["x"], self.enemy.pos["y"], self.enemy.size, self.enemy.size)
        # Check if the rectangles collide
        return player_rect.colliderect(enemy_rect)

    def play_update(self):
        # Handle player input during play mode
        self.handle_input()

        # Update enemy movement
        self.enemy.update_movement()

        # Check for collisions
        collision = self.check_collision()
        if collision:
            if self.mode == "train":
                data_point = {
                    "mode": "play",
                    "player_x": self.player.position["x"],
                    "player_y": self.player.position["y"],
                    "enemy_x": self.enemy.pos["x"],
                    "enemy_y": self.enemy.pos["y"],
                    "collision": True
                }
                self.data_logger.log(data_point)
            else:
                print("Collision detected!")
                self.running = False  # Stop the game on collision
        

    def training_update(self):
        # Increment time for Perlin noise movement
        self.player.noise_time += 0.01

        # Update player position using Perlin noise
        dx_player = pnoise1(self.player.noise_time + self.player.noise_offset_x) * self.player.step
        dy_player = pnoise1(self.player.noise_time + self.player.noise_offset_y) * self.player.step

        self.player.position["x"] = max(0, min(self.screen.get_width() - self.player.size, self.player.position["x"] + dx_player))
        self.player.position["y"] = max(0, min(self.screen.get_height() - self.player.size, self.player.position["y"] + dy_player))

        # Update enemy position using its movement logic
        self.enemy.update_movement()

        # Check for collisions
        collision = self.check_collision()

        # **Data Logging for Training:**
        # Here we log each frame's data. You can include anything you deem relevant.
        data_point = {
            "mode": "train",
            "player_x": self.player.position["x"],
            "player_y": self.player.position["y"],
            "enemy_x": self.enemy.pos["x"],
            "enemy_y": self.enemy.pos["y"],
            "collision": collision
        }
        self.data_logger.log(data_point)

    def handle_input(self):
        keys = pygame.key.get_pressed()

        if keys[pygame.K_LEFT] or keys[pygame.K_a]:
            self.player.position['x'] -= self.player.step
        if keys[pygame.K_RIGHT] or keys[pygame.K_d]:
            self.player.position['x'] += self.player.step
        if keys[pygame.K_UP] or keys[pygame.K_w]:
            self.player.position['y'] -= self.player.step
        if keys[pygame.K_DOWN] or keys[pygame.K_s]:
            self.player.position['y'] += self.player.step
        if keys[pygame.K_ESCAPE]:
            self.running = False

        # Ensure player stays within screen boundaries
        self.player.position['x'] = max(0, min(self.player.position['x'], self.screen.get_width() - self.player.size))
        self.player.position['y'] = max(0, min(self.player.position['y'], self.screen.get_height() - self.player.size))


if __name__ == "__main__":
    game = Game()
    game.run()
