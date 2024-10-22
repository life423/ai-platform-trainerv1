import pygame
import random
from noise import pnoise1


class Enemy:
    def __init__(self, screen_width, screen_height):
        self._screen_width = screen_width
        self._screen_height = screen_height

        # Initial position in the center
        self._initial_position = pygame.Vector2(
            screen_width // 2, screen_height // 2)
        self._position = self._initial_position.copy()

        # Set size to match the player's size
        self._size = pygame.Vector2(50, 50)  # Width and height of the enemy

        # Enemy color (orange)
        self._color = (255, 165, 0)

        # Movement speed
        self._speed = max(2, screen_width // 400)

        # Noise parameters for smooth movement
        self._noise_offset_x = random.uniform(0, 100)
        self._noise_offset_y = random.uniform(0, 100)
        self._time = 0.0  # Time variable for noise-based movement

        # Random movement direction change timer
        self._direction_change_timer = random.randint(30, 100)

        # Current velocity
        self._velocity = pygame.Vector2(0, 0)

        # Rectangle for collision detection and drawing
        self._rect = pygame.Rect(
            self._position.x, self._position.y, self._size.x, self._size.y)

    # Properties with getters and setters
    @property
    def position(self):
        return self._position

    @position.setter
    def position(self, value):
        self._position = value
        self._rect.topleft = self._position  # Update rect position when position changes

    @property
    def velocity(self):
        return self._velocity

    @velocity.setter
    def velocity(self, value):
        self._velocity = value

    @property
    def size(self):
        return self._size

    @property
    def rect(self):
        return self._rect

    @property
    def color(self):
        return self._color

    def reset(self):
        # Reset the enemy to its starting position and reset variables
        self.position = self._initial_position.copy()
        self.velocity = pygame.Vector2(0, 0)
        self._time = 0.0
        self._direction_change_timer = random.randint(30, 100)
        self._noise_offset_x = random.uniform(0, 100)
        self._noise_offset_y = random.uniform(0, 100)

    def update(self):
        # Increment time to get new positions from noise
        self._time += 0.01  # Increment time for smoother changes

        # Generate noise-based movement for smooth base movement
        noise_dx = pnoise1(self._time + self._noise_offset_x) * self._speed
        noise_dy = pnoise1(self._time + self._noise_offset_y) * self._speed

        # Randomly change movement direction every few frames
        self._direction_change_timer -= 1
        if self._direction_change_timer <= 0:
            self.velocity = pygame.Vector2(
                random.choice([-1, 1]) * self._speed,
                random.choice([-1, 1]) * self._speed
            )
            # Reset the timer for the next direction change
            self._direction_change_timer = random.randint(30, 100)

        # Combine noise-based movement with current velocity
        movement = pygame.Vector2(noise_dx, noise_dy) + self.velocity
        self.position += movement

        # Keep the position within bounds
        self.position.x = max(
            0, min(self.position.x, self._screen_width - self._size.x))
        self.position.y = max(
            0, min(self.position.y, self._screen_height - self._size.y))

    def update_combined_movement(self):
        # Movement logic specific for training mode
        self._time += 0.02  # Faster time increment for varied movement

        # Generate noise-based movement
        noise_dx = pnoise1(self._time + self._noise_offset_x) * self._speed * 2
        noise_dy = pnoise1(self._time + self._noise_offset_y) * self._speed * 2

        # Randomly change movement direction more frequently
        self._direction_change_timer -= 1
        if self._direction_change_timer <= 0:
            self.velocity = pygame.Vector2(
                random.uniform(-1, 1) * self._speed * 2,
                random.uniform(-1, 1) * self._speed * 2
            )
            # Reset the timer for the next direction change
            self._direction_change_timer = random.randint(15, 50)

        # Combine noise-based movement with current velocity
        movement = pygame.Vector2(noise_dx, noise_dy) + self.velocity
        self.position += movement

        # Keep the position within bounds
        self.position.x = max(
            0, min(self.position.x, self._screen_width - self._size.x))
        self.position.y = max(
            0, min(self.position.y, self._screen_height - self._size.y))

    def draw(self, screen):
        # Use pygame to draw the enemy rectangle
        pygame.draw.rect(screen, self._color, self._rect)
