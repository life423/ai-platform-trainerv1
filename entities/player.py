import pygame


class Player:
    def __init__(self, screen_width, screen_height):
        self._screen_width = screen_width
        self._screen_height = screen_height
        self._initial_position = pygame.Vector2(
            screen_width // 2, screen_height // 2)
        self._position = self._initial_position.copy()
        self._velocity = pygame.Vector2(0, 0)
        self._speed = 5
        self._size = pygame.Vector2(50, 50)  # Width and height of the player
        self._rect = pygame.Rect(
            self._position.x, self._position.y, self._size.x, self._size.y)

    # Getters and setters using properties
    @property
    def position(self):
        return self._position

    @position.setter
    def position(self, value):
        self._position = value
        self._rect.topleft = self._position

    @property
    def velocity(self):
        return self._velocity

    @velocity.setter
    def velocity(self, value):
        self._velocity = value

    @property
    def speed(self):
        return self._speed

    @speed.setter
    def speed(self, value):
        self._speed = value

    @property
    def rect(self):
        return self._rect

    def handle_input(self):
        keys = pygame.key.get_pressed()
        self._velocity.x = 0
        self._velocity.y = 0

        # Horizontal movement
        if keys[pygame.K_LEFT] or keys[pygame.K_a]:
            self._velocity.x = -1
        if keys[pygame.K_RIGHT] or keys[pygame.K_d]:
            self._velocity.x += 1

        # Vertical movement
        if keys[pygame.K_UP] or keys[pygame.K_w]:
            self._velocity.y = -1
        if keys[pygame.K_DOWN] or keys[pygame.K_s]:
            self._velocity.y += 1

        # Normalize velocity to maintain consistent speed
        if self._velocity.length() > 0:
            self._velocity = self._velocity.normalize() * self._speed

    def update(self):
        self.handle_input()
        self.position += self._velocity

        # Keep the player within the screen boundaries
        self.position.x = max(
            0, min(self.position.x, self._screen_width - self._size.x))
        self.position.y = max(
            0, min(self.position.y, self._screen_height - self._size.y))

        # Update the rectangle's position
        self._rect.topleft = self.position

    def draw(self, screen):
        # Drawing the player as a red rectangle
        pygame.draw.rect(screen, (255, 0, 0), self._rect)

    def reset(self):
        self.position = self._initial_position.copy()
        self.velocity = pygame.Vector2(0, 0)
        self._rect.topleft = self.position
