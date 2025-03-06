import os


class Config:
    def __init__(self):
        # Screen settings
        self.SCREEN_WIDTH = 800
        self.SCREEN_HEIGHT = 600
        self.WINDOW_TITLE = "Pixel Pursuit"
        self.FRAME_RATE = 60
        self.SCREEN_SIZE = (self.SCREEN_WIDTH, self.SCREEN_HEIGHT)

        # Data paths
        self.DATA_PATH = os.path.join("data", "raw", "training_data.json")
        # Model paths are now handled by ModelManager

        # Game settings
        self.WALL_MARGIN = 50
        self.MIN_DISTANCE = 100

        # Speed settings
        self.RANDOM_SPEED_FACTOR_MIN = 0.8
        self.RANDOM_SPEED_FACTOR_MAX = 1.2
        self.ENEMY_MIN_SPEED = 2
        
        # Enemy settings
        self.MAX_ENEMIES = 3              # Maximum number of enemies on screen
        self.ENEMY_SPAWN_INTERVAL = 3000  # Time in ms between enemy spawns
        self.ENEMY_TYPES = ["basic", "fast", "tank"]  # Different enemy types
        
        # Player settings
        self.PLAYER_MAX_HEALTH = 3        # Player's maximum health/lives
        self.PLAYER_INVINCIBLE_TIME = 2000  # Invincibility period after hit (ms)
        
        # Scoring system
        self.POINTS_PER_ENEMY = 100       # Points for destroying an enemy
        self.POINTS_PER_LEVEL = 1000      # Points required to advance to next level
        
        # Power-up settings
        self.POWERUP_SPAWN_CHANCE = 0.05  # Chance to spawn powerup when enemy destroyed
        self.POWERUP_DURATION = 5000      # Duration of power-ups in milliseconds
        self.POWERUP_TYPES = ["shield", "rapid_fire", "speed_boost"]
        
        # UI settings
        self.FONT_SIZE = 24              # Default font size for UI text
        self.FONT_COLOR = (255, 255, 255) # Default font color (white)
        self.SCORE_POSITION = (20, 20)   # Position of score display
        self.HEALTH_POSITION = (20, 50)  # Position of health display


config = Config()
