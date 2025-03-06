# ai_platform_trainer/core/launcher.py
import os
from ai_platform_trainer.core.logging_config import setup_logging
from ai_platform_trainer.gameplay.game import Game
from ai_platform_trainer.core.config_loader import load_config


def main():
    setup_logging()
    config = load_config()
    
    # Set up headless mode if configured
    if config.get("headless", False):
        os.environ["SDL_VIDEODRIVER"] = "dummy"
    
    game = Game(config_params=config)
    game.run()
