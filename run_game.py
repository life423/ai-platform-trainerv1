#!/usr/bin/env python
"""
Simple launcher for AI Platform Trainer.
"""
import os
import sys
import pygame

def run_game():
    # Add the project root to sys.path
    project_root = os.path.dirname(os.path.abspath(__file__))
    if project_root not in sys.path:
        sys.path.insert(0, project_root)
    
    # Initialize pygame
    pygame.init()
    
    # Import and run the game directly
    from ai_platform_trainer.gameplay.game import Game
    
    game = Game()
    game.run()

if __name__ == "__main__":
    run_game()