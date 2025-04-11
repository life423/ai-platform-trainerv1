"""
Direct sprite rendering test for AI Platform Trainer.

This script directly loads and displays sprites on screen to test
if pygame can properly load and display the sprite files.
"""
import os
import sys
import logging
import pygame

# Configure logging
logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s - %(levelname)s - %(message)s'
)

def test_sprite_rendering():
    """Test direct sprite rendering with pygame."""
    # Initialize pygame
    pygame.init()
    screen = pygame.display.set_mode((800, 600))
    pygame.display.set_caption("Sprite Rendering Test")
    clock = pygame.time.Clock()
    
    # Log current working directory
    cwd = os.getcwd()
    logging.info(f"Current working directory: {cwd}")
    
    # Define paths for all entity sprites
    sprite_paths = [
        os.path.join("assets", "sprites", "player", "player.png"),
        os.path.join("assets", "sprites", "enemy", "enemy.png"),
        os.path.join("assets", "sprites", "missile", "missile.png"),
        os.path.join("assets", "sprites", "effects", "explosion_0.png")
    ]
    
    # Try to load sprites directly
    loaded_sprites = []
    for path in sprite_paths:
        abs_path = os.path.abspath(path)
        logging.info(f"Attempting to load sprite from: {abs_path}")
        
        if os.path.exists(abs_path):
            try:
                sprite = pygame.image.load(abs_path).convert_alpha()
                loaded_sprites.append({
                    'surface': sprite,
                    'path': path,
                    'size': sprite.get_size()
                })
                logging.info(f"Successfully loaded sprite: {path}, size: {sprite.get_size()}")
            except pygame.error as e:
                logging.error(f"Failed to load sprite {path}: {e}")
        else:
            logging.error(f"Sprite file not found: {abs_path}")
    
    # If no sprites loaded, try with alternative paths
    if not loaded_sprites:
        logging.warning("No sprites loaded from primary paths, trying alternatives...")
        
        # Try with direct paths in sprites directory
        alt_sprite_paths = [
            os.path.join("assets", "sprites", "player.png"),
            os.path.join("assets", "sprites", "enemy.png"),
            os.path.join("assets", "sprites", "missile.png"),
            os.path.join("assets", "sprites", "explosion_0.png")
        ]
        
        for path in alt_sprite_paths:
            abs_path = os.path.abspath(path)
            logging.info(f"Attempting to load sprite from alternative path: {abs_path}")
            
            if os.path.exists(abs_path):
                try:
                    sprite = pygame.image.load(abs_path).convert_alpha()
                    loaded_sprites.append({
                        'surface': sprite,
                        'path': path,
                        'size': sprite.get_size()
                    })
                    logging.info(f"Successfully loaded sprite: {path}, size: {sprite.get_size()}")
                except pygame.error as e:
                    logging.error(f"Failed to load sprite {path}: {e}")
            else:
                logging.error(f"Alternative sprite file not found: {abs_path}")
    
    # Create placeholder sprites if none loaded
    if not loaded_sprites:
        logging.warning("No sprite files loaded, creating placeholders")
        placeholder_colors = [(0, 128, 255), (255, 50, 50), (255, 255, 50), (255, 150, 0)]
        
        for i, color in enumerate(placeholder_colors):
            surface = pygame.Surface((50, 50), pygame.SRCALPHA)
            pygame.draw.rect(surface, color, pygame.Rect(0, 0, 50, 50))
            loaded_sprites.append({
                'surface': surface,
                'path': f"Placeholder {i}",
                'size': (50, 50)
            })
    
    # Main loop for displaying sprites
    running = True
    y_offset = 50
    while running:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            elif event.type == pygame.KEYDOWN:
                if event.key == pygame.K_ESCAPE:
                    running = False
        
        # Clear screen
        screen.fill((135, 206, 235))  # Sky blue background
        
        # Draw sprites with labels
        y = y_offset
        font = pygame.font.Font(None, 24)
        for sprite_info in loaded_sprites:
            # Draw sprite
            screen.blit(sprite_info['surface'], (350, y))
            
            # Draw text label
            text = font.render(f"Path: {sprite_info['path']}", True, (0, 0, 0))
            screen.blit(text, (50, y + 10))
            
            # Draw size info
            size_text = font.render(f"Size: {sprite_info['size']}", True, (0, 0, 0))
            screen.blit(size_text, (50, y + 30))
            
            y += 100
        
        # Draw instructions
        instructions = font.render("Press ESC to exit", True, (0, 0, 0))
        screen.blit(instructions, (320, 20))
        
        pygame.display.flip()
        clock.tick(60)
    
    pygame.quit()

if __name__ == "__main__":
    try:
        test_sprite_rendering()
    except Exception as e:
        logging.critical(f"Fatal error in sprite test: {e}")