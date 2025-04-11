"""
Create placeholder obstacle sprites for the game.
This script generates simple shapes that can be used until better artwork is available.
"""
import os
from PIL import Image, ImageDraw

# Ensure output directory exists
os.makedirs(os.path.dirname(os.path.abspath(__file__)), exist_ok=True)

def create_rock_sprite(size=40):
    """Create a simple round rock sprite."""
    img = Image.new('RGBA', (size, size), (0, 0, 0, 0))  # Transparent background
    draw = ImageDraw.Draw(img)
    
    # Draw a gray circle with a darker outline
    draw.ellipse([(2, 2), (size-2, size-2)], fill=(130, 130, 130, 255), outline=(80, 80, 80, 255))
    
    # Add some texture/detail
    draw.arc([(8, 8), (size-8, size-8)], 30, 150, fill=(100, 100, 100, 255), width=2)
    draw.arc([(10, 15), (size-15, size-10)], 200, 330, fill=(160, 160, 160, 255), width=2)
    
    return img

def create_wall_sprite(size=40):
    """Create a simple wall/brick sprite."""
    img = Image.new('RGBA', (size, size), (0, 0, 0, 0))  # Transparent background
    draw = ImageDraw.Draw(img)
    
    # Main brick color
    draw.rectangle([(0, 0), (size, size)], fill=(160, 80, 60, 255))
    
    # Draw brick pattern
    brick_height = size // 4
    mortar_width = 2
    
    # Horizontal lines (mortar)
    for y in range(brick_height, size, brick_height):
        draw.line([(0, y), (size, y)], fill=(100, 100, 100, 255), width=mortar_width)
    
    # Vertical lines (alternating)
    for row in range(0, size // brick_height):
        offset = (size // 2) if row % 2 else 0
        if offset > 0:
            draw.line([(offset, row*brick_height), 
                      (offset, row*brick_height + brick_height)], 
                     fill=(100, 100, 100, 255), width=mortar_width)
    
    return img

def create_obstacle_sprite(size=40):
    """Create a generic obstacle (barrier) sprite."""
    img = Image.new('RGBA', (size, size), (0, 0, 0, 0))  # Transparent background
    draw = ImageDraw.Draw(img)
    
    # Yellow/black warning pattern
    draw.rectangle([(0, 0), (size, size)], fill=(50, 50, 50, 255))
    
    # Warning stripes
    stripe_width = size // 6
    for i in range(0, size, stripe_width * 2):
        draw.rectangle([(i, 0), (i + stripe_width, size)], fill=(220, 200, 0, 255))
    
    # Border
    draw.rectangle([(0, 0), (size-1, size-1)], outline=(20, 20, 20, 255), width=2)
    
    return img

# Generate and save the sprites
if __name__ == "__main__":
    sprite_size = 40
    
    # Create and save rock sprite
    rock = create_rock_sprite(sprite_size)
    rock.save(os.path.join(os.path.dirname(__file__), "rock.png"))
    
    # Create and save wall sprite
    wall = create_wall_sprite(sprite_size)
    wall.save(os.path.join(os.path.dirname(__file__), "wall.png"))
    
    # Create and save generic obstacle sprite
    obstacle = create_obstacle_sprite(sprite_size)
    obstacle.save(os.path.join(os.path.dirname(__file__), "obstacle.png"))
    
    print("Obstacle sprites created successfully!")
