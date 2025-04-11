# file: ai_platform_trainer/gameplay/spawner.py
import logging
from ai_platform_trainer.gameplay.spawn_utils import (
    find_valid_spawn_position,
)
from ai_platform_trainer.gameplay.config import config


def spawn_entities(game):
    """
    Spawn the player, enemies and obstacles at random positions at the start of the game,
    ensuring they maintain minimum distance constraints.
    """
    if not game.player:
        logging.error("Player not initialized properly.")
        game.running = False
        return
        
    # Import obstacle class
    from ai_platform_trainer.entities.components.obstacle import Obstacle

    # Spawn player at a random position
    player_pos = find_valid_spawn_position(
        screen_width=game.screen_width,
        screen_height=game.screen_height,
        entity_size=game.player.size,
        margin=config.WALL_MARGIN,
        min_dist=0,
        other_pos=None,
    )
    game.player.position["x"], game.player.position["y"] = player_pos
    logging.info(f"Spawned player at {player_pos}")

    # Handle multiple enemies if available
    if hasattr(game, 'enemies') and game.enemies:
        for i, enemy in enumerate(game.enemies):
            # Find a valid position for this enemy away from player and other enemies
            other_positions = [(game.player.position["x"], game.player.position["y"])]
            
            # Also avoid other enemies that have already been placed
            for j in range(i):
                other_enemy = game.enemies[j]
                other_positions.append((other_enemy.pos["x"], other_enemy.pos["y"]))
            
            # Find position for this enemy
            enemy_pos = find_valid_spawn_position(
                screen_width=game.screen_width,
                screen_height=game.screen_height,
                entity_size=enemy.size,
                margin=config.WALL_MARGIN,
                min_dist=config.MIN_DISTANCE,
                other_pos=other_positions[0],  # Use player position as primary constraint
            )
            
            enemy.pos["x"], enemy.pos["y"] = enemy_pos
            enemy.visible = True
            logging.info(f"Spawned enemy {i+1} at {enemy_pos}")
            
    # For backward compatibility, handle the single enemy case
    elif game.enemy:
        enemy_pos = find_valid_spawn_position(
            screen_width=game.screen_width,
            screen_height=game.screen_height,
            entity_size=game.enemy.size,
            margin=config.WALL_MARGIN,
            min_dist=config.MIN_DISTANCE,
            other_pos=(game.player.position["x"], game.player.position["y"]),
        )
        game.enemy.pos["x"], game.enemy.pos["y"] = enemy_pos
        game.enemy.visible = True
        logging.info(f"Spawned single enemy at {enemy_pos}")
    else:
        logging.warning("No enemies to spawn")
        
    # Spawn obstacles if supported
    if hasattr(game, 'obstacles') and hasattr(game, 'num_obstacles'):
        # Clear any existing obstacles
        game.obstacles = []
        
        # Get all existing entity positions to avoid
        avoid_positions = [(game.player.position["x"], game.player.position["y"])]
        
        if hasattr(game, 'enemies') and game.enemies:
            for enemy in game.enemies:
                avoid_positions.append((enemy.pos["x"], enemy.pos["y"]))
        elif game.enemy:
            avoid_positions.append((game.enemy.pos["x"], game.enemy.pos["y"]))
            
        # Create obstacles
        obstacle_size = 40  # Default size for obstacles
        min_obstacles = min(game.num_obstacles, 8)  # Cap at 8 obstacles
        
        for i in range(min_obstacles):
            # Make some obstacles destructible
            destructible = (i % 3 == 0)  # Every third obstacle is destructible
            
            # Find valid position for obstacle
            obs_pos = find_valid_spawn_position(
                screen_width=game.screen_width,
                screen_height=game.screen_height,
                entity_size=obstacle_size,
                margin=config.WALL_MARGIN,
                min_dist=config.MIN_DISTANCE,
                other_pos=avoid_positions[0],  # Avoid player
            )
            
            # Create obstacle
            obstacle = Obstacle(
                x=obs_pos[0],
                y=obs_pos[1],
                size=obstacle_size,
                destructible=destructible
            )
            
            # Add to game
            game.obstacles.append(obstacle)
            
            # Add position to avoid list for next obstacle
            avoid_positions.append((obs_pos[0], obs_pos[1]))
            
        logging.info(f"Spawned {len(game.obstacles)} obstacles")


def respawn_enemy_at_position(game):
    """
    Find a valid position to respawn the enemy, respecting minimum distance
    from the player, and place the enemy there without showing it yet.
    This helper is used by other respawn functions.
    """
    new_pos = find_valid_spawn_position(
        screen_width=game.screen_width,
        screen_height=game.screen_height,
        entity_size=game.enemy.size,
        margin=config.WALL_MARGIN,
        min_dist=config.MIN_DISTANCE,
        other_pos=(game.player.position["x"], game.player.position["y"]),
    )
    game.enemy.set_position(new_pos[0], new_pos[1])
    return new_pos


def respawn_enemy(game):
    """
    Respawn the enemy at a valid position and show it immediately, without fade-in.
    """
    if not game.enemy or not game.player:
        return

    new_pos = respawn_enemy_at_position(game)
    game.enemy.show()
    game.is_respawning = False
    logging.info(f"Enemy respawned at {new_pos}.")


def respawn_enemy_with_fade_in(game, current_time):
    """
    Respawn the enemy at a valid position and show it with a fade-in effect.
    """
    if not game.enemy or not game.player:
        return

    new_pos = respawn_enemy_at_position(game)
    # Show the enemy with fade-in behavior
    # The fade-in is handled by the enemy.show() method when passed current_time
    game.enemy.show(current_time)

    game.is_respawning = False
    logging.info(f"Enemy respawned at {new_pos} with fade-in.")
