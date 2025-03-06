# Pixel Pursuit - Gameplay Improvements

This document outlines the gameplay improvements added to the Pixel Pursuit game.

## New Features

1. **Multiple Enemies**
   - Multiple enemies can now spawn simultaneously
   - Enemies are spawned based on a configurable interval
   - The maximum number of enemies is configurable

2. **Health/Lives System**
   - Player now has 3 lives (configurable)
   - Visual heart indicators show remaining health
   - Temporary invincibility after taking damage
   - Game over screen when all lives are lost

3. **Score System**
   - Score increases for each enemy destroyed
   - Different enemy types give different point values
   - Score is displayed on screen
   - Final score is shown on the game over screen

4. **Power-ups**
   - Random chance for power-ups to spawn when defeating enemies
   - Three power-up types:
     - Shield: Absorbs one hit
     - Rapid Fire: Increases missile fire rate
     - Speed Boost: Increases player movement speed
   - Visual indicators show active power-ups
   - Power-ups have limited duration

5. **Level Progression**
   - Game gets progressively harder as score increases
   - Enemy spawn rate increases with level
   - Visual level indication

6. **Different Enemy Types**
   - Basic enemy: Standard speed and health
   - Fast enemy: Faster but with less health
   - Tank enemy: Slower but with more health
   - Visual differentiation through colors

7. **Game Over Screen**
   - Shows final score
   - Option to restart (press R)

## Controls

- **Arrow Keys/WASD**: Move player
- **Space**: Shoot missile
- **F**: Toggle fullscreen
- **M**: Return to menu
- **Escape**: Quit game
- **R**: Restart after game over

## Running the Game with New Features

Use the provided batch file to run the game with all the new features:

```
run_game_with_features.bat
```

## Implementation Notes

These features were implemented by:

1. Creating an `EnemyManager` class to handle multiple enemies
2. Adding health, invincibility, and power-up tracking to the player
3. Creating a `PowerupManager` to handle power-up spawning and collection
4. Enhancing the renderer to show UI elements (score, health, power-ups)
5. Implementing level progression based on score
6. Creating a game over state with restart functionality

## Configuration

Game parameters can be adjusted in the `config.py` file:

- `MAX_ENEMIES`: Maximum number of enemies on screen
- `ENEMY_SPAWN_INTERVAL`: Time between enemy spawns (ms)
- `PLAYER_MAX_HEALTH`: Number of player lives
- `POINTS_PER_ENEMY`: Base score for destroying an enemy
- `POWERUP_SPAWN_CHANCE`: Chance of spawning a power-up
- `POWERUP_DURATION`: How long power-ups last (ms)
