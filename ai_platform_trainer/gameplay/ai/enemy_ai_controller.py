from typing import List, Optional

from ai_platform_trainer.entities.ai.enemy_behavior import (
    ChaseBehavior,
    EvadeBehavior,
    CompositeBehavior,
)


def update_enemy_movement(
    enemy,
    player_x: float,
    player_y: float,
    player_speed: float,
    current_time: int,
    missiles: Optional[List] = None,
) -> None:
    """
    Handle the enemy's AI-driven movement using the behavior system.
    
    - 'enemy' is an EnemyPlay instance (access to enemy.pos, enemy.model, etc.)
    - 'player_x', 'player_y' is the player's position
    - 'player_speed' is how fast the player is moving (used to scale enemy 
      speed)
    - 'current_time' might be needed for state updates
    - 'missiles' is a list of active missiles the enemy might need to avoid
    """
    # If the enemy is not visible, skip movement update
    if not enemy.visible:
        return
    
    # If enemy doesn't have a behavior assigned yet, create a composite
    # behavior
    if not hasattr(enemy, 'behavior'):
        chase_behavior = ChaseBehavior(use_model=enemy.model is not None)
        evade_behavior = EvadeBehavior(detection_radius=200.0)
        
        # Composite behavior that combines chasing and evading
        enemy.behavior = CompositeBehavior(
            behaviors=[chase_behavior, evade_behavior],
            # Initial weights (chase more than evade by default)
            weights=[1.0, 0.5]
        )
    
    # Prepare player position in the expected format
    player_pos = {"x": player_x, "y": player_y}
    
    # Default empty list if missiles is None
    missiles_list = missiles or []
    
    # Get movement direction using behavior system
    dx, dy = enemy.behavior.decide_movement(enemy, player_pos, missiles_list)
    
    # Move enemy at 70% of the player's speed
    speed = player_speed * 0.7
    enemy.pos["x"] += dx * speed
    enemy.pos["y"] += dy * speed
    
    # Wrap around screen edges
    enemy.pos["x"], enemy.pos["y"] = enemy.wrap_position(
        enemy.pos["x"], enemy.pos["y"]
    )
