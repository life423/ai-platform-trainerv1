"""
Missile AI Controller module for game missile guidance.

This module provides the AI control logic for in-game missiles,
applying a trained neural network model to determine missile trajectory
adjustments based on current game state.
"""

import math
from typing import Dict, List, Optional

import torch

from ai_platform_trainer.ai.models.simple_missile_model import SimpleMissileModel
from ai_platform_trainer.entities.components.missile import Missile


def update_missile_ai(
    missiles: List["Missile"],
    player_pos: Dict[str, float],
    enemy_pos: Optional[Dict[str, float]],
    shared_input_tensor: torch.Tensor,
    missile_model: SimpleMissileModel,
    model_blend_factor: float = 0.5,
    max_turn_rate: float = 5.0,
) -> None:
    """
    Update missile trajectories using AI model predictions.

    This function processes all active missiles and adjusts their trajectories
    based on a combination of direct targeting and neural network predictions.

    Args:
        missiles: List of missile objects to update
        player_pos: Dictionary containing player x,y coordinates
        enemy_pos: Dictionary containing enemy x,y coordinates (or None if no enemy)
        shared_input_tensor: Pre-allocated tensor for model input to avoid allocations
        missile_model: Trained neural network model for missile guidance
        model_blend_factor: Weight between model output and direct targeting (0-1)
        max_turn_rate: Maximum degrees per update a missile can turn
    """
    for missile in missiles:
        # Calculate current missile direction
        current_angle = math.atan2(missile.vy, missile.vx)

        # Determine target angle (direct path to enemy or continue current trajectory)
        if enemy_pos is None:
            target_angle = current_angle
        else:
            target_angle = math.atan2(
                enemy_pos["y"] - missile.pos["y"], enemy_pos["x"] - missile.pos["x"]
            )

        # Extract position values for model input
        px, py = player_pos["x"], player_pos["y"]

        # If enemy doesn't exist, use projected position based on missile velocity
        ex = enemy_pos["x"] if enemy_pos else missile.pos["x"] + missile.vx
        ey = enemy_pos["y"] if enemy_pos else missile.pos["y"] + missile.vy

        # Calculate distance to target (used as model input feature)
        dist_val = math.hypot(missile.pos["x"] - ex, missile.pos["y"] - ey)

        # Prepare model input tensor with current state
        shared_input_tensor[0] = torch.tensor(
            [
                px,
                py,
                ex,
                ey,
                missile.pos["x"],
                missile.pos["y"],
                current_angle,
                dist_val,
                0.0,
            ]
        )

        # Get AI model prediction (turn rate adjustment)
        with torch.no_grad():
            turn_rate = missile_model(shared_input_tensor).item()

        # Calculate angle difference for direct targeting
        angle_diff = math.degrees(target_angle - current_angle)

        # Blend between model prediction and direct targeting
        blended_turn_rate = (
            model_blend_factor * turn_rate + (1 - model_blend_factor) * angle_diff
        )

        # Constrain turn rate to prevent unrealistic movement
        constrained_turn_rate = max(
            -max_turn_rate, min(max_turn_rate, blended_turn_rate)
        )

        # Apply the turn and update missile velocity components
        new_angle = current_angle + math.radians(constrained_turn_rate)
        missile.vx = missile.speed * math.cos(new_angle)
        missile.vy = missile.speed * math.sin(new_angle)
        missile.last_action = turn_rate  # Store for training data collection
