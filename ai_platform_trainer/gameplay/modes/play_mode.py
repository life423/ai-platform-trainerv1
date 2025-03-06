# file: ai_platform_trainer/gameplay/modes/play_mode.py

import logging
from ai_platform_trainer.gameplay.missile_ai_controller import update_missile_ai


class PlayMode:
    def __init__(self, game):
        """
        Holds 'play' mode logic for the game.
        """
        self.game = game
        
        # Ensure collision manager has dependencies
        if self.game.collision_manager.missile_manager is None:
            self.game.collision_manager.set_missile_manager(
                self.game.missile_manager
            )

    def update(self, current_time: int) -> None:
        """
        The main update loop for 'play' mode, replacing old play_update() 
        logic in game.py.
        """

        # 1) Player movement & input
        if self.game.player and not self.game.player.handle_input():
            logging.info("Player requested to quit. Exiting game.")
            self.game.running = False
            return

        # 2) Enemy movement with missile avoidance
        if self.game.enemy:
            try:
                # Get missiles from missile manager for avoidance
                missiles = self.game.missile_manager.missiles
                
                self.game.enemy.update_movement(
                    self.game.player.position["x"],
                    self.game.player.position["y"],
                    self.game.player.step,
                    current_time,
                    missiles,  # Pass missiles for avoidance behavior
                )
                logging.debug(
                    "Enemy movement updated in play mode with missile avoidance."
                )
            except Exception as e:
                logging.error(f"Error updating enemy movement: {e}")
                self.game.running = False
                return

        # 3) Check for collisions using collision manager
        collision_results = self.game.collision_manager.update(
            self.game, current_time, is_training=False
        )
        
        # Handle player-enemy collision
        if collision_results["player_enemy_collision"]:
            self.game.is_respawning = True
            self.game.respawn_timer = current_time + self.game.respawn_delay

        # 4) Missile AI
        if (
            self.game.missile_model 
            and self.game.player
            and self.game.missile_manager.missiles
        ):
            update_missile_ai(
                self.game.missile_manager.missiles,
                self.game.player.position,
                self.game.enemy.pos if self.game.enemy else None,
                self.game._missile_input,
                self.game.missile_model
            )

        # 5) Misc updates
        # Respawn logic
        self.game.handle_respawn(current_time)

        # If enemy is fading in, keep updating alpha
        if self.game.enemy and self.game.enemy.fading_in:
            self.game.enemy.update_fade_in(current_time)

        # 6) Update missiles through the missile manager
        self.game.missile_manager.update(current_time)
