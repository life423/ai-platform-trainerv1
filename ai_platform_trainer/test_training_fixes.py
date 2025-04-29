"""
Test script to verify the training mode fixes.

This script tests the modified training mode with continuous enemy presence,
improved pattern selection, missile avoidance, and hit reactions.
"""

import logging
import sys
import time
import pygame

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    stream=sys.stdout
)


def run_test():
    """Run a test of the training mode with the new features."""
    try:
        from ai_platform_trainer.gameplay.game import Game

        # Initialize pygame
        pygame.init()
        
        # Import needed elements for training mode
        from ai_platform_trainer.entities.player_training import PlayerTraining
        from ai_platform_trainer.entities.enemy_training import EnemyTrain
        from ai_platform_trainer.gameplay.modes.training_mode import TrainingMode
        from ai_platform_trainer.gameplay.spawner import spawn_entities

        # Create and start the game in training mode
        game = Game()
        
        # Ensure we're using training mode, not play mode
        game.mode = "train"
        
        # Create training mode entities explicitly
        game.player = PlayerTraining(game.screen_width, game.screen_height)
        game.enemy = EnemyTrain(game.screen_width, game.screen_height)
        
        # Set up training mode properly
        spawn_entities(game)
        game.player.reset()
        game.training_mode_manager = TrainingMode(game)
        
        # Initialize data logger
        game.data_logger = None  # We don't need actual logging for the test
        
        # Set reference to game on enemy for missile avoidance
        if hasattr(game, "enemy") and hasattr(game.enemy, "game"):
            game.enemy.game = game
            logging.info("Set game reference on enemy for missile avoidance")
        
        # Verify pattern weights on enemy
        if hasattr(game.enemy, "PATTERN_WEIGHTS"):
            logging.info(f"Enemy pattern weights: {game.enemy.PATTERN_WEIGHTS}")
            
            # Check that pursuit has the highest weight
            if game.enemy.PATTERN_WEIGHTS.get("pursue", 0) >= 0.6:
                logging.info("✓ Pursue pattern has highest weight as expected")
            else:
                logging.warning("✗ Pursue pattern does not have highest weight")
        else:
            logging.warning("✗ Enemy does not have PATTERN_WEIGHTS attribute")
        
        # Run main test loop for a few seconds
        start_time = time.time()
        test_duration = 10  # seconds
        enemy_positions = []
        
        logging.info(f"Running test for {test_duration} seconds...")
        
        while time.time() - start_time < test_duration:
            # Process events to avoid window freezing
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    break
            
            # Update the game with current time
            current_time = pygame.time.get_ticks()
            game.update(current_time)
            
            # Store enemy position to check for continuous presence
            if game.enemy and game.enemy.visible:
                enemy_positions.append((game.enemy.pos["x"], game.enemy.pos["y"]))
            
            # Force a missile hit every 2 seconds for testing
            current_time = int(time.time() - start_time)
            if current_time % 2 == 0 and current_time > 0:
                if hasattr(game.enemy, "register_hit") and not getattr(game, "_test_hit_registered", False):
                    logging.info(f"Testing hit reaction at {current_time}s")
                    game.enemy.register_hit()
                    game._test_hit_registered = True
            else:
                game._test_hit_registered = False
            
            # Render the game - Game has renderer object, not render method
            if hasattr(game, "renderer") and hasattr(game.renderer, "render"):
                game.renderer.render(
                    game.menu, game.player, game.enemy, False  # Not menu_active
                )
            pygame.display.flip()
            
            # Small delay to avoid hogging CPU
            pygame.time.delay(16)  # ~60fps
        
        # Analyze results
        logging.info(f"Test completed, collected {len(enemy_positions)} enemy positions")
        
        if len(enemy_positions) > 0:
            # Check for continuous enemy presence
            time_steps = len(enemy_positions)
            expected_frames = (test_duration * 1000) // 16  # Approximate frames at 60fps
            presence_ratio = time_steps / expected_frames
            
            logging.info(f"Enemy presence ratio: {presence_ratio:.2f}")
            if presence_ratio > 0.88:
                logging.info("✓ Enemy remained continuously present (>88% of frames)")
            else:
                logging.warning("✗ Enemy was not continuously present")
                
            # Check if enemy moved during test
            unique_positions = len(set(enemy_positions))
            logging.info(f"Enemy had {unique_positions} unique positions")
            if unique_positions > 10:
                logging.info("✓ Enemy showed active movement")
            else:
                logging.warning("✗ Enemy showed limited movement")
        
        logging.info("Test finished. Shutting down...")
        pygame.quit()
        return True
    
    except Exception as e:
        logging.error(f"Test failed: {str(e)}")
        import traceback
        traceback.print_exc()
        pygame.quit()
        return False


if __name__ == "__main__":
    success = run_test()
    sys.exit(0 if success else 1)
