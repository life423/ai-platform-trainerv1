#!/usr/bin/env python
"""
Unified Launcher for AI Platform Trainer

Enhanced launcher with new menu system supporting both supervised learning
and reinforcement learning modes.
"""
import sys
import logging
from pathlib import Path


def main():
    """Main entry point for the AI Platform Trainer."""
    # Add the project root to sys.path
    project_root = Path(__file__).parent
    sys.path.insert(0, str(project_root))
    
    try:
        # Import the new game loop system
        from ai_platform_trainer.ui.game_loop import GameLoop
        from ai_platform_trainer.core.render_mode import RenderMode
        
        # Determine render mode from command line
        render_mode = RenderMode.FULL
        if "--headless" in sys.argv:
            render_mode = RenderMode.HEADLESS
        
        # Create and run the enhanced game loop
        game_loop = GameLoop(render_mode=render_mode)
        game_loop.run()
        
        logging.info("Game completed successfully")
        return 0
        
    except KeyboardInterrupt:
        logging.info("Game interrupted by user.")
        return 0
    except Exception as e:
        logging.critical(f"Fatal error: {e}")
        logging.exception("Exception details:")
        return 1


if __name__ == "__main__":
    sys.exit(main())
