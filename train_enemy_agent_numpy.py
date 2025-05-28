import json
import logging
import os

# Assuming numpy_data_validator is in the utils directory
from ai_platform_trainer.utils.numpy_data_validator import NumpyDataValidatorAndTrainer

# Setup basic logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def main():
    """
    Main function to run the NumPy-based enemy AI training pipeline.
    """
    logger.info("Starting NumPy enemy agent training process...")
    
    config_path = "config_enemy_numpy.json" # Assumes this config is in the project root

    if not os.path.exists(config_path):
        logger.error(f"Configuration file not found: {config_path}")
        print(f"ERROR: Configuration file not found: {config_path}")
        return

    try:
        with open(config_path, 'r') as f:
            config = json.load(f)
        logger.info(f"Successfully loaded configuration from {config_path}")
    except json.JSONDecodeError as e:
        logger.error(f"Error decoding JSON from {config_path}: {e}")
        print(f"ERROR: Invalid JSON in {config_path}: {e}")
        return
    except Exception as e:
        logger.error(f"Failed to load configuration from {config_path}: {e}")
        print(f"ERROR: Could not load {config_path}: {e}")
        return

    try:
        # Initialize the orchestrator with the loaded configuration
        orchestrator = NumpyDataValidatorAndTrainer(config=config)
        
        # Run the training pipeline
        success = orchestrator.run_training_pipeline()
        
        if success:
            logger.info("NumPy enemy agent training pipeline completed successfully.")
            print("✅ NumPy enemy agent training pipeline completed successfully.")
        else:
            logger.error("NumPy enemy agent training pipeline failed.")
            print("❌ NumPy enemy agent training pipeline failed.")
            
    except Exception as e:
        logger.exception("An unexpected error occurred during the training process.")
        print(f"❌ An unexpected error occurred: {e}")

if __name__ == "__main__":
    # This allows the script to be run from the project root directory
    # where ai_platform_trainer is a package.
    # Ensure PYTHONPATH is set up correctly if running from elsewhere or if imports fail.
    main()
