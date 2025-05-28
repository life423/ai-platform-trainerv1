import json
import logging
import os
import sys

# Assuming DataValidatorAndTrainer is in ai_platform_trainer.ai.training
# Adjust if the location of the class DataValidatorAndTrainer from user snippet is different
try:
    from ai_platform_trainer.ai.training.data_validator_and_trainer import DataValidatorAndTrainer
except ImportError:
    # Fallback for running from a different context or if path issues exist
    # This might happen if ai_platform_trainer is not in PYTHONPATH when run directly
    # For a robust solution, ensure your project is structured as a package
    # and PYTHONPATH is set, or use relative imports carefully if this script
    # is always run from a specific location relative to the package.
    print("Failed to import DataValidatorAndTrainer directly. Ensure PYTHONPATH is set or run from project root.")
    # As a simple fallback for this script, try a common alternative if it's in utils
    try:
        from ai_platform_trainer.utils.data_validator_and_trainer import DataValidatorAndTrainer
        print("Note: Imported DataValidatorAndTrainer from utils as a fallback.")
    except ImportError:
        print("CRITICAL: Could not import DataValidatorAndTrainer. Check paths and structure.")
        sys.exit(1)


# Setup basic logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def main():
    """
    Main function to run a training pipeline using a configuration file.
    This script is based on the user's `run_training.py` snippet, which
    loads `config.json`.
    """
    logger.info("Starting training process via run_training.py...")
    
    # User's snippet uses "config.json". This might be the original PyTorch config
    # or a generic one. The DataValidatorAndTrainer it calls is now the NumPy one.
    config_path = "config.json" 

    if not os.path.exists(config_path):
        logger.error(f"Configuration file not found: {config_path}")
        print(f"ERROR: Configuration file not found: {config_path}")
        # Attempt to use numpy config if default is missing, as a fallback
        logger.warning(f"Attempting to use 'config_enemy_numpy.json' as a fallback.")
        config_path = "config_enemy_numpy.json"
        if not os.path.exists(config_path):
            logger.error(f"Fallback configuration file also not found: {config_path}")
            print(f"ERROR: Fallback configuration file also not found: {config_path}")
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
        # This now points to the NumPy-based DataValidatorAndTrainer
        orchestrator = DataValidatorAndTrainer(config=config)
        
        # Run the training pipeline
        success = orchestrator.run() # run() method as per user snippet
        
        if success:
            logger.info(f"Training pipeline via {config_path} completed successfully.")
            print(f"✅ Training pipeline via {config_path} completed successfully.")
        else:
            logger.error(f"Training pipeline via {config_path} failed.")
            print(f"❌ Training pipeline via {config_path} failed.")
            
    except Exception as e:
        logger.exception(f"An unexpected error occurred during the training process with {config_path}.")
        print(f"❌ An unexpected error occurred with {config_path}: {e}")

if __name__ == "__main__":
    main()
