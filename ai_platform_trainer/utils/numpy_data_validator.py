import logging
import numpy as np
import os

from ai_platform_trainer.ai.training.numpy_enemy_trainer import EnemyTrainer, EnemyTeacher

logger = logging.getLogger(__name__)

class NumpyDataValidatorAndTrainer:
    """
    Orchestrates the training pipeline for the NumPy-based enemy AI model.
    It handles data generation, validation, model training, and saving.
    """
    def __init__(self, config: dict):
        """
        Initialize the orchestrator.

        Args:
            config (dict): Configuration dictionary, typically loaded from 
                           config_enemy_numpy.json.
        """
        self.config = config
        
        # Ensure 'model' and 'training' keys exist in the config
        if 'model' not in self.config:
            raise ValueError("Missing 'model' configuration in config_enemy_numpy.json")
        if 'training' not in self.config:
            raise ValueError("Missing 'training' configuration in config_enemy_numpy.json")

        model_config = self.config['model']
        self.training_config = self.config['training']

        # Initialize teacher and trainer
        # The EnemyTeacher in numpy_enemy_trainer.py does not take config currently
        teacher = EnemyTeacher() 
        self.trainer = EnemyTrainer(config=model_config, teacher=teacher)
        
        self.model_path = model_config.get('path', 'models/numpy_enemy_model.npz')
        self.min_data_samples = self.training_config.get('min_data_samples', 100)
        self.episodes_to_generate = self.training_config.get('episodes', 100) # Default to 100 if not in config
        self.epochs_to_train = model_config.get('epochs', self.training_config.get('epochs', 10)) # model_config takes precedence for epochs

    def validate_data(self, data: list) -> bool:
        """
        Validates the generated training data.

        Args:
            data (list): A list of (state, action) tuples.

        Returns:
            bool: True if data is valid, False otherwise.
        """
        if not data:
            logger.error("No training data collected.")
            return False

        if len(data) < self.min_data_samples:
            logger.error(
                f"Insufficient training data: collected {len(data)} samples, "
                f"minimum required is {self.min_data_samples}."
            )
            return False

        # Check for action variety
        actions = [action for _, action in data]
        if not actions: # Should be caught by 'if not data' but as a safeguard
            logger.error("No actions found in training data.")
            return False
            
        unique_actions = set(actions)
        if len(unique_actions) < 2 and len(data) > 1 : # Only warn if more than one sample
            logger.warning(
                f"Training data for enemy has low action variety: "
                f"{len(unique_actions)} unique action(s) out of "
                f"{self.config.get('model', {}).get('output_size', 'N/A')} possible actions."
            )
        
        logger.info(f"Data validation passed with {len(data)} samples.")
        return True

    def run_training_pipeline(self) -> bool:
        """
        Runs the full training pipeline: data generation, validation, training, saving.

        Returns:
            bool: True if the pipeline completed successfully, False otherwise.
        """
        logger.info("Starting NumPy enemy AI training pipeline...")
        
        # 1. Generate training data
        logger.info(f"Generating training data for {self.episodes_to_generate} episodes...")
        # Use static_data_path if episodes is 0, as per EnemyTrainer logic
        static_data_path = self.training_config.get("static_data_path")
        if self.episodes_to_generate == 0 and static_data_path and os.path.exists(static_data_path):
            logger.info(f"Attempting to load static data from: {static_data_path}")
            training_data = self.trainer.generate_training_data(episodes=0)
        elif self.episodes_to_generate > 0 :
            training_data = self.trainer.generate_training_data(episodes=self.episodes_to_generate)
        else:
            logger.error("No episodes configured for data generation and no valid static data path found.")
            return False

        if not training_data:
            logger.error("Failed to generate or load training data.")
            return False
        logger.info(f"Generated/loaded {len(training_data)} training samples.")

        # 2. Validate the collected data
        if not self.validate_data(training_data):
            logger.error("Data validation failed. Aborting training.")
            return False

        # 3. Train the model
        logger.info(f"Training model for {self.epochs_to_train} epochs...")
        self.trainer.train_model(data=training_data, epochs=self.epochs_to_train)
        logger.info("Model training complete.")

        # 4. Save the trained model
        # Ensure the directory for the model path exists
        model_dir = os.path.dirname(self.model_path)
        if model_dir and not os.path.exists(model_dir):
            os.makedirs(model_dir)
            logger.info(f"Created directory for model: {model_dir}")
            
        self.trainer.save_model(self.model_path)
        logger.info(f"Trained model saved to {self.model_path}")
        
        return True
