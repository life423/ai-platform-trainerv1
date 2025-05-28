import logging
# Using the NumpyEnemyTrainer and EnemyTeacher (simple version)
# These imports assume this file is in ai_platform_trainer/ai/training/
from .numpy_enemy_trainer import NumpyEnemyTrainer # Use the one I created earlier
from .enemy_teacher import EnemyTeacher # Use the one I just updated

logger = logging.getLogger(__name__)

class DataValidatorAndTrainer:
    """
    Orchestrates the training pipeline for the NumPy-based enemy AI model.
    This version is based on the user's snippet for data_validator_and_trainer.py.
    """
    def __init__(self, config: dict):
        """
        Initialize the orchestrator.

        Args:
            config (dict): Configuration dictionary. Expected keys:
                           'model' (dict): Contains model-specific params like
                                           input_size, hidden_size, output_size,
                                           learning_rate, path (for saving model).
                           'training' (dict): Contains training process params like
                                              episodes, epochs.
        """
        self.config = config # Full config
        self.model_config = self.config.get('model', {})
        self.training_config = self.config.get('training', {}) # User snippet uses config directly for episodes/epochs

        # The user snippet directly instantiates EnemyTeacher without passing specific teacher_config
        self.teacher = EnemyTeacher(config=self.training_config.get('teacher_config'))
        
        # The user snippet passes the 'model' part of the config to EnemyTrainer
        self.trainer = NumpyEnemyTrainer(config=self.model_config, teacher=self.teacher) 
        
        self.model_path = self.model_config.get('path', 'models/numpy_enemy_model.npz') # Default from user snippet
        self.static_data_path = self.training_config.get('static_data_path', "data/raw/enemy_training_data.npz")


    def run(self): # Renamed from run_training_pipeline to match user snippet
        """
        Executes the full training pipeline:
        1. Generates or loads training data.
        2. Validates the data (basic check for sufficiency).
        3. Trains the model.
        4. Saves the trained model.
        """
        logger.info("Starting DataValidatorAndTrainer (NumPy-based) pipeline...")

        # 1. Generate or load training data
        # User snippet uses self.config['episodes'] directly
        num_episodes = self.config.get('episodes', 100) 
        
        training_data = self.trainer.generate_training_data(
            episodes=num_episodes,
            static_data_path=self.static_data_path # Pass static_data_path
        )

        # 2. Validate data (simple check for now)
        min_data_samples = self.training_config.get('min_data_samples', 10)
        if len(training_data) < min_data_samples: # User snippet had < 10
            error_msg = f"Insufficient training data: {len(training_data)} samples. Need at least {min_data_samples}."
            logger.error(error_msg)
            raise ValueError(error_msg) # User snippet raises ValueError

        logger.info(f"Successfully obtained {len(training_data)} training samples.")

        # 3. Train the model
        # User snippet uses self.config['epochs']
        num_epochs = self.config.get('epochs', 10) 
        self.trainer.train_model(data=training_data, epochs=num_epochs)
        
        logger.info("Model training completed.")

        # 4. Save the model
        self.trainer.save_model(path=self.model_path) # User snippet uses self.model_path
        # User snippet has a specific print message
        print(f"✅ Model saved to {self.model_path}") 
        logger.info(f"Model saved to {self.model_path}")
        return True

if __name__ == '__main__':
    # This example will not run correctly as is, because it needs a config.json
    # and the imports are relative. This is just for structure.
    # To run this, you'd typically call it from a script in the project root.
    logging.basicConfig(level=logging.INFO)
    logger.info("DataValidatorAndTrainer module - example usage (requires proper config and context).")
    
    # Example:
    # import json
    # with open("../../config_enemy_numpy.json") as f: # Adjust path to config
    #     config_data = json.load(f)
    # orchestrator = DataValidatorAndTrainer(config=config_data)
    # orchestrator.run()
