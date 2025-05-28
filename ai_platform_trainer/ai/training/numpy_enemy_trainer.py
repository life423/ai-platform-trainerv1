import json
import os
import sys

import numpy as np


# ===== Enemy Teacher (Expert Policy) =====
class EnemyTeacher:
    def reset(self):
        # Not used when loading dummy data
        return np.random.rand(5)

    def decide_action(self, state):
        dx = state[2] - state[0]
        dy = state[3] - state[1]
        # Action mapping: 0=left,1=right,2=down,3=up
        if abs(dx) > abs(dy):
            return 1 if dx > 0 else 0
        else:
            return 3 if dy > 0 else 2

    def step(self, action):
        # Not used when loading dummy data
        next_state = np.random.rand(5)
        done = np.random.rand() < 0.1
        return next_state, done


# ===== Enemy Trainer =====
class EnemyTrainer:
    def __init__(self, config, teacher):
        self.config = config
        self.teacher = teacher
        in_dim = config['input_size']
        hid_dim = config['hidden_size']
        out_dim = config['output_size']
        self.W1 = np.random.randn(in_dim, hid_dim) * 0.1
        self.b1 = np.zeros(hid_dim)
        self.W2 = np.random.randn(hid_dim, out_dim) * 0.1
        self.b2 = np.zeros(out_dim)
        self.learning_rate = config.get('learning_rate', 0.01) # Default if not in config
        self.static_data_path = config.get('static_data_path', "data/raw/enemy_training_data.npz")


    def generate_training_data(self, episodes=100):
        if episodes == 0 and self.static_data_path and os.path.exists(self.static_data_path):
            print(f"Loading dummy data from {self.static_data_path}")
            archive = np.load(self.static_data_path)
            states, actions = archive['states'], archive['actions']
            return list(zip(states, actions))
        print("Generating data with teacher agent...")
        data = []
        for _ in range(episodes):
            state = self.teacher.reset()
            done = False
            while not done:
                action = self.teacher.decide_action(state)
                data.append((state, action))
                state, done = self.teacher.step(action)
        return data

    def train_model(self, data, epochs=10):
        for _ in range(epochs):
            np.random.shuffle(data)
            for state, action in data:
                # Forward pass
                z1 = state.dot(self.W1) + self.b1
                h1 = np.maximum(0, z1)  # ReLU
                z2 = h1.dot(self.W2) + self.b2
                # Softmax
                exp_z = np.exp(z2 - np.max(z2))
                probs = exp_z / np.sum(exp_z)
                # One-hot target
                target = np.zeros_like(probs)
                target[action] = 1.0
                # Backprop
                grad_z2 = probs - target
                grad_W2 = np.outer(h1, grad_z2)
                grad_b2 = grad_z2
                grad_h1 = self.W2.dot(grad_z2)
                grad_z1 = grad_h1 * (z1 > 0)
                grad_W1 = np.outer(state, grad_z1)
                grad_b1 = grad_z1
                # Update
                self.W1 -= self.learning_rate * grad_W1
                self.b1 -= self.learning_rate * grad_b1
                self.W2 -= self.learning_rate * grad_W2
                self.b2 -= self.learning_rate * grad_b2

    def save_model(self, path):
        np.savez(path, W1=self.W1, b1=self.b1, W2=self.W2, b2=self.b2)


# ===== Data Validator & Trainer Orchestration =====
class DataValidatorAndTrainer:
    def __init__(self, config):
        self.config = config
# ===== Main Entrypoint =====
# The main block below is for standalone testing of this trainer file.
# The actual pipeline is typically run via `train_enemy_agent_numpy.py`
if __name__ == "__main__":
    print("--- Starting Standalone EnemyTrainer Test Run ---\n")
    
    # Configuration for standalone run (mimics structure from config_enemy_numpy.json)
    # This is primarily for testing this script directly.
    # For actual training, use train_enemy_agent_numpy.py with config_enemy_numpy.json
    
    test_config_path = "config_enemy_numpy.json" # Use the numpy specific config
    
    if not os.path.exists(test_config_path):
        print(f"ERROR: Test configuration file not found: {test_config_path}")
        print("Please ensure 'config_enemy_numpy.json' exists in the project root.")
        sys.exit(1)

    try:
        with open(test_config_path, 'r') as f:
            config_data = json.load(f)
        print(f"Successfully loaded test configuration from {test_config_path}")
    except Exception as e:
        print(f"Error loading test configuration from {test_config_path}: {e}")
        sys.exit(1)

    # Extract model and training configs
    model_cfg = config_data.get('model', {})
    training_cfg = config_data.get('training', {})

    # Add static_data_path to model_cfg if it's in training_cfg, for EnemyTrainer
    if 'static_data_path' in training_cfg:
        model_cfg['static_data_path'] = training_cfg['static_data_path']

    # Step 1: (Optional) Generate Dummy Data if configured to do so or if needed for test
    # This step might be conditional based on whether static data is used/available
    # For this test run, we assume data generation/loading is handled by EnemyTrainer
    # based on 'episodes' and 'static_data_path' in config.
    
    # Initialize Teacher and Trainer
    teacher_agent = EnemyTeacher() # EnemyTeacher does not take config currently
    enemy_trainer = EnemyTrainer(config=model_cfg, teacher=teacher_agent)

    # Step 2: Generate/Load Data
    print("\nStep 1: Generating/Loading training data...")
    num_episodes_for_data = training_cfg.get('episodes', 0) # Default to 0 to prefer static data
    
    try:
        training_samples = enemy_trainer.generate_training_data(episodes=num_episodes_for_data)
        if not training_samples:
            print("No training data generated or loaded. Aborting.")
            sys.exit(1)
        print(f"Generated/Loaded {len(training_samples)} training samples.")
    except Exception as e:
        print(f"Error during data generation/loading: {e}")
        sys.exit(1)

    # Basic data validation (can be expanded)
    if len(training_samples) < training_cfg.get('min_data_samples', 10): # Min 10 for standalone
        print(f"Insufficient training data ({len(training_samples)} samples). Aborting.")
        sys.exit(1)
    print("Basic data validation passed.")

    # Step 3: Train the Model
    print("\nStep 2: Training the model...")
    num_epochs_for_training = model_cfg.get('epochs', training_cfg.get('epochs', 10))

    try:
        enemy_trainer.train_model(data=training_samples, epochs=num_epochs_for_training)
        print(f"Model training complete after {num_epochs_for_training} epochs.")
    except Exception as e:
        print(f"Error during model training: {e}")
        sys.exit(1)

    # Step 4: Save the Model
    print("\nStep 3: Saving the model...")
    model_save_path = model_cfg.get('path', 'models/numpy_enemy_model_test_standalone.npz')
    
    # Ensure directory exists
    model_save_dir = os.path.dirname(model_save_path)
    if model_save_dir and not os.path.exists(model_save_dir):
        os.makedirs(model_save_dir)
        print(f"Created directory: {model_save_dir}")
        
    try:
        enemy_trainer.save_model(model_save_path)
        print(f"Model saved successfully to {model_save_path}")
    except Exception as e:
        print(f"Error saving model: {e}")
        sys.exit(1)
        
    print("\n--- Standalone EnemyTrainer Test Run Finished ---")
