import os
import sys
import json
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
            # Corrected logic: if dy > 0, enemy is below player, so move UP (action 2)
            # if dy < 0, enemy is above player, so move DOWN (action 3)
            # User snippet had: return 3 if dy > 0 else 2
            # My previous teacher had: return self.ACTION_DOWN if dy > 0 else self.ACTION_UP
            # Assuming standard screen coordinates (y increases downwards)
            # If player.y > enemy.y (dy > 0), player is below enemy, enemy should move DOWN (action 3)
            # If player.y < enemy.y (dy < 0), player is above enemy, enemy should move UP (action 2)
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
        self.learning_rate = config['learning_rate']

    def generate_training_data(self, episodes=100):
        static_path = "data/raw/enemy_training_data.npz"
        if episodes == 0 and os.path.exists(static_path):
            print(f"Loading dummy data from {static_path}")
            archive = np.load(static_path)
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
        # Assuming screen dimensions are known for normalization, e.g., from config or constants
        # For this example, let's use typical screen dimensions if not in config.
        # Ideally, these would come from the game config if available to the trainer.
        screen_width = self.config.get('screen_width', 800) 
        screen_height = self.config.get('screen_height', 600)
        max_dim = max(screen_width, screen_height)

        for _ in range(epochs):
            np.random.shuffle(data)
            for state, action in data:
                # Normalize state: [ex, ey, px, py, dist]
                # ex, px normalized by screen_width
                # ey, py normalized by screen_height
                # dist normalized by max_dim (diagonal)
                normalized_state = np.array([
                    state[0] / screen_width,
                    state[1] / screen_height,
                    state[2] / screen_width,
                    state[3] / screen_height,
                    state[4] / max_dim 
                ], dtype=np.float32)

                # Forward pass
                z1 = normalized_state.dot(self.W1) + self.b1
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
                # Corrected dot product for grad_h1
                grad_h1 = np.dot(grad_z2, self.W2.T) # grad_z2 is (out_dim,), W2.T is (out_dim, hid_dim) -> (hid_dim,)
                grad_z1 = grad_h1 * (z1 > 0)
                grad_W1 = np.outer(normalized_state, grad_z1) # Use normalized_state for gradient calculation
                grad_b1 = grad_z1
                # Update
                self.W1 -= self.learning_rate * grad_W1
                self.b1 -= self.learning_rate * grad_b1
                self.W2 -= self.learning_rate * grad_W2
                self.b2 -= self.learning_rate * grad_b2

    def save_model(self, path):
        # Ensure directory exists
        os.makedirs(os.path.dirname(path), exist_ok=True)
        np.savez(path, W1=self.W1, b1=self.b1, W2=self.W2, b2=self.b2)


# ===== Data Validator & Trainer Orchestration =====
class DataValidatorAndTrainer:
    def __init__(self, config):
        self.config = config
        # Ensure 'model' key exists before accessing sub-keys
        model_config = config.get('model', {})
        self.trainer = EnemyTrainer(model_config, EnemyTeacher())
        self.model_path = model_config.get('path', 'models/numpy_enemy_model.npz') # Default path
        self.episodes = config.get('episodes', 0) # Default to 0 for loading static data
        self.epochs = config.get('epochs', 10) # Default epochs

    def run(self):
        data = self.trainer.generate_training_data(self.episodes)
        if len(data) < 10:
            raise ValueError("Insufficient training data")
        self.trainer.train_model(data, self.epochs)
        self.trainer.save_model(self.model_path)
        print(f"Model saved to {self.model_path}")


# ===== Main Entrypoint =====
if __name__ == "__main__":
    print("--- Starting Enemy Agent Training Orchestration ---\n")
    # Step 1: Generate Dummy Data
    print("Step 1: Generating dummy data for NumPy model...")
    # Ensure the script path is correct for os.system
    dummy_data_script_path = os.path.join("scripts", "tools", "generate_dummy_enemy_data.py")
    result = os.system(f"python {dummy_data_script_path}")
    if result != 0:
        print(f"Dummy data generation failed (exit code {result}). Aborting.")
        sys.exit(1)
    print("Dummy data generation complete.\n")

    # Step 2: Train the Model
    print("Step 2: Running NumPy-based training...")
    config_file_path = "config_enemy_numpy.json" # Using the numpy specific config
    try:
        if not os.path.exists(config_file_path):
            print(f"ERROR: Configuration file '{config_file_path}' not found. Aborting.")
            sys.exit(1)
        with open(config_file_path) as f:
            config = json.load(f)
        
        # Ensure config has necessary keys for DataValidatorAndTrainer
        if 'model' not in config:
            config['model'] = {} # Provide default empty dict if not present
        if 'episodes' not in config: # Default to 0 if not in config
             config['episodes'] = 0
        if 'epochs' not in config: # Default to 10 if not in config
             config['epochs'] = 10


        pipeline = DataValidatorAndTrainer(config)
        pipeline.run()
        print("Enemy AI model training complete.")
    except Exception as e:
        print(f"NumPy-based training failed: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
    print("\n--- Enemy Agent Training Orchestration Finished ---")
