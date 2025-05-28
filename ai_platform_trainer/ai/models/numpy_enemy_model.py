import numpy as np
import os

class NumpyEnemyModel:
    """
    NumPy-based model for enemy AI inference.
    Loads weights from an .npz file and provides a predict method.
    """
    def __init__(self, model_path: str):
        """
        Initialize the model by loading weights from the given path.

        Args:
            model_path (str): Path to the .npz file containing model weights.
        """
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Model file not found at {model_path}")
        
        try:
            weights = np.load(model_path)
            self.W1 = weights['W1']
            self.b1 = weights['b1']
            self.W2 = weights['W2']
            self.b2 = weights['b2']
            print(f"🧠 NumpyEnemyModel loaded successfully from {model_path}")
        except Exception as e:
            raise IOError(f"Error loading model weights from {model_path}: {e}")

    def predict(self, state: np.ndarray) -> int:
        """
        Predicts an action ID given a state.

        Args:
            state (np.ndarray): Input state features (1D array).

        Returns:
            int: The index of the predicted action.
        """
        state = np.asarray(state, dtype=np.float32).flatten()
        # Forward pass
        z1 = np.dot(state, self.W1) + self.b1
        h1 = np.maximum(0, z1)  # ReLU activation
        z2 = np.dot(h1, self.W2) + self.b2
        
        # Return the action with the highest score (logit)
        return int(np.argmax(z2))

if __name__ == '__main__':
    # Example Usage (requires a model file to be present)
    # First, ensure 'models/numpy_enemy_model_test.npz' exists.
    # You can create it by running the NumpyEnemyTrainer example.
    
    print("Running NumpyEnemyModel example...")
    
    # Create a dummy model file for testing if it doesn't exist
    dummy_model_path = "models/numpy_enemy_model_test_infer.npz"
    if not os.path.exists(dummy_model_path):
        print(f"Creating dummy model file for inference test: {dummy_model_path}")
        # Example dimensions
        input_size = 5
        hidden_size = 16
        output_size = 4 # Assuming 4 actions (left, right, up, down)
        
        W1_test = np.random.randn(input_size, hidden_size) * 0.1
        b1_test = np.zeros(hidden_size)
        W2_test = np.random.randn(hidden_size, output_size) * 0.1
        b2_test = np.zeros(output_size)
        os.makedirs(os.path.dirname(dummy_model_path), exist_ok=True)
        np.savez(dummy_model_path, W1=W1_test, b1=b1_test, W2=W2_test, b2=b2_test)
        print("Dummy model file created.")

    try:
        model = NumpyEnemyModel(model_path=dummy_model_path)
        
        # Example state: [enemy_x, enemy_y, player_x, player_y, distance]
        example_state_infer = np.array([100, 200, 150, 210, 50.99], dtype=np.float32)
        
        predicted_action = model.predict(example_state_infer)
        print(f"Sample state: {example_state_infer}")
        print(f"Predicted action ID by NumpyEnemyModel: {predicted_action}")
        
    except Exception as e:
        print(f"Error in NumpyEnemyModel example: {e}")
