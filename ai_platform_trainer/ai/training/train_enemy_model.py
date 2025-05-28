"""
Train an enemy movement prediction model using the EnemyMovementModel
architecture.

This module provides a training pipeline for the enemy movement
prediction model, including data loading, training loop, optimization,
and model saving.
"""
import os
from typing import Optional, List, Dict, Union

import torch
from torch import optim
from torch.utils.data import DataLoader

from .enemy_dataset import EnemyDataset
from ..models.enemy_movement_model import EnemyMovementModel


class EnemyTrainer:
    """
    Trainer for the enemy movement prediction model.

    Encapsulates the training logic for EnemyMovementModel,
    including dataset handling, optimization, and training loop execution.
    """

    def __init__(
        self,
        filename: str = "data/raw/enemy_training_data.json", # Changed default path
        epochs: int = 50,
        batch_size: int = 64,
        lr: float = 0.0005,
        model_save_path: str = "models/enemy_ai_model.pth",
    ) -> None:
        """
        Initialize the trainer with configurable training parameters.

        Args:
            filename: Path to the JSON dataset file for enemy movement.
            epochs: Number of training epochs to run.
            batch_size: Number of samples per training batch.
            lr: Learning rate for the Adam optimizer.
            model_save_path: File path where trained enemy model will be saved.
        """
        self.filename = filename
        self.epochs = epochs
        self.batch_size = batch_size
        self.lr = lr
        self.model_save_path = model_save_path

        self.dataset = EnemyDataset(json_file=self.filename)
        self.dataloader = DataLoader(
            self.dataset, batch_size=self.batch_size, shuffle=True
        )

        # Initialize model and optimizer
        self.model = EnemyMovementModel(
            input_size=5, hidden_size=64, output_size=2  # Changed hidden_size to 64
        )
        self.optimizer = optim.Adam(self.model.parameters(), lr=self.lr)
        self.criterion = torch.nn.MSELoss()  # MSE for dx, dy regression

    def run_training(self) -> Optional[Dict[str, Union[List[float], List[str]]]]:
        """
        Execute the main training loop and save the final model.

        Performs the complete training process, including:
        - Batch iteration through the dataset
        - Forward and backward passes
        - Optimization steps
        - Loss tracking
        - Model saving

        Returns:
            Optional dictionary containing training metrics (e.g., loss history)
        """
        if not hasattr(self, 'dataloader') or self.dataloader is None:
            print("Error: Dataloader not initialized. "
                  "EnemyDataset might be missing or failed to load.")
            return {"error": ["Dataloader not initialized."]} # type: ignore

        if len(self.dataset) == 0: # Use self.dataset directly
            print(f"Error: Dataset at {self.filename} is empty. Cannot train.")
            return {"error": [f"Dataset at {self.filename} is empty."]} # type: ignore

        print(f"Starting enemy model training for {self.epochs} epochs.")
        history: Dict[str, List[float]] = {"loss": []}

        for epoch in range(self.epochs):
            running_loss = 0.0
            total_batches = 0

            for states, target_actions in self.dataloader:
                self.optimizer.zero_grad()

                # Get model predictions
                predictions = self.model(states)

                loss = self.criterion(predictions, target_actions)

                loss.backward()
                self.optimizer.step()

                running_loss += loss.item()
                total_batches += 1

            if total_batches > 0:
                avg_loss = running_loss / total_batches
                history["loss"].append(avg_loss)
                print(f"Epoch {epoch + 1}/{self.epochs}, "
                      f"Avg Loss: {avg_loss:.6f}")
            else:
                # This case should not happen with a real dataloader
                # unless dataset is empty.
                print(f"Epoch {epoch + 1}/{self.epochs}, No data processed.")

        # Ensure the directory for saving the model exists
        os.makedirs(os.path.dirname(self.model_save_path), exist_ok=True)

        # Optionally remove old file
        if os.path.exists(self.model_save_path):
            try:
                os.remove(self.model_save_path)
                print(f"Removed old model file at '{self.model_save_path}'.")
            except OSError as e:
                print(f"Error removing old model file: {e}")

        # Save the model
        try:
            torch.save(self.model.state_dict(), self.model_save_path)
            print(f"Saved new enemy model to '{self.model_save_path}'.")
        except Exception as e:
            print(f"Error saving enemy model: {e}")
            # Return a dict that matches the expected Optional[Dict[str, List[float]]]
            # For simplicity, returning None here, or an error dict if preferred.
            # For example: return {"error_message": [str(e)]} # if List[float] is strict
            return None # Or handle error dict more robustly

        return history


if __name__ == "__main__":
    # Example of how to run the trainer
    print("Running EnemyTrainer with actual data.")
    trainer = EnemyTrainer( # Uses the default filename="data/raw/enemy_training_data.json"
        epochs=10,  # Short epochs for testing
        batch_size=32,
        lr=0.001,
        model_save_path="models/enemy_ai_model.pth"
    )
    training_history_result = trainer.run_training()
    if training_history_result and "error" not in training_history_result: # type: ignore
        print("Enemy model training completed.")
        if "loss" in training_history_result: # type: ignore
            print("Loss history:", training_history_result["loss"]) # type: ignore
    else:
        print("Enemy model training failed or had no data.")
