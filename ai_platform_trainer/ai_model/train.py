#!/usr/bin/env python
"""
Unified training script for AI models.

This script provides a centralized way to train different AI models used in the game:
- Missile prediction model
- Enemy movement model

Usage:
    python -m ai_platform_trainer.ai_model.train --model missile --epochs 20
    python -m ai_platform_trainer.ai_model.train --model enemy --epochs 50
"""

import os
import argparse
import logging
import torch
import torch.nn as nn
import torch.optim as optim
import time
from datetime import datetime
from typing import Dict, Any, Optional

from ai_platform_trainer.data.data_manager import DataManager
from ai_platform_trainer.utils.model_manager import ModelManager
from ai_platform_trainer.ai_model.model_definition.simple_missile_model import SimpleMissileModel
from ai_platform_trainer.ai_model.model_definition.enemy_movement_model import EnemyMovementModel


def train_missile_model(
    data_file: str = "training_data.json",
    epochs: int = 20,
    batch_size: int = 32,
    learning_rate: float = 0.001,
    hidden_size: int = 64,
    save_best: bool = True,
    model_manager: Optional[ModelManager] = None,
) -> SimpleMissileModel:
    """
    Train the missile prediction model.
    
    Args:
        data_file: Path to training data
        epochs: Number of training epochs
        batch_size: Size of mini-batches
        learning_rate: Learning rate for optimizer
        hidden_size: Size of hidden layers
        save_best: Whether to save the best model during training
        model_manager: ModelManager instance for saving the model
        
    Returns:
        The trained model
    """
    logging.info(f"Training missile model with {epochs} epochs")
    
    # Initialize managers if not provided
    if model_manager is None:
        model_manager = ModelManager()
    
    data_manager = DataManager()
    
    # Create dataloader
    dataloader = data_manager.create_missile_dataloader(
        json_file=data_file,
        batch_size=batch_size,
        shuffle=True
    )
    
    # Initialize model and optimizer
    model = SimpleMissileModel(input_size=9, hidden_size=hidden_size, output_size=1)
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    
    # Training loop
    best_loss = float('inf')
    start_time = time.time()
    
    for epoch in range(epochs):
        running_loss = 0.0
        total_batches = 0
        
        for states, actions, weights in dataloader:
            optimizer.zero_grad()
            
            preds = model(states).view(-1)
            actions = actions.view(-1)
            weights = weights.view(-1)
            
            # Weighted MSE
            loss_per_sample = (preds - actions)**2 * weights
            loss = loss_per_sample.mean()
            
            loss.backward()
            optimizer.step()
            
            running_loss += loss.item()
            total_batches += 1
        
        avg_loss = running_loss / total_batches if total_batches > 0 else 0
        logging.info(f"Epoch {epoch}/{epochs - 1}, Avg Loss: {avg_loss:.4f}")
        
        # Save best model
        if save_best and avg_loss < best_loss:
            best_loss = avg_loss
            
            metadata = {
                "epochs": epoch + 1,
                "loss": best_loss,
                "batch_size": batch_size,
                "learning_rate": learning_rate,
                "hidden_size": hidden_size,
                "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                "training_data": data_file
            }
            
            model_manager.save_model("missile", model, metadata=metadata)
            logging.info(f"Saved best model with loss: {best_loss:.4f}")
    
    # Save final model
    training_time = time.time() - start_time
    
    metadata = {
        "epochs": epochs,
        "final_loss": avg_loss,
        "best_loss": best_loss,
        "batch_size": batch_size,
        "learning_rate": learning_rate,
        "hidden_size": hidden_size,
        "training_time_seconds": training_time,
        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "training_data": data_file
    }
    
    save_path = model_manager.save_model("missile", model, metadata=metadata)
    logging.info(f"Training completed in {training_time:.2f} seconds")
    logging.info(f"Final model saved to {save_path}")
    
    return model


def train_enemy_model(
    data_file: str = "training_data.json",
    epochs: int = 50,
    batch_size: int = 32,
    learning_rate: float = 0.001,
    hidden_size: int = 128,
    dropout_prob: float = 0.3,
    save_best: bool = True,
    model_manager: Optional[ModelManager] = None,
) -> EnemyMovementModel:
    """
    Train the enemy movement model.
    
    Args:
        data_file: Path to training data
        epochs: Number of training epochs
        batch_size: Size of mini-batches
        learning_rate: Learning rate for optimizer
        hidden_size: Size of hidden layers
        dropout_prob: Dropout probability
        save_best: Whether to save the best model during training
        model_manager: ModelManager instance for saving the model
        
    Returns:
        The trained model
    """
    logging.info(f"Training enemy movement model with {epochs} epochs")
    
    # Initialize managers if not provided
    if model_manager is None:
        model_manager = ModelManager()
    
    data_manager = DataManager()
    
    # Create dataloader
    dataloader = data_manager.create_enemy_dataloader(
        json_file=data_file,
        batch_size=batch_size,
        shuffle=True
    )
    
    if dataloader is None:
        logging.error("Failed to create dataloader for enemy model")
        return None
    
    # Initialize model and optimizer
    model = EnemyMovementModel(
        input_size=5, 
        hidden_size=hidden_size, 
        output_size=2,
        dropout_prob=dropout_prob
    )
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    criterion = nn.MSELoss()
    
    # Training loop
    best_loss = float('inf')
    start_time = time.time()
    
    for epoch in range(epochs):
        running_loss = 0.0
        total_batches = 0
        
        for features, targets in dataloader:
            optimizer.zero_grad()
            
            # Forward pass
            outputs = model(features)
            loss = criterion(outputs, targets)
            
            # Backward pass and optimize
            loss.backward()
            optimizer.step()
            
            running_loss += loss.item()
            total_batches += 1
        
        avg_loss = running_loss / total_batches if total_batches > 0 else 0
        logging.info(f"Epoch {epoch}/{epochs - 1}, Avg Loss: {avg_loss:.4f}")
        
        # Save best model
        if save_best and avg_loss < best_loss:
            best_loss = avg_loss
            
            metadata = {
                "epochs": epoch + 1,
                "loss": best_loss,
                "batch_size": batch_size,
                "learning_rate": learning_rate,
                "hidden_size": hidden_size,
                "dropout_prob": dropout_prob,
                "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                "training_data": data_file
            }
            
            model_manager.save_model("enemy", model, metadata=metadata)
            logging.info(f"Saved best model with loss: {best_loss:.4f}")
    
    # Save final model
    training_time = time.time() - start_time
    
    metadata = {
        "epochs": epochs,
        "final_loss": avg_loss,
        "best_loss": best_loss,
        "batch_size": batch_size,
        "learning_rate": learning_rate,
        "hidden_size": hidden_size,
        "dropout_prob": dropout_prob,
        "training_time_seconds": training_time,
        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "training_data": data_file
    }
    
    save_path = model_manager.save_model("enemy", model, metadata=metadata)
    logging.info(f"Training completed in {training_time:.2f} seconds")
    logging.info(f"Final model saved to {save_path}")
    
    return model


def main():
    """Parse arguments and train the specified model."""
    parser = argparse.ArgumentParser(description="Train AI models for the game")
    
    parser.add_argument(
        "--model", 
        type=str,
        required=True,
        choices=["missile", "enemy"],
        help="Which model to train (missile or enemy)"
    )
    
    parser.add_argument(
        "--data_file",
        type=str,
        default=None,
        help="JSON file with training data (default: latest in data/raw)"
    )
    
    parser.add_argument(
        "--epochs",
        type=int,
        default=None,
        help="Number of training epochs (default: model-specific)"
    )
    
    parser.add_argument(
        "--batch_size",
        type=int,
        default=32,
        help="Training batch size (default: 32)"
    )
    
    parser.add_argument(
        "--learning_rate",
        type=float,
        default=0.001,
        help="Learning rate (default: 0.001)"
    )
    
    parser.add_argument(
        "--hidden_size",
        type=int,
        default=None,
        help="Size of hidden layers (default: model-specific)"
    )
    
    parser.add_argument(
        "--dropout_prob",
        type=float,
        default=0.3,
        help="Dropout probability (for enemy model only, default: 0.3)"
    )
    
    parser.add_argument(
        "--no_save_best",
        action="store_true",
        help="Don't save best model during training (default: save best)"
    )
    
    parser.add_argument(
        "--models_dir",
        type=str,
        default="models",
        help="Directory to save trained models (default: models)"
    )
    
    args = parser.parse_args()
    
    # Configure basic logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    # Initialize model manager
    model_manager = ModelManager(models_dir=args.models_dir)
    
    # Determine data file
    if args.data_file is None:
        data_manager = DataManager()
        args.data_file = data_manager.get_latest_data_file()
        logging.info(f"Using latest data file: {args.data_file}")
    
    # Set model-specific defaults
    if args.model == "missile":
        if args.epochs is None:
            args.epochs = 20
        if args.hidden_size is None:
            args.hidden_size = 64
        
        train_missile_model(
            data_file=args.data_file,
            epochs=args.epochs,
            batch_size=args.batch_size,
            learning_rate=args.learning_rate,
            hidden_size=args.hidden_size,
            save_best=not args.no_save_best,
            model_manager=model_manager,
        )
    
    elif args.model == "enemy":
        if args.epochs is None:
            args.epochs = 50
        if args.hidden_size is None:
            args.hidden_size = 128
        
        train_enemy_model(
            data_file=args.data_file,
            epochs=args.epochs,
            batch_size=args.batch_size,
            learning_rate=args.learning_rate,
            hidden_size=args.hidden_size,
            dropout_prob=args.dropout_prob,
            save_best=not args.no_save_best,
            model_manager=model_manager,
        )


if __name__ == "__main__":
    main()
