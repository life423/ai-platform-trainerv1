#!/usr/bin/env python
"""
Train both missile and enemy models in sequence.

This script runs the training for both models using the same dataset
and default parameters for each model.

Usage:
    python -m ai_platform_trainer.ai_model.train_all [--epochs EPOCHS] [--data_file DATA_FILE]
"""

import argparse
import logging
from typing import Optional

from ai_platform_trainer.data.data_manager import DataManager
from ai_platform_trainer.ai_model.train import (
    train_missile_model, 
    train_enemy_model
)
from ai_platform_trainer.utils.model_manager import ModelManager


def train_all(
    data_file: Optional[str] = None,
    missile_epochs: int = 20,
    enemy_epochs: int = 50,
    batch_size: int = 32,
    learning_rate: float = 0.001,
) -> None:
    """
    Train both missile and enemy models in sequence.
    
    Args:
        data_file: Path to training data file (if None, uses latest)
        missile_epochs: Number of epochs for missile model
        enemy_epochs: Number of epochs for enemy model
        batch_size: Batch size for training
        learning_rate: Learning rate for optimizers
    """
    # Configure logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    # Initialize managers
    model_manager = ModelManager()
    data_manager = DataManager()
    
    # Determine data file if not specified
    if data_file is None:
        data_file = data_manager.get_latest_data_file()
        logging.info(f"Using latest data file: {data_file}")
    
    # Train missile model
    logging.info("========== TRAINING MISSILE MODEL ==========")
    train_missile_model(
        data_file=data_file,
        epochs=missile_epochs,
        batch_size=batch_size,
        learning_rate=learning_rate,
        model_manager=model_manager,
    )
    
    # Train enemy model
    logging.info("========== TRAINING ENEMY MODEL ==========")
    train_enemy_model(
        data_file=data_file,
        epochs=enemy_epochs,
        batch_size=batch_size,
        learning_rate=learning_rate,
        model_manager=model_manager,
    )
    
    logging.info("========== ALL MODELS TRAINED SUCCESSFULLY ==========")


def main():
    """Parse arguments and train both models."""
    parser = argparse.ArgumentParser(description="Train both missile and enemy models")
    
    parser.add_argument(
        "--data_file",
        type=str,
        default=None,
        help="JSON file with training data (default: latest in data/raw)"
    )
    
    parser.add_argument(
        "--missile_epochs",
        type=int,
        default=20,
        help="Number of training epochs for missile model "
             "(default: 20)"
    )
    
    parser.add_argument(
        "--enemy_epochs",
        type=int,
        default=50,
        help="Number of training epochs for enemy model "
             "(default: 50)"
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
    
    args = parser.parse_args()
    
    train_all(
        data_file=args.data_file,
        missile_epochs=args.missile_epochs,
        enemy_epochs=args.enemy_epochs,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
    )


if __name__ == "__main__":
    main()
