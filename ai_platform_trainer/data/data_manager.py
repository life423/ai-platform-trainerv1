"""
Data Manager for loading and processing training data for AI models.

This module provides a centralized mechanism to:
1. Load training data for different models
2. Process and prepare data for training
3. Create data loaders for training
"""

import json
import os
import logging
from typing import Optional, Tuple
import torch
from torch.utils.data import DataLoader

from ai_platform_trainer.ai_model.missile_dataset import MissileDataset
from ai_platform_trainer.ai_model.enemy_dataset import EnemyDataset


class DataManager:
    """
    Centralized manager for all training data in the game.
    
    This class handles loading, processing, and preparing data
    for training AI models used throughout the game.
    """
    
    def __init__(self, data_dir: str = "data/raw") -> None:
        """
        Initialize the data manager.
        
        Args:
            data_dir: Base directory where training data is stored
        """
        self.data_dir = data_dir
        logging.info("DataManager initialized")
    
    def load_missile_data(self, json_file: str = "training_data.json") -> MissileDataset:
        """
        Load missile training data.
        
        Args:
            json_file: Name of the JSON file containing training data.
                      If not a full path, will be loaded from data_dir.
        
        Returns:
            MissileDataset: Dataset object for missile training
        """
        if not os.path.isabs(json_file):
            json_file = os.path.join(self.data_dir, json_file)
        
        logging.info(f"Loading missile data from {json_file}")
        return MissileDataset(json_file=json_file)
    
    def create_missile_dataloader(
        self, 
        json_file: str = "training_data.json", 
        batch_size: int = 32, 
        shuffle: bool = True
    ) -> DataLoader:
        """
        Create a DataLoader for missile training.
        
        Args:
            json_file: Name of the JSON file containing training data
            batch_size: Size of mini-batches for training
            shuffle: Whether to shuffle the data during training
        
        Returns:
            DataLoader: PyTorch DataLoader for training
        """
        dataset = self.load_missile_data(json_file)
        return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)
    
    def load_enemy_data(self, json_file: str = "training_data.json") -> EnemyDataset:
        """
        Load enemy training data.
        
        Args:
            json_file: Name of the JSON file containing training data.
                      If not a full path, will be loaded from data_dir.
        
        Returns:
            EnemyDataset: Dataset object for enemy movement training
        """
        if not os.path.isabs(json_file):
            json_file = os.path.join(self.data_dir, json_file)
        
        logging.info(f"Loading enemy data from {json_file}")
        return EnemyDataset(json_file=json_file)
    
    def create_enemy_dataloader(
        self,
        json_file: str = "training_data.json",
        batch_size: int = 32,
        shuffle: bool = True
    ) -> Optional[DataLoader]:
        """
        Create a DataLoader for enemy movement training.
        
        Args:
            json_file: Name of the JSON file containing training data
            batch_size: Size of mini-batches for training
            shuffle: Whether to shuffle the data during training
        
        Returns:
            Optional[DataLoader]: PyTorch DataLoader for training or None if data loading failed
        """
        dataset = self.load_enemy_data(json_file)
        
        if len(dataset) == 0:
            return None
        
        return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)
    
    def get_latest_data_file(self) -> str:
        """
        Find the most recently created training data file.
        
        Returns:
            str: Filename of the latest training data file
        """
        data_files = [
            f for f in os.listdir(self.data_dir)
            if f.startswith("training_data_") and f.endswith(".json")
        ]
        
        if not data_files:
            # If no timestamped files, fall back to default
            return "training_data.json"
        
        # Sort by filename (which contains timestamp)
        data_files.sort(reverse=True)
        return data_files[0]
