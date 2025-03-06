"""
Dataset class for loading and preparing enemy movement training data.

This module provides a PyTorch Dataset implementation for enemy training data
that can be used by data loaders for model training.
"""

import json
import torch
from torch.utils.data import Dataset
import logging
from typing import Dict, List, Tuple, Any, Optional


class EnemyDataset(Dataset):
    """
    Dataset for enemy movement training data.
    
    Processes raw JSON game data into feature vectors (player position, enemy position, etc.)
    and target vectors (enemy movement direction).
    """
    
    def __init__(self, json_file: str):
        """
        Initialize the enemy dataset.
        
        Args:
            json_file: Path to JSON file containing training data
        """
        self.features = []
        self.targets = []
        
        try:
            with open(json_file, "r") as f:
                data = json.load(f)
            
            # Process each entry in the data
            for entry in data:
                # Skip entries without enemy data
                if 'enemy_x' not in entry or 'enemy_y' not in entry:
                    continue
                
                # Create feature vector [player_x, player_y, enemy_x, enemy_y, enemy_health]
                feature = [
                    entry['player_x'], 
                    entry['player_y'], 
                    entry['enemy_x'], 
                    entry['enemy_y'],
                    entry.get('enemy_health', 100)  # Default to 100 if not present
                ]
                
                # Target is [dx, dy] - enemy movement direction
                if 'enemy_dx' in entry and 'enemy_dy' in entry:
                    target = [entry['enemy_dx'], entry['enemy_dy']]
                    
                    self.features.append(feature)
                    self.targets.append(target)
            
            # Convert lists to tensors
            if self.features:
                self.features = torch.tensor(self.features, dtype=torch.float32)
                self.targets = torch.tensor(self.targets, dtype=torch.float32)
                logging.info(f"Loaded {len(self.features)} enemy movement training samples")
            else:
                self.features = torch.tensor([])
                self.targets = torch.tensor([])
                logging.warning("No valid enemy movement data found in the dataset")
                
        except Exception as e:
            logging.error(f"Failed to load enemy dataset: {e}")
            self.features = torch.tensor([])
            self.targets = torch.tensor([])
    
    def __len__(self) -> int:
        """Return the number of samples in the dataset."""
        return len(self.features)
    
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Get a sample from the dataset.
        
        Args:
            idx: Index of the sample to get
            
        Returns:
            Tuple[torch.Tensor, torch.Tensor]: Feature and target tensors
        """
        return self.features[idx], self.targets[idx]
    
    def get_all_data(self) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Get all features and targets as tensors.
        
        Returns:
            Tuple[torch.Tensor, torch.Tensor]: All features and targets
        """
        return self.features, self.targets
