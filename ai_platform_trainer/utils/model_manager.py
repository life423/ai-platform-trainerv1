"""
Model Manager for handling model versioning, validation, and lifecycle.

This utility provides a centralized mechanism to:
1. Load and manage all AI models used in the game
2. Save models with versioned filenames
3. Validate models through basic sanity checks
4. Provide a single interface for accessing all models
"""

import glob
import logging
import os
import re
import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union, Type

import torch
import torch.nn as nn

# Import model classes to make them available to the ModelManager
from ai_platform_trainer.ai_model.simple_missile_model import SimpleMissileModel
from ai_platform_trainer.ai_model.model_definition.enemy_movement_model import (
    EnemyMovementModel
)


class ModelManager:
    """
    Centralized manager for all AI models in the game.
    
    This class handles loading, caching, and providing access to
    all AI models used throughout the game, ensuring they're loaded
    only once and properly shared.
    """
    
    def __init__(self, models_dir: str = "models") -> None:
        """
        Initialize the model manager.
        
        Args:
            models_dir: Base directory where models are stored
        """
        self.models_dir = models_dir
        self.loaded_models: Dict[str, nn.Module] = {}
        self.model_metadata: Dict[str, Dict[str, Any]] = {}
        self.model_configs: Dict[str, Dict[str, Any]] = {
            "missile": {
                "class": SimpleMissileModel,
                "path": os.path.join(models_dir, "missile_model.pth"),
                "input_shape": (1, 9)
            },
            "enemy": {
                "class": EnemyMovementModel,
                "path": os.path.join(models_dir, "enemy_ai_model.pth"),
                "args": {"input_size": 5, "hidden_size": 64, "output_size": 2}
            }
        }
        
        logging.info("ModelManager initialized")
    
    def load_model(self, model_name: str) -> Optional[nn.Module]:
        """
        Load a model by name if not already loaded.
        
        Args:
            model_name: Name of the model to load (e.g., 'missile', 'enemy')
            
        Returns:
            Optional[nn.Module]: The loaded model or None if loading failed
        """
        # Return cached model if already loaded
        if model_name in self.loaded_models:
            return self.loaded_models[model_name]
        
        # Check if we have configuration for this model
        if model_name not in self.model_configs:
            logging.error(f"No configuration found for model '{model_name}'")
            return None
        
        config = self.model_configs[model_name]
        model_path = config["path"]
        model_class = config["class"]
        
        # Check if the model file exists
        if not os.path.isfile(model_path):
            logging.warning(f"Model file not found: {model_path}")
            return None
        
        try:
            # Create model instance
            if "args" in config:
                model_instance = model_class(**config["args"])
            else:
                model_instance = model_class()
                
            # Load state dict
            state_dict = torch.load(model_path, map_location="cpu")
            
            # Handle different save formats (direct state_dict or wrapped)
            if isinstance(state_dict, dict) and "state_dict" in state_dict:
                # Save metadata if available
                if "metadata" in state_dict:
                    self.model_metadata[model_name] = state_dict["metadata"]
                model_instance.load_state_dict(state_dict["state_dict"])
            else:
                # Assume it's a direct state_dict
                model_instance.load_state_dict(state_dict)
            
            # Set to evaluation mode
            model_instance.eval()
            
            # Validate model if input shape is provided
            if "input_shape" in config:
                if not self._validate_model(model_instance, config["input_shape"]):
                    logging.warning(f"Model validation failed for {model_name}")
                    return None
            
            # Cache the loaded model
            self.loaded_models[model_name] = model_instance
            logging.info(f"Successfully loaded model '{model_name}' from {model_path}")
            
            return model_instance
            
        except Exception as e:
            logging.error(f"Failed to load model '{model_name}': {e}")
            return None
    
    def get_model(self, model_name: str) -> Optional[nn.Module]:
        """
        Get a model by name, loading it if necessary.
        
        Args:
            model_name: Name of the model to get
            
        Returns:
            Optional[nn.Module]: The model or None if not available
        """
        if model_name not in self.loaded_models:
            return self.load_model(model_name)
        return self.loaded_models[model_name]
    
    def load_all_models(self) -> Dict[str, bool]:
        """
        Load all configured models.
        
        Returns:
            Dict[str, bool]: Map of model names to success status
        """
        results = {}
        for model_name in self.model_configs:
            results[model_name] = self.load_model(model_name) is not None
        return results
    
    def _validate_model(self, model: nn.Module, input_shape: Tuple) -> bool:
        """
        Perform basic validation on a model.
        
        Args:
            model: The model to validate
            input_shape: Input tensor shape for test inference
            
        Returns:
            bool: True if validation passed
        """
        try:
            dummy_input = torch.zeros(input_shape, dtype=torch.float32)
            with torch.no_grad():
                output = model(dummy_input)
            
            # Check if output is a valid tensor
            if isinstance(output, torch.Tensor) or (
                isinstance(output, tuple) and 
                all(isinstance(o, torch.Tensor) for o in output)
            ):
                return True
            else:
                logging.error(f"Model output is not a valid tensor: {type(output)}")
                return False
                
        except Exception as e:
            logging.error(f"Model validation failed: {e}")
            return False
    
    def save_model(self, model_name: str, model: nn.Module, 
                  metadata: Optional[Dict[str, Any]] = None) -> str:
        """
        Save a model with versioning.
        
        Args:
            model_name: Name of the model (e.g., 'missile', 'enemy')
            model: Model to save
            metadata: Optional metadata to include
            
        Returns:
            str: Path to the saved model file or empty string on failure
        """
        # Ensure the models directory exists
        os.makedirs(self.models_dir, exist_ok=True)
        
        # Generate a timestamp for versioning
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # Create versioned filename
        versioned_name = f"{model_name}_{timestamp}.pth"
        versioned_path = os.path.join(self.models_dir, versioned_name)
        
        # Regular filename (for latest version)
        latest_path = os.path.join(self.models_dir, f"{model_name}_model.pth")
        
        # Prepare data to save
        data_to_save = {
            "state_dict": model.state_dict(),
            "model_name": model_name,
            "saved_at": timestamp,
        }
        
        # Add metadata if provided
        if metadata:
            data_to_save["metadata"] = metadata
        
        # Save the model
        try:
            torch.save(data_to_save, versioned_path)
            logging.info(f"Model saved to {versioned_path}")
            
            # Also save as the "latest" version for easy loading
            torch.save(data_to_save, latest_path)
            logging.info(f"Latest model copy saved to {latest_path}")
            
            # Update cached model and metadata
            self.loaded_models[model_name] = model
            if metadata:
                self.model_metadata[model_name] = metadata
                
            return versioned_path
        except Exception as e:
            logging.error(f"Failed to save model '{model_name}': {e}")
            return ""


# Legacy function versions for backward compatibility
def save_model_with_version(
    model: nn.Module,
    base_path: str = "models",
    model_name: str = "model",
    metadata: Optional[Dict[str, Any]] = None,
) -> str:
    """
    Save a model with an automatically versioned filename.
    
    Args:
        model: The PyTorch model to save
        base_path: Directory where models are stored
        model_name: Base name for the model
        metadata: Optional dict of metadata to save with model
        
    Returns:
        str: Path to the saved model file
    """
    manager = ModelManager(base_path)
    return manager.save_model(model_name, model, metadata)


def load_model(
    model_class: Type[nn.Module],
    version: str = "latest",
    base_path: str = "models",
    model_name: str = "model",
) -> Tuple[Optional[nn.Module], Dict[str, Any]]:
    """
    Load a model by version (or latest if not specified).
    
    Args:
        model_class: The PyTorch model class to instantiate
        version: Version to load ('latest' or a timestamp)
        base_path: Directory where models are stored
        model_name: Base name for the model
        
    Returns:
        Tuple of (loaded model, metadata dict)
    """
    # Determine the file path
    if version == "latest":
        model_path = os.path.join(base_path, f"{model_name}_latest.pth")
        # If latest doesn't exist, try to find the most recent versioned file
        if not os.path.exists(model_path):
            model_files = glob.glob(os.path.join(base_path, f"{model_name}_*.pth"))
            if not model_files:
                logging.error(f"No model files found for {model_name}")
                return None, {}
            
            # Sort by timestamp in filename (newest first)
            model_files.sort(reverse=True)
            model_path = model_files[0]
    else:
        # If a specific version was requested
        model_path = os.path.join(base_path, f"{model_name}_{version}.pth")
    
    # Load the model
    if os.path.exists(model_path):
        try:
            data = torch.load(model_path, map_location="cpu")
            model_instance = model_class()
            model_instance.load_state_dict(data["state_dict"])
            model_instance.eval()  # Set to evaluation mode
            
            # Extract metadata
            metadata = {
                "model_path": model_path,
                "model_name": data.get("model_name", model_name),
                "saved_at": data.get("saved_at", "unknown"),
            }
            
            # Add any custom metadata if it exists
            if "metadata" in data:
                metadata.update(data["metadata"])
                
            logging.info(f"Loaded model from {model_path}")
            return model_instance, metadata
        except Exception as e:
            logging.error(f"Failed to load model from {model_path}: {e}")
            return None, {}
    else:
        logging.error(f"Model file not found: {model_path}")
        return None, {}


def list_available_versions(
    base_path: str = "models", model_name: str = "model"
) -> List[Dict[str, Any]]:
    """
    List all available versions of a model.
    
    Args:
        base_path: Directory where models are stored
        model_name: Base name for the model
        
    Returns:
        List of dicts with model version information
    """
    # Find all versioned model files
    pattern = re.compile(rf"{model_name}_(\d{{8}}_\d{{6}})\.pth")
    model_files = glob.glob(os.path.join(base_path, f"{model_name}_*.pth"))
    
    result = []
    for file_path in model_files:
        filename = os.path.basename(file_path)
        # Skip the latest file
        if filename == f"{model_name}_latest.pth":
            continue
        
        match = pattern.match(filename)
        if match:
            timestamp = match.group(1)
            # Get file stats
            stats = os.stat(file_path)
            size_mb = stats.st_size / (1024 * 1024)
            
            result.append({
                "version": timestamp,
                "path": file_path,
                "size_mb": round(size_mb, 2),
                "created": datetime.datetime.fromtimestamp(stats.st_ctime).strftime(
                    "%Y-%m-%d %H:%M:%S"
                ),
            })
    
    # Sort by timestamp (newest first)
    result.sort(key=lambda x: x["version"], reverse=True)
    return result


def validate_model(
    model: nn.Module, input_shape: Tuple, device: str = "cpu"
) -> bool:
    """
    Perform basic validation on a model to ensure it works.
    
    Args:
        model: The PyTorch model to validate
        input_shape: Expected input tensor shape (e.g., (1, 5))
        device: Device to run validation on ('cpu' or 'cuda')
        
    Returns:
        bool: True if validation passed, False otherwise
    """
    try:
        # Create a dummy input tensor
        dummy_input = torch.zeros(input_shape, dtype=torch.float32, device=device)
        
        # Set model to eval mode and move to device
        model.eval()
        model = model.to(device)
        
        # Try to do a forward pass
        with torch.no_grad():
            output = model(dummy_input)
        
        # Check if output is a tensor or tuple of tensors
        if isinstance(output, torch.Tensor):
            return True
        elif isinstance(output, tuple) and all(isinstance(o, torch.Tensor) for o in output):
            return True
        else:
            logging.error(f"Model output is not a tensor or tuple of tensors: {type(output)}")
            return False
            
    except Exception as e:
        logging.error(f"Model validation failed: {e}")
        return False
