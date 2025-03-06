"""
Model Manager for handling model versioning, validation, and lifecycle.

This utility provides functions to:
1. Save models with versioned filenames
2. Load the latest or specific version of a model
3. Validate models through basic sanity checks
"""

import glob
import logging
import os
import re
import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import torch
import torch.nn as nn


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
    # Ensure the models directory exists
    os.makedirs(base_path, exist_ok=True)
    
    # Generate a timestamp for versioning
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # Create versioned filename
    versioned_name = f"{model_name}_{timestamp}.pth"
    versioned_path = os.path.join(base_path, versioned_name)
    
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
        
        # Also save a copy as the "latest" for easy loading
        latest_path = os.path.join(base_path, f"{model_name}_latest.pth")
        torch.save(data_to_save, latest_path)
        logging.info(f"Latest model copy saved to {latest_path}")
        
        return versioned_path
    except Exception as e:
        logging.error(f"Failed to save model: {e}")
        return ""


def load_model(
    model_class: nn.Module,
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
