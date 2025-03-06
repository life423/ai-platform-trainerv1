"""
Configuration loader for the AI Platform Trainer.
This module loads configuration from settings.json and command-line arguments.
"""

import argparse
import json
import logging
import os
from typing import Dict, Any


def load_config() -> Dict[str, Any]:
    """
    Load configuration from settings.json and command-line arguments.
    Command-line arguments have precedence over settings.json.
    
    Returns:
        Dict[str, Any]: Configuration dictionary
    """
    # Default configuration
    config = {
        "headless": False,
        "fullscreen": False,
        "log_data": True,
        "batch_logging": True,
        "training_speed": 1.0,  # For faster than real-time training
    }
    
    # Load from settings.json if it exists
    settings_file = "settings.json"
    if os.path.exists(settings_file):
        try:
            with open(settings_file, "r") as f:
                loaded_settings = json.load(f)
                config.update(loaded_settings)
        except (json.JSONDecodeError, IOError) as e:
            logging.warning(f"Failed to load settings from {settings_file}: {e}")
    
    # Parse command-line arguments
    parser = argparse.ArgumentParser(description="AI Platform Trainer")
    parser.add_argument("--headless", action="store_true", 
                        help="Run in headless mode without display")
    parser.add_argument("--fullscreen", action="store_true", 
                        help="Run in fullscreen mode")
    parser.add_argument("--no-log", dest="log_data", action="store_false", 
                        help="Disable data logging")
    parser.add_argument("--batch-logging", action="store_true", 
                        help="Enable batch logging (save at end)")
    parser.add_argument("--training-speed", type=float, 
                        default=config["training_speed"], 
                        help="Speed multiplier for training "
                             "(e.g., 2.0 runs twice as fast)")
    parser.add_argument("--mode", choices=["train", "play"], 
                        help="Start directly in train or play mode")
    
    args = parser.parse_args()
    
    # Update config with command-line arguments (only if explicitly set)
    for key, value in vars(args).items():
        if value is not None and (key not in config or
                                  value != parser.get_default(key)):
            config[key] = value
    
    # Log the configuration
    logging.info(f"Loaded configuration: {config}")
    
    return config
