# This file has been removed as part of the cleanup process.
# The functionality has been moved to:
# - ai_platform_trainer/core/config_manager.py
# - ai_platform_trainer/engine/core/config_manager.py
#
# For backward compatibility with any code that might still import from this module,
# we're providing the minimal functionality needed.

import json
import os
import warnings
from typing import Dict

warnings.warn(
    "The root config_manager.py is deprecated and has been removed. "
    "Use ai_platform_trainer.core.config_manager or ai_platform_trainer.engine.core.config_manager instead.",
    DeprecationWarning,
    stacklevel=2
)

# Import the proper implementation
from ai_platform_trainer.core.config_manager import get_config_manager

DEFAULT_SETTINGS = {
    "fullscreen": False,
    "width": 1280,
    "height": 720,
}


def load_settings(config_path: str = "settings.json") -> Dict[str, bool | int]:
    """
    Load settings from a JSON file. If the file doesn't exist, returns a default dict.
    """
    warnings.warn(
        "load_settings() from root config_manager.py is deprecated. "
        "Use ai_platform_trainer.core.config_manager.get_config_manager() instead.",
        DeprecationWarning,
        stacklevel=2
    )
    
    if os.path.isfile(config_path):
        with open(config_path, "r") as file:
            return json.load(file)
    return DEFAULT_SETTINGS.copy()


def save_settings(settings: Dict[str, bool | int], config_path: str = "settings.json") -> None:
    """
    Save the settings dictionary to a JSON file.
    """
    warnings.warn(
        "save_settings() from root config_manager.py is deprecated. "
        "Use ai_platform_trainer.core.config_manager.get_config_manager().save() instead.",
        DeprecationWarning,
        stacklevel=2
    )
    
    with open(config_path, "w") as file:
        json.dump(settings, file, indent=4)