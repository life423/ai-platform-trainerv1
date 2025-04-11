"""
Core-to-Engine Adapter Module

This module provides adapters that forward imports from the legacy 'core'
directory to the canonical 'engine/core' directory. This helps maintain
backward compatibility while consolidating code.

Usage:
    Instead of:
        from ai_platform_trainer.core.service_locator import ServiceLocator
    You should use:
        from ai_platform_trainer.engine.core.service_locator import ServiceLocator

    But if legacy code still uses the old import path, this adapter will
    forward the import to the correct location.
"""
import warnings
import importlib
from typing import Any, Optional, Set


# Track which warnings have been shown
_shown_warnings: Set[str] = set()


def _warn_once(message: str) -> None:
    """Show a warning only once per message."""
    if message not in _shown_warnings:
        warnings.warn(message, DeprecationWarning, stacklevel=3)
        _shown_warnings.add(message)


class CoreAdapter:
    """
    Adapter for forwarding imports from core to engine/core.
    
    This class is used to dynamically forward imports from the legacy
    'core' directory to the canonical 'engine/core' directory.
    """
    
    @staticmethod
    def get_module(name: str) -> Any:
        """
        Get a module from engine/core with the given name.
        
        Args:
            name: The name of the module to get
            
        Returns:
            The requested module
        """
        engine_module_name = f"ai_platform_trainer.engine.core.{name}"
        _warn_once(
            f"Importing from 'ai_platform_trainer.core.{name}' is deprecated. "
            f"Use '{engine_module_name}' instead."
        )
        
        try:
            return importlib.import_module(engine_module_name)
        except ImportError as e:
            # If the module doesn't exist in engine/core, try to import from core
            try:
                return importlib.import_module(f"ai_platform_trainer.core.{name}")
            except ImportError:
                # If it doesn't exist in either place, raise the original error
                raise ImportError(
                    f"Failed to import '{name}' from either 'engine/core' or 'core': {e}"
                ) from e


# Create adapter functions for commonly used modules
def get_service_locator() -> Any:
    """Get the ServiceLocator module from engine/core."""
    module = CoreAdapter.get_module("service_locator")
    return module.ServiceLocator


def get_config_manager(config_path: Optional[str] = None) -> Any:
    """
    Get the ConfigManager from engine/core.
    
    Args:
        config_path: Optional path to a config file
        
    Returns:
        ConfigManager instance
    """
    module = CoreAdapter.get_module("config_manager")
    if config_path:
        return module.get_config_manager(config_path)
    return module.get_config_manager()


def setup_logging() -> None:
    """Set up logging using the engine/core logging configuration."""
    module = CoreAdapter.get_module("logging_config")
    return module.setup_logging()
