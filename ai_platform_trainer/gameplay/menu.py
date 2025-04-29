"""
Compatibility module for AI Platform Trainer menu.

This module re-exports the Menu class from the new location to maintain
backward compatibility with existing code during the refactoring process.
"""
import warnings

# Import from new location with prefixed name
from ai_platform_trainer.engine.rendering.menu import Menu as _Menu

# Re-export at module level
Menu = _Menu
