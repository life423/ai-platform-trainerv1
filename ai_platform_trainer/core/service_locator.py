"""
Service Locator Module (Adapter)

This is an adapter module that forwards to the canonical implementation
in engine/core/service_locator.py for backward compatibility.

DEPRECATED: Use ai_platform_trainer.engine.core.service_locator instead.
"""

import warnings
from ai_platform_trainer.core.adapter import CoreAdapter

# Import and re-export the ServiceLocator class from engine/core
_service_locator_module = CoreAdapter.get_module("service_locator")
ServiceLocator = _service_locator_module.ServiceLocator

# Add deprecation warning
warnings.warn(
    "Importing from ai_platform_trainer.core.service_locator is deprecated. "
    "Use ai_platform_trainer.engine.core.service_locator instead.",
    DeprecationWarning,
    stacklevel=2,
)
