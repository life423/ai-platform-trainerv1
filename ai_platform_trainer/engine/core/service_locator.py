# file: ai_platform_trainer/core/service_locator.py
"""
Service Locator pattern implementation for dependency injection.
Provides central registry of services that can be accessed throughout the application.
"""


class ServiceLocator:
    """
    Service locator for dependency injection.
    Maintains a registry of services that can be retrieved by name.
    """

    _services = {}

    @classmethod
    def register(cls, name, service):
        """
        Register a service by name.

        Args:
            name: The name to register the service under
            service: The service instance to register
        """
        cls._services[name] = service

    @classmethod
    def get(cls, name):
        """
        Get a service by name.

        Args:
            name: The name of the service to retrieve

        Returns:
            The registered service, or None if no service with the given name exists
        """
        return cls._services.get(name)
