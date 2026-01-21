"""
Singleton pattern implementation.

Provides thread-safe singleton metaclass and reset functionality.
"""

import threading
from typing import Dict, Type, TypeVar

T = TypeVar("T")


class SingletonMeta(type):
    """
    Thread-safe singleton metaclass.

    Usage:
        class MyClass(metaclass=SingletonMeta):
            pass

        instance1 = MyClass()
        instance2 = MyClass()
        assert instance1 is instance2  # Same instance
    """

    _instances: Dict[Type, object] = {}
    _lock = threading.Lock()

    def __call__(cls, *args, **kwargs):
        if cls not in cls._instances:
            with cls._lock:
                # Double-check locking pattern
                if cls not in cls._instances:
                    cls._instances[cls] = super().__call__(*args, **kwargs)
        return cls._instances[cls]

    @classmethod
    def reset(cls, target_class: Type):
        """
        Reset singleton instance (useful for testing).

        Args:
            target_class: Class to reset
        """
        with cls._lock:
            cls._instances.pop(target_class, None)


def reset_singleton(target_class: Type):
    """
    Reset singleton instance (convenience function).

    Args:
        target_class: Class to reset
    """
    SingletonMeta.reset(target_class)
