
from typing import Any, Dict, List, Optional, Type
import importlib

from modules.common.utils import log_error, log_info
from modules.hmm.signals.strategy import HMMStrategy
from modules.hmm.signals.strategy import HMMStrategy

"""
HMM Strategy Registry Module

Manages registration and retrieval of HMM strategies using the Registry pattern.
Enables dynamic strategy loading from configuration.
"""




class HMMStrategyRegistry:
    """
    Registry for managing HMM strategies.

    Allows strategies to be registered dynamically and loaded from configuration.
    """

    def __init__(self):
        """Initialize empty registry."""
        self._strategies: Dict[str, HMMStrategy] = {}

    def register(self, strategy: HMMStrategy) -> None:
        """
        Register a strategy in the registry.

        Args:
            strategy: HMMStrategy instance to register

        Raises:
            ValueError: If strategy name already exists
        """
        if strategy.name in self._strategies:
            raise ValueError(f"Strategy '{strategy.name}' is already registered")

        self._strategies[strategy.name] = strategy
        log_info(f"Registered HMM strategy: {strategy.name} (weight={strategy.weight}, enabled={strategy.enabled})")

    def get(self, name: str) -> Optional[HMMStrategy]:
        """
        Get strategy by name.

        Args:
            name: Strategy name

        Returns:
            HMMStrategy instance or None if not found
        """
        return self._strategies.get(name)

    def get_all(self) -> List[HMMStrategy]:
        """
        Get all registered strategies.

        Returns:
            List of all registered strategies
        """
        return list(self._strategies.values())

    def get_enabled(self) -> List[HMMStrategy]:
        """
        Get only enabled strategies.

        Returns:
            List of enabled strategies, sorted by weight (descending)
        """
        enabled = [s for s in self._strategies.values() if s.enabled]
        return sorted(enabled, key=lambda s: s.weight, reverse=True)

    def unregister(self, name: str) -> bool:
        """
        Unregister a strategy.

        Args:
            name: Strategy name to unregister

        Returns:
            True if strategy was removed, False if not found
        """
        if name in self._strategies:
            del self._strategies[name]
            log_info(f"Unregistered HMM strategy: {name}")
            return True
        return False

    def clear(self) -> None:
        """Clear all registered strategies."""
        self._strategies.clear()
        log_info("Cleared all HMM strategies from registry")

    def load_from_config(self, config: Dict[str, Dict[str, Any]]) -> None:
        """
        Load strategies from configuration dictionary.

        Config format:
        {
            "strategy_name": {
                "enabled": bool,
                "weight": float,
                "class": "module.path.to.StrategyClass",
                "params": {...}
            }
        }

        Args:
            config: Configuration dictionary
        """
        for strategy_name, strategy_config in config.items():
            if not strategy_config.get("enabled", True):
                log_info(f"Skipping disabled strategy: {strategy_name}")
                continue

            try:
                # Import strategy class
                class_path = strategy_config["class"]
                module_path, class_name = class_path.rsplit(".", 1)

                module = importlib.import_module(module_path)
                strategy_class: Type[HMMStrategy] = getattr(module, class_name)

                # Create strategy instance
                weight = strategy_config.get("weight", 1.0)
                params = strategy_config.get("params", {})

                strategy = strategy_class(name=strategy_name, weight=weight, enabled=True, **params)

                # Register strategy
                self.register(strategy)

            except Exception as e:
                log_error(f"Failed to load strategy '{strategy_name}': {type(e).__name__}: {e}")
                continue

    def __len__(self) -> int:
        """Return number of registered strategies."""
        return len(self._strategies)

    def __contains__(self, name: str) -> bool:
        """Check if strategy is registered."""
        return name in self._strategies

    def __repr__(self) -> str:
        """String representation of registry."""
        return f"HMMStrategyRegistry({len(self._strategies)} strategies)"


# Global registry instance
_default_registry: Optional[HMMStrategyRegistry] = None


def get_default_registry() -> HMMStrategyRegistry:
    """
    Get or create the default global registry.

    Returns:
        Default HMMStrategyRegistry instance
    """
    global _default_registry
    if _default_registry is None:
        _default_registry = HMMStrategyRegistry()
    return _default_registry


__all__ = ["HMMStrategyRegistry", "get_default_registry"]
