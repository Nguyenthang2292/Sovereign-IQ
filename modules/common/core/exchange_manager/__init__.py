"""
Exchange Manager Module - Modular Architecture

This module provides a refactored, modular architecture for managing cryptocurrency exchange connections.
The original monolithic exchange_manager.py has been decomposed into specialized components:

Architecture:
    - base.py: ExchangeWrapper - Core exchange connection lifecycle management
    - connection_factory.py: Factory methods for creating and configuring exchange instances
    - authenticated.py: AuthenticatedExchangeManager - Handles authenticated operations
    - public.py: PublicExchangeManager - Handles public (read-only) operations
    - __init__.py (this file): ExchangeManager - Composite facade maintaining backward compatibility

Design Principles:
    1. Single Responsibility: Each class has a focused purpose
    2. Separation of Concerns: Authenticated vs public operations are isolated
    3. Backward Compatibility: ExchangeManager provides the same public API as before
    4. Thread Safety: All managers are thread-safe using locks
    5. Resource Management: Proper cleanup and connection lifecycle management

Usage:
    # Unified interface (backward compatible)
    manager = ExchangeManager(api_key="...", api_secret="...")

    # Direct access to specialized managers
    manager.authenticated.get_exchange('binance')  # Authenticated operations
    manager.public.get_exchange('kraken')          # Public operations

    # Or use sub-managers independently
    auth_manager = AuthenticatedExchangeManager(api_key="...", api_secret="...")
    public_manager = PublicExchangeManager()

Exported Classes:
    - ExchangeWrapper: Low-level exchange connection wrapper
    - AuthenticatedExchangeManager: Manager for authenticated exchange operations
    - PublicExchangeManager: Manager for public exchange operations
    - ExchangeManager: Composite manager combining authenticated and public managers
"""

import logging
import os
from typing import Optional

from modules.common.domain import normalize_symbol

# Import config with fallbacks
try:
    from config.config_api import get_binance_api_key, get_binance_api_secret
except ImportError:
    # Fallback getter functions if config_api is not available
    def get_binance_api_key():
        return None

    def get_binance_api_secret():
        return None

# Import modular components
from .base import ExchangeWrapper
from .authenticated import AuthenticatedExchangeManager
from .public import PublicExchangeManager

logger = logging.getLogger(__name__)


class ExchangeManager:
    """
    Composite manager that combines AuthenticatedExchangeManager and PublicExchangeManager.

    This class provides a unified interface while maintaining clear separation
    between authenticated and public exchange operations. It serves as a facade
    that delegates operations to specialized sub-managers.

    The ExchangeManager maintains backward compatibility with the original monolithic
    implementation while providing access to the new modular architecture.

    Attributes:
        authenticated (AuthenticatedExchangeManager): Manager for authenticated operations
        public (PublicExchangeManager): Manager for public operations
        api_key (str): Binance API key (for backward compatibility)
        api_secret (str): Binance API secret (for backward compatibility)
        testnet (bool): Testnet flag (for backward compatibility)

    Example:
        >>> # Basic usage (backward compatible)
        >>> manager = ExchangeManager()
        >>> exchange = manager.public.get_exchange('binance')
        >>>
        >>> # With authentication
        >>> manager = ExchangeManager(api_key="...", api_secret="...")
        >>> auth_exchange = manager.authenticated.get_exchange('binance')
        >>>
        >>> # Access specialized managers directly
        >>> manager.authenticated.cleanup_unused_exchanges()
        >>> manager.public.get_exchange_with_fallback('BTC/USDT', '1h')
    """

    def __init__(self, api_key=None, api_secret=None, testnet=False):
        """
        Initialize ExchangeManager with optional credentials.

        Creates both authenticated and public sub-managers. The authenticated manager
        will only be functional if valid credentials are provided.

        Args:
            api_key (str, optional): Binance API key. Can be None to use environment
                variables or config. Defaults to None.
            api_secret (str, optional): Binance API secret. Can be None to use environment
                variables or config. Defaults to None.
            testnet (bool, optional): Use Binance testnet if True. Defaults to False.

        Note:
            If api_key and api_secret are not provided, the manager will attempt to
            retrieve them from:
            1. Environment variables: BINANCE_API_KEY, BINANCE_API_SECRET
            2. Configuration system: get_binance_api_key(), get_binance_api_secret()
        """
        # Initialize sub-managers
        self.authenticated = AuthenticatedExchangeManager(api_key, api_secret, testnet)
        self.public = PublicExchangeManager()

        # Store credentials for backward compatibility
        # Use thread-safe getter functions for API keys
        self.api_key = api_key or os.getenv("BINANCE_API_KEY") or get_binance_api_key()
        self.api_secret = api_secret or os.getenv("BINANCE_API_SECRET") or get_binance_api_secret()
        self.testnet = testnet

    def normalize_symbol(self, market_symbol: str) -> str:
        """
        Normalize market symbol by converting exchange-specific formats to standard format.

        This method handles the conversion of exchange-specific symbol formats (e.g.,
        Binance futures symbols like BTC/USDT:USDT) to a standardized format (BTC/USDT).
        It first removes contract markers (e.g., ':USDT') and then applies the
        normalize_symbol utility function.

        Args:
            market_symbol (str): Market symbol from exchange. Examples:
                - 'BTC/USDT:USDT' (Binance futures)
                - 'ETHUSDT' (Binance spot without separator)
                - 'BTC/USDT' (standard format)

        Returns:
            str: Normalized symbol in format 'BASE/QUOTE' (e.g., 'BTC/USDT')

        Example:
            >>> manager = ExchangeManager()
            >>> manager.normalize_symbol('BTC/USDT:USDT')
            'BTC/USDT'
            >>> manager.normalize_symbol('ETHUSDT')
            'ETH/USDT'
        """
        if ":" in market_symbol:
            market_symbol = market_symbol.split(":")[0]
        return normalize_symbol(market_symbol)

    @property
    def exchange_priority_for_fallback(self):
        """
        Get list of exchange IDs in priority order for OHLCV fallback.

        This property delegates to the public manager's exchange priority configuration.
        The fallback mechanism is used when primary exchanges fail to provide data.

        Returns:
            list[str]: List of exchange IDs in priority order

        Example:
            >>> manager = ExchangeManager()
            >>> priority = manager.exchange_priority_for_fallback
            >>> print(priority)
            ['binance', 'kraken', 'kucoin', ...]
        """
        return self.public.exchange_priority_for_fallback

    @exchange_priority_for_fallback.setter
    def exchange_priority_for_fallback(self, value):
        """
        Set list of exchange IDs in priority order for OHLCV fallback.

        This property setter delegates to the public manager's exchange priority configuration.

        Args:
            value (list[str]): List of exchange IDs in desired priority order

        Example:
            >>> manager = ExchangeManager()
            >>> manager.exchange_priority_for_fallback = ['kraken', 'binance', 'kucoin']
        """
        self.public.exchange_priority_for_fallback = value

    def cleanup_unused_exchanges(self, max_age_hours: Optional[float] = None):
        """
        Cleanup unused exchange connections to free memory.

        This method delegates cleanup operations to both authenticated and public managers.
        It helps prevent memory leaks by removing idle connections.

        For authenticated exchanges, the max_age_hours parameter controls which connections
        are eligible for cleanup based on their last access time. For public exchanges,
        all unused connections are cleaned up (age tracking not yet implemented).

        Args:
            max_age_hours (float, optional): Maximum age in hours for authenticated exchanges.
                - If None: Cleans up all unused exchanges in both managers
                - If specified: Only removes authenticated exchanges older than max_age_hours
                Public exchanges are always cleaned up regardless of age.

        Example:
            >>> manager = ExchangeManager()
            >>> # Cleanup all unused exchanges
            >>> manager.cleanup_unused_exchanges()
            >>>
            >>> # Cleanup authenticated exchanges older than 1 hour
            >>> manager.cleanup_unused_exchanges(max_age_hours=1.0)

        Note:
            This method is thread-safe and can be called from multiple threads.
            It's recommended to call this periodically in long-running applications.
        """
        self.authenticated.cleanup_unused_exchanges(max_age_hours)
        self.public.cleanup_unused_exchanges(max_age_hours)

    def close_exchange(self, exchange_id: str, testnet: bool = False, contract_type: str = None):
        """
        Close and remove a specific exchange connection (both authenticated and public).

        This method attempts to close the exchange connection with the given ID for both
        authenticated and public managers. It ensures proper cleanup of resources including
        closing websocket connections and releasing memory.

        Args:
            exchange_id (str): Exchange identifier (e.g., 'binance', 'kraken', 'kucoin')
            testnet (bool, optional): Testnet flag. Only applies to authenticated exchanges.
                Defaults to False.
            contract_type (str, optional): Contract type (e.g., 'future', 'swap').
                Only applies to authenticated exchanges. Defaults to None.

        Example:
            >>> manager = ExchangeManager()
            >>> # Close public connection
            >>> manager.close_exchange('binance')
            >>>
            >>> # Close authenticated testnet connection
            >>> manager.close_exchange('binance', testnet=True)
            >>>
            >>> # Close authenticated futures connection
            >>> manager.close_exchange('binance', contract_type='future')

        Note:
            - If the exchange doesn't exist in a manager, that manager silently ignores the request
            - This method is thread-safe
            - After closing, the exchange can be re-initialized by calling get_exchange() again
        """
        self.authenticated.close_exchange(exchange_id, testnet, contract_type)
        self.public.close_exchange(exchange_id)


# Export all public classes
__all__ = [
    "ExchangeWrapper",
    "AuthenticatedExchangeManager",
    "PublicExchangeManager",
    "ExchangeManager",
]
