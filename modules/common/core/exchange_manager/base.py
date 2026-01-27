"""
Base infrastructure for exchange management.

This module provides the core infrastructure components used by both
AuthenticatedExchangeManager and PublicExchangeManager:
- ExchangeWrapper: Thread-safe reference counting for exchange instances
- Configuration fallbacks: Default values when config modules are unavailable
- Shared utilities: Common functions used across managers
"""

import logging
import os
import threading
from typing import Optional

import ccxt

logger = logging.getLogger(__name__)

# Import normalize_symbol from utils (core module, should always be available)
from modules.common.domain import normalize_symbol

# Configuration imports with fallbacks
try:
    from config import (
        DEFAULT_CONTRACT_TYPE,
        DEFAULT_EXCHANGE_STRING,
        DEFAULT_REQUEST_PAUSE,
    )
    from config.config_api import (
        get_binance_api_key,
        get_binance_api_secret,
    )
except ImportError:
    # Fallback configuration values when config modules are not available
    DEFAULT_EXCHANGE_STRING = "binance,kraken,kucoin,gate,okx,bybit,mexc,huobi"
    DEFAULT_REQUEST_PAUSE = 0.2
    DEFAULT_CONTRACT_TYPE = "future"
    DEFAULT_EXCHANGES = [
        "binance",
        "kraken",
        "kucoin",
        "gate",
        "okx",
        "bybit",
        "mexc",
        "huobi",
    ]

    # Fallback getter functions if config_api is not available
    def get_binance_api_key():
        return None

    def get_binance_api_secret():
        return None


class ExchangeWrapper:
    """
    Wrapper class to track exchange instance and reference count.
    Ensures thread-safe reference counting for exchange cleanup.
    """

    def __init__(self, exchange: ccxt.Exchange):
        """
        Initialize ExchangeWrapper with an exchange instance.

        Args:
            exchange: The ccxt.Exchange instance to wrap
        """
        self.exchange = exchange
        self._refcount = 0
        self._refcount_lock = threading.Lock()

    def increment_refcount(self) -> int:
        """
        Increment reference count atomically.

        Returns:
            New reference count after increment
        """
        with self._refcount_lock:
            self._refcount += 1
            return self._refcount

    def decrement_refcount(self) -> int:
        """
        Decrement reference count atomically.

        Returns:
            New reference count after decrement
        """
        with self._refcount_lock:
            if self._refcount > 0:
                self._refcount -= 1
            return self._refcount

    def get_refcount(self) -> int:
        """
        Get current reference count.

        Returns:
            Current reference count
        """
        with self._refcount_lock:
            return self._refcount

    def is_in_use(self) -> bool:
        """
        Check if exchange is currently in use.

        Returns:
            True if refcount > 0, False otherwise
        """
        with self._refcount_lock:
            return self._refcount > 0
