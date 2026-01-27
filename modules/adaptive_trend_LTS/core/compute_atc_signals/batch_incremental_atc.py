"""Batch incremental ATC computation for managing multiple symbols.

This module provides BatchIncrementalATC class that manages multiple
IncrementalATC instances for efficient multi-symbol live trading.
"""

from __future__ import annotations

from typing import Dict, Optional, Any

import numpy as np
import pandas as pd

try:
    from modules.common.utils import log_debug, log_info, log_warn, log_error
except ImportError:

    def log_debug(msg: str) -> None:
        print(f"[DEBUG] {msg}")

    def log_info(msg: str) -> None:
        print(f"[INFO] {msg}")

    def log_warn(msg: str) -> None:
        print(f"[WARN] {msg}")

    def log_error(msg: str) -> None:
        print(f"[ERROR] {msg}")


class BatchIncrementalATC:
    """Manages multiple IncrementalATC instances for batch processing.

    Usage:
        batch_atc = BatchIncrementalATC(config)
        batch_atc.add_symbol("BTCUSDT", btc_prices)
        batch_atc.add_symbol("ETHUSDT", eth_prices)
        batch_atc.initialize_all()
        signals = batch_atc.update_all({"BTCUSDT": 50000.0, "ETHUSDT": 3000.0})
    """

    def __init__(self, config: Dict[str, Any]):
        """Initialize batch incremental ATC with configuration.

        Args:
            config: ATC configuration parameters (same as compute_atc_signals)
        """
        self.config = config
        self.instances: Dict[str, "IncrementalATC"] = {}
        self.initialized: bool = False

    def add_symbol(self, symbol: str, prices: pd.Series) -> None:
        """Add a symbol to the batch with historical data.

        Args:
            symbol: Symbol identifier (e.g., "BTCUSDT")
            prices: Historical price series for initialization
        """
        from .incremental_atc import IncrementalATC

        if symbol in self.instances:
            log_warn(f"Symbol {symbol} already exists. Replacing.")
            del self.instances[symbol]

        atc = IncrementalATC(self.config)
        atc.initialize(prices)
        self.instances[symbol] = atc
        log_debug(f"Added symbol {symbol} to batch")

    def update_symbol(self, symbol: str, new_price: float) -> Optional[float]:
        """Update a single symbol with new price.

        Args:
            symbol: Symbol identifier
            new_price: New price value

        Returns:
            Updated signal value, or None if symbol not found
        """
        if symbol not in self.instances:
            log_warn(f"Symbol {symbol} not found in batch")
            return None

        try:
            signal = self.instances[symbol].update(new_price)
            log_debug(f"Updated {symbol} with price {new_price}, signal={signal}")
            return signal
        except Exception as e:
            log_error(f"Error updating {symbol}: {e}")
            return None

    def update_all(self, price_updates: Dict[str, float]) -> Dict[str, float]:
        """Update all symbols with new prices.

        Args:
            price_updates: Dictionary mapping symbol to new price

        Returns:
            Dictionary mapping symbol to updated signal
        """
        signals = {}
        for symbol, price in price_updates.items():
            signal = self.update_symbol(symbol, price)
            if signal is not None:
                signals[symbol] = signal
        return signals

    def get_all_signals(self) -> Dict[str, float]:
        """Get current signals for all symbols.

        Returns:
            Dictionary mapping symbol to current signal value
        """
        return {symbol: atc.state.get("signal", 0.0) for symbol, atc in self.instances.items()}

    def get_symbol_signal(self, symbol: str) -> Optional[float]:
        """Get current signal for a specific symbol.

        Args:
            symbol: Symbol identifier

        Returns:
            Current signal value, or None if symbol not found
        """
        if symbol not in self.instances:
            return None
        return self.instances[symbol].state.get("signal", 0.0)

    def remove_symbol(self, symbol: str) -> bool:
        """Remove a symbol from the batch.

        Args:
            symbol: Symbol identifier

        Returns:
            True if symbol was removed, False if not found
        """
        if symbol in self.instances:
            del self.instances[symbol]
            log_debug(f"Removed symbol {symbol} from batch")
            return True
        log_warn(f"Symbol {symbol} not found in batch")
        return False

    def reset_symbol(self, symbol: str) -> bool:
        """Reset state for a specific symbol.

        Args:
            symbol: Symbol identifier

        Returns:
            True if symbol was reset, False if not found
        """
        if symbol in self.instances:
            self.instances[symbol].reset()
            log_debug(f"Reset symbol {symbol}")
            return True
        log_warn(f"Symbol {symbol} not found in batch")
        return False

    def reset_all(self) -> None:
        """Reset state for all symbols."""
        for symbol in self.instances:
            self.instances[symbol].reset()
        log_debug("Reset all symbols in batch")

    def get_symbol_state(self, symbol: str) -> Optional[Dict[str, Any]]:
        """Get full state for a specific symbol.

        Args:
            symbol: Symbol identifier

        Returns:
            State dictionary, or None if symbol not found
        """
        if symbol not in self.instances:
            return None
        return self.instances[symbol].state.copy()

    def get_all_states(self) -> Dict[str, Dict[str, Any]]:
        """Get full states for all symbols.

        Returns:
            Dictionary mapping symbol to state dictionary
        """
        return {symbol: atc.state.copy() for symbol, atc in self.instances.items()}

    def get_symbol_count(self) -> int:
        """Get the number of symbols in the batch.

        Returns:
            Number of symbols
        """
        return len(self.instances)

    def get_symbols(self) -> list[str]:
        """Get list of all symbols in the batch.

        Returns:
            List of symbol identifiers
        """
        return list(self.instances.keys())
