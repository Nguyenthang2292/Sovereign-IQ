"""Streaming processor for live trading with incremental ATC updates.

This module wraps BatchIncrementalATC to provide a streaming interface
for live trading scenarios, maintaining local state without distributed complexity.
"""

from typing import Dict, List, Optional

from modules.adaptive_trend_LTS.core.compute_atc_signals.batch_incremental_atc import BatchIncrementalATC


class StreamingIncrementalProcessor:
    """Streaming processor for live trading with incremental ATC updates.

    Wraps BatchIncrementalATC to provide a convenient interface for
    streaming live price bars and managing state locally.
    """

    def __init__(self, config: Dict):
        """Initialize streaming processor with configuration.

        Args:
            config: ATC configuration parameters (same as compute_atc_signals)
        """
        self.config = config
        self.batch_atc = BatchIncrementalATC(config)
        self.processed_count = 0

    def initialize_symbol(self, symbol: str, prices):
        """Initialize a symbol with historical data.

        Args:
            symbol: Symbol identifier (e.g., "BTCUSDT")
            prices: Historical price series for initialization
        """
        self.batch_atc.add_symbol(symbol, prices)

    def process_live_bar(self, symbol: str, price: float, timestamp: Optional[float] = None) -> Optional[float]:
        """Process a single live bar for a symbol.

        This is the main interface for streaming live trading data.

        Args:
            symbol: Symbol identifier
            price: New price value
            timestamp: Optional timestamp for the bar (not used in calculation)

        Returns:
            Updated signal value, or None if symbol not found
        """
        signal = self.batch_atc.update_symbol(symbol, price)
        if signal is not None:
            self.processed_count += 1
        return signal

    def process_live_bars(self, price_updates: Dict[str, float]) -> Dict[str, float]:
        """Process multiple live bars in batch.

        Args:
            price_updates: Dictionary mapping symbol to new price

        Returns:
            Dictionary mapping symbol to updated signal
        """
        signals = self.batch_atc.update_all(price_updates)
        self.processed_count += len(price_updates)
        return signals

    def get_signal(self, symbol: str) -> Optional[float]:
        """Get current signal for a specific symbol.

        Args:
            symbol: Symbol identifier

        Returns:
            Current signal value, or None if symbol not found
        """
        return self.batch_atc.get_symbol_signal(symbol)

    def get_all_signals(self) -> Dict[str, float]:
        """Get current signals for all symbols.

        Returns:
            Dictionary mapping symbol to current signal value
        """
        return self.batch_atc.get_all_signals()

    def remove_symbol(self, symbol: str) -> bool:
        """Remove a symbol from the stream.

        Args:
            symbol: Symbol identifier

        Returns:
            True if symbol was removed, False if not found
        """
        return self.batch_atc.remove_symbol(symbol)

    def reset_symbol(self, symbol: str) -> bool:
        """Reset state for a specific symbol.

        Args:
            symbol: Symbol identifier

        Returns:
            True if symbol was reset, False if not found
        """
        return self.batch_atc.reset_symbol(symbol)

    def reset_all(self) -> None:
        """Reset state for all symbols."""
        self.batch_atc.reset_all()

    def get_symbol_state(self, symbol: str) -> Optional[Dict]:
        """Get full state for a specific symbol.

        Args:
            symbol: Symbol identifier

        Returns:
            State dictionary, or None if symbol not found
        """
        return self.batch_atc.get_symbol_state(symbol)

    def get_all_states(self) -> Dict[str, Dict]:
        """Get full states for all symbols.

        Returns:
            Dictionary mapping symbol to state dictionary
        """
        return self.batch_atc.get_all_states()

    def get_symbol_count(self) -> int:
        """Get the number of symbols being tracked.

        Returns:
            Number of symbols
        """
        return self.batch_atc.get_symbol_count()

    def get_symbols(self) -> List[str]:
        """Get list of all symbols being tracked.

        Returns:
            List of symbol identifiers
        """
        return self.batch_atc.get_symbols()

    def get_processed_count(self) -> int:
        """Get total number of bars processed.

        Returns:
            Number of live bars processed since creation
        """
        return self.processed_count

    def get_state(self) -> Dict:
        """Get overall processor state.

        Returns:
            Dictionary with processor state information
        """
        return {
            "symbol_count": self.get_symbol_count(),
            "processed_count": self.processed_count,
            "symbols": self.get_symbols(),
        }
