"""Multi-timeframe wrapper for IncrementalATC."""

from __future__ import annotations

import pandas as pd
from typing import Dict, Any, Union, Optional

try:
    from modules.common.utils import log_debug, log_warn
except ImportError:

    def log_debug(msg: str) -> None:
        print(f"[DEBUG] {msg}")

    def log_warn(msg: str) -> None:
        print(f"[WARN] {msg}")


from .constants import TF_RESOLUTION_MAP
from .core import IncrementalATC


class MultiTimeframeIncrementalATC:
    """Multi-timeframe wrapper for IncrementalATC.

    Maintains one IncrementalATC instance per timeframe and synchronizes
    updates from a base timeframe (e.g., 1m) to higher timeframes only when
    their bars complete.

    Usage:
        mtf = MultiTimeframeIncrementalATC(config, timeframes=["1m", "5m", "15m"])
        mtf.initialize({"1m": prices_1m, "5m": prices_5m, "15m": prices_15m})
        signals = mtf.update(new_price, timeframe="1m")
    """

    def __init__(self, config: Dict[str, Any], timeframes: list[str] | None = None):
        """Initialize multi-timeframe ATC.

        Args:
            config: ATC configuration parameters (same as IncrementalATC)
            timeframes: List of timeframe strings, ordered from base to highest
                       (default: ["1m", "5m", "15m"])
        """
        if timeframes is None:
            timeframes = ["1m", "5m", "15m"]

        self.config = config
        self.timeframes = timeframes
        self.base_tf = timeframes[0]
        self.higher_tfs = timeframes[1:]

        log_debug(f"Initializing MTF with timeframes: {self.timeframes}, base: {self.base_tf}")

        # Create one IncrementalATC per timeframe
        self.atcs = {tf: IncrementalATC(config) for tf in timeframes}

        # Bar counters for each timeframe
        self.bar_counters = {tf: 0 for tf in timeframes}

        # Track last bar prices for each higher timeframe
        self.last_bar_prices: Dict[str, float | None] = {tf: None for tf in self.higher_tfs}

    def _bars_per_tf(self, tf: str) -> int:
        """Get number of minutes in a timeframe bar."""
        return TF_RESOLUTION_MAP.get(tf, 1)

    def _is_bar_completed(self, base_bar_index: int, target_tf: str) -> bool:
        """Check if a higher timeframe bar has completed.

        Args:
            base_bar_index: Current bar index in base timeframe
            target_tf: Target timeframe to check

        Returns:
            True if the target timeframe bar has completed
        """
        base_minutes = self._bars_per_tf(self.base_tf)
        target_minutes = self._bars_per_tf(target_tf)

        if base_minutes == 0 or target_minutes == 0:
            return False

        ratio = target_minutes // base_minutes
        if ratio == 0:
            return False

        return (base_bar_index + 1) % ratio == 0

    def initialize(self, historical_data):
        """Initialize all timeframe ATCs with historical data.

        Args:
            historical_data: Either:
                - Dict mapping timeframe to price series: {"1m": prices_1m, "5m": prices_5m}
                - Single price series (will be used for all timeframes)

        Returns:
            Dict of initialization results per timeframe
        """
        log_debug("Initializing MTF ATC with historical data")

        results = {}

        if isinstance(historical_data, dict):
            for tf, prices in historical_data.items():
                if tf in self.atcs:
                    log_debug(f"Initializing {tf} ATC with {len(prices)} bars")
                    results[tf] = self.atcs[tf].initialize(prices)
        else:
            log_debug(f"Single dataset provided, initializing all TFs with {len(historical_data)} bars")
            for tf in self.timeframes:
                results[tf] = self.atcs[tf].initialize(historical_data)

        # Initialize last bar prices for higher TFs with last prices from data
        for tf in self.higher_tfs:
            if isinstance(historical_data, dict) and tf in historical_data:
                self.last_bar_prices[tf] = historical_data[tf].iloc[-1]
            elif isinstance(historical_data, pd.Series):
                self.last_bar_prices[tf] = historical_data.iloc[-1]

        log_debug("MTF initialization complete")
        return results

    def update(self, new_price: float, timeframe: str | None = None) -> Dict[str, float]:
        """Update ATC signals across all timeframes.

        Args:
            new_price: New price value
            timeframe: Timeframe of the update (default: base_tf)

        Returns:
            Dict of signal values per timeframe
        """
        if timeframe is None:
            timeframe = self.base_tf

        signals = {}

        if timeframe == self.base_tf:
            signals = self._update_from_base(new_price)
        else:
            log_warn(f"Direct updates to non-base timeframe {timeframe} not supported")
            return {tf: self.atcs[tf].state.get("signal", 0.0) for tf in self.timeframes}

        return signals

    def _update_from_base(self, new_price: float) -> Dict[str, float]:
        """Update from base timeframe and sync to higher timeframes.

        Args:
            new_price: New price from base timeframe

        Returns:
            Dict of signal values per timeframe
        """
        signals = {}

        # Update base TF
        base_bar_index = self.bar_counters[self.base_tf]
        base_signal = self.atcs[self.base_tf].update(new_price)
        self.bar_counters[self.base_tf] += 1
        signals[self.base_tf] = base_signal

        # Store last last bar price for higher TFs
        for tf in self.higher_tfs:
            self.last_bar_prices[tf] = new_price

        # Check which higher TFs have completed bars
        for tf in self.higher_tfs:
            if self._is_bar_completed(base_bar_index, tf):
                log_debug(f"Bar completed for {tf} at base bar {base_bar_index}")

                # Push the last closed bar price to higher TF
                bar_price = self.last_bar_prices[tf]
                if bar_price is not None:
                    self.bar_counters[tf] += 1
                    tf_signal = self.atcs[tf].update(bar_price)
                    signals[tf] = tf_signal
                    log_debug(f"{tf} updated with signal {tf_signal}")
            else:
                # Return last known signal for this TF
                signals[tf] = self.atcs[tf].state.get("signal", 0.0)

        log_debug(f"MTF signals: {signals}")
        return signals

    def reset(self):
        """Reset all timeframe ATCs."""
        log_debug("Resetting MTF ATC")
        for tf in self.timeframes:
            self.atcs[tf].reset()
            self.bar_counters[tf] = 0
            self.last_bar_prices[tf] = None

    def get_state(self, tf: str | None = None) -> Dict:
        """Get state for specific timeframe or all timeframes.

        Args:
            tf: Specific timeframe to get state for (default: all)

        Returns:
            State dictionary for requested timeframe(s)
        """
        if tf is not None:
            return self.atcs[tf].state
        return {tf: self.atcs[tf].state for tf in self.timeframes}

    def get_signal(self, tf: str | None = None) -> float | Dict[str, float]:
        """Get current signal for specific timeframe or all timeframes.

        Args:
            tf: Specific timeframe to get signal for (default: all)

        Returns:
            Signal value or dict of signals per timeframe
        """
        if tf is not None:
            return self.atcs[tf].state.get("signal", 0.0)
        return {tf: self.atcs[tf].state.get("signal", 0.0) for tf in self.timeframes}
