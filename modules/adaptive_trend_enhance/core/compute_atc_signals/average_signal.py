"""Average signal calculation for ATC final output.

This module provides the function to calculate the final Average_Signal by
weighting Layer 1 signals with Layer 2 equity curves.
"""

from __future__ import annotations

from typing import Dict

import numpy as np
import pandas as pd

try:
    from modules.common.utils import log_debug, log_warn
except ImportError:
    # Fallback logging if common utils not available
    def log_debug(msg: str) -> None:  # pragma: no cover
        print(f"[DEBUG] {msg}")

    def log_warn(msg: str) -> None:  # pragma: no cover
        print(f"[WARN] {msg}")


def calculate_average_signal(
    layer1_signals: Dict[str, pd.Series],
    layer2_equities: Dict[str, pd.Series],
    ma_configs: list,
    prices: pd.Series,
    long_threshold: float,
    short_threshold: float,
    cutout: int,
    strategy_mode: bool = False,
) -> pd.Series:
    """Calculate the final Average_Signal from Layer 1 signals and Layer 2 equities.

    This function performs vectorized computation of the weighted average signal
    where each Layer 1 signal is weighted by its corresponding Layer 2 equity curve.

    Args:
        layer1_signals: Dictionary of Layer 1 signals keyed by MA type.
        layer2_equities: Dictionary of Layer 2 equity curves keyed by MA type.
        ma_configs: List of (ma_type, length, initial_weight) tuples.
        prices: Price series (for index reference).
        long_threshold: Threshold for LONG signals.
        short_threshold: Threshold for SHORT signals.
        cutout: Number of bars to skip at beginning.
        strategy_mode: If True, shift signal by 1 bar (default: False).

    Returns:
        Series containing the Average_Signal.
    """
    log_debug("Computing Average_Signal (vectorized)...")

    n_bars = len(prices)
    index = prices.index

    # Initialize accumulators
    nom_array = np.zeros(n_bars, dtype=np.float64)
    den_array = np.zeros(n_bars, dtype=np.float64)

    # Pre-calculated thresholds for speed
    # Note: Using strict inequality (> and <) as per original Pine Script logic

    for ma_type, _, _ in ma_configs:
        # Get underlying numpy arrays (faster than Series access)
        signal_vals = layer1_signals[ma_type].values
        equity_vals = layer2_equities[ma_type].values

        # Vectorized cut_signal logic (inlined for performance)
        # Equivalent to: c = x > long ? 1 : x < short ? -1 : 0
        # NaNs will naturally result in 0 (False for both comparisons)

        # Determine signals
        # Use np.select or nested np.where. Nested np.where is often faster for 2 conditions.
        # c = 1 where sig > L, -1 where sig < S, else 0

        # We handle NaN implicitly: NaN > L is False, NaN < S is False -> 0
        with np.errstate(invalid="ignore"):
            cut_vals = np.where(signal_vals > long_threshold, 1.0, np.where(signal_vals < short_threshold, -1.0, 0.0))

        # Apply cutout
        if cutout > 0 and cutout < n_bars:
            cut_vals[:cutout] = 0.0

        # Accumulate
        # nom += signal * equity
        # den += equity
        nom_array += cut_vals * equity_vals
        den_array += equity_vals

    # Calculate final average
    with np.errstate(divide="ignore", invalid="ignore"):
        avg_signal_array = np.divide(nom_array, den_array)
        # Handle division by zero or NaN results
        avg_signal_array = np.where(np.isfinite(avg_signal_array), avg_signal_array, 0.0)

    Average_Signal = pd.Series(avg_signal_array, index=index, dtype="float64")

    if strategy_mode:
        Average_Signal = Average_Signal.shift(1).fillna(0)

    # Optional: Log division by zero stats if needed, but it's handled above

    log_debug("Completed Average_Signal")
    return Average_Signal
