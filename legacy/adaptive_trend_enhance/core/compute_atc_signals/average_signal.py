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
    strategy_mode: bool = False,
    precision: str = "float64",
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

    index = prices.index

    # Initialize accumulators (Task 8.5: Vectorized version)
    dtype = np.float32 if precision == "float32" else np.float64

    # Filter valid configs
    valid_configs = [
        (ma, length, weight) for ma, length, weight in ma_configs if ma in layer1_signals and ma in layer2_equities
    ]

    if not valid_configs:
        return pd.Series(0.0, index=index, dtype=dtype)

    # Convert to 2D matrices for broadcasting (Task 8.5)
    # Shape: (n_mas, n_bars)
    S = np.stack([layer1_signals[m].values for m, _, _ in valid_configs])
    E = np.stack([layer2_equities[m].values for m, _, _ in valid_configs])

    # Vectorized discretization (Task 8.5)
    with np.errstate(invalid="ignore"):
        C = np.where(S > long_threshold, 1.0, np.where(S < short_threshold, -1.0, 0.0))

    # Parallel calculation across components
    # nom = sum(signal_discrete * equity_weight), den = sum(equity_weight)
    nom_array = np.sum(C * E, axis=0)
    den_array = np.sum(E, axis=0)

    # Calculate final average
    with np.errstate(divide="ignore", invalid="ignore"):
        avg_signal_array = np.divide(nom_array, den_array)
        # Handle division by zero or NaN results
        avg_signal_array = np.where(np.isfinite(avg_signal_array), avg_signal_array, 0.0)

    Average_Signal = pd.Series(avg_signal_array, index=index, dtype=dtype)

    if strategy_mode:
        Average_Signal = Average_Signal.shift(1).fillna(0)

    # Optional: Log division by zero stats if needed, but it's handled above

    log_debug("Completed Average_Signal")
    return Average_Signal
