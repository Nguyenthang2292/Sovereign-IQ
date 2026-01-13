"""
Signal Summary Utilities.

This module provides utilities for generating signal summary statistics.
"""

from typing import Any, Dict

import pandas as pd


def get_signal_summary(
    signals: pd.Series,
    signal_strength: pd.Series,
    close: pd.Series,
) -> Dict[str, Any]:
    """
    Generate summary statistics for signal strategy.

    Args:
        signals: Signal series (1 = LONG, -1 = SHORT, 0 = NEUTRAL)
        signal_strength: Signal strength series
        close: Close price series

    Returns:
        Dictionary with summary statistics
    """
    # Input validation
    if (
        not isinstance(signals, pd.Series)
        or not isinstance(signal_strength, pd.Series)
        or not isinstance(close, pd.Series)
    ):
        raise TypeError("All input parameters must be pandas Series")

    if len(signals) == 0:
        return {
            "total_signals": 0,
            "long_signals": 0,
            "short_signals": 0,
            "neutral_signals": 0,
            "avg_signal_strength": 0.0,
            "current_signal": 0,
            "current_strength": 0.0,
        }

    long_count = (signals == 1).sum()
    short_count = (signals == -1).sum()
    neutral_count = (signals == 0).sum()

    # Get current signal (last non-NaN value) - vectorized
    non_nan_signals = signals.dropna()
    if len(non_nan_signals) > 0:
        current_signal = int(non_nan_signals.iloc[-1])
        last_idx = non_nan_signals.index[-1]
        # Check if last_idx exists and value is not NaN
        if last_idx in signal_strength.index:
            strength_value = signal_strength.loc[last_idx]
            current_strength = float(strength_value) if not pd.isna(strength_value) else 0.0
        else:
            current_strength = 0.0
    else:
        current_signal = 0
        current_strength = 0.0

    # Calculate average strength for non-zero signals
    non_zero_signals = signals[signals != 0]
    avg_strength = 0.0
    if len(non_zero_signals) > 0:
        # Ensure index alignment before filtering
        aligned_strength = signal_strength.reindex(signals.index, fill_value=0.0)
        non_zero_strength = aligned_strength[signals != 0]
        avg_strength = non_zero_strength.mean() if len(non_zero_strength) > 0 else 0.0

    return {
        "total_signals": len(signals),
        "long_signals": int(long_count),
        "short_signals": int(short_count),
        "neutral_signals": int(neutral_count),
        "avg_signal_strength": float(avg_strength),
        "current_signal": current_signal,
        "current_strength": float(current_strength),
        "long_percentage": float(long_count / len(signals) * 100) if len(signals) > 0 else 0.0,
        "short_percentage": float(short_count / len(signals) * 100) if len(signals) > 0 else 0.0,
    }


__all__ = [
    "get_signal_summary",
]
