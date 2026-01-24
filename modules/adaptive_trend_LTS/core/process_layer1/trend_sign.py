"""Trend direction determination for Layer 1 processing.

This module provides the trend_sign function to determine trend direction
from signal series.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from modules.common.system import get_memory_manager
from modules.common.utils import log_error, log_warn


def trend_sign(signal: pd.Series, *, strategy: bool = False) -> pd.Series:
    """Determine trend direction from signal series.

    Numeric version (without colors) of Pine Script function:
        trendcol(signal) =>
            c = strategy ? (signal[1] > 0 ? colup : coldw)
                         : (signal > 0) ? colup : coldw

    Args:
        signal: Signal series.
        strategy: If True, uses signal[1] (previous bar) instead of current signal,
            matching Pine Script behavior.

    Returns:
        Series with trend direction values:
        - +1: Bullish trend (signal > 0)
        - -1: Bearish trend (signal < 0)
        - 0: Neutral (signal == 0)

    Raises:
        TypeError: If signal is not a pandas Series.
    """
    if not isinstance(signal, pd.Series):
        raise TypeError(f"signal must be a pandas Series, got {type(signal)}")  # pyright: ignore[reportUnreachable]

    if len(signal) == 0:
        log_warn("Empty signal series provided, returning empty series")
        return pd.Series(dtype="int8", index=signal.index)

    try:
        mem_manager = get_memory_manager()
        with mem_manager.track_memory("trend_sign"):
            base = signal.shift(1) if strategy else signal

            # Vectorize trend sign (Task 8.5)
            v = base.values
            res_vals = np.where(v > 0, 1, np.where(v < 0, -1, 0)).astype(np.int8)
            result = pd.Series(res_vals, index=signal.index, dtype="int8")

        return result

    except Exception as e:
        log_error(f"Error determining trend sign: {e}")
        raise
