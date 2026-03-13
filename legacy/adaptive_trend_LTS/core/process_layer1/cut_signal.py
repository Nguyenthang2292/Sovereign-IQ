"""Signal discretization for Layer 1 processing.

This module provides the cut_signal function to discretize continuous
signals into discrete values {-1, 0, 1}.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from modules.common.system import get_memory_manager
from modules.common.utils import log_error, log_warn


def cut_signal(
    x: pd.Series,
    threshold: float = 0.49,
    long_threshold: float | None = None,
    short_threshold: float | None = None,
    cutout: int = 0,
) -> pd.Series:
    """Discretize continuous signal into {-1, 0, 1} based on threshold.

    Port of Pine Script function:
        Cut(x) =>
            c = x > 0.49 ? 1 : x < -0.49 ? -1 : 0
            c

    Args:
        x: Continuous signal series.
        threshold: Threshold for discretization (default: 0.49).
            Used if long_threshold and short_threshold are not provided.
        long_threshold: Threshold for LONG signals (default: None, uses threshold).
            Values > long_threshold → 1.
        short_threshold: Threshold for SHORT signals (default: None, uses -threshold).
            Values < short_threshold → -1.
        cutout: Number of bars to skip at beginning (force to 0).
            Values > long_threshold → 1, values < short_threshold → -1, else → 0.

    Returns:
        Series with discrete values {-1, 0, 1}.

    Raises:
        ValueError: If threshold is invalid.
        TypeError: If x is not a pandas Series.
    """
    if not isinstance(x, pd.Series):
        raise TypeError(f"x must be a pandas Series, got {type(x)}")  # pyright: ignore[reportUnreachable]

    # Determine actual thresholds
    if long_threshold is None:
        long_threshold = threshold
    if short_threshold is None:
        short_threshold = -threshold

    if long_threshold <= short_threshold:
        raise ValueError(f"long_threshold ({long_threshold}) must be > short_threshold ({short_threshold})")

    if cutout < 0:
        raise ValueError(f"cutout must be >= 0, got {cutout}")

    if len(x) == 0:
        log_warn("Empty signal series provided, returning empty series")
        return pd.Series(dtype="int8", index=x.index)

    try:
        mem_manager = get_memory_manager()
        with mem_manager.track_memory("cut_signal"):
            # Vectorize discretization (Task 8.5)
            # c = x > long ? 1 : x < short ? -1 : 0
            v = x.values
            c_vals = np.select([v > long_threshold, v < short_threshold], [1, -1], default=0).astype(np.int8)

            c = pd.Series(c_vals, index=x.index, dtype="int8")

        # Enforce cutout: set first 'cutout' bars to 0
        if cutout > 0 and cutout < len(c):
            c.iloc[:cutout] = 0
            # Also ensure NaN handling aligns if needed, though int8 has no NaN

        # Check for excessive NaN values
        nan_count = x.isna().sum()
        if nan_count > 0:
            nan_pct = (nan_count / len(x)) * 100
            if nan_pct > 10:  # Warn if more than 10% NaN
                log_warn(
                    f"Input signal contains {nan_count} NaN values ({nan_pct:.1f}%). "
                    f"These will be treated as 0 (no signal)."
                )

        return c

    except Exception as e:
        log_error(f"Error discretizing signal: {e}")
        raise
