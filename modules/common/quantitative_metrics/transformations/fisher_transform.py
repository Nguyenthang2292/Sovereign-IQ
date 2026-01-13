"""
Fisher Transform indicator for technical analysis.

The Fisher Transform is a mathematical transformation that converts prices into
a Gaussian normal distribution, making it easier to identify turning points.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

# Try to import Numba for JIT compilation, fallback if not available
try:
    from numba import njit

    NUMBA_AVAILABLE = True

    @njit(cache=True)
    def _fisher_transform_core_jit(hl2: np.ndarray, high_: np.ndarray, low_: np.ndarray, n: int) -> np.ndarray:
        """
        Core Fisher Transform calculation using Numba JIT for performance.

        This function performs the recursive Fisher Transform calculation on numpy arrays.
        It's JIT-compiled for maximum performance.

        Args:
            hl2: Array of (high + low) / 2 values
            high_: Array of rolling max values
            low_: Array of rolling min values
            n: Length of arrays

        Returns:
            Array of Fisher Transform values
        """
        value = np.zeros(n, dtype=np.float64)
        fish1 = np.zeros(n, dtype=np.float64)

        for i in range(1, n):
            # Check for NaN or invalid values
            if np.isnan(high_[i]) or np.isnan(low_[i]) or np.isnan(hl2[i]):
                value[i] = value[i - 1] if i > 0 else 0.0
                fish1[i] = fish1[i - 1] if i > 0 else 0.0
                continue

            # Normalize
            if high_[i] == low_[i] or abs(high_[i] - low_[i]) < 1e-10:
                normalized = 0.0
            else:
                normalized = (hl2[i] - low_[i]) / (high_[i] - low_[i]) - 0.5

            # Update value with recursive smoothing
            prev_value = value[i - 1] if i > 0 and not np.isnan(value[i - 1]) else 0.0
            new_value = 0.66 * normalized + 0.67 * prev_value

            # Clamp to avoid infinite values
            if new_value > 0.99:
                new_value = 0.999
            elif new_value < -0.99:
                new_value = -0.999
            value[i] = new_value

            # Calculate Fisher Transform
            prev_fish = fish1[i - 1] if i > 0 and not np.isnan(fish1[i - 1]) else 0.0
            val_abs = abs(value[i])

            if val_abs >= 1.0 or not np.isfinite(value[i]):
                fish1[i] = prev_fish
            else:
                # Safe log calculation
                denominator = 1.0 - value[i]
                if abs(denominator) < 1e-10:
                    fish1[i] = prev_fish
                else:
                    ratio = (1.0 + value[i]) / denominator
                    if ratio > 0:
                        log_val = np.log(ratio)
                        fish1[i] = 0.5 * log_val + 0.5 * prev_fish
                    else:
                        fish1[i] = prev_fish

        return fish1
except ImportError:
    NUMBA_AVAILABLE = False
    _fisher_transform_core_jit = None


def calculate_fisher_transform(high: pd.Series, low: pd.Series, close: pd.Series, length: int = 9) -> pd.Series:
    """
    Calculate Fisher Transform applied to hl2 over length bars.

    The Fisher Transform is a mathematical transformation that converts prices into
    a Gaussian normal distribution, making it easier to identify turning points.

    Uses Numba JIT compilation for the core recursive calculation if available,
    falling back to pure Python if Numba is not installed.

    Args:
        high: High price series (pd.Series). Must not be None or empty.
        low: Low price series (pd.Series). Must not be None or empty.
        close: Close price series (pd.Series). Must not be None or empty.
        length: Period for Fisher Transform calculation (must be > 0). Default: 9

    Returns:
        Series with Fisher Transform values. Returns empty Series if input is invalid.

    Example:
        >>> high = pd.Series([100, 102, 101, 105, 106], dtype=float)
        >>> low = pd.Series([98, 99, 100, 101, 102], dtype=float)
        >>> close = pd.Series([99, 101, 100.5, 103, 104], dtype=float)
        >>> fisher = calculate_fisher_transform(high, low, close, length=3)
        >>> # Returns Fisher Transform values
    """
    # Validate inputs: None check
    if high is None or low is None or close is None:
        return pd.Series(dtype=float)

    # Validate inputs: type check
    if not isinstance(high, pd.Series) or not isinstance(low, pd.Series) or not isinstance(close, pd.Series):
        return pd.Series(dtype=float)

    # Validate inputs: empty series check
    if len(high) == 0 or len(low) == 0 or len(close) == 0:
        return pd.Series(dtype=float)

    # Validate length: must be positive
    if length <= 0:
        return pd.Series(dtype=float)

    hl2 = (high + low) / 2
    high_ = hl2.rolling(window=length, min_periods=1).max()
    low_ = hl2.rolling(window=length, min_periods=1).min()

    # Convert to numpy arrays for JIT-compiled function
    hl2_arr = hl2.values
    high_arr = high_.values
    low_arr = low_.values
    n = len(close)

    # Use JIT-compiled core function if Numba is available
    if NUMBA_AVAILABLE and _fisher_transform_core_jit is not None:
        fish1_arr = _fisher_transform_core_jit(hl2_arr, high_arr, low_arr, n)
    else:
        # Fallback to original implementation if Numba is not available
        value = np.zeros(n, dtype=np.float64)
        fish1_arr = np.zeros(n, dtype=np.float64)

        for i in range(1, n):
            if np.isnan(high_arr[i]) or np.isnan(low_arr[i]) or np.isnan(hl2_arr[i]):
                value[i] = value[i - 1] if i > 0 else 0.0
                fish1_arr[i] = fish1_arr[i - 1] if i > 0 else 0.0
                continue

            if high_arr[i] == low_arr[i] or abs(high_arr[i] - low_arr[i]) < 1e-10:
                normalized = 0.0
            else:
                normalized = (hl2_arr[i] - low_arr[i]) / (high_arr[i] - low_arr[i]) - 0.5

            prev_value = value[i - 1] if i > 0 and not np.isnan(value[i - 1]) else 0.0
            new_value = 0.66 * normalized + 0.67 * prev_value

            # Clamp
            if new_value > 0.99:
                new_value = 0.999
            elif new_value < -0.99:
                new_value = -0.999
            value[i] = new_value

            prev_fish = fish1_arr[i - 1] if i > 0 and not np.isnan(fish1_arr[i - 1]) else 0.0
            val_abs = abs(value[i])

            if val_abs >= 1.0 or not np.isfinite(value[i]):
                fish1_arr[i] = prev_fish
            else:
                denominator = 1.0 - value[i]
                if abs(denominator) < 1e-10:
                    fish1_arr[i] = prev_fish
                else:
                    ratio = (1.0 + value[i]) / denominator
                    if ratio > 0:
                        log_val = np.log(ratio)
                        fish1_arr[i] = 0.5 * log_val + 0.5 * prev_fish
                    else:
                        fish1_arr[i] = prev_fish

    # Convert back to pandas Series
    return pd.Series(fish1_arr, index=close.index)


__all__ = [
    "calculate_fisher_transform",
    "NUMBA_AVAILABLE",
]
