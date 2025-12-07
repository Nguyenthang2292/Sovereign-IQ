"""Utility functions for Adaptive Trend Classification (ATC).

This module provides core utility functions used throughout the ATC system:
- rate_of_change: Calculate percentage price change
- diflen: Calculate length offsets for Moving Averages based on robustness
- exp_growth: Calculate exponential growth factor over time
"""

from __future__ import annotations

from typing import Optional, Tuple

import numpy as np
import pandas as pd

from modules.common.utils import log_warn, log_error


def rate_of_change(prices: pd.Series) -> pd.Series:
    """Calculate percentage rate of change for price series.

    Equivalent to Pine Script global variable:
        R = (close - close[1]) / close[1]

    Args:
        prices: Price series (typically close prices).

    Returns:
        Series containing percentage change values. First value will be NaN.

    Raises:
        ValueError: If prices is empty.
        TypeError: If prices is not a pandas Series.
    """
    if not isinstance(prices, pd.Series):
        raise TypeError(f"prices must be a pandas Series, got {type(prices)}")
    
    if prices is None or len(prices) == 0:
        log_warn("Empty prices series provided for rate_of_change, returning empty series")
        return pd.Series(dtype="float64", index=prices.index if hasattr(prices, 'index') else pd.RangeIndex(0, 0))
    
    try:
        result = prices.pct_change()
        
        # Check for excessive NaN values (should only be first value)
        nan_count = result.isna().sum()
        if nan_count > 1:
            log_warn(
                f"rate_of_change contains {nan_count} NaN values. "
                f"Expected only 1 (first value). This may indicate data quality issues."
            )
        
        return result
    
    except Exception as e:
        log_error(f"Error calculating rate_of_change: {e}")
        raise


def diflen(length: int, robustness: str = "Medium") -> Tuple[int, int, int, int, int, int, int, int]:
    """Calculate length offsets for Moving Averages based on robustness setting.

    Port of Pine Script `diflen(length)` function. Returns 8 length values
    (4 positive offsets and 4 negative offsets) based on the robustness parameter.

    Args:
        length: Base length for Moving Average (must be > 0).
        robustness: Robustness setting determining offset spread:
            - "Narrow": Small offsets (±1, ±2, ±3, ±4)
            - "Medium": Medium offsets (±1, ±2, ±4, ±6)
            - "Wide": Large offsets (±1, ±3, ±5, ±7)

    Returns:
        Tuple of 8 integers: (L1, L2, L3, L4, L_1, L_2, L_3, L_4)
        where L1-L4 are positive offsets and L_1-L_4 are negative offsets.
        All returned values are guaranteed to be > 0.

    Raises:
        ValueError: If length is invalid or robustness is invalid.
        TypeError: If length is not an integer.
    """
    if not isinstance(length, int):
        raise TypeError(f"length must be an integer, got {type(length)}")
    
    if length <= 0:
        raise ValueError(f"length must be > 0, got {length}")
    
    robustness = robustness or "Medium"
    
    if not isinstance(robustness, str):
        raise TypeError(f"robustness must be a string, got {type(robustness)}")
    
    VALID_ROBUSTNESS = {"Narrow", "Medium", "Wide"}
    robustness_normalized = robustness.strip() if isinstance(robustness, str) else str(robustness)
    
    if robustness_normalized not in VALID_ROBUSTNESS:
        log_warn(
            f"Invalid robustness '{robustness}'. Valid values: {', '.join(VALID_ROBUSTNESS)}. "
            f"Using default 'Medium'."
        )
        robustness_normalized = "Medium"

    try:
        if robustness_normalized == "Narrow":
            L1, L_1 = length + 1, length - 1
            L2, L_2 = length + 2, length - 2
            L3, L_3 = length + 3, length - 3
            L4, L_4 = length + 4, length - 4
        elif robustness_normalized == "Medium":
            L1, L_1 = length + 1, length - 1
            L2, L_2 = length + 2, length - 2
            L3, L_3 = length + 4, length - 4
            L4, L_4 = length + 6, length - 6
        else:  # "Wide" or any other value (fallback to Wide)
            L1, L_1 = length + 1, length - 1
            L2, L_2 = length + 3, length - 3
            L3, L_3 = length + 5, length - 5
            L4, L_4 = length + 7, length - 7

        # Ensure all lengths are positive (negative offsets should still be > 0)
        lengths = [L1, L2, L3, L4, L_1, L_2, L_3, L_4]
        min_length = min(lengths)
        
        if min_length <= 0:
            # Adjust negative offsets to be at least 1
            L_1 = max(1, L_1)
            L_2 = max(1, L_2)
            L_3 = max(1, L_3)
            L_4 = max(1, L_4)
            
            log_warn(
                f"Some calculated lengths were <= 0 for length={length}, robustness={robustness_normalized}. "
                f"Adjusted negative offsets to minimum 1. "
                f"Result: L_1={L_1}, L_2={L_2}, L_3={L_3}, L_4={L_4}"
            )

        return L1, L2, L3, L4, L_1, L_2, L_3, L_4
    
    except Exception as e:
        log_error(f"Error calculating diflen: {e}")
        raise


def exp_growth(
    L: float,
    index: Optional[pd.Index] = None,
    *,
    cutout: int = 0,
) -> pd.Series:
    """Calculate exponential growth factor over time.

    Port of Pine Script function:
        e(L) =>
            bars = bar_index == 0 ? 1 : bar_index
            x = 1.0
            if time >= cuttime
                x := math.pow(math.e, L * (bar_index - cutout))
            x

    In TradingView, `time` and `bar_index` are global environment variables.
    Here we approximate using positional indices (0, 1, 2, ...) of the Series.

    Args:
        L: Lambda (growth rate parameter, must be finite).
        index: Time/bar index of the data. If None, creates empty RangeIndex.
        cutout: Number of bars to skip at the beginning (bars before cutout
            will have value 1.0, must be >= 0).

    Returns:
        Series containing exponential growth factors e^(L * (bar_index - cutout))
        for bars >= cutout, and 1.0 for bars < cutout.

    Raises:
        ValueError: If L is invalid, cutout is invalid, or overflow occurs.
        TypeError: If L is not a number or cutout is not an integer.
    """
    if not isinstance(L, (int, float)) or np.isnan(L) or np.isinf(L):
        raise ValueError(f"L must be a finite number, got {L}")
    
    if not isinstance(cutout, int) or cutout < 0:
        raise ValueError(f"cutout must be a non-negative integer, got {cutout}")
    
    if index is None:
        index = pd.RangeIndex(0, 0)

    try:
        # Use position 0..n-1 as equivalent to `bar_index`
        bars = pd.Series(range(len(index)), index=index, dtype="float64")
        # In Pine: if bar_index == 0 then bars = 1, else = bar_index
        bars = bars.where(bars != 0, 1.0)

        # Condition "has passed cutout"
        active = bars >= cutout
        x = pd.Series(1.0, index=index, dtype="float64")
        
        # Calculate exponential growth for active bars
        if active.any():
            # Calculate exponent to check for overflow
            exponents = L * (bars[active] - cutout)
            
            # Check for potential overflow (exp > 700 will overflow float64)
            max_exponent = exponents.max() if len(exponents) > 0 else 0
            if max_exponent > 700:
                log_warn(
                    f"Potential overflow in exp_growth: max exponent = {max_exponent:.2f}. "
                    f"Values > 700 may result in inf. L={L}, max_bar={bars[active].max()}, cutout={cutout}"
                )
            
            # Calculate exponential growth
            growth_values = np.e ** exponents
            
            # Check for overflow/inf values
            inf_count = np.isinf(growth_values).sum()
            if inf_count > 0:
                log_warn(
                    f"exp_growth produced {inf_count} inf values. "
                    f"This may indicate overflow. Consider reducing L or cutout."
                )
                # Replace inf with a large but finite value
                growth_values = np.where(np.isinf(growth_values), np.finfo(np.float64).max, growth_values)
            
            x.loc[active] = growth_values.astype("float64")
        
        return x
    
    except OverflowError as e:
        log_error(f"Overflow error in exp_growth: {e}. L={L}, cutout={cutout}")
        raise ValueError(f"Overflow in exponential calculation. L={L} may be too large.") from e
    except Exception as e:
        log_error(f"Error calculating exp_growth: {e}")
        raise


__all__ = [
    "rate_of_change",
    "diflen",
    "exp_growth",
]

