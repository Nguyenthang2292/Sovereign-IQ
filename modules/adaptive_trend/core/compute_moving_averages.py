"""Moving Average calculations for Adaptive Trend Classification (ATC).

This module provides functions to calculate various types of Moving Averages:
- calculate_kama_atc: KAMA (Kaufman Adaptive Moving Average) tuned for ATC
- ma_calculation: Calculate different MA types (EMA, HMA, WMA, DEMA, LSMA, KAMA)
- set_of_moving_averages: Generate a set of 9 MAs from a base length with offsets
"""

from __future__ import annotations

from typing import Optional, Tuple

import numpy as np
import pandas as pd
import pandas_ta as ta

from modules.adaptive_trend.utils import diflen
from modules.common.utils import log_error, log_warn

try:
    from numba import njit

    _HAS_NUMBA = True
except ImportError:
    _HAS_NUMBA = False

    def njit(*args, **kwargs):
        def decorator(func):
            return func

        return decorator


def calculate_kama_atc(
    prices: pd.Series,
    length: int = 28,
) -> Optional[pd.Series]:
    """Calculate KAMA (Kaufman Adaptive Moving Average) for ATC.

    Direct port of Pine Script KAMA function to ensure exact matching:
        kama(source, length) =>
            fast= 0.666
            slow = 0.064
            noisex = math.abs(source - source[1])
            KAMA = 0.0
            signal = math.abs(source - source[length])
            noise = math.sum(noisex, length)
            ratio = noise != 0 ? signal / noise : 0
            smooth = math.pow(ratio * (fast - slow) + slow, 2)
            KAMA := nz(KAMA[1]) + smooth * (source - nz(KAMA[1]))
            KAMA

    Args:
        prices: Price series (typically close prices).
        length: KAMA window length (efficiency ratio period).

    Returns:
        KAMA Series with same index as prices, or None if calculation fails.

    Raises:
        ValueError: If length is invalid or prices is empty.
        TypeError: If prices is not a pandas Series.
    """
    if not isinstance(prices, pd.Series):
        raise TypeError(f"prices must be a pandas Series, got {type(prices)}")

    if len(prices) == 0:
        log_warn("Empty prices series provided for KAMA calculation")
        return None

    if length <= 0:
        raise ValueError(f"length must be > 0, got {length}")

    if length > len(prices):
        log_warn(
            f"KAMA length ({length}) is greater than prices length ({len(prices)}). "
            f"This may result in insufficient data for calculation."
        )

    try:
        prices_array = prices.values.astype("float64")
        kama_array = _calculate_kama_atc_core(prices_array, length)
        return pd.Series(kama_array, index=prices.index)

    except Exception as e:
        log_error(f"Error calculating KAMA: {e}")
        raise


# @njit(cache=True)
def _calculate_kama_atc_core(
    prices_array: np.ndarray,
    length: int,
) -> np.ndarray:
    """Core KAMA calculation optimized with Numba."""
    n = len(prices_array)
    kama = np.full(n, np.nan, dtype=np.float64)

    if n < 1:
        return kama

    fast = 0.666
    slow = 0.064

    for i in range(n):
        if i == 0:
            kama[i] = prices_array[i]
            continue

        if i < length:
            kama[i] = kama[i - 1]
            continue

        # Calculate noise: PineScript math.sum(math.abs(src - src[1]), length)
        noise = 0.0
        for j in range(i - length + 1, i + 1):
            if j <= 0:
                continue
            noise += abs(prices_array[j] - prices_array[j - 1])

        signal = abs(prices_array[i] - prices_array[i - length])
        ratio = 0.0 if noise == 0 else signal / noise

        # Use float64 power directly
        smooth = (ratio * (fast - slow) + slow) ** 2

        prev_kama = kama[i - 1]
        # Use np.isnan on float64 explicitly
        if np.isnan(prev_kama):
            prev_kama = prices_array[i]

        kama[i] = prev_kama + (smooth * (prices_array[i] - prev_kama))

    return kama


def ma_calculation(
    source: pd.Series,
    length: int,
    ma_type: str,
) -> Optional[pd.Series]:
    """Calculate Moving Average based on specified type.

    Port of Pine Script function:
        ma_calculation(source, length, ma_type) =>
            if ma_type == "EMA"
                ta.ema(source, length)
            else if ma_type == "HMA"
                ta.sma(source, length)  # Note: Uses SMA, not classic Hull MA
            else if ma_type == "WMA"
                ta.wma(source, length)
            else if ma_type == "DEMA"
                ta.dema(source, length)
            else if ma_type == "LSMA"
                lsma(source, length)
            else if ma_type == "KAMA"
                kama(source, length)
            else
                na

    Notes:
    - **DEVIATION FROM PINESCRIPT**: HMA uses TRUE Hull Moving Average (ta.hma) instead of SMA.
      The original PineScript source uses ta.sma() for "HMA", but this Python implementation
      uses the correct Hull formula for better trend following and reduced lag.
    - LSMA uses `ta.linreg`, equivalent to `lsma()` in Pine.
    - KAMA calls `calculate_kama_atc` with normalized fast/slow parameters.

    Args:
        source: Source price series.
        length: Window length for Moving Average.
        ma_type: Type of MA: "EMA", "HMA", "WMA", "DEMA", "LSMA", or "KAMA"
            (case-insensitive).

    Returns:
        Moving Average Series, or None if calculation fails or invalid ma_type.

    Raises:
        ValueError: If length is invalid, source is empty, or ma_type is invalid.
        TypeError: If source is not a pandas Series.
    """
    if not isinstance(source, pd.Series):
        raise TypeError(f"source must be a pandas Series, got {type(source)}")  # pyright: ignore[reportUnreachable]

    if len(source) == 0:
        log_warn("Empty source series provided for MA calculation")
        return None

    if length <= 0:
        raise ValueError(f"length must be > 0, got {length}")

    if length > len(source):
        log_warn(
            f"MA length ({length}) is greater than source length ({len(source)}). "
            f"This may result in insufficient data for calculation."
        )

    if not isinstance(ma_type, str) or not ma_type.strip():
        raise ValueError(f"ma_type must be a non-empty string, got {ma_type}")

    ma = ma_type.upper().strip()
    VALID_MA_TYPES = {"EMA", "HMA", "WMA", "DEMA", "LSMA", "KAMA"}

    if ma not in VALID_MA_TYPES:
        log_warn(f"Invalid ma_type '{ma_type}'. Valid types: {', '.join(VALID_MA_TYPES)}. Returning None.")
        return None

    try:
        if ma == "EMA":
            result = ta.ema(source, length=length)
        elif ma == "HMA":
            # DEVIATION FROM PINESCRIPT SOURCE:
            # The original Pine Script source (source_pine.txt) uses ta.sma() for "HMA".
            # This Python implementation uses TRUE Hull Moving Average (ta.hma) for correctness.
            #
            # PineScript source line: else if ma_type == "HMA" ta.sma(source, length)
            # Python implementation: ta.hma(source, length)
            #
            # Rationale: Using true HMA provides better trend following and reduces lag,
            # which is the intended purpose of Hull Moving Average. The PineScript version
            # likely used SMA as a simplification or placeholder.
            #
            # Impact: This will produce different signals compared to the original PineScript.
            # All Python versions (Original, Enhanced, Rust) now use consistent TRUE HMA.
            result = ta.hma(source, length=length)
            if result is None:
                log_warn(f"HMA calculation failed, falling back to SMA for length={length}")
                result = ta.sma(source, length=length)
        elif ma == "WMA":
            result = ta.wma(source, length=length)
        elif ma == "DEMA":
            result = ta.dema(source, length=length)
        elif ma == "LSMA":
            # LSMA ~ Linear Regression (Least Squares Moving Average)
            result = ta.linreg(source, length=length)
        elif ma == "KAMA":
            result = calculate_kama_atc(source, length=length)
        else:
            # This should never happen due to validation above, but kept for safety
            return None

        if result is None:
            log_warn(f"MA calculation ({ma}) returned None for length={length}")
        elif len(result) == 0:
            log_warn(f"MA calculation ({ma}) returned empty series for length={length}")
        elif not isinstance(result, pd.Series):
            log_warn(f"MA calculation ({ma}) returned unexpected type {type(result)}, expected pandas Series")
            return None

        return result

    except Exception as e:
        log_error(f"Error calculating {ma} MA with length={length}: {e}")
        raise


def set_of_moving_averages(
    length: int,
    source: pd.Series,
    ma_type: str,
    robustness: str = "Medium",
) -> Optional[Tuple[pd.Series, ...]]:
    """Generate a set of 9 Moving Averages with different length offsets.

    Port of Pine Script function:
        SetOfMovingAverages(length, source, ma_type) =>
            [L1,L2,L3,L4,L_1,L_2,L_3,L_4] = diflen(length)
            MA   = ma_calculation(source, length, ma_type)
            MA1  = ma_calculation(source, L1,     ma_type)
            ...
            [MA, MA1, MA2, MA3, MA4, MA_1, MA_2, MA_3, MA_4]

    Calculates 9 MAs: one at base length, four with positive offsets (L1-L4),
    and four with negative offsets (L_1-L_4).

    Args:
        length: Base length for Moving Average.
        source: Source price series.
        ma_type: Type of MA: "EMA", "HMA", "WMA", "DEMA", "LSMA", or "KAMA".
        robustness: Robustness setting determining offset spread:
            "Narrow", "Medium", or "Wide".

    Returns:
        Tuple of 9 MA Series: (MA, MA1, MA2, MA3, MA4, MA_1, MA_2, MA_3, MA_4),
        or None if source is empty or invalid.

    Raises:
        ValueError: If length is invalid, source is empty, or robustness is invalid.
        TypeError: If source is not a pandas Series.
    """
    # Input validation
    if not isinstance(source, pd.Series):
        raise TypeError(f"source must be a pandas Series, got {type(source)}")  # pyright: ignore[reportUnreachable]

    if len(source) == 0:
        log_warn("Empty source series provided for set_of_moving_averages")
        return None

    if length <= 0:
        raise ValueError(f"length must be > 0, got {length}")

    if not isinstance(ma_type, str) or not ma_type.strip():
        raise ValueError(f"ma_type must be a non-empty string, got {ma_type}")

    VALID_ROBUSTNESS = {"Narrow", "Medium", "Wide"}
    if robustness not in VALID_ROBUSTNESS:
        log_warn(
            f"Invalid robustness '{robustness}'. Valid values: {', '.join(VALID_ROBUSTNESS)}. Using default 'Medium'."
        )
        robustness = "Medium"

    try:
        # Calculate length offsets
        L1, L2, L3, L4, L_1, L_2, L_3, L_4 = diflen(length, robustness=robustness)

        # Validate offsets are positive (negative offsets from diflen should still be > 0)
        lengths = [length, L1, L2, L3, L4, L_1, L_2, L_3, L_4]
        if any(len_val <= 0 for len_val in lengths):
            invalid_lengths = [len_val for len_val in lengths if len_val <= 0]
            raise ValueError(f"Invalid length offsets calculated: {invalid_lengths}. All lengths must be > 0.")

        # Calculate all MAs (optimized with list comprehension)
        ma_lengths = [length, L1, L2, L3, L4, L_1, L_2, L_3, L_4]
        ma_names = ["MA", "MA1", "MA2", "MA3", "MA4", "MA_1", "MA_2", "MA_3", "MA_4"]

        mas = []
        failed_calculations = []

        for ma_len, ma_name in zip(ma_lengths, ma_names):
            ma_result = ma_calculation(source, ma_len, ma_type)
            if ma_result is None:
                failed_calculations.append(f"{ma_name} (length={ma_len})")
                log_warn(f"Failed to calculate {ma_name} ({ma_type}, length={ma_len}).")
            mas.append(ma_result)

        # Check if any MA calculation failed
        if all(ma is None for ma in mas):
            log_error(f"All MA calculations failed for ma_type={ma_type}, length={length}.")
            return None

        # Raise error if any MA calculation failed (don't return partial tuple)
        if failed_calculations:
            failed_list = ", ".join(failed_calculations)
            error_msg = (
                f"Failed to calculate {len(failed_calculations)} out of 9 MAs "
                f"for ma_type={ma_type}, length={length}. "
                f"Failed: {failed_list}. "
                f"Cannot proceed with partial MA set as it will cause TypeErrors downstream."
            )
            log_error(error_msg)
            raise ValueError(error_msg)

        # Unpack for return tuple (maintaining original variable names)
        MA, MA1, MA2, MA3, MA4, MA_1, MA_2, MA_3, MA_4 = mas

        return MA, MA1, MA2, MA3, MA4, MA_1, MA_2, MA_3, MA_4

    except Exception as e:
        log_error(f"Error calculating set of moving averages: {e}")
        raise


__all__ = [
    "calculate_kama_atc",
    "ma_calculation",
    "set_of_moving_averages",
]
