"""Moving Average calculations for Adaptive Trend Classification (ATC).

This module provides functions to calculate various types of Moving Averages:
- calculate_kama_atc: KAMA (Kaufman Adaptive Moving Average) tuned for ATC
- ma_calculation: Calculate different MA types (EMA, HMA, WMA, DEMA, LSMA, KAMA)
- set_of_moving_averages: Generate a set of 9 MAs from a base length with offsets
"""

from __future__ import annotations

from typing import Optional

import pandas as pd
import pandas_ta as ta

from modules.common.indicators.momentum import calculate_kama_series
from .utils import diflen


def calculate_kama_atc(
    prices: pd.Series,
    length: int = 28,
) -> Optional[pd.Series]:
    """Calculate KAMA (Kaufman Adaptive Moving Average) for ATC.

    Uses KAMA formula from `momentum.calculate_kama_series` with parameters
    chosen to match Pine Script behavior:
    - length: Window length (default 28, equivalent to Pine `kama_len`)
    - fast: 2 → fast_sc ≈ 0.666 (matches Pine: 0.666)
    - slow: 30 → slow_sc ≈ 0.064 (matches Pine: 0.064)

    Args:
        prices: Price series (typically close prices).
        length: KAMA window length, equivalent to Pine `kama_len`.

    Returns:
        KAMA Series with same index as prices, or None if calculation fails.
    """
    if prices is None or len(prices) == 0:
        return None

    return calculate_kama_series(
        prices=prices,
        period=length,
        fast=2,
        slow=30,
    )


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
    - HMA maps to SMA (not classic Hull MA) to match original script behavior.
    - LSMA uses `ta.linreg`, equivalent to `lsma()` in Pine.
    - KAMA calls `calculate_kama_atc` with normalized fast/slow parameters.

    Args:
        source: Source price series.
        length: Window length for Moving Average.
        ma_type: Type of MA: "EMA", "HMA", "WMA", "DEMA", "LSMA", or "KAMA"
            (case-insensitive).

    Returns:
        Moving Average Series, or None if calculation fails or invalid ma_type.
    """
    if source is None or len(source) == 0:
        return None

    ma = ma_type.upper()

    if ma == "EMA":
        return ta.ema(source, length=length)
    if ma == "HMA":
        # Pine: HMA branch đang dùng ta.sma, không phải Hull MA chuẩn.
        return ta.sma(source, length=length)
    if ma == "WMA":
        return ta.wma(source, length=length)
    if ma == "DEMA":
        return ta.dema(source, length=length)
    if ma == "LSMA":
        # LSMA ~ Linear Regression (Least Squares Moving Average)
        return ta.linreg(source, length=length)
    if ma == "KAMA":
        return calculate_kama_atc(source, length=length)

    return None


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
    """
    if source is None or len(source) == 0:
        return None

    L1, L2, L3, L4, L_1, L_2, L_3, L_4 = diflen(length, robustness=robustness)

    MA = ma_calculation(source, length, ma_type)
    MA1 = ma_calculation(source, L1, ma_type)
    MA2 = ma_calculation(source, L2, ma_type)
    MA3 = ma_calculation(source, L3, ma_type)
    MA4 = ma_calculation(source, L4, ma_type)
    MA_1 = ma_calculation(source, L_1, ma_type)
    MA_2 = ma_calculation(source, L_2, ma_type)
    MA_3 = ma_calculation(source, L_3, ma_type)
    MA_4 = ma_calculation(source, L_4, ma_type)

    return MA, MA1, MA2, MA3, MA4, MA_1, MA_2, MA_3, MA_4


__all__ = [
    "calculate_kama_atc",
    "ma_calculation",
    "set_of_moving_averages",
]

