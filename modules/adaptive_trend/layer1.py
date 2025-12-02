"""Layer 1 Processing functions for Adaptive Trend Classification (ATC).

This module provides functions for processing signals from multiple Moving
Averages in Layer 1 of the ATC system:
- weighted_signal: Calculate weighted average signal from multiple signals and weights
- cut_signal: Discretize continuous signal into {-1, 0, 1}
- trend_sign: Determine trend direction (+1 for bullish, -1 for bearish, 0 for neutral)
- _layer1_signal_for_ma: Calculate Layer 1 signal for a specific MA type
"""

from __future__ import annotations

from typing import Iterable, Tuple

import pandas as pd

from .equity import equity_series
from .signals import generate_signal_from_ma
from .utils import rate_of_change


def weighted_signal(
    signals: Iterable[pd.Series],
    weights: Iterable[pd.Series],
) -> pd.Series:
    """Calculate weighted average signal from multiple signals and weights.

    Port of Pine Script function:
        Signal(m1, w1, ..., m9, w9) =>
            n = Σ (mi * wi)      # Weighted sum
            d = Σ wi             # Sum of weights
            sig = math.round(n/d, 2)
            sig

    Typically receives 9 signal series and 9 weight series (equity curves).

    Args:
        signals: Iterable of signal series (typically 9 series).
        weights: Iterable of weight series (typically equity curves).

    Returns:
        Weighted signal series rounded to 2 decimal places.

    Raises:
        ValueError: If signals and weights have different lengths.
    """
    signals = list(signals)
    weights = list(weights)
    if len(signals) != len(weights):
        raise ValueError("signals và weights phải có cùng độ dài")

    if not signals:
        return pd.Series(dtype="float64")

    num = None
    den = None
    for m, w in zip(signals, weights):
        term = m * w
        num = term if num is None else num + term
        den = w if den is None else den + w

    sig = num / den
    return sig.round(2)


def cut_signal(x: pd.Series, threshold: float = 0.49) -> pd.Series:
    """Discretize continuous signal into {-1, 0, 1} based on threshold.

    Port of Pine Script function:
        Cut(x) =>
            c = x > 0.49 ? 1 : x < -0.49 ? -1 : 0
            c

    Args:
        x: Continuous signal series.
        threshold: Threshold for discretization (default: 0.49).
            Values > threshold → 1, values < -threshold → -1, else → 0.

    Returns:
        Series with discrete values {-1, 0, 1}.
    """
    c = pd.Series(0, index=x.index, dtype="int8")
    c[x > threshold] = 1
    c[x < -threshold] = -1
    return c


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
    """
    base = signal.shift(1) if strategy else signal
    result = pd.Series(0, index=signal.index, dtype="int8")
    result[base > 0] = 1
    result[base < 0] = -1
    return result


def _layer1_signal_for_ma(
    prices: pd.Series,
    ma_tuple: Tuple[pd.Series, ...],
    *,
    L: float,
    De: float,
    cutout: int = 0,
) -> Tuple[pd.Series, Tuple[pd.Series, ...], Tuple[pd.Series, ...]]:
    """Calculate Layer 1 signal for a specific Moving Average type.

    Port of Pine Script logic block:
        E   = eq(1, signal(MA),   R), sE   = signal(MA)
        E1  = eq(1, signal(MA1),  R), sE1  = signal(MA1)
        ...
        EMA_Signal = Signal(sE, E, sE1, E1, ..., sE_4, E_4)

    For each of the 9 MAs:
    1. Generate signal from price/MA crossover
    2. Calculate equity curve from signal
    3. Weight signals by their equity curves to get final Layer 1 signal

    Args:
        prices: Price series (typically close prices).
        ma_tuple: Tuple of 9 MA Series: (MA, MA1, MA2, MA3, MA4, MA_1, MA_2, MA_3, MA_4).
        L: Lambda (growth rate) for equity calculations.
        De: Decay factor for equity calculations.
        cutout: Number of bars to skip at beginning.

    Returns:
        Tuple containing:
        - signal_series: Weighted Layer 1 signal for this MA type
        - signals_tuple: Tuple of 9 individual signals (s, s1, s2, s3, s4, s_1, s_2, s_3, s_4)
        - equity_tuple: Tuple of 9 equity curves (E, E1, E2, E3, E4, E_1, E_2, E_3, E_4)
    """
    (
        MA,
        MA1,
        MA2,
        MA3,
        MA4,
        MA_1,
        MA_2,
        MA_3,
        MA_4,
    ) = ma_tuple

    R = rate_of_change(prices)

    s = generate_signal_from_ma(prices, MA)
    s1 = generate_signal_from_ma(prices, MA1)
    s2 = generate_signal_from_ma(prices, MA2)
    s3 = generate_signal_from_ma(prices, MA3)
    s4 = generate_signal_from_ma(prices, MA4)
    s_1 = generate_signal_from_ma(prices, MA_1)
    s_2 = generate_signal_from_ma(prices, MA_2)
    s_3 = generate_signal_from_ma(prices, MA_3)
    s_4 = generate_signal_from_ma(prices, MA_4)

    E = equity_series(1.0, s, R, L=L, De=De, cutout=cutout)
    E1 = equity_series(1.0, s1, R, L=L, De=De, cutout=cutout)
    E2 = equity_series(1.0, s2, R, L=L, De=De, cutout=cutout)
    E3 = equity_series(1.0, s3, R, L=L, De=De, cutout=cutout)
    E4 = equity_series(1.0, s4, R, L=L, De=De, cutout=cutout)
    E_1 = equity_series(1.0, s_1, R, L=L, De=De, cutout=cutout)
    E_2 = equity_series(1.0, s_2, R, L=L, De=De, cutout=cutout)
    E_3 = equity_series(1.0, s_3, R, L=L, De=De, cutout=cutout)
    E_4 = equity_series(1.0, s_4, R, L=L, De=De, cutout=cutout)

    signal_series = weighted_signal(
        signals=[s, s1, s2, s3, s4, s_1, s_2, s_3, s_4],
        weights=[E, E1, E2, E3, E4, E_1, E_2, E_3, E_4],
    )

    return (
        signal_series,
        (s, s1, s2, s3, s4, s_1, s_2, s_3, s_4),
        (E, E1, E2, E3, E4, E_1, E_2, E_3, E_4),
    )


__all__ = [
    "weighted_signal",
    "cut_signal",
    "trend_sign",
    "_layer1_signal_for_ma",
]

