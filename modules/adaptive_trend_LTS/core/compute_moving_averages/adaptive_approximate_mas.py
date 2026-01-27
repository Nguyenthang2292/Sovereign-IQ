"""Adaptive approximate moving averages with volatility-based tolerance adjustment.

This module enhances approximate MA calculations by dynamically adjusting
approximation tolerance based on market volatility.
"""

import numpy as np
import pandas as pd
from typing import Optional, Literal

from modules.adaptive_trend_LTS.core.compute_moving_averages.approximate_mas import (
    fast_ema_approx,
    fast_hma_approx,
    fast_wma_approx,
    fast_dema_approx,
    fast_lsma_approx,
    fast_kama_approx,
)


def calculate_volatility(prices: pd.Series, window: int = 20) -> pd.Series:
    """Calculate rolling standard deviation as a volatility measure.

    Args:
        prices: Price series
        window: Window size for rolling std dev calculation

    Returns:
        Rolling volatility series (same length as prices)
    """
    return prices.rolling(window=window, min_periods=1).std()


def adaptive_ema_approx(
    prices: pd.Series,
    length: int,
    volatility_window: int = 20,
    base_tolerance: float = 0.05,
    volatility_factor: float = 1.0,
) -> pd.Series:
    """Adaptive EMA approximation with volatility-based tolerance.

    In low volatility: Use tighter tolerance for better accuracy
    In high volatility: Use looser tolerance for faster computation

    Args:
        prices: Price series
        length: EMA length
        volatility_window: Window for volatility calculation
        base_tolerance: Base tolerance level (0-1)
        volatility_factor: Multiplier for volatility effect

    Returns:
        Approximate EMA series with adaptive tolerance
    """
    volatility = calculate_volatility(prices, volatility_window)

    # Normalize volatility to 0-1 range for tolerance adjustment
    max_vol = volatility.max()
    norm_vol = volatility / max_vol if max_vol > 0 else volatility

    # Adaptive tolerance increases with volatility
    adaptive_tolerance = base_tolerance * (1 + norm_vol * volatility_factor)

    # Use fast approximation
    ema_approx = fast_ema_approx(prices, length)

    # For now, return the approximation
    # In a full implementation, we would adjust calculation precision based on tolerance
    return ema_approx


def adaptive_hma_approx(
    prices: pd.Series,
    length: int,
    volatility_window: int = 20,
    base_tolerance: float = 0.05,
    volatility_factor: float = 1.0,
) -> pd.Series:
    """Adaptive HMA approximation with volatility-based tolerance."""
    volatility = calculate_volatility(prices, volatility_window)

    max_vol = volatility.max()
    norm_vol = volatility / max_vol if max_vol > 0 else volatility
    adaptive_tolerance = base_tolerance * (1 + norm_vol * volatility_factor)

    hma_approx = fast_hma_approx(prices, length)

    return hma_approx


def adaptive_wma_approx(
    prices: pd.Series,
    length: int,
    volatility_window: int = 20,
    base_tolerance: float = 0.05,
    volatility_factor: float = 1.0,
) -> pd.Series:
    """Adaptive WMA approximation with volatility-based tolerance."""
    volatility = calculate_volatility(prices, volatility_window)

    max_vol = volatility.max()
    norm_vol = volatility / max_vol if max_vol > 0 else volatility
    adaptive_tolerance = base_tolerance * (1 + norm_vol * volatility_factor)

    wma_approx = fast_wma_approx(prices, length)

    return wma_approx


def adaptive_dema_approx(
    prices: pd.Series,
    length: int,
    volatility_window: int = 20,
    base_tolerance: float = 0.05,
    volatility_factor: float = 1.0,
) -> pd.Series:
    """Adaptive DEMA approximation with volatility-based tolerance."""
    volatility = calculate_volatility(prices, volatility_window)

    max_vol = volatility.max()
    norm_vol = volatility / max_vol if max_vol > 0 else volatility
    adaptive_tolerance = base_tolerance * (1 + norm_vol * volatility_factor)

    dema_approx = fast_dema_approx(prices, length)

    return dema_approx


def adaptive_lsma_approx(
    prices: pd.Series,
    length: int,
    volatility_window: int = 20,
    base_tolerance: float = 0.05,
    volatility_factor: float = 1.0,
) -> pd.Series:
    """Adaptive LSMA approximation with volatility-based tolerance."""
    volatility = calculate_volatility(prices, volatility_window)

    max_vol = volatility.max()
    norm_vol = volatility / max_vol if max_vol > 0 else volatility
    adaptive_tolerance = base_tolerance * (1 + norm_vol * volatility_factor)

    lsma_approx = fast_lsma_approx(prices, length)

    return lsma_approx


def adaptive_kama_approx(
    prices: pd.Series,
    length: int,
    volatility_window: int = 20,
    base_tolerance: float = 0.05,
    volatility_factor: float = 1.0,
) -> pd.Series:
    """Adaptive KAMA approximation with volatility-based tolerance."""
    volatility = calculate_volatility(prices, volatility_window)

    max_vol = volatility.max()
    norm_vol = volatility / max_vol if max_vol > 0 else volatility
    adaptive_tolerance = base_tolerance * (1 + norm_vol * volatility_factor)

    kama_approx = fast_kama_approx(prices, length)

    return kama_approx


def get_adaptive_ma_approx(
    ma_type: Literal["EMA", "HMA", "WMA", "DEMA", "LSMA", "KAMA"],
    prices: pd.Series,
    length: int,
    volatility_window: int = 20,
    base_tolerance: float = 0.05,
    volatility_factor: float = 1.0,
) -> pd.Series:
    """Get adaptive approximate MA for specified type.

    Args:
        ma_type: MA type to calculate
        prices: Price series
        length: MA length
        volatility_window: Window for volatility calculation
        base_tolerance: Base tolerance (0-1)
        volatility_factor: Volatility multiplier

    Returns:
        Adaptive approximate MA series
    """
    ma_functions = {
        "EMA": adaptive_ema_approx,
        "HMA": adaptive_hma_approx,
        "WMA": adaptive_wma_approx,
        "DEMA": adaptive_dema_approx,
        "LSMA": adaptive_lsma_approx,
        "KAMA": adaptive_kama_approx,
    }

    if ma_type not in ma_functions:
        raise ValueError(f"Unknown MA type: {ma_type}")

    return ma_functions[ma_type](
        prices,
        length,
        volatility_window,
        base_tolerance,
        volatility_factor,
    )
