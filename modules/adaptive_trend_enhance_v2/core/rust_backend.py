"""
Python wrapper for Rust extensions with fallback to Numba.
"""

from __future__ import annotations

import warnings

import numpy as np
import pandas as pd
import pandas_ta as ta

try:
    from atc_rust import (
        calculate_equity_rust,
        calculate_kama_rust,
        process_signal_persistence_rust,
        calculate_ema_rust,
        calculate_wma_rust,
        calculate_dema_rust,
        calculate_lsma_rust,
        calculate_hma_rust,
    )

    RUST_AVAILABLE = True
except ImportError:
    warnings.warn("Rust extensions not available, falling back to Numba")
    RUST_AVAILABLE = False


def calculate_equity(
    r_values: np.ndarray,
    sig_prev: np.ndarray,
    starting_equity: float,
    decay_multiplier: float,
    cutout: int,
    use_rust: bool = True,
) -> np.ndarray:
    """
    Calculate equity with optional Rust backend.

    Args:
        r_values: Array of return values (n_bars,).
        sig_prev: Array of previous period signals (n_bars,).
        starting_equity: Initial equity value.
        decay_multiplier: Decay factor (1.0 - De).
        cutout: Number of bars to skip at beginning.
        use_rust: If True, attempts to use Rust backend. Falls back to Numba if False or unavailable.

    Returns:
        np.ndarray: Array of equity values.
    """
    if use_rust and RUST_AVAILABLE:
        # Rust expectation: r_values (n_bars,), sig_prev (n_bars,)
        return calculate_equity_rust(r_values, sig_prev, starting_equity, decay_multiplier, cutout)
    else:
        from .compute_equity.core import _calculate_equity_core

        return _calculate_equity_core(r_values, sig_prev, starting_equity, decay_multiplier, cutout)


def calculate_kama(
    prices: np.ndarray,
    length: int,
    use_rust: bool = True,
) -> np.ndarray:
    """
    Calculate KAMA with optional Rust backend.

    Args:
        prices: Array of price values (n_bars,).
        length: Efficiency ratio length.
        use_rust: If True, attempts to use Rust backend.

    Returns:
        np.ndarray: Array of KAMA values.
    """
    if use_rust and RUST_AVAILABLE:
        return calculate_kama_rust(prices, length)
    else:
        from .compute_moving_averages._numba_cores import _calculate_kama_atc_core

        return _calculate_kama_atc_core(prices, length)


def process_signal_persistence(
    up: np.ndarray,
    down: np.ndarray,
    use_rust: bool = True,
) -> np.ndarray:
    """
    Process signal persistence with optional Rust backend.

    Args:
        up: Boolean array of bullish crossover events.
        down: Boolean array of bearish crossunder events.
        use_rust: If True, attempts to use Rust backend.

    Returns:
        np.ndarray: Array of persistent signals (1, -1, or last state).
    """
    if use_rust and RUST_AVAILABLE:
        return process_signal_persistence_rust(up, down)
    else:
        # Fallback to local numba kernel if needed or existing one
        from .signal_detection.generate_signal import _apply_signal_persistence

        out = np.zeros(len(up), dtype=np.int8)
        _apply_signal_persistence(up, down, out)
        return out


def calculate_ema(
    prices: np.ndarray,
    length: int,
    use_rust: bool = True,
) -> np.ndarray:
    """
    Calculate EMA with optional Rust backend.

    Args:
        prices: Array of price values (n_bars,).
        length: Period for EMA calculation.
        use_rust: If True, attempts to use Rust backend.

    Returns:
        np.ndarray: Array of EMA values.
    """
    if use_rust and RUST_AVAILABLE:
        return calculate_ema_rust(prices, length)
    else:
        series = pd.Series(prices)
        result = ta.ema(series, length=length)
        return result.values if result is not None else np.full(len(prices), np.nan)


def calculate_wma(
    prices: np.ndarray,
    length: int,
    use_rust: bool = True,
) -> np.ndarray:
    """
    Calculate WMA with optional Rust backend.

    Args:
        prices: Array of price values (n_bars,).
        length: Period for WMA calculation.
        use_rust: If True, attempts to use Rust backend.

    Returns:
        np.ndarray: Array of WMA values.
    """
    if use_rust and RUST_AVAILABLE:
        return calculate_wma_rust(prices, length)
    else:
        series = pd.Series(prices)
        result = ta.wma(series, length=length)
        return result.values if result is not None else np.full(len(prices), np.nan)


def calculate_dema(
    prices: np.ndarray,
    length: int,
    use_rust: bool = True,
) -> np.ndarray:
    """
    Calculate DEMA with optional Rust backend.

    Args:
        prices: Array of price values (n_bars,).
        length: Period for DEMA calculation.
        use_rust: If True, attempts to use Rust backend.

    Returns:
        np.ndarray: Array of DEMA values.
    """
    if use_rust and RUST_AVAILABLE:
        return calculate_dema_rust(prices, length)
    else:
        series = pd.Series(prices)
        result = ta.dema(series, length=length)
        return result.values if result is not None else np.full(len(prices), np.nan)


def calculate_lsma(
    prices: np.ndarray,
    length: int,
    use_rust: bool = True,
) -> np.ndarray:
    """
    Calculate LSMA with optional Rust backend.

    Args:
        prices: Array of price values (n_bars,).
        length: Period for LSMA calculation.
        use_rust: If True, attempts to use Rust backend.

    Returns:
        np.ndarray: Array of LSMA values.
    """
    if use_rust and RUST_AVAILABLE:
        return calculate_lsma_rust(prices, length)
    else:
        series = pd.Series(prices)
        result = ta.linreg(series, length=length)
        return result.values if result is not None else np.full(len(prices), np.nan)


def calculate_hma(
    prices: np.ndarray,
    length: int,
    use_rust: bool = True,
) -> np.ndarray:
    """
    Calculate HMA with optional Rust backend.

    Args:
        prices: Array of price values (n_bars,).
        length: Period for HMA calculation.
        use_rust: If True, attempts to use Rust backend.

    Returns:
        np.ndarray: Array of HMA values.
    """
    if use_rust and RUST_AVAILABLE:
        return calculate_hma_rust(prices, length)
    else:
        # Fallback to pandas_ta HMA (matching ma_calculation_enhanced.py behavior)
        series = pd.Series(prices)
        result = ta.hma(series, length=length)
        if result is None:
            # Secondary fallback to SMA if HMA fails
            result = ta.sma(series, length=length)
        return result.values if result is not None else np.full(len(prices), np.nan)
