"""
Python wrapper for Rust extensions with fallback to Numba.
"""

from __future__ import annotations

import warnings
from typing import Optional, Union

import numpy as np
import pandas as pd
import pandas_ta as ta

try:
    from atc_rust import (  # type: ignore[import-untyped]
        calculate_dema_rust,
        calculate_ema_rust,
        calculate_equity_rust,
        calculate_hma_rust,
        calculate_kama_rust,
        calculate_lsma_rust,
        calculate_wma_rust,
        process_signal_persistence_rust,
    )

    RUST_AVAILABLE = True
except ImportError:
    warnings.warn("Rust extensions not available, falling back to Numba")
    RUST_AVAILABLE = False
    calculate_dema_rust = None  # type: ignore[misc]
    calculate_ema_rust = None  # type: ignore[misc]
    calculate_equity_rust = None  # type: ignore[misc]
    calculate_hma_rust = None  # type: ignore[misc]
    calculate_kama_rust = None  # type: ignore[misc]
    calculate_lsma_rust = None  # type: ignore[misc]
    calculate_wma_rust = None  # type: ignore[misc]
    process_signal_persistence_rust = None  # type: ignore[misc]


def _ensure_numpy_array(data: Union[pd.Series, np.ndarray]) -> np.ndarray:
    """Convert pandas Series to numpy array if needed."""
    if isinstance(data, pd.Series):
        return data.to_numpy()
    return data


def calculate_equity(
    r_values: Union[np.ndarray, pd.Series],
    sig_prev: Union[np.ndarray, pd.Series],
    starting_equity: float,
    decay_multiplier: float,
    cutout: int,
    use_rust: bool = True,
    floor_val: Optional[float] = None,
) -> np.ndarray:
    """
    Calculate equity with optional Rust CPU backend.

    Args:
        r_values: Array of return values (n_bars,).
        sig_prev: Array of previous period signals (n_bars,).
        starting_equity: Initial equity value.
        decay_multiplier: Decay factor (1.0 - De).
        cutout: Number of bars to skip at beginning.
        use_rust: If True, attempts to use Rust CPU backend.
        floor_val: Optional equity floor.

    Returns:
        np.ndarray: Array of equity values.
    """
    # Convert to numpy arrays if needed
    r_vals = _ensure_numpy_array(r_values)
    sig_p = _ensure_numpy_array(sig_prev)

    # Rust backend uses a hardcoded equity floor of 0.25; configurable floor_val
    # is only applied in the Python path. Use Python path when a custom floor is requested.
    rust_ok = use_rust and RUST_AVAILABLE
    if rust_ok and floor_val is not None and floor_val != 0.25:
        rust_ok = False  # Fall back to Python so equity_floor is respected

    if rust_ok:
        assert calculate_equity_rust is not None  # narrow type after RUST_AVAILABLE check
        return calculate_equity_rust(r_vals, sig_p, starting_equity, decay_multiplier, cutout)
    else:
        from .compute_equity.core import _calculate_equity_core

        return _calculate_equity_core(
            r_vals, sig_p, starting_equity, decay_multiplier, cutout, floor_val=floor_val
        )


def calculate_kama(
    prices: Union[np.ndarray, pd.Series],
    length: int,
    use_rust: bool = True,
) -> np.ndarray:
    """
    Calculate KAMA with optional Rust CPU backend.

    Args:
        prices: Array of price values (n_bars,).
        length: Efficiency ratio length.
        use_rust: If True, attempts to use Rust CPU backend.

    Returns:
        np.ndarray: Array of KAMA values.
    """
    # Convert to numpy array if needed
    prices_arr = _ensure_numpy_array(prices)

    if use_rust and RUST_AVAILABLE:
        assert calculate_kama_rust is not None
        return calculate_kama_rust(prices_arr, length)
    else:
        from .compute_moving_averages._numba_cores import _calculate_kama_atc_core

        return _calculate_kama_atc_core(prices_arr, length)


def process_signal_persistence(
    up: Union[np.ndarray, pd.Series],
    down: Union[np.ndarray, pd.Series],
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
    # Convert to numpy arrays if needed
    up_arr = _ensure_numpy_array(up)
    down_arr = _ensure_numpy_array(down)

    if use_rust and RUST_AVAILABLE:
        assert process_signal_persistence_rust is not None
        return process_signal_persistence_rust(up_arr, down_arr)
    else:
        # Fallback to local numba kernel if needed or existing one
        from .signal_detection.generate_signal import _apply_signal_persistence

        out = np.zeros(len(up_arr), dtype=np.int8)
        _apply_signal_persistence(up_arr, down_arr, out)
        return out


def calculate_ema(
    prices: Union[np.ndarray, pd.Series],
    length: int,
    use_rust: bool = True,
) -> np.ndarray:
    """
    Calculate EMA with optional Rust CPU backend.

    Args:
        prices: Array of price values (n_bars,).
        length: Period for EMA calculation.
        use_rust: If True, attempts to use Rust CPU backend.

    Returns:
        np.ndarray: Array of EMA values.
    """
    # Convert to numpy array if needed
    prices_arr = _ensure_numpy_array(prices)

    if use_rust and RUST_AVAILABLE:
        assert calculate_ema_rust is not None
        return calculate_ema_rust(prices_arr, length)
    else:
        series = pd.Series(prices_arr)
        result = ta.ema(series, length=length)
        return result.values if result is not None else np.full(len(prices_arr), np.nan)  # type: ignore[return-value]


def calculate_wma(
    prices: Union[np.ndarray, pd.Series],
    length: int,
    use_rust: bool = True,
) -> np.ndarray:
    """
    Calculate WMA with optional Rust CPU backend.

    Args:
        prices: Array of price values (n_bars,).
        length: Period for WMA calculation.
        use_rust: If True, attempts to use Rust CPU backend.

    Returns:
        np.ndarray: Array of WMA values.
    """
    # Convert to numpy array if needed
    prices_arr = _ensure_numpy_array(prices)

    if use_rust and RUST_AVAILABLE:
        assert calculate_wma_rust is not None
        return calculate_wma_rust(prices_arr, length)
    else:
        series = pd.Series(prices_arr)
        result = ta.wma(series, length=length)
        return result.values if result is not None else np.full(len(prices_arr), np.nan)  # type: ignore[return-value]


def calculate_dema(
    prices: Union[np.ndarray, pd.Series],
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
    # Convert to numpy array if needed
    prices_arr = _ensure_numpy_array(prices)

    if use_rust and RUST_AVAILABLE:
        assert calculate_dema_rust is not None
        return calculate_dema_rust(prices_arr, length)
    else:
        series = pd.Series(prices_arr)
        result = ta.dema(series, length=length)
        return result.values if result is not None else np.full(len(prices_arr), np.nan)  # type: ignore[return-value]


def calculate_lsma(
    prices: Union[np.ndarray, pd.Series],
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
    # Convert to numpy array if needed
    prices_arr = _ensure_numpy_array(prices)

    if use_rust and RUST_AVAILABLE:
        assert calculate_lsma_rust is not None
        return calculate_lsma_rust(prices_arr, length)
    else:
        series = pd.Series(prices_arr)
        result = ta.linreg(series, length=length)
        return result.values if result is not None else np.full(len(prices_arr), np.nan)  # type: ignore[return-value]


def calculate_hma(
    prices: Union[np.ndarray, pd.Series],
    length: int,
    use_rust: bool = True,
) -> np.ndarray:
    """
    Calculate HMA with optional Rust CPU backend.

    Args:
        prices: Array of price values (n_bars,).
        length: Period for HMA calculation.
        use_rust: If True, attempts to use Rust CPU backend.

    Returns:
        np.ndarray: Array of HMA values.
    """
    # Convert to numpy array if needed
    prices_arr = _ensure_numpy_array(prices)

    if use_rust and RUST_AVAILABLE:
        assert calculate_hma_rust is not None
        return calculate_hma_rust(prices_arr, length)
    else:
        # Fallback to pandas_ta HMA (matching ma_calculation_enhanced.py behavior)
        series = pd.Series(prices_arr)
        result = ta.hma(series, length=length)
        if result is None:
            # Secondary fallback to SMA if HMA fails
            result = ta.sma(series, length=length)
        return result.values if result is not None else np.full(len(prices_arr), np.nan)  # type: ignore[return-value]
