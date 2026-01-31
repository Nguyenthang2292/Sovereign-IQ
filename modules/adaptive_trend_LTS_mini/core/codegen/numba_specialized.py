"""Numba JIT-specialized ATC computations for hot path configurations.

This module provides JIT-compiled implementations of ATC computations
for frequently used configurations. Functions are compiled at first use
and cached for performance.
"""

from __future__ import annotations

import importlib.util
from typing import Tuple

import numpy as np

# Check if Numba is available using find_spec as recommended
if importlib.util.find_spec("numba"):
    from numba import njit

    NUMBA_AVAILABLE = True
else:
    NUMBA_AVAILABLE = False

    # Fallback decorator if Numba not available
    def njit(*args, **kwargs):
        def decorator(func):
            return func

        return decorator


@njit
def compute_ema_jit(prices: np.ndarray, length: int) -> np.ndarray:
    """Compute EMA using JIT-compiled implementation.

    Args:
        prices: Price array
        length: EMA period

    Returns:
        EMA array
    """
    n = len(prices)
    ema = np.zeros(n, dtype=np.float64)
    alpha = 2.0 / (length + 1.0)

    # Initialize first value
    if n > 0:
        ema[0] = prices[0]

    # Compute EMA recursively
    for i in range(1, n):
        ema[i] = alpha * prices[i] + (1.0 - alpha) * ema[i - 1]

    return ema


@njit
def compute_ema_only_atc_jit(
    prices: np.ndarray,
    ema_len: int,
    lambda_scaled: float,
    decay_scaled: float,
    long_threshold: float,
    short_threshold: float,
    cutout: int,
    strategy_mode: bool,
) -> Tuple[np.ndarray, np.ndarray]:
    """Compute ATC signals for EMA-only configuration using JIT.

    This is a simplified ATC computation that only uses EMA, suitable
    for fast scanning and filtering.

    Args:
        prices: Price array
        ema_len: EMA period
        lambda_scaled: Lambda parameter (scaled)
        decay_scaled: Decay parameter (scaled)
        long_threshold: Long signal threshold
        short_threshold: Short signal threshold
        cutout: Number of bars to skip
        strategy_mode: If True, shift signal by 1 bar

    Returns:
        Tuple of (ema_signal, ema_equity) arrays
    """
    n = len(prices)

    # Compute EMA
    ema = compute_ema_jit(prices, ema_len)

    # Compute rate of change
    roc = np.zeros(n, dtype=np.float64)
    for i in range(1, n):
        if prices[i - 1] != 0:
            roc[i] = (prices[i] - prices[i - 1]) / prices[i - 1]

    # Layer 1 signal computation (simplified EMA-only)
    ema_signal = np.zeros(n, dtype=np.float64)
    for i in range(1, n):
        # Compare EMA with price
        diff = ema[i] - prices[i]

        # Apply adaptive lambda scaling
        signal = diff * lambda_scaled

        # Decay from previous
        if i > 0:
            signal = signal * (1.0 - decay_scaled) + ema_signal[i - 1] * decay_scaled

        ema_signal[i] = signal

    # Layer 2 equity computation
    ema_equity = np.zeros(n, dtype=np.float64)
    starting_equity = 1.0

    for i in range(cutout, n):
        # Use signal from previous bar
        if i > 0:
            sig_prev = ema_signal[i - 1]
        else:
            sig_prev = 0.0

        # Apply signal to equity
        if sig_prev > long_threshold:
            ema_equity[i] = ema_equity[i - 1] * (1.0 + roc[i])
        elif sig_prev < short_threshold:
            ema_equity[i] = ema_equity[i - 1] * (1.0 - roc[i])
        else:
            ema_equity[i] = ema_equity[i - 1]

    # Strategy mode: shift signal by 1 bar
    if strategy_mode and n > 1:
        ema_signal_shifted = np.zeros_like(ema_signal)
        ema_signal_shifted[1:] = ema_signal[:-1]
        ema_signal = ema_signal_shifted

    # Apply cutout
    ema_signal[:cutout] = 0.0
    ema_equity[:cutout] = starting_equity

    return ema_signal, ema_equity


def compute_ema_only_atc(
    prices: np.ndarray,
    ema_len: int = 28,
    lambda_param: float = 0.02,
    decay: float = 0.03,
    long_threshold: float = 0.1,
    short_threshold: float = -0.1,
    cutout: int = 0,
    strategy_mode: bool = False,
) -> Tuple[np.ndarray, np.ndarray]:
    """Compute ATC signals for EMA-only configuration.

    This is a Python wrapper that calls the JIT-compiled implementation.
    The JIT compilation happens on first call, subsequent calls are fast.

    Args:
        prices: Price array
        ema_len: EMA period (default: 28)
        lambda_param: Lambda parameter (default: 0.02)
        decay: Decay parameter (default: 0.03)
        long_threshold: Long signal threshold (default: 0.1)
        short_threshold: Short signal threshold (default: -0.1)
        cutout: Number of bars to skip (default: 0)
        strategy_mode: If True, shift signal by 1 bar (default: False)

    Returns:
        Tuple of (ema_signal, ema_equity) arrays

    Raises:
        ValueError: If prices is empty or ema_len <= 0
    """
    if not NUMBA_AVAILABLE:
        raise ImportError("Numba is required for JIT specialization. Install with: pip install numba")

    if len(prices) == 0:
        raise ValueError("Price array cannot be empty")

    if ema_len <= 0:
        raise ValueError("EMA length must be positive")

    # Scale parameters
    lambda_scaled = lambda_param / 1000.0
    decay_scaled = decay / 100.0

    # Call JIT-compiled function
    ema_signal, ema_equity = compute_ema_only_atc_jit(
        prices.astype(np.float64),
        ema_len,
        lambda_scaled,
        decay_scaled,
        long_threshold,
        short_threshold,
        cutout,
        strategy_mode,
    )

    return ema_signal, ema_equity


__all__ = [
    "compute_ema_only_atc",
    "compute_ema_only_atc_jit",
    "compute_ema_jit",
]
