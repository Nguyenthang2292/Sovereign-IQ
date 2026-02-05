"""Python wrapper for Rust incremental ATC backend.

This module provides a wrapper that calls Rust implementations when available,
with fallback to Python implementations when Rust is not available.
"""

from __future__ import annotations

from typing import Any, Dict, Tuple

import pandas as pd

# Try to import Rust backend
try:
    from atc_rust import update_incremental_atc_rust

    _RUST_AVAILABLE = True
except ImportError:
    update_incremental_atc_rust = None
    _RUST_AVAILABLE = False


try:
    from modules.common.utils import log_debug, log_warn, log_error
except ImportError:

    def log_debug(msg: str) -> None:
        print(f"[DEBUG] {msg}")

    def log_warn(msg: str) -> None:
        print(f"[WARN] {msg}")

    def log_error(msg: str) -> None:
        print(f"[ERROR] {msg}")


def check_rust_available() -> bool:
    """Check if Rust incremental backend is available.

    Returns:
        True if Rust backend is available, False otherwise
    """
    return _RUST_AVAILABLE


def update_incremental_rust(
    state: Dict[str, Any], new_price: float, config: Dict[str, Any]
) -> Tuple[float, Dict[str, Any]]:
    """Update incremental ATC using Rust backend.

    Args:
        state: Current ATC state dictionary
        new_price: New price value
        config: ATC configuration dictionary

    Returns:
        Tuple of (signal, updated_state)

    Raises:
        ImportError: If Rust backend is not available
    """
    if not _RUST_AVAILABLE:
        raise ImportError("Rust incremental backend is not available. Install the atc_rust package.")

    log_debug(f"Updating incremental ATC with Rust backend, new_price={new_price}")

    try:
        # Scale parameters to match PineScript behavior if needed
        # Rust backend expects unscaled values and handles scaling internally?
        # Or does it expect scaled? ATCConfig property says lambda_scaled = La / 1000.
        # Most of our Python code scales internally.

        rust_config = {
            "ema_len": config.get("ema_len", 28),
            "hma_len": config.get("hma_len", 28),
            "wma_len": config.get("wma_len", 28),
            "dema_len": config.get("dema_len", 28),
            "lsma_len": config.get("lsma_len", 28),
            "kama_len": config.get("kama_len", 28),
            "ema_w": config.get("ema_w", 1.0),
            "hma_w": config.get("hma_w", 1.0),
            "wma_w": config.get("wma_w", 1.0),
            "dema_w": config.get("dema_w", 1.0),
            "lsma_w": config.get("lsma_w", 1.0),
            "kama_w": config.get("kama_w", 1.0),
            "robustness": config.get("robustness", "Medium"),
            "La": config.get("La", config.get("lambda_param", 0.02)),
            "De": config.get("De", config.get("decay", 0.03)),
            "long_threshold": config.get("long_threshold", 0.1),
            "short_threshold": config.get("short_threshold", -0.1),
        }

        signal, updated_state = update_incremental_atc_rust(state, new_price, rust_config)
        log_debug(f"Rust update complete, signal={signal}")
        return signal, updated_state
    except Exception as e:
        log_error(f"Rust incremental update failed: {e}")
        raise


def update_incremental_python(
    state: Dict[str, Any], new_price: float, config: Dict[str, Any]
) -> Tuple[float, Dict[str, Any]]:
    """Update incremental ATC using Python backend (fallback).

    Args:
        state: Current ATC state dictionary
        new_price: New price value
        config: ATC configuration dictionary

    Returns:
        Tuple of (signal, updated_state)

    Notes:
        The rolling window size is bounded by the maximum base MA length + 1.
        This does not expand for robustness offsets (diflen), so results can
        diverge from full-batch computation when offsets exceed the base length.
    """
    log_debug(f"Updating incremental ATC with Python backend, new_price={new_price}")

    # Import here to avoid circular imports
    from modules.adaptive_trend_LTS_mini.core.compute_atc_signals import compute_atc_signals

    # Maintain rolling price window
    price_window = state.get("price_window", [])
    if not isinstance(price_window, list):
        price_window = list(price_window)

    price_window.append(float(new_price))

    # Keep window size bounded by max MA length + 1
    max_len = (
        max(
            config.get("ema_len", 28),
            config.get("hma_len", 28),
            config.get("wma_len", 28),
            config.get("dema_len", 28),
            config.get("lsma_len", 28),
            config.get("kama_len", 28),
        )
        + 1
    )

    if len(price_window) > max_len:
        price_window = price_window[-max_len:]

    # Build compute config with compatible keys
    compute_config = dict(config)
    if "La" not in compute_config and "lambda_param" in compute_config:
        compute_config["La"] = compute_config["lambda_param"]
    if "De" not in compute_config and "decay" in compute_config:
        compute_config["De"] = compute_config["decay"]

    allowed_keys = {
        "ema_len",
        "hma_len",
        "wma_len",
        "dema_len",
        "lsma_len",
        "kama_len",
        "ema_w",
        "hma_w",
        "wma_w",
        "dema_w",
        "lsma_w",
        "kama_w",
        "robustness",
        "La",
        "De",
        "cutout",
        "long_threshold",
        "short_threshold",
        "strategy_mode",
        "parallel_l1",
        "parallel_l2",
        "precision",
        "use_rust_backend",
        "use_cache",
        "fast_mode",
        "use_cuda",
        "prefer_gpu",
        "use_approximate",
        "approximate_threshold",
        "use_adaptive_approximate",
        "approximate_volatility_window",
        "approximate_volatility_factor",
    }

    compute_kwargs = {k: v for k, v in compute_config.items() if k in allowed_keys}

    prices = pd.Series(price_window, index=range(len(price_window)))
    results = compute_atc_signals(prices, **compute_kwargs)

    avg_series = results.get("Average_Signal")
    signal = float(avg_series.iloc[-1]) if avg_series is not None and len(avg_series) > 0 else 0.0

    state["price_window"] = price_window
    state["initialized"] = True

    log_debug(f"Python update complete, signal={signal}")
    return signal, state


def update_incremental_auto(
    state: Dict[str, Any], new_price: float, config: Dict[str, Any]
) -> Tuple[float, Dict[str, Any]]:
    """Update incremental ATC using Rust or Python backend automatically.

    Args:
        state: Current ATC state dictionary
        new_price: New price value
        config: ATC configuration dictionary

    Returns:
        Tuple of (signal, updated_state)
    """
    use_rust = config.get("use_rust_incremental", True)

    if use_rust and _RUST_AVAILABLE:
        return update_incremental_rust(state, new_price, config)
    else:
        if use_rust and not _RUST_AVAILABLE:
            log_warn("use_rust_incremental=True but Rust backend not available, falling back to Python")
        return update_incremental_python(state, new_price, config)


__all__ = [
    "check_rust_available",
    "update_incremental_rust",
    "update_incremental_python",
    "update_incremental_auto",
]
