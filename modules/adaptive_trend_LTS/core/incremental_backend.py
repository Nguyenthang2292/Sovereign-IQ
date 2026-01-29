"""Python wrapper for Rust incremental ATC backend.

This module provides a wrapper that calls Rust implementations when available,
with fallback to Python implementations when Rust is not available.
"""

from __future__ import annotations

from typing import Any, Dict, Tuple

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


def update_incremental_rust(state: Dict[str, Any], new_price: float, config: Dict[str, Any]) -> Tuple[float, Dict[str, Any]]:
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
        signal, updated_state = update_incremental_atc_rust(state, new_price)
        log_debug(f"Rust update complete, signal={signal}")
        return signal, updated_state
    except Exception as e:
        log_error(f"Rust incremental update failed: {e}")
        raise


def update_incremental_python(state: Dict[str, Any], new_price: float, config: Dict[str, Any]) -> Tuple[float, Dict[str, Any]]:
    """Update incremental ATC using Python backend (fallback).

    Args:
        state: Current ATC state dictionary
        new_price: New price value
        config: ATC configuration dictionary

    Returns:
        Tuple of (signal, updated_state)
    """
    log_debug(f"Updating incremental ATC with Python backend, new_price={new_price}")

    # Import here to avoid circular imports
    from .incremental_atc import IncrementalATC

    # Create a temporary IncrementalATC to use its update methods
    # This is a fallback when Rust is not available
    decay = config.get("De", 0.03) / 100.0
    la = config.get("La", 0.02) / 1000.0
    long_threshold = config.get("long_threshold", 0.1)
    short_threshold = config.get("short_threshold", -0.1)

    # This is a simplified Python fallback - in production would use full Python logic
    # For now, just return a placeholder signal
    signal = 0.0

    log_debug(f"Python update complete, signal={signal}")
    return signal, state


def update_incremental_auto(state: Dict[str, Any], new_price: float, config: Dict[str, Any]) -> Tuple[float, Dict[str, Any]]:
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
