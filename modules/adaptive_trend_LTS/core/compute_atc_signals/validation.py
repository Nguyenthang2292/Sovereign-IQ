"""Input validation utilities for ATC signal computation.

This module provides validation functions to ensure inputs to compute_atc_signals
are valid before processing.
"""

from __future__ import annotations

from typing import Optional

import pandas as pd

try:
    from modules.common.utils import log_error, log_warn
except ImportError:
    # Fallback logging if common utils not available
    def log_warn(msg: str) -> None:  # pragma: no cover
        print(f"[WARN] {msg}")

    def log_error(msg: str) -> None:  # pragma: no cover
        print(f"[ERROR] {msg}")


def validate_atc_inputs(
    prices: Optional[pd.Series],
    src: Optional[pd.Series],
    robustness: str,
    cutout: int,
) -> tuple[pd.Series, pd.Series, str, int]:
    """Validate and normalize inputs for ATC computation.

    Args:
        prices: Price series.
        src: Source series (optional, defaults to prices).
        robustness: Robustness level ("Narrow", "Medium", "Wide").
        cutout: Number of bars to skip at beginning.

    Returns:
        Tuple of (validated_prices, validated_src, validated_robustness, validated_cutout).

    Raises:
        ValueError: If any input is invalid.
    """
    # Validate prices
    if prices is None or len(prices) == 0:
        log_error("prices cannot be empty or None")
        raise ValueError("prices cannot be empty or None")

    # Validate src
    if src is None:
        src = prices

    if len(src) == 0:
        log_error(f"src cannot be empty. Type of src: {type(src)}, Type of prices: {type(prices)}")
        if hasattr(src, "shape"):
            log_error(f"src shape: {src.shape}")
        if hasattr(prices, "shape"):
            log_error(f"prices shape: {prices.shape}")
        raise ValueError("src cannot be empty")

    # Validate robustness
    if robustness not in ("Narrow", "Medium", "Wide"):
        log_warn(f"robustness '{robustness}' is invalid, using 'Medium'")
        robustness = "Medium"

    # Validate cutout
    if cutout < 0:
        log_warn(f"cutout {cutout} < 0, setting to 0")
        cutout = 0
    if cutout >= len(prices):
        log_error(f"cutout ({cutout}) >= prices length ({len(prices)})")
        raise ValueError(f"cutout ({cutout}) must be less than prices length ({len(prices)})")

    return prices, src, robustness, cutout
