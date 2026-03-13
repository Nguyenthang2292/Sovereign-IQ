"""Constants and utility functions for incremental ATC computation."""

from __future__ import annotations

import numpy as np
from typing import Dict, Any

# Timeframe resolution mapping (minutes)
TF_RESOLUTION_MAP = {
    "1m": 1,
    "5m": 5,
    "15m": 15,
    "30m": 30,
    "1h": 60,
    "4h": 240,
    "1d": 1440,
}

# Robustness offsets for MA length calculations
ROBUSTNESS_OFFSETS = {
    "Narrow": 4,
    "Medium": 6,
    "Wide": 7,
}

# Maximum exponent for growth factor calculation
MAX_EXPONENT = 20.0


def get_initial_weights(config: Dict[str, Any]) -> Dict[str, float]:
    """Get initial equity weights per MA type from config."""
    return {
        "EMA": float(config.get("ema_w", 1.0)),
        "HMA": float(config.get("hma_w", 1.0)),
        "WMA": float(config.get("wma_w", 1.0)),
        "DEMA": float(config.get("dema_w", 1.0)),
        "LSMA": float(config.get("lsma_w", 1.0)),
        "KAMA": float(config.get("kama_w", 1.0)),
    }


def get_scaled_params(config: Dict[str, Any]) -> tuple[float, float]:
    """Return scaled lambda and decay parameters (PineScript-compatible)."""
    la = config.get("La", config.get("lambda_param", 0.02))
    de = config.get("De", config.get("decay", 0.03))
    return la / 1000.0, de / 100.0


def calculate_growth_factor(bar_index: int, cutout: int, L: float) -> float:
    """Compute exp growth factor for a single bar index.

    FIX #5: Improved overflow handling with reasonable upper bound.

    Args:
        bar_index: Current bar index
        cutout: Cutout threshold
        L: Scaled lambda parameter

    Returns:
        Growth factor value
    """
    bar_val = 1 if bar_index == 0 else bar_index
    if bar_val < cutout:
        return 1.0

    exponent = L * (bar_val - cutout)

    # FIX #5: Prevent extreme growth factors
    # Cap exponent at 20 to prevent overflow (e^20 ≈ 4.8e8)
    # NOTE: Growth Overflow Prevention
    # This cap at e^20 (~485 million) is a safeguard against numerical instability
    # in very long backtests. While 485M seems large, it prevents float64 overflow
    # and keeps equity values within manageable ranges.
    if exponent > MAX_EXPONENT:
        try:
            from modules.common.utils import log_warn
        except ImportError:

            def log_warn(msg: str, *args: object) -> None:
                print(f"[WARN] {msg % args if args else msg}")

        log_warn(
            f"Growth factor exponent {exponent:.2f} exceeds maximum {MAX_EXPONENT}. "
            f"Capping to prevent overflow (bar_index={bar_index}, L={L})."
        )
        exponent = MAX_EXPONENT

    try:
        growth = float(np.exp(exponent))
        # Additional safety check
        if not np.isfinite(growth):
            try:
                from modules.common.utils import log_warn
            except ImportError:

                def log_warn(msg: str, *args: object) -> None:
                    print(f"[WARN] {msg % args if args else msg}")

            log_warn(f"Growth factor is not finite: {growth}, using max safe value")
            growth = np.exp(MAX_EXPONENT)
        return growth
    except OverflowError:
        # Fallback to max safe value
        try:
            from modules.common.utils import log_warn
        except ImportError:

            def log_warn(msg: str, *args: object) -> None:
                print(f"[WARN] {msg % args if args else msg}")

        log_warn("OverflowError in growth factor calculation, using max safe value")
        return float(np.exp(MAX_EXPONENT))
