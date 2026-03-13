"""
Layer 2 equity calculation utilities for Adaptive Trend Classification (ATC).
"""

from __future__ import annotations

from typing import Dict, Optional

import pandas as pd

try:
    from modules.common.utils import log_debug, log_warn
except ImportError:
    # Fallback logging if common utils not available
    def log_debug(msg: str, *args: object) -> None:  # pragma: no cover
        print(f"[DEBUG] {msg}")

    def log_warn(msg: str, *args: object) -> None:  # pragma: no cover
        print(f"[WARN] {msg}")


from modules.adaptive_trend_LTS_mini.core.compute_equity import equity_series
from modules.common.system import get_memory_manager, temp_series


@temp_series
def calculate_layer2_equities(
    layer1_signals: Dict[str, pd.Series],
    ma_configs: list,
    rate_of_change_series: pd.Series,
    lambda_val: float,
    decay_val: float,
    cutout: int = 0,
    parallel: bool = True,
    precision: str = "float64",
    use_rust_backend: bool = True,
    floor_val: Optional[float] = None,
) -> Dict[str, pd.Series]:
    """
    Calculate Layer 2 equity curves based on Layer 1 signal performance.

    Port of Pine Script Layer 2 calculation:
        EMA_S = eq(ema_w,  EMA_Signal,  R)
        HMA_S = eq(hma_w,  HMA_Signal,  R)
        ...

    This function calculates the equity curve for each MA type based on how well
    its Layer 1 signal performed. The equity curve serves as a dynamic weight
    in the final signal aggregation.

    Args:
        layer1_signals: Dictionary of Layer 1 signals keyed by MA type (e.g., "EMA", "HMA").
        ma_configs: List of (ma_type, length, initial_weight) tuples.
        rate_of_change_series: Rate of change series (calculated once and reused).
        lambda_val: Lambda (growth rate) for exponential growth factor.
        decay_val: Decay factor for equity calculations.
        cutout: Number of bars to skip at beginning.
        parallel: If True, calculate equities in parallel (default: True).

    Returns:
        Dictionary of Layer 2 equity curves keyed by MA type.

    Raises:
        ValueError: If ma_configs contains invalid entries.
    """
    log_debug("Computing Layer 2 equity weights...")
    layer2_equities: Dict[str, pd.Series] = {}

    mem_manager = get_memory_manager()

    with mem_manager.track_memory("calculate_layer2_equities"):
        for ma_type, _, initial_weight in ma_configs:
            if ma_type not in layer1_signals:
                log_warn(f"Layer 1 signal for {ma_type} not found, skipping")
                continue

            equity = equity_series(
                starting_equity=initial_weight,
                sig=layer1_signals[ma_type],
                rate_of_change_series=rate_of_change_series,
                lambda_val=lambda_val,
                decay_val=decay_val,
                cutout=cutout,
                verbose=False,
            )
            layer2_equities[ma_type] = equity

    log_debug("Completed Layer 2 equity weights")
    return layer2_equities


__all__ = ["calculate_layer2_equities"]
