"""
Layer 2 equity calculation utilities for Adaptive Trend Classification (ATC).
"""

from __future__ import annotations

from typing import Dict

import pandas as pd

try:
    from modules.common.utils import log_debug, log_warn
except ImportError:
    # Fallback logging if common utils not available
    def log_debug(msg: str) -> None:  # pragma: no cover
        print(f"[DEBUG] {msg}")

    def log_warn(msg: str) -> None:  # pragma: no cover
        print(f"[WARN] {msg}")

from modules.adaptive_trend_enhance.core.compute_equity import equity_series
from modules.adaptive_trend_enhance.core.hardware_manager import get_hardware_manager
from modules.adaptive_trend_enhance.core.memory_manager import get_memory_manager

try:
    from concurrent.futures import ThreadPoolExecutor, as_completed

    THREADING_AVAILABLE = True
except ImportError:  # pragma: no cover
    THREADING_AVAILABLE = False


def calculate_layer2_equities(
    layer1_signals: Dict[str, pd.Series],
    ma_configs: list,
    R: pd.Series,
    L: float,
    De: float,
    cutout: int,
    parallel: bool = True,
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
        R: Rate of change series (calculated once and reused).
        L: Lambda (growth rate) for exponential growth factor.
        De: Decay factor for equity calculations.
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
    hw_manager = get_hardware_manager()

    with mem_manager.track_memory("calculate_layer2_equities"):
        if parallel and THREADING_AVAILABLE and len(ma_configs) > 1:
            # Parallel processing for multiple MA types
            def _calculate_equity_for_ma(ma_type, initial_weight):
                if ma_type not in layer1_signals:
                    return ma_type, None
                equity = equity_series(
                    starting_equity=initial_weight,
                    sig=layer1_signals[ma_type],
                    R=R,
                    L=L,
                    De=De,
                    cutout=cutout,
                )
                return ma_type, equity

            # Get optimal worker count
            config = hw_manager.get_optimal_workload_config(len(ma_configs))
            max_workers = min(config.num_threads, len(ma_configs))

            with ThreadPoolExecutor(max_workers=max_workers) as executor:
                futures = {
                    executor.submit(_calculate_equity_for_ma, ma_type, initial_weight): ma_type
                    for ma_type, _, initial_weight in ma_configs
                }

                for future in as_completed(futures):
                    ma_type, equity = future.result()
                    if equity is not None:
                        layer2_equities[ma_type] = equity
        else:
            # Sequential processing
            for ma_type, _, initial_weight in ma_configs:
                if ma_type not in layer1_signals:
                    log_warn(f"Layer 1 signal for {ma_type} not found, skipping")
                    continue

                equity = equity_series(
                    starting_equity=initial_weight,
                    sig=layer1_signals[ma_type],
                    R=R,
                    L=L,
                    De=De,
                    cutout=cutout,
                )
                layer2_equities[ma_type] = equity

    log_debug("Completed Layer 2 equity weights")
    return layer2_equities


__all__ = ["calculate_layer2_equities"]
