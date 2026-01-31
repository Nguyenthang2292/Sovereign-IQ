"""
Layer 2 equity calculation utilities for Adaptive Trend Classification (ATC).
"""

from __future__ import annotations

from typing import Dict

import numpy as np
import pandas as pd

try:
    from modules.common.utils import log_debug, log_warn
except ImportError:
    # Fallback logging if common utils not available
    def log_debug(msg: str) -> None:  # pragma: no cover
        print(f"[DEBUG] {msg}")

    def log_warn(msg: str) -> None:  # pragma: no cover
        print(f"[WARN] {msg}")


from modules.adaptive_trend_enhance.core.compute_equity import _calculate_equities_parallel, equity_series
from modules.common.system import get_memory_manager, get_series_pool, temp_series


@temp_series
def calculate_layer2_equities(
    layer1_signals: Dict[str, pd.Series],
    ma_configs: list,
    R: pd.Series,
    L: float,
    De: float,
    parallel: bool = True,
    precision: str = "float64",
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

    with mem_manager.track_memory("calculate_layer2_equities"):
        if parallel and len(ma_configs) > 1:
            # Prepare batch data for vectorized parallel calculation
            dtype = np.float32 if precision == "float32" else np.float64
            ma_types = [cfg[0] for cfg in ma_configs if cfg[0] in layer1_signals]
            initial_weights = np.array([cfg[2] for cfg in ma_configs if cfg[0] in layer1_signals], dtype=dtype)

            if not ma_types:
                return {}

            # Stack signals into matrix (n_signals, n_bars)
            n_bars = len(R)
            n_signals = len(ma_types)
            n_bars = len(R)
            n_signals = len(ma_types)
            signals_matrix = np.empty((n_signals, n_bars), dtype=dtype)

            for i, ma_type in enumerate(ma_types):
                signals_matrix[i] = layer1_signals[ma_type].values

            # Shift signals by 1 period (sig[1] in Pine Script)
            # Parallel worker will handle this if we shift here or inside
            # _calculate_equities_parallel assumes sig_prev_values is already shifted!
            # Let's check _calculate_equity_core in equity_series.py:
            # sig_shifted = sig.shift(1)
            # ... calls _calculate_equity_core(sig_prev_values=sig_shifted.values)

            signals_prev = np.empty_like(signals_matrix)
            signals_prev[:, 1:] = signals_matrix[:, :-1]
            # First value is not used in equity calculation, but set to 0.0 to avoid nan
            signals_prev[:, 0] = 0.0

            # Get growth factor
            from modules.adaptive_trend.utils import exp_growth

            growth = exp_growth(L=L, index=R.index, cutout=0)
            r_adjusted = (R * growth).values
            d = 1.0 - De

            # Calculate all equities in parallel using Numba
            # Result matrix (n_signals, n_bars)
            equity_matrix = _calculate_equities_parallel(
                starting_equities=initial_weights,
                sig_prev_values=signals_prev,
                r_values=r_adjusted,
                decay_multiplier=d,
                cutout=0,
            )

            # Replace any NaN with 0.0 for consistency
            equity_matrix = np.nan_to_num(equity_matrix, nan=0.0)

            # Convert back to dictionary of Series
            series_pool = get_series_pool()
            for i, ma_type in enumerate(ma_types):
                # Acquire from pool to be efficient
                # Re-wrapping the data into a Series with appropriate index
                # Note: We use the pooled series to avoid creating new pd.Index objects if possible,
                # but pandas often creates them anyway. The key is reusing the backing storage.
                equity_series_obj = series_pool.acquire(n_bars, dtype=dtype, index=R.index)
                equity_series_obj[:] = equity_matrix[i]
                layer2_equities[ma_type] = equity_series_obj
        else:
            # Sequential processing (fallback or single MA)
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
                )
                layer2_equities[ma_type] = equity

    log_debug("Completed Layer 2 equity weights")
    return layer2_equities


__all__ = ["calculate_layer2_equities"]
