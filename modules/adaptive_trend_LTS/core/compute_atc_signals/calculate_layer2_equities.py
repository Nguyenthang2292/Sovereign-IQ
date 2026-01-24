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


from modules.adaptive_trend_enhance.core.compute_equity import _calculate_equities_parallel
from modules.adaptive_trend_LTS.core.rust_backend import calculate_equity
from modules.common.system import get_memory_manager, get_series_pool, temp_series


@temp_series
def calculate_layer2_equities(
    layer1_signals: Dict[str, pd.Series],
    ma_configs: list,
    R: pd.Series,
    L: float,
    De: float,
    cutout: int = 0,
    parallel: bool = True,
    precision: str = "float64",
    use_cuda: bool = False,
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
        if parallel and not use_cuda and len(ma_configs) > 1:
            # Prepare batch data for vectorized parallel calculation
            dtype = np.float32 if precision == "float32" else np.float64
            ma_types = [cfg[0] for cfg in ma_configs if cfg[0] in layer1_signals]
            initial_weights = np.array([cfg[2] for cfg in ma_configs if cfg[0] in layer1_signals], dtype=dtype)

            if not ma_types:
                return {}

            # Stack signals into matrix (n_signals, n_bars)
            n_bars = len(R)
            n_signals = len(ma_types)
            signals_matrix = np.empty((n_signals, n_bars), dtype=dtype)

            for i, ma_type in enumerate(ma_types):
                signals_matrix[i] = layer1_signals[ma_type].values

            # Shift signals by 1 period (sig[1] in Pine Script)
            # Parallel worker will handle this if we shift here or inside
            # _calculate_equities_parallel assumes sig_prev_values is already shifted!
            signals_prev = np.empty_like(signals_matrix)
            signals_prev[:, 1:] = signals_matrix[:, :-1]
            # First value is not used in equity calculation, but set to NaN to match Original
            signals_prev[:, 0] = np.nan

            # Get growth factor
            from modules.adaptive_trend.utils import exp_growth

            growth = exp_growth(L=L, index=R.index, cutout=cutout)
            r_adjusted = (R * growth).values
            d = 1.0 - De

            # Calculate all equities in parallel using Numba
            # Result matrix (n_signals, n_bars)
            equity_matrix = _calculate_equities_parallel(
                starting_equities=initial_weights,
                sig_prev_values=signals_prev,
                r_values=r_adjusted,
                decay_multiplier=d,
                cutout=cutout,
            )

            # Convert back to dictionary of Series
            series_pool = get_series_pool()
            for i, ma_type in enumerate(ma_types):
                # Acquire from pool to be efficient
                equity_series_obj = series_pool.acquire(n_bars, dtype=dtype, index=R.index)
                equity_series_obj.iloc[:] = equity_matrix[i]
                layer2_equities[ma_type] = equity_series_obj
        else:
            # Sequential processing (fallback or single MA)
            # R multiplied by e(L) (growth factor)
            from modules.adaptive_trend.utils import exp_growth

            growth = exp_growth(L=L, index=R.index, cutout=cutout)
            r_adjusted = R * growth
            d = 1.0 - De

            for ma_type, _, initial_weight in ma_configs:
                if ma_type not in layer1_signals:
                    log_warn(f"Layer 1 signal for {ma_type} not found, skipping")
                    continue

                sig = layer1_signals[ma_type]

                # Shift signals by 1 period (sig[1] in Pine Script)
                sig_shifted = sig.shift(1)  # Leave first as NaN to match Original

                # Calculate equity using Rust backend (handles CPU/CUDA internally)
                equity_values = calculate_equity(
                    r_values=r_adjusted.values,
                    sig_prev=sig_shifted.values,
                    starting_equity=initial_weight,
                    decay_multiplier=d,
                    cutout=cutout,
                    use_rust=True,
                    use_cuda=use_cuda,
                )

                equity = pd.Series(equity_values, index=R.index, dtype=np.float64)
                layer2_equities[ma_type] = equity

    log_debug("Completed Layer 2 equity weights")
    return layer2_equities


__all__ = ["calculate_layer2_equities"]
