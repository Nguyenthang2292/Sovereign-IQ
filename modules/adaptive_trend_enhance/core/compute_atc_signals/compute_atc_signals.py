"""Adaptive Trend Classification (ATC) - Main computation entrypoint.

This module orchestrates the full ATC pipeline:
1. Input validation
2. Moving averages computation
3. Layer 1 signal calculation
4. Layer 2 equity calculation
5. Final Average_Signal calculation
"""

from __future__ import annotations

from typing import Dict, Optional

import pandas as pd

try:
    from modules.common.utils import log_debug, log_error, log_info
except ImportError:
    # Fallback logging if common utils not available
    def log_debug(msg: str) -> None:  # pragma: no cover
        print(f"[DEBUG] {msg}")

    def log_info(msg: str) -> None:  # pragma: no cover
        print(f"[INFO] {msg}")

    def log_error(msg: str) -> None:  # pragma: no cover
        print(f"[ERROR] {msg}")


from modules.adaptive_trend_enhance.core.compute_moving_averages import set_of_moving_averages
from modules.adaptive_trend_enhance.core.process_layer1 import _layer1_signal_for_ma
from modules.adaptive_trend_enhance.utils.rate_of_change import rate_of_change
from modules.common.system import cleanup_series, get_memory_manager, get_series_pool, temp_series

from .average_signal import calculate_average_signal
from .calculate_layer2_equities import calculate_layer2_equities
from .validation import validate_atc_inputs


@temp_series
def compute_atc_signals(
    prices: pd.Series,
    src: Optional[pd.Series] = None,
    *,
    ema_len: int = 28,
    hull_len: int = 28,
    wma_len: int = 28,
    dema_len: int = 28,
    lsma_len: int = 28,
    kama_len: int = 28,
    ema_w: float = 1.0,
    hma_w: float = 1.0,
    wma_w: float = 1.0,
    dema_w: float = 1.0,
    lsma_w: float = 1.0,
    kama_w: float = 1.0,
    robustness: str = "Medium",
    La: float = 0.02,
    De: float = 0.03,
    cutout: int = 0,
    long_threshold: float = 0.1,
    short_threshold: float = -0.1,
    strategy_mode: bool = False,
) -> dict[str, pd.Series]:
    """Compute Adaptive Trend Classification (ATC) signals.

    Args:
        prices: Price series for ATC calculation.
        src: Source series (optional, defaults to prices).
        ema_len: EMA length (default: 28).
        hull_len: HMA length (default: 28).
        wma_len: WMA length (default: 28).
        dema_len: DEMA length (default: 28).
        lsma_len: LSMA length (default: 28).
        kama_len: KAMA length (default: 28).
        ema_w: EMA initial weight (default: 1.0).
        hma_w: HMA initial weight (default: 1.0).
        wma_w: WMA initial weight (default: 1.0).
        dema_w: DEMA initial weight (default: 1.0).
        lsma_w: LSMA initial weight (default: 1.0).
        kama_w: KAMA initial weight (default: 1.0).
        robustness: Robustness level - "Narrow", "Medium", or "Wide" (default: "Medium").
        La: Lambda parameter (default: 0.02).
        De: Decay parameter (default: 0.03).
        cutout: Number of bars to skip at beginning (default: 0).
        long_threshold: Threshold for LONG signals (default: 0.1).
        short_threshold: Threshold for SHORT signals (default: -0.1).
        strategy_mode: If True, shift signal by 1 bar (default: False).

    Returns:
        Dictionary containing:
        - {MA_TYPE}_Signal: Layer 1 signal for each MA type
        - {MA_TYPE}_S: Layer 2 equity for each MA type
        - Average_Signal: Final weighted average signal

    Raises:
        ValueError: If inputs are invalid.
    """
    log_debug(f"Starting ATC signal computation for {len(prices)} bars")

    # Validate inputs
    prices, src, robustness, cutout = validate_atc_inputs(prices, src, robustness, cutout)

    # Apply PineScript scaling to Lambda and Decay
    La_scaled = La / 1000.0
    De_scaled = De / 100.0

    log_info(
        f"Parameters: robustness={robustness}, La_scaled={La_scaled}, De_scaled={De_scaled}, "
        f"cutout={cutout}, strategy_mode={strategy_mode}"
    )

    # Define configuration for each MA type
    ma_configs = [
        ("EMA", ema_len, ema_w),
        ("HMA", hull_len, hma_w),
        ("WMA", wma_len, wma_w),
        ("DEMA", dema_len, dema_w),
        ("LSMA", lsma_len, lsma_w),
        ("KAMA", kama_len, kama_w),
    ]

    # Use memory manager for orchestration
    mem_manager = get_memory_manager()

    # DECLARE MOVING AVERAGES (SetOfMovingAverages)
    log_debug("Computing Moving Averages...")
    ma_tuples: Dict[str, tuple] = {}
    with mem_manager.track_memory("set_of_moving_averages_all"):
        for ma_type, length, _ in ma_configs:
            ma_tuple = set_of_moving_averages(length, src, ma_type, robustness=robustness)
            if ma_tuple is None:
                log_error(f"Cannot compute {ma_type} with length={length}")
                raise ValueError(f"Cannot compute {ma_type} with length={length}")
            ma_tuples[ma_type] = ma_tuple
    log_debug(f"Computed {len(ma_tuples)} MA types")

    # MAIN CALCULATIONS - Adaptability Layer 1
    log_debug("Computing rate_of_change (reused for Layer 1 and Layer 2)...")
    R = rate_of_change(prices)

    log_debug("Computing Layer 1 signals...")
    layer1_signals: Dict[str, pd.Series] = {}
    with mem_manager.track_memory("layer1_signals_all"):
        series_pool = get_series_pool()
        for ma_type, _, _ in ma_configs:
            signal, signals_tuple, equity_tuple = _layer1_signal_for_ma(
                prices, ma_tuples[ma_type], L=La_scaled, De=De_scaled, cutout=cutout, R=R
            )
            layer1_signals[ma_type] = signal

            # Release intermediate component signals and equities back to pool
            # These are the 9 variants (s, s1...) and (E, E1...) which are not needed individually anymore
            for s in signals_tuple:
                series_pool.release(s)
            for e in equity_tuple:
                series_pool.release(e)

    log_debug("Completed Layer 1 signals")

    # Adaptability Layer 2
    layer2_equities = calculate_layer2_equities(
        layer1_signals=layer1_signals,
        ma_configs=ma_configs,
        R=R,
        L=La_scaled,
        De=De_scaled,
        cutout=cutout,
    )

    # FINAL CALCULATIONS - Average Signal
    Average_Signal = calculate_average_signal(
        layer1_signals=layer1_signals,
        layer2_equities=layer2_equities,
        ma_configs=ma_configs,
        prices=prices,
        long_threshold=long_threshold,
        short_threshold=short_threshold,
        cutout=cutout,
        strategy_mode=strategy_mode,
    )

    # Build result dictionary
    result: Dict[str, pd.Series] = {}
    for ma_type, _, _ in ma_configs:
        result[f"{ma_type}_Signal"] = layer1_signals[ma_type]
        result[f"{ma_type}_S"] = layer2_equities[ma_type]

    result["Average_Signal"] = Average_Signal

    cleanup_series(R)
    log_info(f"Completed ATC signal computation for {len(prices)} bars")
    return result


__all__ = ["compute_atc_signals"]
