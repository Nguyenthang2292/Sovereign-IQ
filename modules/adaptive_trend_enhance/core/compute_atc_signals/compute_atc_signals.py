"""
Adaptive Trend Classification (ATC) - Main computation entrypoint.
"""

from __future__ import annotations

from typing import Dict, Optional

import numpy as np
import pandas as pd

try:
    from modules.common.utils import log_debug, log_error, log_info, log_warn
except ImportError:
    # Fallback logging if common utils not available
    def log_debug(msg: str) -> None:  # pragma: no cover
        print(f"[DEBUG] {msg}")

    def log_info(msg: str) -> None:  # pragma: no cover
        print(f"[INFO] {msg}")

    def log_warn(msg: str) -> None:  # pragma: no cover
        print(f"[WARN] {msg}")

    def log_error(msg: str) -> None:  # pragma: no cover
        print(f"[ERROR] {msg}")


from modules.adaptive_trend_enhance.core.compute_moving_averages import set_of_moving_averages
from modules.adaptive_trend_enhance.core.memory_manager import get_memory_manager
from modules.adaptive_trend_enhance.core.process_layer1 import _layer1_signal_for_ma, cut_signal
from modules.adaptive_trend_enhance.utils.rate_of_change import rate_of_change

from .calculate_layer2_equities import calculate_layer2_equities


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
    """
    Compute Adaptive Trend Classification (ATC) signals.
    """
    log_debug(f"Starting ATC signal computation for {len(prices)} bars")

    if prices is None or len(prices) == 0:
        log_error("prices cannot be empty or None")
        raise ValueError("prices cannot be empty or None")

    if src is None:
        src = prices

    if len(src) == 0:
        log_error("src cannot be empty")
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
        for ma_type, _, _ in ma_configs:
            signal, _, _ = _layer1_signal_for_ma(
                prices, ma_tuples[ma_type], L=La_scaled, De=De_scaled, cutout=cutout, R=R
            )
            layer1_signals[ma_type] = signal
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

    # FINAL CALCULATIONS - Vectorized for performance
    log_debug("Computing Average_Signal (vectorized)...")

    n_bars = len(prices)
    index = prices.index

    nom_array = np.zeros(n_bars, dtype=np.float64)
    den_array = np.zeros(n_bars, dtype=np.float64)

    for ma_type, _, _ in ma_configs:
        signal = layer1_signals[ma_type]
        equity = layer2_equities[ma_type]

        cut_sig = cut_signal(
            signal,
            long_threshold=long_threshold,
            short_threshold=short_threshold,
            cutout=cutout,
        )

        cut_sig_values = cut_sig.values
        equity_values = equity.values

        nom_array += cut_sig_values * equity_values
        den_array += equity_values

    with np.errstate(divide="ignore", invalid="ignore"):
        avg_signal_array = np.divide(nom_array, den_array)
        avg_signal_array = np.where(np.isfinite(avg_signal_array), avg_signal_array, 0.0)

    Average_Signal = pd.Series(avg_signal_array, index=index, dtype="float64")

    if strategy_mode:
        Average_Signal = Average_Signal.shift(1).fillna(0)

    zero_divisions = np.sum(den_array == 0)
    if zero_divisions > 0:
        log_warn(f"Detected {zero_divisions} division by zero cases, replaced with 0")

    log_debug("Completed Average_Signal")

    result: Dict[str, pd.Series] = {}
    for ma_type, _, _ in ma_configs:
        result[f"{ma_type}_Signal"] = layer1_signals[ma_type]
        result[f"{ma_type}_S"] = layer2_equities[ma_type]

    result["Average_Signal"] = Average_Signal

    log_info(f"Completed ATC signal computation for {len(prices)} bars")
    return result


__all__ = ["compute_atc_signals"]
