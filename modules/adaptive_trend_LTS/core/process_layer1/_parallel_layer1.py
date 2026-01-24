"""Parallel implementation of Layer 1 ATC signals using ProcessPoolExecutor and shared memory.

This module provides the _layer1_parallel_atc_signals function to calculate
Layer 1 signals for all MA types in parallel.
"""

from __future__ import annotations

import multiprocessing as mp
from concurrent.futures import ProcessPoolExecutor, as_completed
from typing import Dict, List, Optional, Tuple

import pandas as pd

from modules.adaptive_trend_enhance.core.process_layer1.layer1_signal import _layer1_signal_for_ma
from modules.common.system.shared_memory_utils import (
    cleanup_shared_memory,
    reconstruct_series_from_shared_memory,
    setup_shared_memory_for_series,
)
from modules.common.utils import log_warn


def _layer1_worker(
    ma_type: str,
    ma_tuple: Tuple[pd.Series, ...],
    prices_shm_info: Dict,
    r_shm_info: Dict,
    l_val: float,
    de_val: float,
    precision: str = "float64",
) -> Tuple[str, pd.Series]:
    """Worker function for parallel Layer 1 signal calculation.

    Args:
        ma_type: Type of MA (e.g., "EMA")
        ma_tuple: Tuple of 9 MA Series
        prices_shm_info: Shared memory info for prices Series
        r_shm_info: Shared memory info for R (rate of change) Series
        l_val: Lambda value
        de_val: Decay value
        cutout: Cutout bars

    Returns:
        Tuple of (ma_type, signal_series)
    """
    # Reconstruct Series from shared memory
    prices = reconstruct_series_from_shared_memory(prices_shm_info)
    r_series = reconstruct_series_from_shared_memory(r_shm_info)

    # Calculate Layer 1 signal
    signal, _, _ = _layer1_signal_for_ma(
        prices=prices,
        ma_tuple=ma_tuple,
        L=l_val,
        De=de_val,
        R=r_series,
    )

    return ma_type, signal


def _layer1_parallel_atc_signals(
    prices: pd.Series,
    ma_tuples: Dict[str, Tuple[pd.Series, ...]],
    ma_configs: List[Tuple[str, int, float]],
    R: pd.Series,
    L: float,
    De: float,
    max_workers: Optional[int] = None,
    precision: str = "float64",
) -> Dict[str, pd.Series]:
    """Calculate Layer 1 signals for all MA types in parallel using shared memory.

    Args:
        prices: Price Series
        ma_tuples: Dictionary of MA tuples keyed by MA type
        ma_configs: List of (ma_type, length, weight) tuples
        R: Rate of change Series
        L: Lambda value
        De: Decay value
        cutout: Cutout bars
        max_workers: Maximum number of worker processes

    Returns:
        Dictionary of Layer 1 signals keyed by MA type
    """
    if max_workers is None:
        max_workers = min(len(ma_configs), mp.cpu_count())

    layer1_signals: Dict[str, pd.Series] = {}

    # Setup shared memory for inputs
    prices_shm = setup_shared_memory_for_series(prices)
    r_shm = setup_shared_memory_for_series(R)

    try:
        # Use ProcessPoolExecutor for CPU-bound parallel processing
        with ProcessPoolExecutor(max_workers=max_workers) as executor:
            futures = {
                executor.submit(
                    _layer1_worker,
                    ma_type,
                    ma_tuples[ma_type],
                    prices_shm,
                    r_shm,
                    L,
                    De,
                    precision,
                ): ma_type
                for ma_type, _, _ in ma_configs
                if ma_type in ma_tuples
            }

            for future in as_completed(futures):
                ma_type = futures[future]
                try:
                    res_ma_type, signal = future.result()
                    layer1_signals[res_ma_type] = signal
                except Exception as e:
                    log_warn(f"Error in Layer 1 worker for {ma_type}: {e}")
                    # Fallback to sequential for this specific MA if it failed
                    signal, _, _ = _layer1_signal_for_ma(
                        prices=prices,
                        ma_tuple=ma_tuples[ma_type],
                        L=L,
                        De=De,
                        R=R,
                    )
                    layer1_signals[ma_type] = signal

    finally:
        # Always cleanup shared memory
        cleanup_shared_memory(prices_shm)
        cleanup_shared_memory(r_shm)

    return layer1_signals
