"""
Set of Moving Averages Calculation (Enhanced).

This module provides the logic to calculate a "set" of moving averages, which consists
of 9 variations of a base moving average (1 central + 4 larger + 4 smaller).
These variations are generated based on a 'robustness' parameter and are used
to create the adaptive bands for the trend classification.

Key Features:
- Calculates 9 MAs in parallel using ThreadPoolExecutor.
- Supports specific GPU acceleration via `prefer_gpu` flag.
- Integrates with HardwareManager for optimal workload distribution.
"""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from typing import Optional, Tuple

import pandas as pd

from modules.adaptive_trend_enhance.utils import diflen
from modules.common.system import get_hardware_manager
from modules.common.utils import log_error, log_warn

from .ma_calculation_enhanced import ma_calculation_enhanced


def set_of_moving_averages_enhanced(
    length: int,
    source: pd.Series,
    ma_type: str,
    robustness: str = "Medium",
    use_cache: bool = True,
    use_parallel: bool = True,
    prefer_gpu: bool = True,
) -> Optional[Tuple[pd.Series, ...]]:
    """
    Calculate a set of 9 moving averages with varying lengths based on robustness.

    The set includes:
    1. The primary MA with the specified length.
    2. 4 MAs with lengths increasing from the primary length (MA1-MA4).
    3. 4 MAs with lengths decreasing from the primary length (MA_1-MA_4).

    The variations are determined by the `diflen` utility based on the `robustness` level.

    Args:
        length: Base length for the moving average.
        source: Input price series.
        ma_type: Type of moving average ("EMA", "HMA", etc.).
        robustness: "Narrow", "Medium", or "Wide" (default: "Medium").
            Controls the spread of the varying lengths.
        use_cache: If True, uses cached results if available (default: True).
        use_parallel: If True, computes the 9 MAs in parallel threads (default: True).
        prefer_gpu: If True, attempts to use GPU acceleration (default: True).

    Returns:
        Tuple of 9 pandas Series: (MA, MA1, MA2, MA3, MA4, MA_1, MA_2, MA_3, MA_4)
        Returns None if calculation fails or input is invalid.

    Raises:
        TypeError: If source is not a pandas Series.
        ValueError: If length <= 0 or if any calculation fails.
    """
    if not isinstance(source, pd.Series):
        raise TypeError(f"source must be a pandas Series, got {type(source)}")

    if len(source) == 0:
        log_warn("Empty source series for set_of_moving_averages")
        return None

    if length <= 0:
        raise ValueError(f"length must be > 0, got {length}")

    VALID_ROBUSTNESS = {"Narrow", "Medium", "Wide"}
    if robustness not in VALID_ROBUSTNESS:
        log_warn(f"Invalid robustness '{robustness}'. Using 'Medium'.")
        robustness = "Medium"

    try:
        # REMOVED: with track_memory("set_of_moving_averages"):
        L1, L2, L3, L4, L_1, L_2, L_3, L_4 = diflen(length, robustness=robustness)

        ma_lengths = [length, L1, L2, L3, L4, L_1, L_2, L_3, L_4]
        ma_names = ["MA", "MA1", "MA2", "MA3", "MA4", "MA_1", "MA_2", "MA_3", "MA_4"]

        if any(len_val <= 0 for len_val in ma_lengths):
            invalid = [ma_l for ma_l in ma_lengths if ma_l <= 0]
            raise ValueError(f"Invalid length offsets: {invalid}")

        if use_parallel:
            hw_mgr = get_hardware_manager()
            # Note: get_optimal_workload_config might still call get_resources(), but it's once per MA set (6 per symbol)
            # We could optimize this too but it's less critical than the inner logs.
            config = hw_mgr.get_optimal_workload_config(workload_size=9, prefer_gpu=prefer_gpu)

            with ThreadPoolExecutor(max_workers=config.num_threads) as executor:
                futures = [
                    executor.submit(ma_calculation_enhanced, source, ma_len, ma_type, use_cache, prefer_gpu)
                    for ma_len in ma_lengths
                ]
                mas = [f.result() for f in futures]
        else:
            mas = [ma_calculation_enhanced(source, ma_len, ma_type, use_cache, prefer_gpu) for ma_len in ma_lengths]

        failed_calculations = [f"{ma_names[i]} (length={ma_lengths[i]})" for i, ma in enumerate(mas) if ma is None]

        if failed_calculations:
            failed_list = ", ".join(failed_calculations)
            error_msg = (
                f"Failed to calculate {len(failed_calculations)} out of 9 MAs "
                f"for ma_type={ma_type}, length={length}. "
                f"Failed: {failed_list}"
            )
            log_error(error_msg)
            raise ValueError(error_msg)

        MA, MA1, MA2, MA3, MA4, MA_1, MA_2, MA_3, MA_4 = mas
        return MA, MA1, MA2, MA3, MA4, MA_1, MA_2, MA_3, MA_4
    except Exception as e:
        log_error(f"Error calculating set of moving averages: {e}")
        raise


__all__ = ["set_of_moving_averages_enhanced"]
