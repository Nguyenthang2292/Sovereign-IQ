"""Adaptive Trend Classification (ATC) - Main computation entrypoint.

This module orchestrates the full ATC pipeline:
1. Input validation
2. Moving averages computation
3. Layer 1 signal calculation
4. Layer 2 equity calculation
5. Final Average_Signal calculation
"""

from __future__ import annotations

import warnings
from contextlib import nullcontext
from typing import Dict, List, Optional, Tuple

import pandas as pd

try:
    from modules.common.utils import log_debug, log_error, log_info, log_warn
except ImportError:
    # Fallback logging if common utils not available
    def log_debug(msg: str) -> None:  # pragma: no cover
        print(f"[DEBUG] {msg}")

    def log_info(msg: str) -> None:  # pragma: no cover
        print(f"[INFO] {msg}")

    def log_error(msg: str) -> None:  # pragma: no cover
        print(f"[ERROR] {msg}")

    def log_warn(msg: str) -> None:  # pragma: no cover
        print(f"[WARN] {msg}")


from modules.adaptive_trend_LTS_mini.core.compute_moving_averages import set_of_moving_averages
from modules.adaptive_trend_LTS_mini.core.process_layer1 import _layer1_signal_for_ma
from modules.adaptive_trend_LTS_mini.utils.diflen import diflen
from modules.adaptive_trend_LTS_mini.utils.rate_of_change import rate_of_change
from modules.common.system import (
    cleanup_series,
    get_hardware_manager,
    get_memory_manager,
    get_series_pool,
)

from .average_signal import calculate_average_signal
from .calculate_layer2_equities import calculate_layer2_equities
from .validation import validate_atc_inputs


def _validate_and_scale_params(
    prices: pd.Series,
    src: Optional[pd.Series],
    robustness: str,
    cutout: int,
    lambda_param: float,
    decay: float,
) -> Tuple[pd.Series, pd.Series, str, int, float, float]:
    """Validate and scale parameters for ATC computation."""
    prices, src, robustness, cutout = validate_atc_inputs(prices, src, robustness, cutout)

    # Validate to prevent double-scaling
    if lambda_param < 0.0001:
        log_warn(
            f"lambda_param={lambda_param} appears to be already scaled (expected unscaled value like 0.02). "
            f"Double-scaling will produce incorrect results. Using lambda_param as-is."
        )
    if decay < 0.0001:
        log_warn(
            f"decay={decay} appears to be already scaled (expected unscaled value like 0.03). "
            f"Double-scaling will produce incorrect results. Using decay as-is."
        )

    lambda_scaled = lambda_param / 1000.0
    decay_scaled = decay / 100.0

    return prices, src, robustness, cutout, lambda_scaled, decay_scaled


def _compute_ma_tuples(
    prices: pd.Series,
    src: Optional[pd.Series],
    ma_configs: List[Tuple[str, int, float]],
    robustness: str,
    use_adaptive_approximate: bool,
    use_approximate: bool,
    approximate_threshold: float,
    approximate_volatility_window: int,
    approximate_volatility_factor: float,
    use_rust_backend: bool,
    use_cache: bool,
    fast_mode: bool,
) -> Dict[str, tuple]:
    """Compute Moving Average tuples for all configured types."""
    ma_tuples: Dict[str, tuple] = {}
    mem_manager = get_memory_manager()
    # Avoid tracking overhead in performance-critical path when fast_mode is True
    context_ma = nullcontext() if fast_mode else mem_manager.track_memory("set_of_moving_averages_all")

    with context_ma:
        if use_adaptive_approximate:
            from modules.adaptive_trend_LTS_mini.core.compute_moving_averages.adaptive_approximate_mas import (
                adaptive_dema_approx,
                adaptive_ema_approx,
                adaptive_hma_approx,
                adaptive_kama_approx,
                adaptive_lsma_approx,
                adaptive_wma_approx,
            )

            ma_source = src if src is not None else prices

            def make_approx_tuple(func, length, **kwargs):
                _diflen = diflen(length, robustness=robustness)
                if _diflen is None:
                    raise ValueError(f"diflen returned None for length={length}")
                L1, L2, L3, L4, L_1, L_2, L_3, L_4 = _diflen
                lengths = [length, L1, L2, L3, L4, L_1, L_2, L_3, L_4]
                return tuple(func(ma_source, ln, **kwargs) for ln in lengths)

            ma_tuples["EMA"] = make_approx_tuple(
                adaptive_ema_approx,
                ma_configs[0][1],
                volatility_window=approximate_volatility_window,
                volatility_factor=approximate_volatility_factor,
                base_tolerance=approximate_threshold,
            )
            ma_tuples["HMA"] = make_approx_tuple(
                adaptive_hma_approx,
                ma_configs[1][1],
                volatility_window=approximate_volatility_window,
                volatility_factor=approximate_volatility_factor,
                base_tolerance=approximate_threshold,
            )
            ma_tuples["WMA"] = make_approx_tuple(
                adaptive_wma_approx,
                ma_configs[2][1],
                volatility_window=approximate_volatility_window,
                volatility_factor=approximate_volatility_factor,
                base_tolerance=approximate_threshold,
            )
            ma_tuples["DEMA"] = make_approx_tuple(
                adaptive_dema_approx,
                ma_configs[3][1],
                volatility_window=approximate_volatility_window,
                volatility_factor=approximate_volatility_factor,
                base_tolerance=approximate_threshold,
            )
            ma_tuples["LSMA"] = make_approx_tuple(
                adaptive_lsma_approx,
                ma_configs[4][1],
                volatility_window=approximate_volatility_window,
                volatility_factor=approximate_volatility_factor,
                base_tolerance=approximate_threshold,
            )
            ma_tuples["KAMA"] = make_approx_tuple(
                adaptive_kama_approx,
                ma_configs[5][1],
                volatility_window=approximate_volatility_window,
                volatility_factor=approximate_volatility_factor,
                base_tolerance=approximate_threshold,
            )
        elif use_approximate:
            from modules.adaptive_trend_LTS_mini.core.compute_moving_averages.approximate_mas import (
                fast_dema_approx,
                fast_ema_approx,
                fast_hma_approx,
                fast_kama_approx,
                fast_lsma_approx,
                fast_wma_approx,
            )

            ma_source_basic = src if src is not None else prices

            def make_approx_tuple_basic(func, length):
                _diflen = diflen(length, robustness=robustness)
                if _diflen is None:
                    raise ValueError(f"diflen returned None for length={length}")
                L1, L2, L3, L4, L_1, L_2, L_3, L_4 = _diflen
                lengths = [length, L1, L2, L3, L4, L_1, L_2, L_3, L_4]
                return tuple(func(ma_source_basic, leng, tolerance=approximate_threshold) for leng in lengths)

            ma_tuples["EMA"] = make_approx_tuple_basic(fast_ema_approx, ma_configs[0][1])
            ma_tuples["HMA"] = make_approx_tuple_basic(fast_hma_approx, ma_configs[1][1])
            ma_tuples["WMA"] = make_approx_tuple_basic(fast_wma_approx, ma_configs[2][1])
            ma_tuples["DEMA"] = make_approx_tuple_basic(fast_dema_approx, ma_configs[3][1])
            ma_tuples["LSMA"] = make_approx_tuple_basic(fast_lsma_approx, ma_configs[4][1])
            ma_tuples["KAMA"] = make_approx_tuple_basic(fast_kama_approx, ma_configs[5][1])
        else:
            source_series = src if src is not None else prices
            for ma_type, length, _ in ma_configs:
                ma_tuple = set_of_moving_averages(
                    length,
                    source=source_series,
                    ma_type=ma_type,
                    robustness=robustness,
                    use_cache=use_cache,
                    use_rust=use_rust_backend,
                )
                if ma_tuple is None:
                    log_error(f"Cannot compute {ma_type} with length={length}")
                    raise ValueError(f"Cannot compute {ma_type} with length={length}")
                ma_tuples[ma_type] = ma_tuple

    return ma_tuples


# Default thresholds for auto-enabling Layer 1 parallel processing (when parallel_l1 is None)
DEFAULT_MIN_BARS_PARALLEL_L1 = 500
DEFAULT_MIN_CORES_PARALLEL_L1 = 4


def _compute_layer1(
    prices: pd.Series,
    ma_tuples: Dict[str, tuple],
    ma_configs: List[Tuple[str, int, float]],
    rate_of_change_series: pd.Series,
    lambda_scaled: float,
    decay_scaled: float,
    parallel_l1: Optional[bool],
    fast_mode: bool,
    min_bars_parallel_l1: int = DEFAULT_MIN_BARS_PARALLEL_L1,
    min_cores_parallel_l1: int = DEFAULT_MIN_CORES_PARALLEL_L1,
) -> Dict[str, pd.Series]:
    """Compute Layer 1 signals."""
    layer1_signals: Dict[str, pd.Series] = {}
    mem_manager = get_memory_manager()
    context_l1 = nullcontext() if fast_mode else mem_manager.track_memory("layer1_signals_all")

    with context_l1:
        series_pool = get_series_pool()
        import multiprocessing as mp

        is_child_process = mp.current_process().daemon or mp.current_process().name != "MainProcess"

        if parallel_l1 is None:
            hw_manager = get_hardware_manager()
            n_bars = len(prices)
            n_cores = hw_manager.get_resources().cpu_cores
            use_parallel_l1 = (
                n_bars >= min_bars_parallel_l1
                and not is_child_process
                and n_cores >= min_cores_parallel_l1
            )
        else:
            use_parallel_l1 = parallel_l1

        if use_parallel_l1:
            from modules.adaptive_trend_LTS_mini.core.process_layer1 import _layer1_parallel_atc_signals

            layer1_signals = _layer1_parallel_atc_signals(
                prices=prices,
                ma_tuples=ma_tuples,
                ma_configs=ma_configs,
                rate_of_change_series=rate_of_change_series,
                lambda_val=lambda_scaled,
                decay_val=decay_scaled,
            )
        else:
            pool_release_failures = 0
            for ma_type, _, _ in ma_configs:
                signal, signals_tuple, equity_tuple = _layer1_signal_for_ma(
                    prices,
                    ma_tuples[ma_type],
                    lambda_val=lambda_scaled,
                    decay_val=decay_scaled,
                    rate_of_change_series=rate_of_change_series,
                )
                layer1_signals[ma_type] = signal

                try:
                    for s in signals_tuple:
                        series_pool.release(s)
                    for e in equity_tuple:
                        series_pool.release(e)
                except Exception as e:
                    pool_release_failures += 1
                    log_warn(f"Error releasing series to pool for {ma_type}: {e}")
            if pool_release_failures > 0:
                log_warn(
                    f"Series pool release failures: {pool_release_failures} "
                    "(may indicate memory pressure or pool errors)"
                )

    return layer1_signals


def _build_ma_configs(
    ema_len: int,
    hma_len: int,
    wma_len: int,
    dema_len: int,
    lsma_len: int,
    kama_len: int,
    ema_w: float,
    hma_w: float,
    wma_w: float,
    dema_w: float,
    lsma_w: float,
    kama_w: float,
) -> List[Tuple[str, int, float]]:
    """Build MA config list from length/weight parameters."""
    return [
        ("EMA", ema_len, ema_w),
        ("HMA", hma_len, hma_w),
        ("WMA", wma_len, wma_w),
        ("DEMA", dema_len, dema_w),
        ("LSMA", lsma_len, lsma_w),
        ("KAMA", kama_len, kama_w),
    ]


def _run_layer2_and_average(
    layer1_signals: Dict[str, pd.Series],
    ma_configs: List[Tuple[str, int, float]],
    rate_of_change_series: pd.Series,
    lambda_scaled: float,
    decay_scaled: float,
    cutout: int,
    parallel_l2: Optional[bool],
    precision: str,
    use_rust_backend: bool,
    equity_floor: float,
    prices: pd.Series,
    long_threshold: float,
    short_threshold: float,
    strategy_mode: bool,
) -> Tuple[Dict[str, pd.Series], pd.Series]:
    """Compute Layer 2 equities and Average_Signal."""
    layer2_equities = calculate_layer2_equities(
        layer1_signals=layer1_signals,
        ma_configs=ma_configs,
        rate_of_change_series=rate_of_change_series,
        lambda_val=lambda_scaled,
        decay_val=decay_scaled,
        cutout=cutout,
        parallel=parallel_l2,
        precision=precision,
        use_rust_backend=use_rust_backend,
        floor_val=equity_floor,
    )
    average_signal = calculate_average_signal(
        layer1_signals=layer1_signals,
        layer2_equities=layer2_equities,
        ma_configs=ma_configs,
        prices=prices,
        long_threshold=long_threshold,
        short_threshold=short_threshold,
        cutout=cutout,
        strategy_mode=strategy_mode,
        precision=precision,
    )
    return layer2_equities, average_signal


def _build_atc_result(
    layer1_signals: Dict[str, pd.Series],
    layer2_equities: Dict[str, pd.Series],
    ma_configs: List[Tuple[str, int, float]],
    average_signal: pd.Series,
) -> Dict[str, pd.Series]:
    """Build final result dict from layer1, layer2, and average signal."""
    result: Dict[str, pd.Series] = {}
    for ma_type, _, _ in ma_configs:
        result[f"{ma_type}_Signal"] = layer1_signals[ma_type]
        result[f"{ma_type}_S"] = layer2_equities[ma_type]
    result["Average_Signal"] = average_signal
    return result


def compute_atc_signals(
    prices: pd.Series,
    src: Optional[pd.Series] = None,
    *,
    ema_len: int = 28,
    hma_len: int = 28,
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
    lambda_param: float = 0.02,
    decay: float = 0.03,
    decay_rate: Optional[float] = None,  # Deprecated: use decay=
    cutout: int = 0,
    long_threshold: float = 0.1,
    short_threshold: float = -0.1,
    strategy_mode: bool = False,
    parallel_l1: Optional[bool] = None,
    min_bars_parallel_l1: Optional[int] = None,
    min_cores_parallel_l1: Optional[int] = None,
    parallel_l2: Optional[bool] = True,
    precision: str = "float64",
    use_rust_backend: bool = True,
    use_cache: bool = True,
    fast_mode: bool = True,
    use_approximate: bool = False,
    approximate_threshold: float = 0.05,
    use_adaptive_approximate: bool = False,
    approximate_volatility_window: int = 20,
    approximate_volatility_factor: float = 1.0,
    equity_floor: float = 0.25,
) -> dict[str, pd.Series]:
    """Compute Adaptive Trend Classification (ATC) signals."""
    if decay_rate is not None:
        warnings.warn(
            "decay_rate= is deprecated, use decay= instead.",
            DeprecationWarning,
            stacklevel=2,
        )
        decay = decay_rate
    log_debug(f"Starting ATC signal computation for {len(prices)} bars")
    prices, src, robustness, cutout, lambda_scaled, decay_scaled = _validate_and_scale_params(
        prices, src, robustness, cutout, lambda_param, decay
    )
    log_info(
        f"Parameters: robustness={robustness}, lambda_scaled={lambda_scaled}, decay_scaled={decay_scaled}, "
        f"cutout={cutout}, strategy_mode={strategy_mode}, equity_floor={equity_floor}"
    )

    ma_configs = _build_ma_configs(
        ema_len, hma_len, wma_len, dema_len, lsma_len, kama_len,
        ema_w, hma_w, wma_w, dema_w, lsma_w, kama_w,
    )
    ma_tuples = _compute_ma_tuples(
        prices, src, ma_configs, robustness,
        use_adaptive_approximate, use_approximate, approximate_threshold,
        approximate_volatility_window, approximate_volatility_factor,
        use_rust_backend, use_cache, fast_mode,
    )
    log_debug(f"Computed {len(ma_tuples)} MA types")

    rate_of_change_series = rate_of_change(prices)
    min_bars = min_bars_parallel_l1 if min_bars_parallel_l1 is not None else DEFAULT_MIN_BARS_PARALLEL_L1
    min_cores = min_cores_parallel_l1 if min_cores_parallel_l1 is not None else DEFAULT_MIN_CORES_PARALLEL_L1
    layer1_signals = _compute_layer1(
        prices, ma_tuples, ma_configs, rate_of_change_series,
        lambda_scaled, decay_scaled, parallel_l1, fast_mode,
        min_bars_parallel_l1=min_bars,
        min_cores_parallel_l1=min_cores,
    )
    log_debug("Completed Layer 1 signals")

    layer2_equities, average_signal = _run_layer2_and_average(
        layer1_signals, ma_configs, rate_of_change_series,
        lambda_scaled, decay_scaled, cutout, parallel_l2, precision,
        use_rust_backend, equity_floor, prices,
        long_threshold, short_threshold, strategy_mode,
    )
    result = _build_atc_result(layer1_signals, layer2_equities, ma_configs, average_signal)

    cleanup_series(rate_of_change_series)
    log_info(f"Completed ATC signal computation for {len(prices)} bars")
    return result


__all__ = ["compute_atc_signals"]
