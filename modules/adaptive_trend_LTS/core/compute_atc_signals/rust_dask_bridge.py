"""Bridge between Rust batch processing and Dask for optimal performance."""

from __future__ import annotations

import gc
from typing import Dict, Optional

import dask.bag as db
import numpy as np
import pandas as pd

try:
    from modules.common.utils import log_error, log_info, log_warn
except ImportError:

    def log_info(msg: str) -> None:
        print(f"[INFO] {msg}")

    def log_error(msg: str) -> None:
        print(f"[ERROR] {msg}")

    def log_warn(msg: str) -> None:
        print(f"[WARN] {msg}")


try:
    import atc_rust

    HAS_RUST = True
except ImportError:
    HAS_RUST = False
    log_warn("Rust extensions not available, falling back to Python")


def _process_partition_with_rust_cpu(
    partition_data: Dict[str, np.ndarray | pd.Series],
    config: dict,
) -> Dict[str, Dict[str, pd.Series]]:
    """Process a partition using Rust batch processing (CPU).

    Args:
        partition_data: Dictionary mapping symbol to price data (Series or ndarray)
        config: ATC configuration parameters

    Returns:
        Dictionary mapping symbol to result dict with Average_Signal Series
    """
    if not HAS_RUST:
        return _process_partition_python(partition_data, config)

    try:
        params = config.copy()

        la = params.get("La", params.get("la", 0.02))
        de = params.get("De", params.get("de", 0.03))

        la_scaled = la / 1000.0
        de_scaled = de / 100.0

        symbols_numpy = {}
        for s, v in partition_data.items():
            if v is not None:
                if isinstance(v, pd.Series):
                    if v.empty:
                        continue
                    symbols_numpy[s] = v.values.astype(np.float64)
                elif isinstance(v, np.ndarray):
                    if v.size == 0:
                        continue
                    symbols_numpy[s] = v.astype(np.float64)
                else:
                    symbols_numpy[s] = np.array(v, dtype=np.float64)

        if not symbols_numpy:
            return {}

        batch_results = atc_rust.compute_atc_signals_batch_cpu(
            symbols_numpy,
            ema_len=params.get("ema_len", 28),
            hull_len=params.get("hull_len", 28),
            wma_len=params.get("wma_len", 28),
            dema_len=params.get("dema_len", 28),
            lsma_len=params.get("lsma_len", 28),
            kama_len=params.get("kama_len", 28),
            robustness=params.get("robustness", "Medium"),
            la=la_scaled,
            de=de_scaled,
            long_threshold=params.get("long_threshold", 0.1),
            short_threshold=params.get("short_threshold", -0.1),
        )

        results = {}
        for symbol, signal_array in batch_results.items():
            # CRITICAL FIX: Preserve original pandas index from input data
            # This ensures downstream comparisons work correctly by index alignment
            orig_data = partition_data.get(symbol)
            if orig_data is not None and hasattr(orig_data, "index"):
                # Restore original index from pandas Series
                results[symbol] = {"Average_Signal": pd.Series(signal_array, index=orig_data.index)}
            else:
                # Fallback: create Series with default index
                results[symbol] = {"Average_Signal": pd.Series(signal_array)}

        gc.collect()
        return results

    except Exception as e:
        log_error(f"Rust processing failed: {e}, falling back to Python")
        return _process_partition_python(partition_data, config)


def _process_partition_with_rust_cuda(
    partition_data: Dict[str, np.ndarray | pd.Series],
    config: dict,
) -> Dict[str, Dict[str, pd.Series]]:
    """Process a partition using Rust batch processing (CUDA).

    Args:
        partition_data: Dictionary mapping symbol to price data (Series or ndarray)
        config: ATC configuration parameters

    Returns:
        Dictionary mapping symbol to result dict with Average_Signal Series
    """
    if not HAS_RUST:
        return _process_partition_python(partition_data, config)

    try:
        params = config.copy()

        la = params.get("La", params.get("la", 0.02))
        de = params.get("De", params.get("de", 0.03))

        la_scaled = la / 1000.0
        de_scaled = de / 100.0

        symbols_numpy = {}
        for s, v in partition_data.items():
            if v is not None:
                if isinstance(v, pd.Series):
                    if v.empty:
                        continue
                    symbols_numpy[s] = v.values.astype(np.float64)
                elif isinstance(v, np.ndarray):
                    if v.size == 0:
                        continue
                    symbols_numpy[s] = v.astype(np.float64)
                else:
                    symbols_numpy[s] = np.array(v, dtype=np.float64)

        if not symbols_numpy:
            return {}

        batch_results = atc_rust.compute_atc_signals_batch(
            symbols_numpy,
            ema_len=params.get("ema_len", 28),
            hull_len=params.get("hull_len", 28),
            wma_len=params.get("wma_len", 28),
            dema_len=params.get("dema_len", 28),
            lsma_len=params.get("lsma_len", 28),
            kama_len=params.get("kama_len", 28),
            robustness=params.get("robustness", "Medium"),
            la=la_scaled,
            de=de_scaled,
            long_threshold=params.get("long_threshold", 0.1),
            short_threshold=params.get("short_threshold", -0.1),
        )

        results = {}
        for symbol, signal_array in batch_results.items():
            # CRITICAL FIX: Preserve original pandas index from input data
            # This ensures downstream comparisons work correctly by index alignment
            orig_data = partition_data.get(symbol)
            if orig_data is not None and hasattr(orig_data, "index"):
                # Restore original index from pandas Series
                results[symbol] = {"Average_Signal": pd.Series(signal_array, index=orig_data.index)}
            else:
                # Fallback: create Series with default index
                results[symbol] = {"Average_Signal": pd.Series(signal_array)}

        gc.collect()
        return results

    except Exception as e:
        log_error(f"Rust CUDA processing failed: {e}, falling back to CPU Rust")
        return _process_partition_with_rust_cpu(partition_data, config)


def _process_partition_python(
    partition_data: Dict[str, np.ndarray | pd.Series],
    config: dict,
) -> Dict[str, Dict[str, pd.Series]]:
    """Process a partition using Python backend (fallback).

    Args:
        partition_data: Dictionary mapping symbol to price data (Series or ndarray)
        config: ATC configuration parameters

    Returns:
        Dictionary mapping symbol to result dict with Average_Signal Series
    """
    try:
        from modules.adaptive_trend_enhance.core.compute_atc_signals import compute_atc_signals

        # Filter out parameters that adaptive_trend_enhance doesn't accept
        python_config = {
            k: v
            for k, v in config.items()
            if k
            not in (
                "use_rust_backend",
                "use_cuda",
            )  # These are LTS-specific, not in enhance module
        }

        results = {}
        for symbol, prices in partition_data.items():
            try:
                if prices is None:
                    continue
                if isinstance(prices, (pd.Series, np.ndarray)):
                    if isinstance(prices, pd.Series):
                        if prices.empty:
                            continue
                    elif isinstance(prices, np.ndarray):
                        if prices.size == 0:
                            continue

                # Preserve original index if available
                orig_index = None
                if isinstance(prices, pd.Series):
                    orig_index = prices.index
                    prices_series = prices
                elif isinstance(prices, np.ndarray):
                    prices_series = pd.Series(prices)
                else:
                    prices_series = prices

                result = compute_atc_signals(prices=prices_series, **python_config)

                if result:
                    avg_signal = result.get("Average_Signal", pd.Series())
                    # CRITICAL FIX: Preserve original pandas index from input data
                    if orig_index is not None and len(avg_signal) == len(orig_index):
                        avg_signal = pd.Series(avg_signal.values, index=orig_index)
                    results[symbol] = {"Average_Signal": avg_signal}
            except Exception as e:
                log_error(f"Error processing {symbol} in Python fallback: {e}")

        gc.collect()
        return results

    except Exception as e:
        log_error(f"Python fallback failed: {e}")
        return {}


def auto_tune_partition_size(
    total_symbols: int,
    available_memory_gb: float = 8.0,
    target_memory_per_partition_gb: float = 0.5,
) -> int:
    """Auto-determine optimal partition size based on available memory.

    Args:
        total_symbols: Total number of symbols
        available_memory_gb: Available RAM in GB
        target_memory_per_partition_gb: Target memory per partition

    Returns:
        Optimal partition size
    """
    memory_per_symbol_mb = 1.0
    memory_per_symbol_gb = memory_per_symbol_mb / 1024.0

    max_symbols_per_partition = int(target_memory_per_partition_gb / memory_per_symbol_gb)

    min_partition_size = 10
    max_partition_size = 200

    partition_size = max(
        min_partition_size,
        min(max_symbols_per_partition, max_partition_size),
    )

    return partition_size


def process_symbols_rust_dask(
    symbols_data: Dict[str, pd.Series],
    config: dict,
    use_cuda: bool = False,
    npartitions: Optional[int] = None,
    partition_size: int = 50,
    use_fallback: bool = True,
) -> Dict[str, Dict[str, pd.Series]]:
    """Process symbols using Rust + Dask hybrid approach.

    Args:
        symbols_data: Dictionary of symbol -> price Series
        config: ATC configuration
        use_cuda: Use CUDA backend (default: False)
        npartitions: Number of Dask partitions (auto if None)
        partition_size: Symbols per partition
        use_fallback: Use Python fallback on Rust errors (default: True)

    Returns:
        Dictionary mapping symbol -> {"Average_Signal": pd.Series}
    """
    if not symbols_data:
        return {}

    symbols_count = len(symbols_data)

    if npartitions is None:
        npartitions = max(1, symbols_count // partition_size)

    log_info(f"Processing {symbols_count} symbols with Rust+Dask ({npartitions} partitions)")

    def _prepare_partition(items: list) -> Dict[str, np.ndarray]:
        partition_dict = {}
        for symbol, prices in items:
            if prices is None:
                continue
            if isinstance(prices, pd.Series):
                if prices.empty:
                    continue
                partition_dict[symbol] = prices.values
            elif isinstance(prices, np.ndarray):
                if prices.size == 0:
                    continue
                partition_dict[symbol] = prices
            else:
                partition_dict[symbol] = np.array(prices, dtype=np.float64)
        return partition_dict

    def _process_partition(partition_dict: Dict[str, np.ndarray]) -> Dict[str, Dict[str, pd.Series]]:
        if not partition_dict:
            return {}

        if use_cuda and HAS_RUST:
            return _process_partition_with_rust_cuda(partition_dict, config)
        elif HAS_RUST:
            return _process_partition_with_rust_cpu(partition_dict, config)
        else:
            return _process_partition_python(partition_dict, config)

    symbols_items = list(symbols_data.items())

    symbols_bag = db.from_sequence(symbols_items, npartitions=npartitions)

    partitions_bag = symbols_bag.map_partitions(lambda items: [_prepare_partition(items)])

    results_bag = partitions_bag.map(_process_partition)

    results_list = results_bag.compute()

    final_results = {}
    for partition_results in results_list:
        if isinstance(partition_results, dict):
            final_results.update(partition_results)

    log_info(f"Rust+Dask batch processing completed: {len(final_results)} symbols processed")
    return final_results
