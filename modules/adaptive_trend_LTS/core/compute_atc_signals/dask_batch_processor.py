"""Dask-based batch processor for out-of-core symbol processing."""

from __future__ import annotations

import gc
from typing import Dict, Optional

import dask.bag as db
import numpy as np
import pandas as pd

from .batch_processor import process_symbols_batch_cuda, process_symbols_batch_rust
from .compute_atc_signals import compute_atc_signals

try:
    from modules.common.utils import log_error, log_info, log_warn
except ImportError:

    def log_info(message: str) -> None:
        print(f"[INFO] {message}")

    def log_error(message: str) -> None:
        print(f"[ERROR] {message}")

    def log_warn(message: str) -> None:
        print(f"[WARN] {message}")


def _process_partition_with_backend(
    partition_items: list,
    config: dict,
    use_rust: bool,
    use_cuda: bool,
    use_fallback: bool,
) -> Dict[str, Dict[str, pd.Series]]:
    """Process a partition of symbols with specified backend.

    Args:
        partition_items: List of (symbol, prices) tuples
        config: ATC configuration parameters
        use_rust: Use Rust backend (CPU multi-threaded)
        use_cuda: Use CUDA backend
        use_fallback: Use fallback Python processor if backend fails

    Returns:
        Dictionary mapping symbol -> {"Average_Signal": pd.Series}
    """
    if not partition_items:
        return {}

    partition_dict = dict(partition_items)

    try:
        if use_cuda:
            return process_symbols_batch_cuda(partition_dict, config)
        elif use_rust:
            return process_symbols_batch_rust(partition_dict, config)
        else:
            return _process_partition_python(partition_dict, config)
    except Exception as e:
        if use_fallback:
            log_warn(f"Backend processing failed: {e}. Falling back to Python")
            return _process_partition_python(partition_dict, config)
        else:
            log_error(f"Error processing partition: {e}")
            raise

    finally:
        gc.collect()


def _process_partition_python(
    symbols_data: Dict[str, pd.Series],
    config: dict,
) -> Dict[str, Dict[str, pd.Series]]:
    """Process partition using Python backend (fallback)."""
    results = {}
    for symbol, prices in symbols_data.items():
        try:
            if prices is None or (isinstance(prices, pd.Series) and prices.empty):
                continue
            result = compute_atc_signals(prices, **config)
            if result:
                results[symbol] = {"Average_Signal": result.get("Average_Signal", pd.Series())}
        except Exception as e:
            log_error(f"Error processing {symbol} in Python fallback: {e}")
    return results


def process_symbols_batch_dask(
    symbols_data: Dict[str, pd.Series],
    config: dict,
    use_rust: bool = True,
    use_cuda: bool = False,
    npartitions: Optional[int] = None,
    partition_size: int = 50,
    use_fallback: bool = True,
) -> Dict[str, Dict[str, pd.Series]]:
    """Process symbols in batches using Dask for out-of-core processing.

    Args:
        symbols_data: Dictionary of symbol -> price Series
        config: ATC configuration
        use_rust: Use Rust backend (default: True)
        use_cuda: Use CUDA backend if available (default: False)
        npartitions: Number of Dask partitions (auto if None)
        partition_size: Symbols per partition
        use_fallback: Use Python fallback on backend errors (default: True)

    Returns:
        Dictionary mapping symbol -> {"Average_Signal": pd.Series}
    """
    if not symbols_data:
        return {}

    symbols_count = len(symbols_data)

    if npartitions is None:
        npartitions = max(1, symbols_count // partition_size)

    log_info(f"Processing {symbols_count} symbols with Dask ({npartitions} partitions)")

    symbols_items = list(symbols_data.items())

    symbols_bag = db.from_sequence(symbols_items, npartitions=npartitions)

    results_bag = symbols_bag.map_partitions(
        lambda items: [_process_partition_with_backend(items, config, use_rust, use_cuda, use_fallback)]
    )

    results_list = results_bag.compute()

    final_results = {}
    for partition_results in results_list:
        if isinstance(partition_results, dict):
            final_results.update(partition_results)

    log_info(f"Dask batch processing completed: {len(final_results)} symbols processed")
    return final_results
