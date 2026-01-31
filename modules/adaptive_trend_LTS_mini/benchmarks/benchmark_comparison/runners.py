import threading
import time
import traceback
from typing import Any, Callable, Dict, Tuple

import pandas as pd
import psutil

from modules.common.utils import log_error, log_info, log_success


def _measure_memory_rss_mb() -> float:
    """Get current process RSS memory in MB."""
    return psutil.Process().memory_info().rss / 1024 / 1024


class MemoryMonitor:
    """Monitors peak memory usage in a background thread."""

    def __init__(self, interval: float = 0.01):
        self.interval = interval
        self._stop_event = threading.Event()
        self._peak_memory = 0.0
        self._start_memory = 0.0
        self._thread = None

    def __enter__(self):
        self._start_memory = _measure_memory_rss_mb()
        self._peak_memory = self._start_memory
        self._stop_event.clear()
        self._thread = threading.Thread(target=self._monitor)
        self._thread.daemon = True
        self._thread.start()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self._stop_event.set()
        if self._thread:
            self._thread.join()

    def _monitor(self):
        while not self._stop_event.is_set():
            current_mem = _measure_memory_rss_mb()
            self._peak_memory = max(self._peak_memory, current_mem)
            time.sleep(self.interval)

    @property
    def peak_usage(self) -> float:
        """Returns peak memory usage (delta from start) in MB."""
        return max(0.0, self._peak_memory - self._start_memory)

    @property
    def absolute_peak(self) -> float:
        """Returns absolute peak memory in MB."""
        return self._peak_memory


def _run_serial_benchmark(
    name: str, compute_func: Callable[..., Any], prices_data: Dict[str, pd.Series], config: Dict[str, Any]
) -> Tuple[Dict[str, Any], float, float]:
    """Generic runner for serial (per-symbol) execution benchmarks.

    Handles timing, memory tracking, error counting, and progress logging.
    """
    log_info(f"Running {name} adaptive_trend_LTS_mini module...")

    results = {}

    # Error tracking
    error_count = 0
    max_errors = max(5, int(len(prices_data) * 0.1))

    # Progress logging setup
    total_symbols = len(prices_data)
    # Log at 10%, 20%, etc., but at least every 10 items
    log_interval = max(10, total_symbols // 10)

    start_time = time.time()

    # Monitor memory during execution
    with MemoryMonitor() as mem_mon:
        for idx, (symbol, prices) in enumerate(prices_data.items(), 1):
            try:
                # Execute computation
                result = compute_func(prices=prices, **config)
                results[symbol] = result

                # Progress logging
                if idx % log_interval == 0 or idx == total_symbols:
                    log_info(f"{name}: Processed {idx}/{total_symbols} symbols")

            except Exception as e:
                log_error(f"{name} failed for {symbol}: {e}")
                traceback.print_exc()
                results[symbol] = None

                # Fail-fast check
                error_count += 1
                if error_count > max_errors:
                    log_error(f"{name}: Aborting test - too many errors ({error_count})")
                    break

        total_memory_used = mem_mon.peak_usage

    execution_time = time.time() - start_time

    log_success(f"{name} module completed in {execution_time:.2f}s, peak memory: {total_memory_used:.1f} MB")
    return results, execution_time, total_memory_used


def _run_batch_benchmark(
    name: str,
    batch_func: Callable[..., Dict[str, Any]],
    prices_data: Dict[str, pd.Series],
    config: Dict[str, Any],
    **kwargs,
) -> Tuple[Dict[str, Any], float, float]:
    """Generic runner for batch execution benchmarks."""
    log_info(f"Running {name} adaptive_trend_LTS_mini module...")

    results = {}
    start_time = time.time()

    try:
        # Execute batch computation with memory monitoring
        with MemoryMonitor() as mem_mon:
            results = batch_func(prices_data, config, **kwargs)
            total_memory_used = mem_mon.peak_usage

        log_info(f"{name}: Processed {len(results)}/{len(prices_data)} symbols")

    except Exception as e:
        log_error(f"{name} processing failed: {e}")
        traceback.print_exc()
        results = {}
        total_memory_used = 0.0

    execution_time = time.time() - start_time

    log_success(f"{name} module completed in {execution_time:.2f}s, peak memory: {total_memory_used:.1f} MB")
    return results, execution_time, total_memory_used


def run_original_module(
    prices_data: Dict[str, pd.Series], config: Dict[str, Any]
) -> Tuple[Dict[str, Any], float, float]:
    """Run original adaptive_trend module."""
    from modules.adaptive_trend_LTS_mini.core.compute_atc_signals import compute_atc_signals as compute_atc_original

    return _run_serial_benchmark("Original", compute_atc_original, prices_data, config)


def run_rust_module(prices_data: Dict[str, pd.Series], config: Dict[str, Any]) -> Tuple[Dict[str, Any], float, float]:
    """Run Rust-accelerated adaptive_trend_LTS module."""
    from modules.adaptive_trend_LTS_mini.core.compute_atc_signals import compute_atc_signals as compute_atc_rust

    return _run_serial_benchmark("Rust", compute_atc_rust, prices_data, config)


def run_approximate_module(
    prices_data: Dict[str, pd.Series], config: Dict[str, Any]
) -> Tuple[Dict[str, Any], float, float]:
    """Run Approximate MAs adaptive_trend_LTS module."""
    from modules.adaptive_trend_LTS_mini.core.compute_atc_signals import compute_atc_signals as compute_atc_rust

    return _run_serial_benchmark("Approximate", compute_atc_rust, prices_data, config)


def run_adaptive_approximate_module(
    prices_data: Dict[str, pd.Series], config: Dict[str, Any]
) -> Tuple[Dict[str, Any], float, float]:
    """Run Adaptive Approximate MAs adaptive_trend_LTS module."""
    from modules.adaptive_trend_LTS_mini.core.compute_atc_signals import compute_atc_signals as compute_atc_rust

    return _run_serial_benchmark("Adaptive Approx", compute_atc_rust, prices_data, config)


def run_rust_batch_module(
    prices_data: Dict[str, pd.Series], config: Dict[str, Any]
) -> Tuple[Dict[str, Any], float, float]:
    """Run Rust-accelerated adaptive_trend_LTS module using Rayon batch processing."""
    from modules.adaptive_trend_LTS_mini.core.compute_atc_signals.batch_processor import process_symbols_batch_rust

    return _run_batch_benchmark("Rust (Rayon)", process_symbols_batch_rust, prices_data, config)


def run_dask_module(prices_data: Dict[str, pd.Series], config: dict) -> Tuple[Dict[str, Dict], float, float]:
    """Run Dask-based adaptive_trend_LTS module for out-of-core processing."""
    try:
        from modules.adaptive_trend_LTS_mini.core.compute_atc_signals.dask_batch_processor import (
            process_symbols_batch_dask,
        )
    except ImportError:
        log_error("Dask batch processor not available")
        return {}, 0.0, 0.0

    # CPU-only fallback for mini version defaults
    kwargs = {"use_rust": True, "npartitions": None, "partition_size": 50}
    return _run_batch_benchmark("Dask", process_symbols_batch_dask, prices_data, config, **kwargs)


def run_rust_dask_module(prices_data: Dict[str, pd.Series], config: dict) -> Tuple[Dict[str, Dict], float, float]:
    """Run Rust+Dask hybrid adaptive_trend_LTS module."""
    try:
        from modules.adaptive_trend_LTS_mini.core.compute_atc_signals.rust_dask_bridge import process_symbols_rust_dask
    except ImportError:
        log_error("Rust-Dask bridge not available")
        return {}, 0.0, 0.0

    kwargs = {"npartitions": None, "partition_size": 50, "use_fallback": True}
    return _run_batch_benchmark("Rust+Dask", process_symbols_rust_dask, prices_data, config, **kwargs)
