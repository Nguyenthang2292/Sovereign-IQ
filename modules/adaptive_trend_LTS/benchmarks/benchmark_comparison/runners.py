"""Benchmark runner functions for all 8 versions."""

import time
from typing import Dict, Tuple

import pandas as pd

from modules.common.utils import log_error, log_info, log_success


def run_original_module(prices_data: Dict[str, pd.Series], config: dict) -> Tuple[Dict[str, Dict], float, float]:
    """Run original adaptive_trend module.

    Args:
        prices_data: Dictionary of symbol -> price Series
        config: ATC configuration parameters

    Returns:
        Tuple of (results_dict, execution_time_seconds, peak_memory_mb)
    """
    log_info("Running original adaptive_trend module...")
    from modules.adaptive_trend.core import compute_atc_signals as compute_atc_original

    results = {}
    start_time = time.time()

    # Track memory (approximate)
    import psutil

    process = psutil.Process()
    mem_before = process.memory_info().rss / 1024 / 1024  # MB

    for idx, (symbol, prices) in enumerate(prices_data.items(), 1):
        try:
            result = compute_atc_original(prices=prices, **config)
            results[symbol] = result

            if idx % 100 == 0:
                log_info(f"Original: Processed {idx}/{len(prices_data)} symbols")

        except Exception as e:
            import traceback

            log_error(f"Original failed for {symbol}: {e}")
            traceback.print_exc()
            results[symbol] = None

    end_time = time.time()
    mem_after = process.memory_info().rss / 1024 / 1024  # MB

    execution_time = end_time - start_time
    peak_memory = mem_after - mem_before

    log_success(f"Original module completed in {execution_time:.2f}s")
    return results, execution_time, peak_memory


def run_enhanced_module(prices_data: Dict[str, pd.Series], config: dict) -> Tuple[Dict[str, Dict], float, float]:
    """Run enhanced adaptive_trend_enhance module.

    Args:
        prices_data: Dictionary of symbol -> price Series
        config: ATC configuration parameters

    Returns:
        Tuple of (results_dict, execution_time_seconds, peak_memory_mb)
    """
    log_info("Running enhanced adaptive_trend_enhance module...")
    from modules.adaptive_trend_enhance.core import compute_atc_signals as compute_atc_enhanced

    results = {}
    start_time = time.time()

    # Track memory (approximate)
    import psutil

    process = psutil.Process()
    mem_before = process.memory_info().rss / 1024 / 1024  # MB

    for idx, (symbol, prices) in enumerate(prices_data.items(), 1):
        try:
            result = compute_atc_enhanced(prices=prices, **config)
            results[symbol] = result

            if idx % 100 == 0:
                log_info(f"Enhanced: Processed {idx}/{len(prices_data)} symbols")

        except Exception as e:
            import traceback

            log_error(f"Enhanced failed for {symbol}: {e}")
            traceback.print_exc()
            results[symbol] = None

    end_time = time.time()
    mem_after = process.memory_info().rss / 1024 / 1024  # MB

    execution_time = end_time - start_time
    peak_memory = mem_after - mem_before

    log_success(f"Enhanced module completed in {execution_time:.2f}s")
    return results, execution_time, peak_memory


def run_rust_module(prices_data: Dict[str, pd.Series], config: dict) -> Tuple[Dict[str, Dict], float, float]:
    """Run Rust-accelerated adaptive_trend_LTS module.

    Args:
        prices_data: Dictionary of symbol -> price Series
        config: ATC configuration parameters

    Returns:
        Tuple of (results_dict, execution_time_seconds, peak_memory_mb)
    """
    log_info("Running Rust-accelerated adaptive_trend_LTS module...")
    from modules.adaptive_trend_LTS.core.compute_atc_signals import compute_atc_signals as compute_atc_rust

    results = {}
    start_time = time.time()

    # Track memory (approximate)
    import psutil

    process = psutil.Process()
    mem_before = process.memory_info().rss / 1024 / 1024  # MB

    for idx, (symbol, prices) in enumerate(prices_data.items(), 1):
        try:
            result = compute_atc_rust(prices=prices, **config)
            results[symbol] = result

            if idx % 100 == 0:
                log_info(f"Rust: Processed {idx}/{len(prices_data)} symbols")

        except Exception as e:
            import traceback

            log_error(f"Rust failed for {symbol}: {e}")
            traceback.print_exc()
            results[symbol] = None

    end_time = time.time()
    mem_after = process.memory_info().rss / 1024 / 1024  # MB

    execution_time = end_time - start_time
    peak_memory = mem_after - mem_before

    log_success(f"Rust module completed in {execution_time:.2f}s")
    return results, execution_time, peak_memory


def run_cuda_module(prices_data: Dict[str, pd.Series], config: dict) -> Tuple[Dict[str, Dict], float, float]:
    """Run CUDA-accelerated adaptive_trend_LTS module using concurrent batch processing.

    Args:
        prices_data: Dictionary of symbol -> price Series
        config: ATC configuration parameters

    Returns:
        Tuple of (results_dict, execution_time_seconds, peak_memory_mb)
    """
    log_info("Running CUDA-accelerated adaptive_trend_LTS module (Concurrent Batch)...")
    from modules.adaptive_trend_LTS.core.compute_atc_signals.batch_processor import process_symbols_batch_cuda

    start_time = time.time()

    # Track memory (approximate)
    import psutil

    process = psutil.Process()
    mem_before = process.memory_info().rss / 1024 / 1024  # MB

    try:
        # Use concurrent batch processor
        # This uses ThreadPoolExecutor to overlap Rust-CUDA tasks
        results = process_symbols_batch_cuda(prices_data, config, num_threads=4)

        log_info(f"CUDA: Processed {len(results)}/{len(prices_data)} symbols")

    except Exception as e:
        import traceback

        log_error(f"CUDA batch processing failed: {e}")
        traceback.print_exc()
        results = {}

    end_time = time.time()
    mem_after = process.memory_info().rss / 1024 / 1024  # MB

    execution_time = end_time - start_time
    peak_memory = mem_after - mem_before

    log_success(f"CUDA module completed in {execution_time:.2f}s")
    return results, execution_time, peak_memory


def run_rust_batch_module(prices_data: Dict[str, pd.Series], config: dict) -> Tuple[Dict[str, Dict], float, float]:
    """Run Rust-accelerated adaptive_trend_LTS module using Rayon batch processing.

    Args:
        prices_data: Dictionary of symbol -> price Series
        config: ATC configuration parameters

    Returns:
        Tuple of (results_dict, execution_time_seconds, peak_memory_mb)
    """
    log_info("Running Rust-accelerated adaptive_trend_LTS module (Rayon Batch)...")
    from modules.adaptive_trend_LTS.core.compute_atc_signals.batch_processor import process_symbols_batch_rust

    start_time = time.time()

    # Track memory (approximate)
    import psutil

    process = psutil.Process()
    mem_before = process.memory_info().rss / 1024 / 1024  # MB

    try:
        # Use Rayon batch processor
        results = process_symbols_batch_rust(prices_data, config)

        log_info(f"Rust (Rayon): Processed {len(results)}/{len(prices_data)} symbols")

    except Exception as e:
        import traceback

        log_error(f"Rust Rayon batch processing failed: {e}")
        traceback.print_exc()
        results = {}

    end_time = time.time()
    mem_after = process.memory_info().rss / 1024 / 1024  # MB

    execution_time = end_time - start_time
    peak_memory = mem_after - mem_before

    log_success(f"Rust Rayon module completed in {execution_time:.2f}s")
    return results, execution_time, peak_memory


def run_dask_module(prices_data: Dict[str, pd.Series], config: dict) -> Tuple[Dict[str, Dict], float, float]:
    """Run Dask-based adaptive_trend_LTS module for out-of-core processing.

    Args:
        prices_data: Dictionary of symbol -> price Series
        config: ATC configuration parameters

    Returns:
        Tuple of (results_dict, execution_time_seconds, peak_memory_mb)
    """
    log_info("Running Dask-based adaptive_trend_LTS module (Out-of-Core)...")
    try:
        from modules.adaptive_trend_LTS.core.compute_atc_signals.dask_batch_processor import (
            process_symbols_batch_dask,
        )
    except ImportError:
        log_error("Dask batch processor not available")
        return {}, 0.0, 0.0

    start_time = time.time()

    # Track memory (approximate)
    import psutil

    process = psutil.Process()
    mem_before = process.memory_info().rss / 1024 / 1024  # MB

    try:
        # Use Dask batch processor (Python fallback)
        results = process_symbols_batch_dask(
            prices_data, config, use_rust=False, use_cuda=False, npartitions=None, partition_size=50
        )

        log_info(f"Dask: Processed {len(results)}/{len(prices_data)} symbols")

    except Exception as e:
        import traceback

        log_error(f"Dask batch processing failed: {e}")
        traceback.print_exc()
        results = {}

    end_time = time.time()
    mem_after = process.memory_info().rss / 1024 / 1024  # MB

    execution_time = end_time - start_time
    peak_memory = mem_after - mem_before

    log_success(f"Dask module completed in {execution_time:.2f}s")
    return results, execution_time, peak_memory


def run_rust_dask_module(prices_data: Dict[str, pd.Series], config: dict) -> Tuple[Dict[str, Dict], float, float]:
    """Run Rust+Dask hybrid adaptive_trend_LTS module.

    Args:
        prices_data: Dictionary of symbol -> price Series
        config: ATC configuration parameters

    Returns:
        Tuple of (results_dict, execution_time_seconds, peak_memory_mb)
    """
    log_info("Running Rust+Dask hybrid adaptive_trend_LTS module...")
    try:
        from modules.adaptive_trend_LTS.core.compute_atc_signals.rust_dask_bridge import process_symbols_rust_dask
    except ImportError:
        log_error("Rust-Dask bridge not available")
        return {}, 0.0, 0.0

    start_time = time.time()

    # Track memory (approximate)
    import psutil

    process = psutil.Process()
    mem_before = process.memory_info().rss / 1024 / 1024  # MB

    try:
        # Use Rust+Dask hybrid (CPU Rust with Dask partitions)
        results = process_symbols_rust_dask(
            prices_data, config, use_cuda=False, npartitions=None, partition_size=50, use_fallback=True
        )

        log_info(f"Rust+Dask: Processed {len(results)}/{len(prices_data)} symbols")

    except Exception as e:
        import traceback

        log_error(f"Rust+Dask hybrid processing failed: {e}")
        traceback.print_exc()
        results = {}

    end_time = time.time()
    mem_after = process.memory_info().rss / 1024 / 1024  # MB

    execution_time = end_time - start_time
    peak_memory = mem_after - mem_before

    log_success(f"Rust+Dask module completed in {execution_time:.2f}s")
    return results, execution_time, peak_memory


def run_cuda_dask_module(prices_data: Dict[str, pd.Series], config: dict) -> Tuple[Dict[str, Dict], float, float]:
    """Run CUDA+Dask hybrid adaptive_trend_LTS module.

    Args:
        prices_data: Dictionary of symbol -> price Series
        config: ATC configuration parameters

    Returns:
        Tuple of (results_dict, execution_time_seconds, peak_memory_mb)
    """
    log_info("Running CUDA+Dask hybrid adaptive_trend_LTS module...")
    try:
        from modules.adaptive_trend_LTS.core.compute_atc_signals.rust_dask_bridge import process_symbols_rust_dask
    except ImportError:
        log_error("Rust-Dask bridge not available")
        return {}, 0.0, 0.0

    start_time = time.time()

    # Track memory (approximate)
    import psutil

    process = psutil.Process()
    mem_before = process.memory_info().rss / 1024 / 1024  # MB

    try:
        # Use CUDA+Dask hybrid (CUDA Rust with Dask partitions)
        results = process_symbols_rust_dask(
            prices_data, config, use_cuda=True, npartitions=None, partition_size=50, use_fallback=True
        )

        log_info(f"CUDA+Dask: Processed {len(results)}/{len(prices_data)} symbols")

    except Exception as e:
        import traceback

        log_error(f"CUDA+Dask hybrid processing failed: {e}")
        traceback.print_exc()
        results = {}

    end_time = time.time()
    mem_after = process.memory_info().rss / 1024 / 1024  # MB

    execution_time = end_time - start_time
    peak_memory = mem_after - mem_before

    log_success(f"CUDA+Dask module completed in {execution_time:.2f}s")
    return results, execution_time, peak_memory


def run_rust_cuda_dask_module(
    prices_data: Dict[str, pd.Series], config: dict
) -> Tuple[Dict[str, Dict], float, float]:
    """Run Rust+CUDA+Dask hybrid adaptive_trend_LTS module (all three optimizations).

    Args:
        prices_data: Dictionary of symbol -> price Series
        config: ATC configuration parameters

    Returns:
        Tuple of (results_dict, execution_time_seconds, peak_memory_mb)
    """
    log_info("Running Rust+CUDA+Dask hybrid adaptive_trend_LTS module (All Three)...")
    try:
        from modules.adaptive_trend_LTS.core.compute_atc_signals.rust_dask_bridge import process_symbols_rust_dask
    except ImportError:
        log_error("Rust-Dask bridge not available")
        return {}, 0.0, 0.0

    start_time = time.time()

    # Track memory (approximate)
    import psutil

    process = psutil.Process()
    mem_before = process.memory_info().rss / 1024 / 1024  # MB

    try:
        # Use Rust+CUDA+Dask hybrid (CUDA Rust with Dask partitions for out-of-core)
        # This combines all three optimizations: Rust speed, CUDA acceleration, Dask memory management
        results = process_symbols_rust_dask(
            prices_data, config, use_cuda=True, npartitions=None, partition_size=50, use_fallback=True
        )

        log_info(f"Rust+CUDA+Dask: Processed {len(results)}/{len(prices_data)} symbols")

    except Exception as e:
        import traceback

        log_error(f"Rust+CUDA+Dask hybrid processing failed: {e}")
        traceback.print_exc()
        results = {}

    end_time = time.time()
    mem_after = process.memory_info().rss / 1024 / 1024  # MB

    execution_time = end_time - start_time
    peak_memory = mem_after - mem_before

    log_success(f"Rust+CUDA+Dask module completed in {execution_time:.2f}s")
    return results, execution_time, peak_memory
