"""Main scanning function for ATC symbol scanner.

This module provides the scan_all_symbols function, the main public API
for scanning all symbols and finding LONG/SHORT signals.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Optional, Tuple

import numpy as np
import pandas as pd

if TYPE_CHECKING:
    from modules.common.core.data_fetcher import DataFetcher

try:
    from modules.common.utils import log_error, log_progress, log_success, log_warn
except ImportError:

    def log_error(message: str) -> None:
        print(f"[ERROR] {message}")

    def log_warn(message: str) -> None:
        print(f"[WARN] {message}")

    def log_success(message: str) -> None:
        print(f"[SUCCESS] {message}")

    def log_progress(message: str) -> None:
        print(f"[PROGRESS] {message}")


from modules.adaptive_trend_LTS.utils.config import ATCConfig
from modules.common.system import get_hardware_manager, get_memory_manager

from .asyncio_scan import _scan_asyncio
from .dask_scan import _scan_dask
from .gpu_scan import _scan_gpu_batch
from .processpool import _scan_processpool
from .sequential import _scan_sequential
from .threadpool import _scan_threadpool


def scan_all_symbols(
    data_fetcher: "DataFetcher",
    atc_config: ATCConfig,
    max_symbols: Optional[int] = None,
    min_signal: float = 0.01,
    execution_mode: str = "threadpool",
    max_workers: Optional[int] = None,
    batch_size: int = 100,
    npartitions: Optional[int] = None,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Scan all futures symbols and filter those with LONG/SHORT signals.

    Fetches OHLCV data for multiple symbols, calculates ATC signals for each,
    and returns DataFrames containing symbols with signals above the threshold,
    separated into LONG (trend > 0) and SHORT (trend < 0) signals.

    Execution modes:
    - "sequential": Process symbols one by one (safest, avoids rate limits)
    - "threadpool": Use ThreadPoolExecutor for parallel data fetching (default, faster)
    - "asyncio": Use asyncio for parallel data fetching (fastest, but requires async support)
    - "processpool": Use ProcessPoolExecutor for parallel data fetching
    - "gpu_batch": Use GPU for batch processing (requires CUDA)
    - "dask": Use Dask for out-of-core processing with large symbol lists

    Args:
        data_fetcher: DataFetcher instance for fetching market data.
        atc_config: ATCConfig object containing all ATC parameters.
        max_symbols: Maximum number of symbols to scan (None = all symbols).
        min_signal: Minimum signal strength to include (must be >= 0).
        execution_mode: Execution mode - "sequential", "threadpool", "asyncio", "processpool", "gpu_batch", or "dask" (default: "threadpool").
        max_workers: Maximum number of worker threads/processes for parallel execution.
                    If None, uses default (min(32, num_symbols + 4) for threadpool).
        batch_size: Number of symbols to process in each batch before forcing GC (default: 100).
                    Larger batches use more memory but may be faster. Smaller batches use less memory.
        npartitions: Number of Dask partitions for Dask mode (auto if None).

    Returns:
        Tuple of two DataFrames:
        - long_signals_df: Symbols with bullish signals (trend > 0), sorted by signal strength
        - short_signals_df: Symbols with bearish signals (trend < 0), sorted by signal strength

        Each DataFrame contains columns: symbol, signal, trend, price, exchange.

    Raises:
        ValueError: If any parameter is invalid.
        TypeError: If data_fetcher is None or missing required methods.
        AttributeError: If data_fetcher doesn't have required methods.
    """
    # Input validation
    if data_fetcher is None:
        raise ValueError("data_fetcher cannot be None")

    if not isinstance(atc_config, ATCConfig):
        raise ValueError(f"atc_config must be an ATCConfig instance, got {type(atc_config)}")

    # Validate data_fetcher has required methods
    required_methods = ["list_binance_futures_symbols", "fetch_ohlcv_with_fallback_exchange"]
    for method_name in required_methods:
        if not hasattr(data_fetcher, method_name):
            raise AttributeError(f"data_fetcher must have method '{method_name}', got {type(data_fetcher)}")

    if not isinstance(atc_config.timeframe, str) or not atc_config.timeframe.strip():
        raise ValueError(f"atc_config.timeframe must be a non-empty string, got {atc_config.timeframe}")

    if not isinstance(atc_config.limit, int) or atc_config.limit <= 0:
        raise ValueError(f"atc_config.limit must be a positive integer, got {atc_config.limit}")

    # Validate all MA lengths
    ma_lengths = {
        "ema_len": atc_config.ema_len,
        "hma_len": atc_config.hma_len,
        "wma_len": atc_config.wma_len,
        "dema_len": atc_config.dema_len,
        "lsma_len": atc_config.lsma_len,
        "kama_len": atc_config.kama_len,
    }
    for name, length in ma_lengths.items():
        if not isinstance(length, int) or length <= 0:
            raise ValueError(f"atc_config.{name} must be a positive integer, got {length}")

    # Validate robustness
    VALID_ROBUSTNESS = {"Narrow", "Medium", "Wide"}
    if atc_config.robustness not in VALID_ROBUSTNESS:
        raise ValueError(f"atc_config.robustness must be one of {VALID_ROBUSTNESS}, got {atc_config.robustness}")

    # Validate lambda_param
    if (
        not isinstance(atc_config.lambda_param, (int, float))
        or np.isnan(atc_config.lambda_param)
        or np.isinf(atc_config.lambda_param)
    ):
        raise ValueError(f"atc_config.lambda_param must be a finite number, got {atc_config.lambda_param}")

    # Validate decay
    if not isinstance(atc_config.decay, (int, float)) or not (0 <= atc_config.decay <= 1):
        raise ValueError(f"atc_config.decay must be between 0 and 1, got {atc_config.decay}")

    # Validate cutout
    if not isinstance(atc_config.cutout, int) or atc_config.cutout < 0:
        raise ValueError(f"atc_config.cutout must be a non-negative integer, got {atc_config.cutout}")

    # Validate max_symbols
    if max_symbols is not None and (not isinstance(max_symbols, int) or max_symbols <= 0):
        raise ValueError(f"max_symbols must be a positive integer or None, got {max_symbols}")

    # Validate min_signal
    if not isinstance(min_signal, (int, float)) or min_signal < 0:
        raise ValueError(f"min_signal must be a non-negative number, got {min_signal}")

    # Validate execution_mode
    VALID_MODES = {"sequential", "threadpool", "asyncio", "processpool", "gpu_batch", "dask", "auto"}
    if execution_mode not in VALID_MODES:
        raise ValueError(f"execution_mode must be one of {VALID_MODES}, got {execution_mode}")

    # Validate max_workers
    if max_workers is not None and (not isinstance(max_workers, int) or max_workers <= 0):
        raise ValueError(f"max_workers must be a positive integer or None, got {max_workers}")

    try:
        hw_manager = get_hardware_manager()
        mem_manager = get_memory_manager()

        log_progress("Fetching futures symbols from Binance...")
        all_symbols = data_fetcher.list_binance_futures_symbols(
            max_candidates=None,  # Get all symbols first
            progress_label="Symbol Discovery",
        )

        if not all_symbols:
            log_error("No symbols found")
            return pd.DataFrame(), pd.DataFrame()

        # Limit symbols if max_symbols specified
        if max_symbols and max_symbols > 0:
            symbols = all_symbols[:max_symbols]
            log_success(f"Found {len(all_symbols)} futures symbols, scanning first {len(symbols)} symbols")
        else:
            symbols = all_symbols
            log_success(f"Found {len(symbols)} futures symbols")

        # Use hardware manager to determine optimal execution mode if auto
        if execution_mode == "auto":
            execution_mode = hw_manager.get_optimal_execution_mode(len(symbols))
            log_progress(f"Auto-selected execution mode: {execution_mode}")

        # Use hardware manager to determine optimal workers if not provided
        if max_workers is None:
            config = hw_manager.get_optimal_workload_config(len(symbols))
            max_workers = config.num_threads if execution_mode == "threadpool" else config.num_processes

        log_progress(
            f"Scanning {len(symbols)} symbols for ATC signals using {execution_mode} mode (workers={max_workers})..."
        )

        # Use safe memory operation for the entire scan
        with mem_manager.safe_memory_operation(f"scan_all_symbols:{len(symbols)}"):
            # Route to appropriate execution method
            if execution_mode == "sequential":
                results, skipped_count, error_count, skipped_symbols = _scan_sequential(
                    symbols, data_fetcher, atc_config, min_signal, batch_size
                )
            elif execution_mode == "threadpool":
                results, skipped_count, error_count, skipped_symbols = _scan_threadpool(
                    symbols, data_fetcher, atc_config, min_signal, max_workers, batch_size
                )
            elif execution_mode == "asyncio":
                results, skipped_count, error_count, skipped_symbols = _scan_asyncio(
                    symbols, data_fetcher, atc_config, min_signal, max_workers, batch_size
                )
            elif execution_mode == "processpool":
                results, skipped_count, error_count, skipped_symbols = _scan_processpool(
                    symbols, data_fetcher, atc_config, min_signal, max_workers, batch_size
                )
            elif execution_mode == "gpu_batch":
                results, skipped_count, error_count, skipped_symbols = _scan_gpu_batch(
                    symbols, data_fetcher, atc_config, min_signal, batch_size
                )
            elif execution_mode == "dask":
                results, skipped_count, error_count, skipped_symbols = _scan_dask(
                    symbols, data_fetcher, atc_config, min_signal, npartitions, batch_size
                )
            else:
                # Fallback to sequential
                results, skipped_count, error_count, skipped_symbols = _scan_sequential(
                    symbols, data_fetcher, atc_config, min_signal, batch_size
                )

            total = len(symbols)

            # Summary logging
            log_progress(
                f"Scan complete: {total} total, {len(results)} signals found, "
                f"{skipped_count} skipped, {error_count} errors"
            )

            if skipped_count > 0 and len(skipped_symbols) <= 10:
                log_warn(f"Skipped symbols: {', '.join(skipped_symbols)}")
            elif skipped_count > 10:
                log_warn(f"Skipped {skipped_count} symbols (first 10: {', '.join(skipped_symbols[:10])}...)")

            if not results:
                log_warn("No signals found above threshold")
                return pd.DataFrame(), pd.DataFrame()

            # Create DataFrame using from_records for better performance
            # Type is explicit to avoid inference overhead
            if results:
                results_df = pd.DataFrame.from_records(results)
            else:
                results_df = pd.DataFrame()

            # Fix: Guard against empty results_df before accessing columns
            # This prevents KeyError when results_df is empty (no valid signals)
            if results_df.empty:
                return pd.DataFrame(), pd.DataFrame()

            # Fix: Guard against empty results_df before accessing columns
            # This prevents KeyError when results_df is empty (no valid signals)
            if results_df.empty:
                return pd.DataFrame(), pd.DataFrame()

            # Filter LONG and SHORT signals
            long_signals = results_df[results_df["trend"] > 0].copy()
            short_signals = results_df[results_df["trend"] < 0].copy()

            # Sort by signal strength (absolute value)
            long_signals = long_signals.sort_values("signal", ascending=False).reset_index(drop=True)
            short_signals = short_signals.sort_values("signal", ascending=True).reset_index(drop=True)

            log_success(f"Found {len(long_signals)} LONG signals and {len(short_signals)} SHORT signals")

            # Final memory log
            mem_manager.log_memory_stats()

            return long_signals, short_signals

    except KeyboardInterrupt:
        log_warn("Scan interrupted by user")
        return pd.DataFrame(), pd.DataFrame()
    except Exception as e:
        log_error(f"Fatal error scanning symbols: {type(e).__name__}: {e}")
        import traceback

        log_error(f"Traceback: {traceback.format_exc()}")
        return pd.DataFrame(), pd.DataFrame()
