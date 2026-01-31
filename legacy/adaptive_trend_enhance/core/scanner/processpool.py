"""ProcessPool-based scanning implementation for ATC scanner.

This module provides the _scan_processpool function for parallel symbol
scanning using ProcessPoolExecutor.
"""

from __future__ import annotations

import gc
import multiprocessing as mp
from concurrent.futures import ProcessPoolExecutor, as_completed
from typing import TYPE_CHECKING, Any, Dict, List, Optional, Tuple

if TYPE_CHECKING:
    from modules.common.core.data_fetcher import DataFetcher

try:
    from modules.common.utils import log_debug, log_progress, log_warn
except ImportError:

    def log_warn(message: str) -> None:
        print(f"[WARN] {message}")

    def log_progress(message: str) -> None:
        print(f"[PROGRESS] {message}")

    def log_debug(message: str) -> None:
        print(f"[DEBUG] {message}")


from modules.adaptive_trend_enhance.utils.config import ATCConfig

from .process_symbol import _process_symbol

# Global data fetcher for worker processes
_worker_data_fetcher: Optional[DataFetcher] = None


def _init_worker(api_key: Optional[str], api_secret: Optional[str], testnet: bool):
    """Initialize a worker process with its own DataFetcher."""
    global _worker_data_fetcher
    try:
        from modules.common.core.data_fetcher import DataFetcher
        from modules.common.core.exchange_manager import (
            AuthenticatedExchangeManager,
            ExchangeManager,
            PublicExchangeManager,
        )

        # Initialize managers
        # We use a fresh pool of connections per process
        public_mgr = PublicExchangeManager()
        auth_mgr = AuthenticatedExchangeManager(api_key=api_key, api_secret=api_secret, testnet=testnet)
        ex_mgr = ExchangeManager(auth_mgr, public_mgr)

        _worker_data_fetcher = DataFetcher(ex_mgr)
    except Exception as e:
        # We can't easily log from here if loggers aren't initialized
        print(f"Failed to initialize worker process: {e}")


def _worker_task(symbol: str, atc_config: ATCConfig, min_signal: float) -> Optional[Dict[str, Any]]:
    """Worker task to process a single symbol."""
    global _worker_data_fetcher
    if _worker_data_fetcher is None:
        return None

    # To avoid nested ProcessPoolExecutor (Level 2), we might want to
    # disable it here for signals calculation if we are already in a process pool.
    # However, compute_atc_signals handles this by check if len(prices) > 500
    # and we can also set an environment variable or flag.

    return _process_symbol(symbol, _worker_data_fetcher, atc_config, min_signal)


def _scan_processpool(
    symbols: List[str],
    data_fetcher: "DataFetcher",
    atc_config: ATCConfig,
    min_signal: float,
    max_workers: Optional[int],
    batch_size: int = 100,
) -> Tuple[List[Dict[str, Any]], int, int, List[str]]:
    """Scan symbols using ProcessPoolExecutor with batched processing."""
    if max_workers is None:
        # Default to CPU count - 1 to leave room for main process
        max_workers = max(1, mp.cpu_count() - 1)

    # Extract credentials for worker initialization
    api_key = None
    api_secret = None
    testnet = False

    try:
        if hasattr(data_fetcher, "exchange_manager") and data_fetcher.exchange_manager.authenticated:
            auth = data_fetcher.exchange_manager.authenticated
            api_key = auth.default_api_key
            api_secret = auth.default_api_secret
            testnet = auth.testnet
    except Exception:
        pass

    results = []
    skipped_count = 0
    error_count = 0
    skipped_symbols = []
    total = len(symbols)
    completed = 0

    log_debug(f"Starting ProcessPool scan with {max_workers} workers")

    # Use spawn context if available to be safer across platforms
    ctx = mp.get_context("spawn")

    # Create executor once and reuse across batches
    with ProcessPoolExecutor(
        max_workers=max_workers, mp_context=ctx, initializer=_init_worker, initargs=(api_key, api_secret, testnet)
    ) as executor:
        # Process in batches
        for batch_start in range(0, total, batch_size):
            batch_end = min(batch_start + batch_size, total)
            batch_symbols = symbols[batch_start:batch_end]

            # Submit batch tasks
            future_to_symbol = {
                executor.submit(_worker_task, symbol, atc_config, min_signal): symbol for symbol in batch_symbols
            }

            # Process completed tasks
            try:
                for future in as_completed(future_to_symbol):
                    symbol = future_to_symbol[future]
                    completed += 1

                    try:
                        result = future.result()
                        if result is None:
                            skipped_count += 1
                            skipped_symbols.append(symbol)
                        else:
                            results.append(result)
                    except Exception as e:
                        error_count += 1
                        skipped_symbols.append(symbol)
                        log_warn(f"Error processing symbol {symbol} in ProcessPool: {type(e).__name__}: {e}")

                    # Progress update every 10 symbols
                    if completed % 10 == 0 or completed == total:
                        log_progress(
                            f"Scanned {completed}/{total} symbols... "
                            f"Found {len(results)} signals, "
                            f"Skipped {skipped_count}, Errors {error_count}"
                        )
            except KeyboardInterrupt:
                log_warn("Scan interrupted by user")
                # Executor context exit will handle shutdown
                raise  # Re-raise to break outer loop

            # Force garbage collection in main process after each batch
            gc.collect()

    return results, skipped_count, error_count, skipped_symbols
