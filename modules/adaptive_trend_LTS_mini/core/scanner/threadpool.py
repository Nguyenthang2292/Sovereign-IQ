"""ThreadPool-based scanning implementation for ATC scanner.

This module provides the _scan_threadpool function for parallel symbol
scanning using ThreadPoolExecutor.
"""

from __future__ import annotations

import gc
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import TYPE_CHECKING, Optional, Tuple

if TYPE_CHECKING:
    from modules.common.core.data_fetcher import DataFetcher

try:
    from modules.common.utils import log_progress, log_warn
except ImportError:

    def log_warn(msg: str) -> None:
        print(f"[WARN] {msg}")

    def log_progress(msg: str) -> None:
        print(f"[PROGRESS] {msg}")


from modules.adaptive_trend_LTS_mini.utils.config import ATCConfig

from .process_symbol import _process_symbol


def _scan_threadpool(
    symbols: list,
    data_fetcher: "DataFetcher",
    atc_config: ATCConfig,
    min_signal: float,
    max_workers: Optional[int],
    batch_size: int = 100,
) -> Tuple[list, int, int, list]:
    """Scan symbols using ThreadPoolExecutor with batched processing.

    Uses a single ThreadPoolExecutor for the entire scan to avoid
    the overhead of creating/destroying pools for each batch.
    """
    if max_workers is None:
        max_workers = min(32, len(symbols) + 4)

    results = []
    skipped_count = 0
    error_count = 0
    skipped_symbols = []
    total = len(symbols)
    completed = 0
    all_futures = {}

    # Use a single ThreadPoolExecutor for the entire scan
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        # Submit all tasks in batches (submit doesn't block)
        for batch_start in range(0, total, batch_size):
            batch_end = min(batch_start + batch_size, total)
            batch_symbols = symbols[batch_start:batch_end]

            # Submit batch tasks to the shared executor
            for symbol in batch_symbols:
                future = executor.submit(_process_symbol, symbol, data_fetcher, atc_config, min_signal)
                all_futures[future] = symbol

        # Process completed tasks as they finish
        try:
            for future in as_completed(all_futures):
                symbol = all_futures[future]
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
                    log_warn(f"Error processing symbol {symbol}: {type(e).__name__}: {e}. Skipping and continuing...")

                # Progress update every 10 symbols
                if completed % 10 == 0 or completed == total:
                    log_progress(
                        f"Scanned {completed}/{total} symbols... "
                        f"Found {len(results)} signals, "
                        f"Skipped {skipped_count}, Errors {error_count}"
                    )
        except KeyboardInterrupt:
            log_warn("Scan interrupted by user")
            # Cancel remaining tasks
            for future in all_futures:
                future.cancel()

    # Garbage collect only once after entire scan completes
    # (or remove entirely if memory pressure is not an issue)
    gc.collect()

    return results, skipped_count, error_count, skipped_symbols
