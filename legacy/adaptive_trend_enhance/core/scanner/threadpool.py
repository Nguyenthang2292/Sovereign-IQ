"""ThreadPool-based scanning implementation for ATC scanner.

This module provides the _scan_threadpool function for parallel symbol
scanning using ThreadPoolExecutor.
"""

from __future__ import annotations

import gc
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import TYPE_CHECKING, Any, Dict, Optional, Tuple

if TYPE_CHECKING:
    from modules.common.core.data_fetcher import DataFetcher

try:
    from modules.common.utils import log_progress, log_warn
except ImportError:

    def log_warn(message: str) -> None:
        print(f"[WARN] {message}")

    def log_progress(message: str) -> None:
        print(f"[PROGRESS] {message}")


from modules.adaptive_trend_enhance.utils.config import ATCConfig
from .process_symbol import _process_symbol


def _scan_threadpool(
    symbols: list,
    data_fetcher: "DataFetcher",
    atc_config: ATCConfig,
    min_signal: float,
    max_workers: Optional[int],
    batch_size: int = 100,
    ohlcv_cache: Optional[Dict[str, pd.DataFrame]] = None,
) -> Tuple[list, int, int, list]:
    """Scan symbols using ThreadPoolExecutor with batched processing."""
    if max_workers is None:
        max_workers = min(32, len(symbols) + 4)

    results = []
    skipped_count = 0
    error_count = 0
    skipped_symbols = []
    total = len(symbols)
    completed = 0

    # Process in batches to reduce memory usage
    for batch_start in range(0, total, batch_size):
        batch_end = min(batch_start + batch_size, total)
        batch_symbols = symbols[batch_start:batch_end]

        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            # Submit batch tasks
            future_to_symbol = {
                executor.submit(_process_symbol, symbol, data_fetcher, atc_config, min_signal, ohlcv_cache): symbol
                for symbol in batch_symbols
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
                        log_warn(
                            f"Error processing symbol {symbol}: {type(e).__name__}: {e}. Skipping and continuing..."
                        )

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
                for future in future_to_symbol:
                    future.cancel()
                break

        # Force garbage collection after each batch
        gc.collect()

    return results, skipped_count, error_count, skipped_symbols
