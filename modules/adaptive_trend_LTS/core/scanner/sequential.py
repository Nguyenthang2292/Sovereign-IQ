"""Sequential scanning implementation for ATC scanner.

This module provides the _scan_sequential function and batched processing
for sequential symbol scanning.
"""

from __future__ import annotations

import gc
from typing import TYPE_CHECKING, Any, Dict, Iterator, Optional, Tuple

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


def _process_symbols_batched(
    symbols: list,
    data_fetcher: "DataFetcher",
    atc_config: ATCConfig,
    min_signal: float,
    batch_size: int = 100,
) -> Iterator[Optional[Dict[str, Any]]]:
    """Process symbols in batches using a generator to reduce memory usage.

    This generator function processes symbols in batches and yields results
    one at a time, allowing for memory cleanup between batches.

    Args:
        symbols: List of symbols to process
        data_fetcher: DataFetcher instance
        atc_config: ATCConfig object
        min_signal: Minimum signal strength threshold
        batch_size: Number of symbols to process before forcing GC (default: 100)

    Yields:
        Optional[Dict[str, Any]]: Result for each symbol, or None if skipped/error
    """
    total = len(symbols)
    processed = 0

    for i in range(0, total, batch_size):
        batch = symbols[i : i + batch_size]
        batch_num = (i // batch_size) + 1
        total_batches = (total + batch_size - 1) // batch_size

        # Process batch
        for symbol in batch:
            try:
                result = _process_symbol(symbol, data_fetcher, atc_config, min_signal)
                processed += 1
                yield result
            except Exception as e:
                processed += 1
                log_warn(f"Error processing symbol {symbol}: {type(e).__name__}: {e}. Skipping and continuing...")
                yield None

        # Force garbage collection after each batch to free memory
        gc.collect()

        # Progress update
        if processed % 10 == 0 or processed == total:
            log_progress(f"Processed {processed}/{total} symbols (batch {batch_num}/{total_batches})...")


def _scan_sequential(
    symbols: list,
    data_fetcher: "DataFetcher",
    atc_config: ATCConfig,
    min_signal: float,
    batch_size: int = 100,
) -> Tuple[list, int, int, list]:
    """Scan symbols sequentially using batched generator processing."""
    results = []
    skipped_count = 0
    error_count = 0
    skipped_symbols = []
    total = len(symbols)

    try:
        for idx, result in enumerate(
            _process_symbols_batched(symbols, data_fetcher, atc_config, min_signal, batch_size), 1
        ):
            if result is None:
                skipped_count += 1
                if idx <= len(symbols):
                    skipped_symbols.append(symbols[idx - 1] if idx > 0 else "UNKNOWN")
            else:
                results.append(result)

            # Progress update every 10 symbols
            if idx % 10 == 0 or idx == total:
                log_progress(
                    f"Scanned {idx}/{total} symbols... "
                    f"Found {len(results)} signals, "
                    f"Skipped {skipped_count}, Errors {error_count}"
                )
    except KeyboardInterrupt:
        log_warn("Scan interrupted by user")

    return results, skipped_count, error_count, skipped_symbols
