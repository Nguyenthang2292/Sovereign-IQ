"""Asyncio-based scanning implementation for ATC scanner.

This module provides the _scan_asyncio function and async helpers for
parallel symbol scanning using asyncio.
"""

from __future__ import annotations

import asyncio
import gc
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


async def _process_symbol_async(
    symbol: str,
    data_fetcher: "DataFetcher",
    atc_config: ATCConfig,
    min_signal: float,
    loop: asyncio.AbstractEventLoop,
) -> Optional[Dict[str, Any]]:
    """Async wrapper for _process_symbol using asyncio.to_thread."""
    try:
        result = await loop.run_in_executor(
            None,
            _process_symbol,
            symbol,
            data_fetcher,
            atc_config,
            min_signal,
        )
        return result
    except Exception:
        return None


def _scan_asyncio(
    symbols: list,
    data_fetcher: "DataFetcher",
    atc_config: ATCConfig,
    min_signal: float,
    max_workers: Optional[int],
    batch_size: int = 100,
) -> Tuple[list, int, int, list]:
    """Scan symbols using asyncio with batched processing."""

    async def _async_scan():
        loop = asyncio.get_event_loop()
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

            if max_workers is not None:
                # Use semaphore to limit concurrent tasks
                semaphore = asyncio.Semaphore(max_workers)

                async def _process_with_semaphore(symbol):
                    async with semaphore:
                        result = await _process_symbol_async(symbol, data_fetcher, atc_config, min_signal, loop)
                        return symbol, result

                tasks = [_process_with_semaphore(symbol) for symbol in batch_symbols]
            else:
                # Wrap to include symbol
                async def _wrap_with_symbol(symbol):
                    result = await _process_symbol_async(symbol, data_fetcher, atc_config, min_signal, loop)
                    return symbol, result

                tasks = [_wrap_with_symbol(symbol) for symbol in batch_symbols]

            try:
                # Process results as they complete
                for coro in asyncio.as_completed(tasks):
                    try:
                        symbol, result = await coro
                        completed += 1

                        if result is None:
                            skipped_count += 1
                            skipped_symbols.append(symbol)
                        else:
                            results.append(result)

                        # Progress update every 10 symbols
                        if completed % 10 == 0 or completed == total:
                            log_progress(
                                f"Scanned {completed}/{total} symbols... "
                                f"Found {len(results)} signals, "
                                f"Skipped {skipped_count}, Errors {error_count}"
                            )
                    except Exception as e:
                        error_count += 1
                        completed += 1
                        # Try to get symbol from task if possible
                        skipped_symbols.append("UNKNOWN")
                        log_warn(f"Error processing symbol: {type(e).__name__}: {e}. Skipping and continuing...")
            except KeyboardInterrupt:
                log_warn("Scan interrupted by user")
                # Cancel remaining tasks
                for task in tasks:
                    if not task.done():
                        task.cancel()
                break

            # Force garbage collection after each batch
            gc.collect()

        return results, skipped_count, error_count, skipped_symbols

    # Run async function
    try:
        return asyncio.run(_async_scan())
    except RuntimeError:
        # If we're already in an event loop, use nest_asyncio or create new loop
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        try:
            return loop.run_until_complete(_async_scan())
        finally:
            loop.close()
