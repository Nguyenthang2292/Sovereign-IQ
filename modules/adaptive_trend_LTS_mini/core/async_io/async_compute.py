"""
Async I/O and Parallelism Abstraction for Adaptive Trend LTS

This module provides async wrappers and parallel execution helpers
to improve performance for I/O-bound and CPU-bound workloads.
"""

import asyncio
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor
from functools import partial
from typing import Any, Callable, Dict, Optional, TypeVar, List

from typing import ParamSpec

import pandas as pd

from modules.common.ui.logging import log_info, log_error

P = ParamSpec("P")
R = TypeVar("R")


class AsyncComputeManager:
    """Manager for async and parallel computation."""

    def __init__(
        self,
        max_threads: int = 10,
        max_processes: Optional[int] = None,
        enable_processes: bool = False,
    ):
        """
        Initialize AsyncComputeManager.

        Args:
            max_threads: Maximum number of threads
            max_processes: Maximum number of processes (None = CPU count)
            enable_processes: Whether to create process pool (lazy by default)
        """
        self.thread_executor = ThreadPoolExecutor(max_workers=max_threads)
        self._process_executor: Optional[ProcessPoolExecutor] = None
        self._max_processes = max_processes

        if enable_processes:
            self._process_executor = ProcessPoolExecutor(max_workers=max_processes)
            log_info(
                f"AsyncComputeManager initialized with {max_threads} threads and {max_processes or 'auto'} processes"
            )
        else:
            log_info(
                f"AsyncComputeManager initialized with {max_threads} threads (processes disabled)"
            )

    @property
    def process_executor(self) -> ProcessPoolExecutor:
        """Lazy-load process executor."""
        if self._process_executor is None:
            self._process_executor = ProcessPoolExecutor(max_workers=self._max_processes)
            log_info("Process pool executor created on demand")
        return self._process_executor

    async def run_in_thread(
        self, func: Callable[P, R], *args: P.args, **kwargs: P.kwargs
    ) -> R:
        """Run a synchronous function in a thread pool."""
        loop = asyncio.get_running_loop()
        bound_func = partial(func, *args, **kwargs)
        return await loop.run_in_executor(self.thread_executor, bound_func)

    async def run_in_process(
        self, func: Callable[P, R], *args: P.args, **kwargs: P.kwargs
    ) -> R:
        """Run a synchronous function in a process pool."""
        loop = asyncio.get_running_loop()
        bound_func = partial(func, *args, **kwargs)
        return await loop.run_in_executor(self.process_executor, bound_func)

    async def _compute_all_async(
        self,
        symbols_data: Dict[str, pd.Series],
        compute_func: Callable,
        return_exceptions: bool = False,
        **kwargs,
    ) -> Dict[str, Any]:
        """Internal helper to compute all symbols at once."""
        bound_func = partial(compute_func, **kwargs)
        tasks = [
            self.run_in_thread(bound_func, prices) for prices in symbols_data.values()
        ]

        results = await asyncio.gather(*tasks, return_exceptions=return_exceptions)
        result_dict = dict(zip(symbols_data.keys(), results))

        # Log failures if not returning exceptions
        if not return_exceptions:
            for symbol, result in result_dict.items():
                if isinstance(result, Exception):
                    log_error(f"Failed to compute signals for {symbol}: {result}")

        return result_dict

    async def compute_batch_async(
        self,
        symbols_data: Dict[str, pd.Series],
        compute_func: Callable,
        batch_size: Optional[int] = None,
        return_exceptions: bool = False,
        progress_callback: Optional[Callable[[str, int, int], None]] = None,
        **kwargs,
    ) -> Dict[str, Any]:
        """
        Compute signals for multiple symbols concurrently using threads.
        Suitable for I/O-bound or GIL-releasing (Rust/CUDA) workloads.

        Args:
            symbols_data: Dictionary of symbol -> price data
            compute_func: Function to compute signals
            batch_size: If set, process symbols in batches of this size
            return_exceptions: If True, return exceptions instead of raising
            progress_callback: Optional callback(symbol, completed, total)
            **kwargs: Additional arguments for compute_func

        Returns:
            Dictionary of symbol -> results (or exceptions if return_exceptions=True)
        """
        # If granular progress tracking is needed
        if progress_callback:
            bound_func = partial(compute_func, **kwargs)
            
            async def wrapped_task(sym, p):
                try:
                    res = await self.run_in_thread(bound_func, p)
                    return sym, res
                except Exception as e:
                    if return_exceptions:
                        return sym, e
                    raise e

            # If batch size is set, we still process in batches but we need to manage the overall progress
            if batch_size is not None and batch_size > 0:
                results = {}
                symbols_list = list(symbols_data.items())
                total = len(symbols_list)
                completed = 0
                
                for i in range(0, total, batch_size):
                    batch_items = symbols_list[i : i + batch_size]
                    batch_wrapped_tasks = [
                        wrapped_task(s, p) for s, p in batch_items
                    ]
                    
                    for coro in asyncio.as_completed(batch_wrapped_tasks):
                        try:
                            sym, res = await coro
                            results[sym] = res
                            completed += 1
                            progress_callback(sym, completed, total)
                        except Exception as e:
                            # Should only happen if return_exceptions=False and task failed
                             if not return_exceptions:
                                 log_error(f"Task failed in progress tracking: {e}")
                                 raise e
                return results

            else:
                # Process all at once with progress
                wrapped_tasks = [
                    wrapped_task(s, p) for s, p in symbols_data.items()
                ]
                results = {}
                completed = 0
                total = len(wrapped_tasks)
                
                for coro in asyncio.as_completed(wrapped_tasks):
                    try:
                        sym, res = await coro
                        results[sym] = res
                        completed += 1
                        progress_callback(sym, completed, total)
                    except Exception as e:
                        if not return_exceptions:
                             log_error(f"Task failed in progress tracking: {e}")
                             raise e
                return results

        # No progress callback logic
        if batch_size is None or batch_size <= 0:
            return await self._compute_all_async(
                symbols_data, compute_func, return_exceptions, **kwargs
            )

        # Batch processing without granular progress (just log info per batch)
        results = {}
        symbols_list = list(symbols_data.items())
        total_batches = (len(symbols_list) + batch_size - 1) // batch_size

        for i in range(0, len(symbols_list), batch_size):
            batch = dict(symbols_list[i : i + batch_size])
            batch_results = await self._compute_all_async(
                batch, compute_func, return_exceptions, **kwargs
            )
            results.update(batch_results)
            log_info(f"Processed batch {i // batch_size + 1}/{total_batches}")

        return results

    def shutdown(self, wait: bool = True) -> None:
        """Shutdown all executors."""
        log_info("Shutting down AsyncComputeManager executors")
        self.thread_executor.shutdown(wait=wait)
        if self._process_executor:
            self._process_executor.shutdown(wait=wait)

    def __enter__(self):
        """Context manager entry."""
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit - ensures cleanup."""
        self.shutdown()
        return False

    async def __aenter__(self):
        """Async context manager entry."""
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Async context manager exit - ensures cleanup."""
        self.shutdown()
        return False


# Wrapper for compute_atc_signals
async def compute_atc_signals_async(prices: pd.Series, **kwargs) -> Dict[str, pd.Series]:
    """Async wrapper for compute_atc_signals."""
    from functools import partial

    from modules.adaptive_trend_LTS_mini.core.compute_atc_signals.compute_atc_signals import (
        compute_atc_signals,
    )

    # Run in default executor (usually thread pool)
    loop = asyncio.get_running_loop()
    return await loop.run_in_executor(
        None, partial(compute_atc_signals, prices, **kwargs)
    )


async def run_batch_atc_async(
    symbols_data: Dict[str, pd.Series],
    manager: Optional[AsyncComputeManager] = None,
    **kwargs,
) -> Dict[str, Dict[str, pd.Series]]:
    """Compute ATC signals for multiple symbols concurrently."""
    from modules.adaptive_trend_LTS_mini.core.compute_atc_signals.compute_atc_signals import (
        compute_atc_signals,
    )

    should_cleanup = False
    if manager is None:
        manager = AsyncComputeManager()
        should_cleanup = True

    try:
        return await manager.compute_batch_async(
            symbols_data, compute_atc_signals, **kwargs
        )
    finally:
        if should_cleanup:
            manager.shutdown()
