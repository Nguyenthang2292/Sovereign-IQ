"""
Async I/O and Parallelism Abstraction for Adaptive Trend LTS

This module provides async wrappers and parallel execution helpers
to improve performance for I/O-bound and CPU-bound workloads.
"""

import asyncio
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor
from typing import Any, Callable, Dict, Optional

import pandas as pd

from modules.common.ui.logging import log_info


class AsyncComputeManager:
    """Manager for async and parallel computation."""

    def __init__(self, max_threads: int = 10, max_processes: Optional[int] = None):
        self.thread_executor = ThreadPoolExecutor(max_workers=max_threads)
        self.process_executor = ProcessPoolExecutor(max_workers=max_processes)
        log_info(f"AsyncComputeManager initialized with {max_threads} threads and {max_processes or 'auto'} processes")

    async def run_in_thread(self, func: Callable, *args, **kwargs) -> Any:
        """Run a synchronous function in a thread pool."""
        loop = asyncio.get_running_loop()
        return await loop.run_in_executor(self.thread_executor, func, *args, **kwargs)

    async def run_in_process(self, func: Callable, *args, **kwargs) -> Any:
        """Run a synchronous function in a process pool."""
        loop = asyncio.get_running_loop()
        return await loop.run_in_executor(self.process_executor, func, *args, **kwargs)

    async def compute_batch_async(
        self, symbols_data: Dict[str, pd.Series], compute_func: Callable, **kwargs
    ) -> Dict[str, Any]:
        """
        Compute signals for multiple symbols concurrently using threads.
        Suitable for I/O-bound or GIL-releasing (Rust/CUDA) workloads.
        """
        tasks = []
        for symbol, prices in symbols_data.items():
            tasks.append(self.run_in_thread(compute_func, prices, **kwargs))

        results = await asyncio.gather(*tasks)
        return dict(zip(symbols_data.keys(), results))


# Wrapper for compute_atc_signals
async def compute_atc_signals_async(prices: pd.Series, **kwargs) -> Dict[str, pd.Series]:
    """Async wrapper for compute_atc_signals."""
    from functools import partial

    from modules.adaptive_trend_LTS.core.compute_atc_signals.compute_atc_signals import compute_atc_signals

    # Run in default executor (usually thread pool)
    loop = asyncio.get_running_loop()
    return await loop.run_in_executor(None, partial(compute_atc_signals, prices, **kwargs))


async def run_batch_atc_async(symbols_data: Dict[str, pd.Series], **kwargs) -> Dict[str, Dict[str, pd.Series]]:
    """Compute ATC signals for multiple symbols concurrently."""
    from modules.adaptive_trend_LTS.core.compute_atc_signals.compute_atc_signals import compute_atc_signals

    tasks = []
    for symbol, prices in symbols_data.items():
        tasks.append(compute_atc_signals_async(prices, **kwargs))

    results = await asyncio.gather(*tasks)
    return dict(zip(symbols_data.keys(), results))
