"""Async wrapper for IncrementalATC with asyncio support.

This module provides async/await interfaces for incremental ATC updates,
making it compatible with asyncio event loops, WebSocket handlers, and
async web frameworks (FastAPI, etc.).

Usage:
    # Basic async usage
    atc = AsyncIncrementalATC(config)
    await atc.initialize(prices)
    signal = await atc.update(new_price)

    # WebSocket integration
    async for price in websocket_stream:
        signal = await atc.update(price)
        await websocket.send_json({"signal": signal})

    # Multi-timeframe async
    mtf = AsyncMultiTimeframeIncrementalATC(config, ["1m", "5m", "15m"])
    await mtf.initialize(historical_data)
    signals = await mtf.update(new_price)
"""

from __future__ import annotations

import asyncio
from concurrent.futures import Executor
from pathlib import Path
from typing import Any, Callable, Dict, Optional, Union

import pandas as pd

from .core import IncrementalATC
from .multi_timeframe import MultiTimeframeIncrementalATC

try:
    from modules.common.utils import log_debug
except ImportError:

    def log_debug(msg: str) -> None:
        print(f"[DEBUG] {msg}")


class AsyncIncrementalATC:
    """Async wrapper for IncrementalATC.

    Provides async/await interface for incremental ATC updates while
    maintaining thread safety for the underlying synchronous implementation.

    All compute-intensive operations are offloaded to a thread pool executor
    to avoid blocking the asyncio event loop.

    Example:
        >>> config = {"ema_len": 28, "hma_len": 28, ...}
        >>> atc = AsyncIncrementalATC(config)
        >>> await atc.initialize(historical_prices)
        >>> signal = await atc.update(99.5)
        >>> print(f"New signal: {signal}")
    """

    def __init__(
        self,
        config: Dict[str, Any],
        executor: Optional[Executor] = None,
    ):
        """Initialize async incremental ATC wrapper.

        Args:
            config: ATC configuration dictionary
            executor: Optional executor for running sync code (defaults to loop's executor)
        """
        self._config = config
        self._executor = executor
        self._atc = IncrementalATC(config)
        log_debug("AsyncIncrementalATC initialized")

    @property
    def state(self) -> Dict[str, Any]:
        """Access current state dictionary."""
        return self._atc.state

    async def initialize(
        self, prices: pd.Series, loop: Optional[asyncio.AbstractEventLoop] = None
    ) -> Dict[str, pd.Series]:
        """Initialize with historical data (async).

        Args:
            prices: Historical price series
            loop: Optional event loop (defaults to current loop)

        Returns:
            Full calculation results
        """
        if loop is None:
            loop = asyncio.get_running_loop()

        log_debug(f"Async initializing with {len(prices)} bars")
        results = await loop.run_in_executor(
            self._executor, self._atc.initialize, prices
        )
        log_debug("Async initialization complete")
        return results

    async def update(
        self, new_price: float, loop: Optional[asyncio.AbstractEventLoop] = None
    ) -> float:
        """Update with new price (async, O(1) operation).

        Args:
            new_price: New price value
            loop: Optional event loop (defaults to current loop)

        Returns:
            Updated signal value
        """
        if loop is None:
            loop = asyncio.get_running_loop()

        signal = await loop.run_in_executor(self._executor, self._atc.update, new_price)
        return signal

    async def batch_update(
        self, new_prices: Any, loop: Optional[asyncio.AbstractEventLoop] = None
    ) -> list[float]:
        """Update with multiple prices (async).

        Args:
            new_prices: Iterable of price values
            loop: Optional event loop (defaults to current loop)

        Returns:
            List of signal values
        """
        if loop is None:
            loop = asyncio.get_running_loop()

        signals = await loop.run_in_executor(
            self._executor, self._atc.batch_update, new_prices
        )
        return signals

    async def reset(self, loop: Optional[asyncio.AbstractEventLoop] = None):
        """Reset state (async).

        Args:
            loop: Optional event loop (defaults to current loop)
        """
        if loop is None:
            loop = asyncio.get_running_loop()

        await loop.run_in_executor(self._executor, self._atc.reset)
        log_debug("Async reset complete")

    async def save_state(
        self, path: Union[str, Path], loop: Optional[asyncio.AbstractEventLoop] = None
    ) -> None:
        """Save state to file (async).

        Args:
            path: File path for saving state
            loop: Optional event loop (defaults to current loop)
        """
        if loop is None:
            loop = asyncio.get_running_loop()

        await loop.run_in_executor(self._executor, self._atc.save_state, path)
        log_debug(f"Async state saved to {path}")

    @classmethod
    async def load_state(
        cls,
        path: Union[str, Path],
        executor: Optional[Executor] = None,
        loop: Optional[asyncio.AbstractEventLoop] = None,
    ) -> "AsyncIncrementalATC":
        """Load state from file (async).

        Args:
            path: File path to load state from
            executor: Optional executor for running sync code
            loop: Optional event loop (defaults to current loop)

        Returns:
            Restored AsyncIncrementalATC instance
        """
        if loop is None:
            loop = asyncio.get_running_loop()

        sync_atc = await loop.run_in_executor(None, IncrementalATC.load_state, path)
        async_atc = cls(sync_atc.config, executor=executor)
        async_atc._atc = sync_atc
        log_debug(f"Async state loaded from {path}")
        return async_atc


class AsyncMultiTimeframeIncrementalATC:
    """Async wrapper for MultiTimeframeIncrementalATC.

    Provides async/await interface for multi-timeframe incremental updates.

    Example:
        >>> config = {"ema_len": 28, ...}
        >>> mtf = AsyncMultiTimeframeIncrementalATC(config, ["1m", "5m", "15m"])
        >>> await mtf.initialize({"1m": prices_1m, "5m": prices_5m, "15m": prices_15m})
        >>> signals = await mtf.update(99.5)
        >>> print(f"Signals: {signals}")
    """

    def __init__(
        self,
        config: Dict[str, Any],
        timeframes: Optional[list[str]] = None,
        executor: Optional[Executor] = None,
    ):
        """Initialize async multi-timeframe ATC wrapper.

        Args:
            config: ATC configuration dictionary
            timeframes: List of timeframe strings (default: ["1m", "5m", "15m"])
            executor: Optional executor for running sync code
        """
        self._config = config
        self._executor = executor
        self._mtf = MultiTimeframeIncrementalATC(config, timeframes)
        self.timeframes = self._mtf.timeframes
        log_debug(f"AsyncMultiTimeframeIncrementalATC initialized with TFs: {self.timeframes}")

    async def initialize(
        self,
        historical_data: Union[Dict[str, pd.Series], pd.Series],
        loop: Optional[asyncio.AbstractEventLoop] = None,
    ) -> Dict[str, Dict[str, pd.Series]]:
        """Initialize all timeframes with historical data (async).

        Args:
            historical_data: Either dict mapping timeframe to prices, or single series
            loop: Optional event loop (defaults to current loop)

        Returns:
            Dict of initialization results per timeframe
        """
        if loop is None:
            loop = asyncio.get_running_loop()

        log_debug("Async MTF initializing")
        results = await loop.run_in_executor(
            self._executor, self._mtf.initialize, historical_data
        )
        log_debug("Async MTF initialization complete")
        return results

    async def update(
        self,
        new_price: float,
        timeframe: Optional[str] = None,
        loop: Optional[asyncio.AbstractEventLoop] = None,
    ) -> Dict[str, float]:
        """Update signals across all timeframes (async).

        Args:
            new_price: New price value
            timeframe: Source timeframe (default: base timeframe)
            loop: Optional event loop (defaults to current loop)

        Returns:
            Dict of signal values per timeframe
        """
        if loop is None:
            loop = asyncio.get_running_loop()

        signals = await loop.run_in_executor(
            self._executor, self._mtf.update, new_price, timeframe
        )
        return signals

    async def reset(self, loop: Optional[asyncio.AbstractEventLoop] = None):
        """Reset all timeframes (async).

        Args:
            loop: Optional event loop (defaults to current loop)
        """
        if loop is None:
            loop = asyncio.get_running_loop()

        await loop.run_in_executor(self._executor, self._mtf.reset)
        log_debug("Async MTF reset complete")

    async def get_state(
        self, tf: Optional[str] = None, loop: Optional[asyncio.AbstractEventLoop] = None
    ) -> Dict:
        """Get state for specific or all timeframes (async).

        Args:
            tf: Specific timeframe (default: all)
            loop: Optional event loop (defaults to current loop)

        Returns:
            State dictionary
        """
        if loop is None:
            loop = asyncio.get_running_loop()

        state = await loop.run_in_executor(self._executor, self._mtf.get_state, tf)
        return state

    async def get_signal(
        self, tf: Optional[str] = None, loop: Optional[asyncio.AbstractEventLoop] = None
    ) -> Union[float, Dict[str, float]]:
        """Get signal for specific or all timeframes (async).

        Args:
            tf: Specific timeframe (default: all)
            loop: Optional event loop (defaults to current loop)

        Returns:
            Signal value or dict of signals
        """
        if loop is None:
            loop = asyncio.get_running_loop()

        signal = await loop.run_in_executor(self._executor, self._mtf.get_signal, tf)
        return signal


# Convenience function for stream processing
async def process_price_stream(
    atc: AsyncIncrementalATC,
    price_stream: Any,
    on_signal: Optional[Callable[..., Any]] = None,
) -> None:
    """Process a stream of prices through async ATC.

    Args:
        atc: AsyncIncrementalATC instance (already initialized)
        price_stream: Async iterable of price values
        on_signal: Optional callback for each signal (async or sync)

    Example:
        >>> atc = AsyncIncrementalATC(config)
        >>> await atc.initialize(historical_prices)
        >>> async def print_signal(signal):
        ...     print(f"Signal: {signal}")
        >>> await process_price_stream(atc, websocket_prices, on_signal=print_signal)
    """
    async for price in price_stream:
        signal = await atc.update(price)
        if on_signal is not None:
            if asyncio.iscoroutinefunction(on_signal):
                await on_signal(signal)
            else:
                on_signal(signal)
