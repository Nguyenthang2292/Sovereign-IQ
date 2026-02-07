"""Async API surface for incremental ATC.

This module provides a clean, single-import interface for async incremental ATC usage.

Usage:
    from modules.adaptive_trend_LTS_mini.core.compute_atc_signals.incremental import (
        AsyncIncrementalATC,
        AsyncMultiTimeframeIncrementalATC,
        process_price_stream,
    )

    # Or from the async_api directly
    from modules.adaptive_trend_LTS_mini.core.compute_atc_signals.incremental.async_api import (
        AsyncIncrementalATC,
        process_price_stream,
    )

Example:
    >>> from modules.adaptive_trend_LTS_mini.core.compute_atc_signals.incremental import AsyncIncrementalATC
    >>> import asyncio
    >>>
    >>> config = {"ema_len": 28, "hma_len": 28, "wma_len": 28}
    >>> atc = AsyncIncrementalATC(config)
    >>>
    >>> # Initialize with historical data
    >>> historical_prices = pd.Series([100.0, 101.0, 99.5, 102.0])
    >>> asyncio.run(atc.initialize(historical_prices))
    >>>
    >>> # Update with new price
    >>> signal = asyncio.run(atc.update(103.5))
    >>> print(f"Signal: {signal}")

WebSocket Integration Example:
    >>> async def handle_websocket(websocket):
    ...     atc = AsyncIncrementalATC(config)
    ...     await atc.initialize(historical_data)
    ...
    ...     async def send_signal(signal):
    ...         await websocket.send_json({"signal": signal})
    ...
    ...     # Process incoming price stream
    ...     await process_price_stream(atc, websocket, on_signal=send_signal)
"""

from __future__ import annotations

from .async_wrapper import (
    AsyncIncrementalATC,
    AsyncMultiTimeframeIncrementalATC,
    process_price_stream,
)

__all__ = [
    "AsyncIncrementalATC",
    "AsyncMultiTimeframeIncrementalATC",
    "process_price_stream",
]
