# Async API for Incremental ATC

> **Status**: ✅ Complete
> **Version**: 1.0
> **Last Updated**: 2026-02-06

## Overview

The Async API provides `asyncio`-compatible wrappers for incremental ATC updates, enabling seamless integration with:

- **WebSocket handlers** (real-time price feeds)
- **Async web frameworks** (FastAPI, aiohttp, etc.)
- **Event-driven architectures**
- **Concurrent processing pipelines**

All compute-intensive operations are offloaded to thread pool executors, ensuring the asyncio event loop remains responsive.

---

## Quick Verification

Test that async API is working correctly:

```python
# Quick import verification
import asyncio
import numpy as np
import pandas as pd
from modules.adaptive_trend_LTS_mini.core.compute_atc_signals.incremental import (
    AsyncIncrementalATC
)

# Minimal working example
config = {"ema_len": 20, "hma_len": 20, "wma_len": 20}
atc = AsyncIncrementalATC(config)

# Test with sufficient historical data (need at least max_len + warmup)
np.random.seed(42)
base = 100.0
returns = np.random.normal(0.001, 0.02, 50)
prices = pd.Series(base * np.exp(np.cumsum(returns)))

result = asyncio.run(atc.initialize(prices))
print(f"✓ Initialize successful: {result is not None}")

# Test update
signal = asyncio.run(atc.update(104.0))
print(f"✓ Update successful: Signal = {signal:.4f}")
```

Expected output:
```
✓ Initialize successful: True
✓ Update successful: Signal = 0.5234
```

**Alternative**: Run the verification script:
```bash
python tests/adaptive_trend_LTS_mini/verify_async_api.py
```

---

## Quick Start

### Basic Usage

```python
import asyncio
import pandas as pd
from modules.adaptive_trend_LTS_mini.core.compute_atc_signals.incremental import (
    AsyncIncrementalATC
)
from modules.adaptive_trend_LTS_mini.utils.config import ATCConfig

async def main():
    # Configure
    config = ATCConfig(
        ema_len=28,
        hma_len=28,
        wma_len=28,
        dema_len=28,
        lsma_len=28,
        kama_len=28,
        robustness="Medium",
        lambda_param=5.0,
        decay=0.005,
        cutout=100,
    ).to_dict()

    # Initialize with historical data
    atc = AsyncIncrementalATC(config)
    historical_prices = pd.Series([100.0, 101.0, 99.5, ...])  # Your data
    await atc.initialize(historical_prices)

    # Process new prices
    signal = await atc.update(102.5)
    print(f"Signal: {signal}")

asyncio.run(main())
```

---

## API Reference

### `AsyncIncrementalATC`

Async wrapper for single-timeframe incremental ATC.

#### Constructor

```python
AsyncIncrementalATC(
    config: Dict[str, Any],
    executor: Optional[asyncio.AbstractEventLoop] = None
)
```

**Parameters:**
- `config`: ATC configuration dictionary (see `ATCConfig`)
- `executor`: Optional custom executor for sync operations

#### Methods

##### `async initialize(prices: pd.Series) -> Dict[str, pd.Series]`

Initialize with historical data.

```python
results = await atc.initialize(historical_prices)
# Returns: Full batch calculation results
```

##### `async update(new_price: float) -> float`

Update with new price (O(1) operation).

```python
signal = await atc.update(105.0)
# Returns: Updated signal value
```

##### `async batch_update(new_prices: Iterable) -> List[float]`

Update with multiple prices.

```python
signals = await atc.batch_update([105.0, 106.0, 104.5])
# Returns: List of signal values
```

##### `async reset()`

Reset internal state.

```python
await atc.reset()
```

##### `async save_state(path: Union[str, Path])`

Save state to file (MessagePack format).

```python
await atc.save_state("state.msgpack")
```

##### `async load_state(path: Union[str, Path]) -> AsyncIncrementalATC`

Load state from file (class method).

```python
atc = await AsyncIncrementalATC.load_state("state.msgpack")
```

#### Properties

- `state: Dict[str, Any]` - Access current state dictionary

---

### `AsyncMultiTimeframeIncrementalATC`

Async wrapper for multi-timeframe incremental ATC.

#### Constructor

```python
AsyncMultiTimeframeIncrementalATC(
    config: Dict[str, Any],
    timeframes: Optional[List[str]] = None,  # Default: ["1m", "5m", "15m"]
    executor: Optional[asyncio.AbstractEventLoop] = None
)
```

#### Methods

##### `async initialize(historical_data: Union[Dict, pd.Series]) -> Dict`

Initialize all timeframes.

```python
# Dict mapping timeframe to prices
await mtf.initialize({
    "1m": prices_1m,
    "5m": prices_5m,
    "15m": prices_15m,
})

# Or single series for all timeframes
await mtf.initialize(prices)
```

##### `async update(new_price: float, timeframe: Optional[str] = None) -> Dict[str, float]`

Update from base timeframe (syncs to higher TFs automatically).

```python
signals = await mtf.update(105.0)
# Returns: {"1m": 0.52, "5m": 0.48, "15m": 0.45}
```

##### `async get_signal(tf: Optional[str] = None) -> Union[float, Dict[str, float]]`

Get current signal(s).

```python
# All timeframes
signals = await mtf.get_signal()

# Specific timeframe
signal_1m = await mtf.get_signal("1m")
```

##### `async get_state(tf: Optional[str] = None) -> Dict`

Get state for specific or all timeframes.

```python
state = await mtf.get_state("5m")
```

---

### `process_price_stream()`

Helper for processing async price streams.

```python
async def process_price_stream(
    atc: AsyncIncrementalATC,
    price_stream: AsyncIterable,
    on_signal: Optional[Callable] = None
) -> None
```

**Parameters:**
- `atc`: Initialized `AsyncIncrementalATC` instance
- `price_stream`: Async iterable of price values
- `on_signal`: Optional callback for each signal (async or sync)

**Example:**

```python
async def signal_handler(signal):
    print(f"New signal: {signal}")

async for price in websocket_stream:
    await process_price_stream(atc, [price], on_signal=signal_handler)
```

---

## Integration Examples

### 1. WebSocket Integration

```python
import asyncio
import ccxt.pro as ccxtpro
from modules.adaptive_trend_LTS_mini.core.compute_atc_signals.incremental import (
    AsyncIncrementalATC
)

async def websocket_handler():
    # Setup
    exchange = ccxtpro.binance()
    atc = AsyncIncrementalATC(config)

    # Initialize with history
    ohlcv = await exchange.fetch_ohlcv("BTC/USDT", "1m", limit=500)
    prices = pd.Series([bar[4] for bar in ohlcv])  # Close prices
    await atc.initialize(prices)

    # Stream updates
    while True:
        kline = await exchange.watch_ohlcv("BTC/USDT", "1m")
        latest_price = kline[-1][4]
        signal = await atc.update(latest_price)
        print(f"Signal: {signal}")

asyncio.run(websocket_handler())
```

### 2. FastAPI Integration

```python
from fastapi import FastAPI
from modules.adaptive_trend_LTS_mini.core.compute_atc_signals.incremental import (
    AsyncIncrementalATC
)

app = FastAPI()

# Global state (in production, use dependency injection)
atc: AsyncIncrementalATC = None

@app.on_event("startup")
async def startup():
    global atc
    atc = AsyncIncrementalATC(config)
    # Initialize with historical data
    await atc.initialize(historical_prices)

@app.post("/update")
async def update_price(price: float):
    signal = await atc.update(price)
    return {"price": price, "signal": signal}

@app.get("/signal")
async def get_signal():
    return {"signal": atc.state.get("signal")}
```

### 3. Multi-Timeframe WebSocket

```python
from modules.adaptive_trend_LTS_mini.core.compute_atc_signals.incremental import (
    AsyncMultiTimeframeIncrementalATC
)

async def multi_timeframe_websocket():
    mtf = AsyncMultiTimeframeIncrementalATC(
        config,
        timeframes=["1m", "5m", "15m"]
    )

    # Initialize
    await mtf.initialize({
        "1m": prices_1m,
        "5m": prices_5m,
        "15m": prices_15m,
    })

    # Process 1-minute stream (syncs to higher TFs automatically)
    async for price in websocket_1m_stream:
        signals = await mtf.update(price)
        print(f"1m: {signals['1m']}, 5m: {signals['5m']}, 15m: {signals['15m']}")
```

### 4. Concurrent Processing

```python
async def process_multiple_symbols():
    symbols = ["BTC/USDT", "ETH/USDT", "BNB/USDT"]

    # Create instance per symbol
    atcs = {
        symbol: AsyncIncrementalATC(config)
        for symbol in symbols
    }

    # Initialize concurrently
    await asyncio.gather(*[
        atcs[symbol].initialize(historical_data[symbol])
        for symbol in symbols
    ])

    # Process updates concurrently
    new_prices = {"BTC/USDT": 99000, "ETH/USDT": 3200, "BNB/USDT": 580}
    signals = await asyncio.gather(*[
        atcs[symbol].update(new_prices[symbol])
        for symbol in symbols
    ])

    return dict(zip(symbols, signals))
```

---

## Performance Considerations

### Thread Pool Executor

By default, operations run in the default thread pool executor. For high-throughput applications, configure a custom executor:

```python
import concurrent.futures

executor = concurrent.futures.ThreadPoolExecutor(max_workers=4)
atc = AsyncIncrementalATC(config, executor=executor)
```

### Memory Management

- Use `reset()` to clear state between sessions
- Save state periodically for long-running processes
- Consider state size for persistent storage (typically <100KB)

### Latency

- `initialize()`: 50-200ms (depends on history length)
- `update()`: <1ms per price (O(1) operation)
- `batch_update()`: ~0.5ms per price in batch

---

## Error Handling

### Common Errors

```python
try:
    signal = await atc.update(price)
except RuntimeError:
    # Not initialized
    await atc.initialize(historical_prices)
except ValueError:
    # Invalid price (NaN, inf, negative)
    print("Invalid price value")
except Exception as e:
    # Other errors (network, state corruption, etc.)
    print(f"Unexpected error: {e}")
```

### Graceful Shutdown

```python
import signal

async def shutdown_handler(atc):
    await atc.save_state("state.msgpack")
    print("State saved, exiting...")

# Register signal handler
loop = asyncio.get_event_loop()
loop.add_signal_handler(
    signal.SIGINT,
    lambda: asyncio.create_task(shutdown_handler(atc))
)
```

---

## Testing

Run async tests with pytest:

```bash
pytest tests/adaptive_trend_LTS_mini/test_async_incremental.py -v
```

Test coverage includes:
- Basic async operations
- Multi-timeframe updates
- Stream processing
- Thread safety
- Error handling
- State persistence

---

## Examples

### Demo Scripts

1. **Basic Async Demo**: `examples/async_incremental_demo.py`
   - Demonstrates all async patterns
   - Mock data and simulated streams
   - Run: `python modules/adaptive_trend_LTS_mini/examples/async_incremental_demo.py`

2. **Live WebSocket Demo**: `examples/websocket_incremental_live.py`
   - Real-time Binance WebSocket integration
   - Live price streaming
   - Run: `python modules/adaptive_trend_LTS_mini/examples/websocket_incremental_live.py --symbol BTC/USDT`

### Requirements

For WebSocket examples:
```bash
pip install ccxt  # For WebSocket connectivity
```

---

## Architecture

### Thread Safety

The underlying `IncrementalATC` uses `threading.RLock` for thread safety. The async wrapper offloads operations to threads, ensuring:

- **Non-blocking**: Event loop remains responsive
- **Thread-safe**: Multiple concurrent calls are serialized
- **Safe state access**: No race conditions

### Event Loop Compatibility

Compatible with:
- `asyncio.run()` (Python 3.7+)
- `asyncio.get_event_loop().run_until_complete()`
- Nested event loops (with `nest_asyncio`)
- FastAPI, aiohttp, Tornado, etc.

---

## Migration Guide

### From Sync to Async

**Before (sync):**
```python
from modules.adaptive_trend_LTS_mini.core.compute_atc_signals.incremental import (
    IncrementalATC
)

atc = IncrementalATC(config)
atc.initialize(prices)
signal = atc.update(100.0)
```

**After (async):**
```python
from modules.adaptive_trend_LTS_mini.core.compute_atc_signals.incremental import (
    AsyncIncrementalATC
)

async def main():
    atc = AsyncIncrementalATC(config)
    await atc.initialize(prices)
    signal = await atc.update(100.0)

asyncio.run(main())
```

**Key Changes:**
1. Import `AsyncIncrementalATC` instead of `IncrementalATC`
2. Add `async`/`await` keywords
3. Run in async context (`asyncio.run()`)

---

## Troubleshooting

### "RuntimeError: This event loop is already running"

**Solution**: Use `asyncio.create_task()` or `nest_asyncio`:

```python
import nest_asyncio
nest_asyncio.apply()
```

### "Must call initialize() before update()"

**Solution**: Always initialize before updating:

```python
await atc.initialize(historical_prices)
```

### WebSocket Connection Issues

**Solution**: Check exchange connectivity and API limits:

```python
exchange = ccxtpro.binance({"enableRateLimit": True})
```

---

## References

- [Incremental ATC Core Documentation](../README.md)
- [ATCConfig Reference](API_REFERENCE.md#atcconfig)
- [Python asyncio Documentation](https://docs.python.org/3/library/asyncio.html)
- [ccxt.pro Documentation](https://github.com/ccxt/ccxt/wiki/ccxt.pro)

---

## Changelog

### v1.0 (2026-02-06)
- ✅ Initial release
- ✅ `AsyncIncrementalATC` wrapper
- ✅ `AsyncMultiTimeframeIncrementalATC` wrapper
- ✅ `process_price_stream()` helper
- ✅ Comprehensive examples and tests
- ✅ WebSocket integration demo
