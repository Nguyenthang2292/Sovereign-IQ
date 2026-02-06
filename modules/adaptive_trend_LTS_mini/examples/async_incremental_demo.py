"""Demo: Async Incremental ATC with WebSocket and FastAPI integration examples.

This demo showcases multiple ways to use AsyncIncrementalATC:
1. Basic async usage with simulated price stream
2. Multi-timeframe async updates
3. WebSocket integration pattern (mock)
4. FastAPI integration pattern (mock)

Run with:
    python modules/adaptive_trend_LTS_mini/examples/async_incremental_demo.py
"""

import asyncio
from pathlib import Path
import sys
import numpy as np
import pandas as pd

# Add project root to path
project_root = Path(__file__).resolve().parents[4]
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

try:
    from modules.adaptive_trend_LTS_mini.core.compute_atc_signals.incremental import (
        AsyncIncrementalATC,
        AsyncMultiTimeframeIncrementalATC,
        process_price_stream,
    )
    from modules.adaptive_trend_LTS_mini.utils.config import ATCConfig
except ImportError:
    # Try relative import if running from examples directory
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    from core.compute_atc_signals.incremental import (
        AsyncIncrementalATC,
        AsyncMultiTimeframeIncrementalATC,
        process_price_stream,
    )
    from utils.config import ATCConfig


def generate_sample_prices(base_price: float = 100.0, num_bars: int = 200) -> pd.Series:
    """Generate sample price data for demonstration."""
    np.random.seed(42)
    returns = np.random.normal(0.0005, 0.02, num_bars)
    prices = base_price * np.exp(np.cumsum(returns))
    return pd.Series(prices)


async def simulate_price_stream(prices: pd.Series, delay_ms: int = 100):
    """Simulate a real-time price stream with delays."""
    for price in prices:
        await asyncio.sleep(delay_ms / 1000.0)
        yield float(price)


# =============================================================================
# Demo 1: Basic Async Usage
# =============================================================================


async def demo_basic_async():
    """Demonstrate basic async incremental ATC usage."""
    print("\n" + "=" * 80)
    print("DEMO 1: Basic Async Incremental ATC")
    print("=" * 80)

    # Setup configuration
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
        use_rust_backend=False,
        use_o1_mas=True,
        use_rust_incremental=False,
    ).to_dict()

    # Generate sample data
    historical_prices = generate_sample_prices(100.0, 200)
    new_prices = generate_sample_prices(historical_prices.iloc[-1], 50)

    print(f"\n[1] Initializing with {len(historical_prices)} historical bars...")
    atc = AsyncIncrementalATC(config)
    await atc.initialize(historical_prices)
    print(f"    Initial signal: {atc.state['signal']:.4f}")

    print(f"\n[2] Processing {len(new_prices)} new prices asynchronously...")
    signals = []
    for i, price in enumerate(new_prices[:10]):  # Show first 10
        signal = await atc.update(price)
        signals.append(signal)
        if i % 3 == 0:
            print(f"    Bar {i + 1}: Price={price:.2f}, Signal={signal:.4f}")

    print(f"\n[3] Final signal: {signals[-1]:.4f}")
    print("    ✓ Basic async usage complete!")


# =============================================================================
# Demo 2: Multi-Timeframe Async
# =============================================================================


async def demo_multi_timeframe():
    """Demonstrate multi-timeframe async incremental ATC."""
    print("\n" + "=" * 80)
    print("DEMO 2: Multi-Timeframe Async Incremental ATC")
    print("=" * 80)

    config = ATCConfig(
        ema_len=20,
        hma_len=20,
        wma_len=20,
        dema_len=20,
        lsma_len=20,
        kama_len=20,
        robustness="Low",
        lambda_param=3.0,
        decay=0.01,
        cutout=50,
        use_rust_backend=False,
        use_o1_mas=True,
        use_rust_incremental=False,
    ).to_dict()

    # Generate sample data for different timeframes
    prices_1m = generate_sample_prices(100.0, 500)
    prices_5m = prices_1m[::5]  # Downsample for 5m
    prices_15m = prices_1m[::15]  # Downsample for 15m

    print(f"\n[1] Initializing multi-timeframe ATC...")
    mtf = AsyncMultiTimeframeIncrementalATC(
        config, timeframes=["1m", "5m", "15m"]
    )

    historical_data = {
        "1m": prices_1m.iloc[:400],
        "5m": prices_5m.iloc[:80],
        "15m": prices_15m.iloc[:26],
    }

    await mtf.initialize(historical_data)
    print(f"    Timeframes: {mtf.timeframes}")
    initial_signals = await mtf.get_signal()
    print(f"    Initial signals: {initial_signals}")

    print(f"\n[2] Simulating 1-minute updates...")
    # Simulate 15 new 1-minute bars
    new_1m_prices = prices_1m.iloc[400:415]

    for i, price in enumerate(new_1m_prices):
        signals = await mtf.update(price, timeframe="1m")
        if i % 5 == 0 or (i + 1) % 5 == 0:  # Show every 5th and 15th
            print(f"    Bar {i + 1}: Price={price:.2f}")
            print(f"        1m={signals['1m']:.4f}, 5m={signals['5m']:.4f}, 15m={signals['15m']:.4f}")

    print("\n    ✓ Multi-timeframe async complete!")


# =============================================================================
# Demo 3: WebSocket Pattern (Mock)
# =============================================================================


async def demo_websocket_pattern():
    """Demonstrate WebSocket integration pattern."""
    print("\n" + "=" * 80)
    print("DEMO 3: WebSocket Integration Pattern (Mock)")
    print("=" * 80)

    config = ATCConfig(
        ema_len=14,
        hma_len=14,
        wma_len=14,
        dema_len=14,
        lsma_len=14,
        kama_len=14,
        robustness="Low",
        lambda_param=2.0,
        decay=0.01,
        cutout=30,
        use_rust_backend=False,
        use_o1_mas=True,
        use_rust_incremental=False,
    ).to_dict()

    # Mock WebSocket callbacks
    received_signals = []

    async def on_price_update(price: float, atc: AsyncIncrementalATC):
        """Callback for each price update from WebSocket."""
        signal = await atc.update(price)
        received_signals.append(signal)
        # In real WebSocket, you'd send to clients here
        return {"price": price, "signal": signal, "timestamp": "mock"}

    # Initialize
    historical_prices = generate_sample_prices(100.0, 150)
    atc = AsyncIncrementalATC(config)
    await atc.initialize(historical_prices)

    print(f"\n[1] Simulating WebSocket price stream...")
    print("    (In production, this would connect to Binance/exchange WebSocket)")

    # Simulate incoming WebSocket messages
    new_prices = generate_sample_prices(historical_prices.iloc[-1], 20)

    for i, price in enumerate(new_prices):
        await asyncio.sleep(0.05)  # Simulate network delay
        result = await on_price_update(price, atc)

        if i % 5 == 0:
            print(f"    WS Message {i + 1}: {result}")

    print(f"\n[2] Received {len(received_signals)} signals via WebSocket")
    print(f"    Latest signal: {received_signals[-1]:.4f}")
    print("    ✓ WebSocket pattern complete!")


# =============================================================================
# Demo 4: FastAPI Integration Pattern (Mock)
# =============================================================================


async def demo_fastapi_pattern():
    """Demonstrate FastAPI integration pattern."""
    print("\n" + "=" * 80)
    print("DEMO 4: FastAPI Integration Pattern (Mock)")
    print("=" * 80)

    config = ATCConfig(
        ema_len=21,
        hma_len=21,
        wma_len=21,
        dema_len=21,
        lsma_len=21,
        kama_len=21,
        robustness="Medium",
        lambda_param=4.0,
        decay=0.005,
        cutout=80,
        use_rust_backend=False,
        use_o1_mas=True,
        use_rust_incremental=False,
    ).to_dict()

    # Simulate FastAPI global state (in real app, use app.state or dependency injection)
    class MockAppState:
        def __init__(self):
            self.atc: AsyncIncrementalATC = None
            self.is_initialized = False

    app_state = MockAppState()

    # Mock FastAPI endpoint functions
    async def startup_event():
        """Mock FastAPI startup event."""
        print("\n[FastAPI] Startup: Initializing ATC...")
        historical_prices = generate_sample_prices(100.0, 200)
        app_state.atc = AsyncIncrementalATC(config)
        await app_state.atc.initialize(historical_prices)
        app_state.is_initialized = True
        print(f"[FastAPI] ATC initialized with signal: {app_state.atc.state['signal']:.4f}")

    async def post_price_endpoint(price: float):
        """Mock POST /price endpoint."""
        if not app_state.is_initialized:
            return {"error": "ATC not initialized"}

        signal = await app_state.atc.update(price)
        return {
            "price": price,
            "signal": signal,
            "state": "updated",
        }

    async def get_signal_endpoint():
        """Mock GET /signal endpoint."""
        if not app_state.is_initialized:
            return {"error": "ATC not initialized"}

        return {
            "signal": app_state.atc.state.get("signal"),
            "average_signal": app_state.atc.state.get("average_signal"),
            "initialized": app_state.is_initialized,
        }

    # Simulate FastAPI lifecycle
    print("\n[1] Simulating FastAPI app lifecycle...")
    await startup_event()

    print("\n[2] Simulating API requests...")
    new_prices = generate_sample_prices(100.0, 10)

    for i, price in enumerate(new_prices):
        result = await post_price_endpoint(float(price))
        if i % 3 == 0:
            print(f"    POST /price: {result}")

        if i == 5:
            signal_result = await get_signal_endpoint()
            print(f"    GET /signal: {signal_result}")

    print("\n    ✓ FastAPI pattern complete!")


# =============================================================================
# Demo 5: Stream Processing with Callback
# =============================================================================


async def demo_stream_processing():
    """Demonstrate stream processing with callback."""
    print("\n" + "=" * 80)
    print("DEMO 5: Stream Processing with Callback")
    print("=" * 80)

    config = ATCConfig(
        ema_len=18,
        hma_len=18,
        wma_len=18,
        dema_len=18,
        lsma_len=18,
        kama_len=18,
        robustness="Low",
        lambda_param=3.5,
        decay=0.008,
        cutout=60,
        use_rust_backend=False,
        use_o1_mas=True,
        use_rust_incremental=False,
    ).to_dict()

    # Setup
    historical_prices = generate_sample_prices(100.0, 180)
    new_prices = generate_sample_prices(historical_prices.iloc[-1], 30)

    atc = AsyncIncrementalATC(config)
    await atc.initialize(historical_prices)

    print(f"\n[1] Processing price stream with callback...")

    # Track signals
    processed_signals = []

    async def signal_callback(signal: float):
        """Callback for each signal."""
        processed_signals.append(signal)
        if len(processed_signals) % 10 == 0:
            print(f"    Processed {len(processed_signals)} signals, latest={signal:.4f}")

    # Process stream
    price_stream = simulate_price_stream(new_prices, delay_ms=50)
    await process_price_stream(atc, price_stream, on_signal=signal_callback)

    print(f"\n[2] Stream processing complete!")
    print(f"    Total signals processed: {len(processed_signals)}")
    print(f"    Signal range: [{min(processed_signals):.4f}, {max(processed_signals):.4f}]")
    print("    ✓ Stream processing complete!")


# =============================================================================
# Main Demo Runner
# =============================================================================


async def main():
    """Run all demos."""
    print("\n" + "=" * 80)
    print("ASYNC INCREMENTAL ATC DEMO SUITE")
    print("=" * 80)
    print("\nThis demo showcases async/await integration patterns for incremental ATC.")
    print("All demos use mock data and simulated streams for demonstration purposes.\n")

    try:
        await demo_basic_async()
        await demo_multi_timeframe()
        await demo_websocket_pattern()
        await demo_fastapi_pattern()
        await demo_stream_processing()

        print("\n" + "=" * 80)
        print("ALL DEMOS COMPLETE! ✓")
        print("=" * 80)
        print("\nKey Takeaways:")
        print("  • AsyncIncrementalATC wraps sync operations in thread pool executor")
        print("  • Compatible with asyncio event loops (WebSocket, FastAPI, etc.)")
        print("  • Multi-timeframe support with AsyncMultiTimeframeIncrementalATC")
        print("  • Stream processing helper: process_price_stream()")
        print("  • All operations are thread-safe and non-blocking")
        print("\n")

    except Exception as e:
        print(f"\n❌ Demo failed with error: {e}")
        import traceback

        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    asyncio.run(main())
