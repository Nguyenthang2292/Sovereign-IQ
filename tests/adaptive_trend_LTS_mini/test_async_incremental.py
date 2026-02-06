"""Tests for async incremental ATC wrapper.

Test coverage:
- AsyncIncrementalATC basic operations
- AsyncMultiTimeframeIncrementalATC operations
- Stream processing with callbacks
- Thread safety and concurrent updates
"""

import asyncio
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from modules.adaptive_trend_LTS_mini.core.compute_atc_signals.incremental import (
    AsyncIncrementalATC,
    AsyncMultiTimeframeIncrementalATC,
    process_price_stream,
)
from modules.adaptive_trend_LTS_mini.utils.config import ATCConfig


# ============================================================================
# Fixtures
# ============================================================================


@pytest.fixture
def sample_config():
    """Create sample ATC configuration."""
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
    ).to_dict()
    # Add incremental-specific parameters
    config["use_o1_mas"] = True
    config["use_rust_incremental"] = False
    return config


@pytest.fixture
def sample_prices():
    """Generate sample price data."""
    np.random.seed(42)
    base_price = 100.0
    num_bars = 200
    returns = np.random.normal(0.0005, 0.02, num_bars)
    prices = base_price * np.exp(np.cumsum(returns))
    return pd.Series(prices)


# ============================================================================
# Test AsyncIncrementalATC
# ============================================================================


@pytest.mark.asyncio
async def test_async_initialize(sample_config, sample_prices):
    """Test async initialization."""
    atc = AsyncIncrementalATC(sample_config)
    results = await atc.initialize(sample_prices)

    assert results is not None
    assert "Average_Signal" in results
    assert atc.state["initialized"] is True
    assert len(atc.state["price_history"]) > 0


@pytest.mark.asyncio
async def test_async_update(sample_config, sample_prices):
    """Test async update operation."""
    atc = AsyncIncrementalATC(sample_config)
    await atc.initialize(sample_prices)

    new_price = 105.5
    signal = await atc.update(new_price)

    assert isinstance(signal, (float, np.floating))
    assert np.isfinite(signal)
    assert atc.state["price_history"][-1] == new_price


@pytest.mark.asyncio
async def test_async_batch_update(sample_config, sample_prices):
    """Test async batch update."""
    atc = AsyncIncrementalATC(sample_config)
    await atc.initialize(sample_prices)

    new_prices = [105.0, 106.0, 104.5, 107.0]
    signals = await atc.batch_update(new_prices)

    assert len(signals) == len(new_prices)
    assert all(np.isfinite(s) for s in signals)


@pytest.mark.asyncio
async def test_async_reset(sample_config, sample_prices):
    """Test async reset operation."""
    atc = AsyncIncrementalATC(sample_config)
    await atc.initialize(sample_prices)
    await atc.update(105.0)

    await atc.reset()

    assert atc.state["initialized"] is False
    assert len(atc.state["price_history"]) == 0


@pytest.mark.asyncio
async def test_async_save_load_state(sample_config, sample_prices):
    """Test async state persistence."""
    atc = AsyncIncrementalATC(sample_config)
    await atc.initialize(sample_prices)
    await atc.update(105.0)

    original_signal = atc.state["signal"]

    # Save state
    with tempfile.TemporaryDirectory() as tmpdir:
        state_path = Path(tmpdir) / "state.msgpack"
        await atc.save_state(state_path)

        # Load state
        restored_atc = await AsyncIncrementalATC.load_state(state_path)

        assert restored_atc.state["initialized"] is True
        assert restored_atc.state["signal"] == original_signal
        assert len(restored_atc.state["price_history"]) == len(atc.state["price_history"])


# ============================================================================
# Test AsyncMultiTimeframeIncrementalATC
# ============================================================================


@pytest.mark.asyncio
async def test_async_mtf_initialize(sample_config):
    """Test multi-timeframe async initialization."""
    np.random.seed(42)
    prices = pd.Series(100 * np.exp(np.cumsum(np.random.normal(0, 0.01, 300))))

    mtf = AsyncMultiTimeframeIncrementalATC(
        sample_config, timeframes=["1m", "5m", "15m"]
    )

    historical_data = {
        "1m": prices[:240],
        "5m": prices[:240:5],
        "15m": prices[:240:15],
    }

    results = await mtf.initialize(historical_data)

    assert "1m" in results
    assert "5m" in results
    assert "15m" in results


@pytest.mark.asyncio
async def test_async_mtf_update(sample_config):
    """Test multi-timeframe async update."""
    np.random.seed(42)
    prices = pd.Series(100 * np.exp(np.cumsum(np.random.normal(0, 0.01, 300))))

    mtf = AsyncMultiTimeframeIncrementalATC(
        sample_config, timeframes=["1m", "5m"]
    )

    historical_data = {
        "1m": prices[:200],
        "5m": prices[:200:5],
    }

    await mtf.initialize(historical_data)

    # Update with new price
    signals = await mtf.update(105.0)

    assert "1m" in signals
    assert "5m" in signals
    assert all(np.isfinite(s) for s in signals.values())


@pytest.mark.asyncio
async def test_async_mtf_get_signal(sample_config):
    """Test multi-timeframe get_signal."""
    np.random.seed(42)
    prices = pd.Series(100 * np.exp(np.cumsum(np.random.normal(0, 0.01, 200))))

    mtf = AsyncMultiTimeframeIncrementalATC(
        sample_config, timeframes=["1m", "5m"]
    )

    await mtf.initialize(prices[:150])
    await mtf.update(105.0)

    # Get all signals
    signals = await mtf.get_signal()
    assert "1m" in signals
    assert "5m" in signals

    # Get specific signal
    signal_1m = await mtf.get_signal("1m")
    assert isinstance(signal_1m, (float, np.floating))


# ============================================================================
# Test Stream Processing
# ============================================================================


@pytest.mark.asyncio
async def test_process_price_stream(sample_config, sample_prices):
    """Test stream processing with callback."""
    atc = AsyncIncrementalATC(sample_config)
    await atc.initialize(sample_prices)

    # Track signals
    received_signals = []

    async def callback(signal):
        received_signals.append(signal)

    # Create async stream
    async def price_stream():
        for price in [105.0, 106.0, 104.5]:
            yield price

    await process_price_stream(atc, price_stream(), on_signal=callback)

    assert len(received_signals) == 3
    assert all(np.isfinite(s) for s in received_signals)


@pytest.mark.asyncio
async def test_stream_processing_sync_callback(sample_config, sample_prices):
    """Test stream processing with synchronous callback."""
    atc = AsyncIncrementalATC(sample_config)
    await atc.initialize(sample_prices)

    received_signals = []

    def sync_callback(signal):  # Non-async callback
        received_signals.append(signal)

    async def price_stream():
        for price in [105.0, 106.0]:
            yield price

    await process_price_stream(atc, price_stream(), on_signal=sync_callback)

    assert len(received_signals) == 2


# ============================================================================
# Test Thread Safety and Concurrency
# ============================================================================


@pytest.mark.asyncio
async def test_concurrent_updates(sample_config, sample_prices):
    """Test concurrent updates are handled safely."""
    atc = AsyncIncrementalATC(sample_config)
    await atc.initialize(sample_prices)

    # Sequential updates (simulating concurrent access)
    tasks = [
        atc.update(100.0 + i * 0.5)
        for i in range(10)
    ]

    signals = await asyncio.gather(*tasks)

    assert len(signals) == 10
    assert all(np.isfinite(s) for s in signals)


@pytest.mark.asyncio
async def test_multiple_instances(sample_config, sample_prices):
    """Test multiple independent async instances."""
    atc1 = AsyncIncrementalATC(sample_config)
    atc2 = AsyncIncrementalATC(sample_config)

    await asyncio.gather(
        atc1.initialize(sample_prices),
        atc2.initialize(sample_prices),
    )

    signals = await asyncio.gather(
        atc1.update(105.0),
        atc2.update(106.0),
    )

    assert len(signals) == 2
    # Signals can differ due to different price updates
    assert signals[0] != signals[1] or sample_prices.iloc[-1] == 105.0


# ============================================================================
# Test Error Handling
# ============================================================================


@pytest.mark.asyncio
async def test_update_before_initialize(sample_config):
    """Test error handling when updating before initialization."""
    atc = AsyncIncrementalATC(sample_config)

    with pytest.raises(RuntimeError, match="Must call initialize"):
        await atc.update(100.0)


@pytest.mark.asyncio
async def test_invalid_price(sample_config, sample_prices):
    """Test error handling with invalid price values."""
    atc = AsyncIncrementalATC(sample_config)
    await atc.initialize(sample_prices)

    with pytest.raises(ValueError):
        await atc.update(float('nan'))

    with pytest.raises(ValueError):
        await atc.update(float('inf'))

    with pytest.raises(ValueError):
        await atc.update(-100.0)  # Negative price


# ============================================================================
# Integration Tests
# ============================================================================


@pytest.mark.asyncio
async def test_full_async_workflow(sample_config):
    """Test complete async workflow."""
    np.random.seed(42)

    # Generate data
    historical = pd.Series(100 * np.exp(np.cumsum(np.random.normal(0, 0.01, 200))))
    new_prices = pd.Series(100 * np.exp(np.cumsum(np.random.normal(0, 0.01, 20))))

    # Initialize
    atc = AsyncIncrementalATC(sample_config)
    await atc.initialize(historical)

    # Process updates
    signals = []
    for price in new_prices:
        signal = await atc.update(price)
        signals.append(signal)

    # Verify
    assert len(signals) == len(new_prices)
    assert atc.state["initialized"] is True

    # Save and restore
    with tempfile.TemporaryDirectory() as tmpdir:
        state_path = Path(tmpdir) / "state.msgpack"
        await atc.save_state(state_path)

        restored = await AsyncIncrementalATC.load_state(state_path)
        final_signal = await restored.update(105.0)

        assert np.isfinite(final_signal)


if __name__ == "__main__":
    # Run tests with pytest
    pytest.main([__file__, "-v", "-s"])
