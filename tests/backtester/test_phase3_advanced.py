"""
Advanced Phase 3 tests demonstrating lazy loading, memory monitoring, and TTL caching.

These tests showcase the final 10-20% memory optimization.
Run with: pytest tests/backtester/test_phase3_advanced.py -v
"""

import pytest


def test_lazy_data_generator(lazy_data_generator):
    """Test lazy data generator creates data on demand."""
    generator = lazy_data_generator()

    # Get first chunk
    chunk1 = next(generator)
    assert len(chunk1) == 100  # First 100 periods
    assert "open" in chunk1.columns
    assert "close" in chunk1.columns

    # Get second chunk
    chunk2 = next(generator)
    assert len(chunk2) == 100  # Next 100 periods (101-200)
    assert chunk2.index[0] > chunk1.index[-1]  # Sequential dates

    # Verify continuity (second chunk starts where first ended)
    assert abs(chunk2.iloc[0]["open"] - chunk1.iloc[-1]["close"]) < 1.0


def test_lazy_mock_data_fetcher(lazy_mock_data_fetcher):
    """Test lazy data fetcher with on-demand data generation."""
    # First fetch
    df1, exchange = lazy_mock_data_fetcher.fetch_ohlcv_with_fallback_exchange("BTC/USDT", limit=50)
    assert len(df1) == 50
    assert exchange == "binance"

    # Second fetch with different limit
    df2, _ = lazy_mock_data_fetcher.fetch_ohlcv_with_fallback_exchange("BTC/USDT", limit=25)
    assert len(df2) == 25

    # Same limit should use cache
    df3, _ = lazy_mock_data_fetcher.fetch_ohlcv_with_fallback_exchange("BTC/USDT", limit=50)
    assert len(df3) == 50
    assert df3.equals(df1)  # Should be cached


def test_memory_monitor_basic(memory_monitor):
    """Test basic memory monitoring functionality."""
    initial_memory = memory_monitor.get_memory_usage()

    # Create some data to use memory
    data = list(range(10000))  # Small memory usage

    final_memory = memory_monitor.get_memory_usage()

    # Memory should not be negative (basic sanity check)
    assert final_memory >= 0
    assert initial_memory >= 0
    del data  # Cleanup


def test_memory_monitor_tracking(memory_monitor):
    """Test memory tracking with context manager."""
    with memory_monitor.track() as tracker:
        # Create data inside tracking context
        large_list = list(range(50000))
        del large_list  # Cleanup

    # Should have tracked memory usage
    assert hasattr(tracker, "initial_memory")
    assert hasattr(tracker, "final_memory")
    assert tracker.peak_memory >= 0


def test_ttl_cache_basic(ttl_cache):
    """Test basic TTL cache functionality."""
    # Initially empty
    assert ttl_cache.get("test_key") is None

    # Set value
    ttl_cache.set("test_key", "test_value")
    assert ttl_cache.get("test_key") == "test_value"

    # Size should be 1
    assert ttl_cache.size() == 1


def test_ttl_cache_expiration(ttl_cache):
    """Test TTL cache expiration."""
    # Set with very short TTL for testing
    ttl_cache._ttl = 0.1  # 100ms

    ttl_cache.set("short_key", "expires_quickly")
    assert ttl_cache.get("short_key") == "expires_quickly"

    # Wait for expiration
    import time

    time.sleep(0.2)

    # Should be expired
    assert ttl_cache.get("short_key") is None

    # Reset TTL
    ttl_cache._ttl = 600


def test_cached_data_factory(cached_data_factory):
    """Test cached data factory with TTL."""
    call_count = [0]

    def expensive_operation():
        call_count[0] += 1
        return f"data_{call_count[0]}"

    # First call should create data
    result1 = cached_data_factory("test_factory_key", expensive_operation)
    assert result1 == "data_1"
    assert call_count[0] == 1

    # Second call should use cache
    result2 = cached_data_factory("test_factory_key", expensive_operation)
    assert result2 == "data_1"  # Same data
    assert call_count[0] == 1  # Not called again


def test_memory_monitor_assertion(memory_monitor):
    """Test memory monitoring assertions."""
    # This should pass (very high limit)
    memory_monitor.assert_memory_under(10000)  # 10GB limit

    # Test with reasonable limit
    current = memory_monitor.get_memory_usage()
    memory_monitor.assert_memory_under(current + 1000)  # Should pass


@pytest.mark.memory_intensive
def test_combined_phase3_features(lazy_mock_data_fetcher, memory_monitor, cached_data_factory):
    """Integration test combining all Phase 3 features."""
    with memory_monitor.track() as tracker:
        # Use lazy fetcher
        df, _ = lazy_mock_data_fetcher.fetch_ohlcv_with_fallback_exchange("BTC/USDT", limit=200)

        # Use cached factory
        processed_data = cached_data_factory(
            "processed_large_dataset",
            lambda: df.rolling(20).mean(),  # Expensive operation
        )

        assert len(df) == 200
        assert len(processed_data) == 200
        assert "open" in processed_data.columns
        assert "close" in processed_data.columns

    # Memory tracking should work (if psutil available)
    if hasattr(tracker, "peak_memory"):
        assert tracker.peak_memory >= 0
        print(f"\n📊 Phase 3 test memory usage: {tracker.peak_memory:.2f}MB")
    else:
        print("\n📊 Memory monitoring not available (psutil not installed)")
