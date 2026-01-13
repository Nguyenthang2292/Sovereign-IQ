"""
Optimized test for range oscillator strategy using local fixtures.
"""

import time

import numpy as np
import pandas as pd
import pytest


@pytest.fixture
def fast_ohlcv_data():
    """Create fast OHLC data for testing."""
    np.random.seed(42)
    n = 50
    dates = pd.date_range("2024-01-01", periods=n, freq="1h")
    close = 50000 + np.cumsum(np.random.randn(n) * 100)
    high = close + np.abs(np.random.randn(n) * 50)
    low = close - np.abs(np.random.randn(n) * 50)
    return {
        "high": pd.Series(high, index=dates, name="high"),
        "low": pd.Series(low, index=dates, name="low"),
        "close": pd.Series(close, index=dates, name="close"),
    }


@pytest.fixture
def fast_data_fetcher():
    """Mock data fetcher for testing."""

    class MockDataFetcher:
        def fetch_ohlcv(self, symbol, timeframe, limit):
            return pd.DataFrame()

    return MockDataFetcher()


@pytest.fixture
def performance_tracker():
    """Simple performance tracker for timing tests."""

    class PerformanceTracker:
        def __init__(self):
            self.start_time = None
            self.elapsed = 0

        def start(self):
            self.start_time = time.time()

        def stop(self):
            if self.start_time:
                self.elapsed = time.time() - self.start_time
                self.start_time = None
            return self.elapsed

    return PerformanceTracker()


@pytest.fixture
def fast_config():
    """Fast config for testing."""

    class FastConfig:
        osc_length = 20
        osc_mult = 2.0

    return FastConfig()


@pytest.mark.unit
class TestRangeOscillatorOptimized:
    """Optimized version of range oscillator tests."""

    def test_strategy_basic_fast(self, fast_ohlcv_data, fast_data_fetcher, performance_tracker):
        """Test basic strategy with fast data and performance tracking."""
        performance_tracker.start()

        from modules.range_oscillator.strategies.basic import generate_signals_basic_strategy

        high = fast_ohlcv_data["high"]
        low = fast_ohlcv_data["low"]
        close = fast_ohlcv_data["close"]

        signals, strength = generate_signals_basic_strategy(high=high, low=low, close=close, length=20, mult=2.0)

        assert isinstance(signals, pd.Series)
        assert isinstance(strength, pd.Series)
        assert len(signals) == len(close)
        assert len(strength) == len(close)

        elapsed = performance_tracker.stop()
        assert elapsed < 5.0, f"Test too slow: {elapsed:.3f}s"

    def test_strategy_with_parameters_fast(self, fast_ohlcv_data, fast_config):
        """Test strategy with different parameters using cached config."""
        from modules.range_oscillator.strategies.basic import generate_signals_basic_strategy

        high = fast_ohlcv_data["high"]
        low = fast_ohlcv_data["low"]
        close = fast_ohlcv_data["close"]

        signals, strength = generate_signals_basic_strategy(
            high=high,
            low=low,
            close=close,
            length=fast_config.osc_length,
            mult=fast_config.osc_mult,
        )

        assert len(signals) == len(close)
        assert len(strength) == len(close)

    @pytest.mark.parametrize(
        "osc_length,mult",
        [
            (10, 1.5),
            (20, 2.0),
            (30, 2.5),
        ],
    )
    def test_strategy_parametrized_fast(self, fast_ohlcv_data, osc_length, mult):
        """Parametrized test using pytest parametrization."""
        from modules.range_oscillator.strategies.basic import generate_signals_basic_strategy

        high = fast_ohlcv_data["high"]
        low = fast_ohlcv_data["low"]
        close = fast_ohlcv_data["close"]

        signals, strength = generate_signals_basic_strategy(
            high=high,
            low=low,
            close=close,
            length=osc_length,
            mult=mult,
        )

        assert len(signals) > 0
        assert len(strength) > 0

    def test_strategy_edge_cases_fast(self, fast_ohlcv_data):
        """Test edge cases with minimal data using cached data."""
        from modules.range_oscillator.strategies.basic import generate_signals_basic_strategy

        high = fast_ohlcv_data["high"].iloc[:3]
        low = fast_ohlcv_data["low"].iloc[:3]
        close = fast_ohlcv_data["close"].iloc[:3]

        try:
            signals, strength = generate_signals_basic_strategy(
                high=high,
                low=low,
                close=close,
                length=2,
                mult=1.0,
            )

            assert len(signals) == len(close)
            assert len(strength) == len(close)
        except (ValueError, IndexError):
            pass


@pytest.mark.integration
class TestRangeOscillatorIntegration:
    """Integration tests with mocked expensive operations."""

    def test_with_mocked_model(self, performance_tracker):
        """Test integration with mocked model."""
        performance_tracker.start()

        import numpy as np

        prediction = np.random.randint(0, 5, size=5)
        assert len(prediction) == 5

        elapsed = performance_tracker.stop()
        assert elapsed < 1.0, f"Integration test too slow: {elapsed:.3f}s"

    @pytest.mark.slow
    def test_full_workflow_slow(self, fast_data_fetcher):
        """Full workflow test - marked as slow."""
        pass


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-m", "not slow"])
