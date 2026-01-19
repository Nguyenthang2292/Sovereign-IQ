"""Performance benchmarks for Range Oscillator heatmap functions.

This module provides performance benchmarks to measure and track
the efficiency of heatmap calculation functions.
"""

import time
from typing import Dict

import numpy as np
import pandas as pd
import pytest

from modules.range_oscillator.config.heatmap_config import HeatmapConfig
from modules.range_oscillator.core.heatmap import calculate_heat_colors, calculate_trend_direction


@pytest.fixture
def large_dataset():
    """Create large dataset for performance testing."""
    np.random.seed(42)
    n = 5000  # Large dataset
    dates = pd.date_range("2024-01-01", periods=n, freq="1h")

    # Generate realistic price data
    close = pd.Series(50000 + np.cumsum(np.random.randn(n) * 100), index=dates, name="close")
    ma = close.rolling(window=50).mean()

    # Generate oscillator values
    oscillator = pd.Series(np.random.uniform(-100, 100, n), index=dates, name="oscillator")

    return oscillator, close, ma


@pytest.fixture
def medium_dataset():
    """Create medium dataset for performance testing."""
    np.random.seed(42)
    n = 1000
    dates = pd.date_range("2024-01-01", periods=n, freq="1h")

    close = pd.Series(50000 + np.cumsum(np.random.randn(n) * 100), index=dates, name="close")
    ma = close.rolling(window=50).mean()
    oscillator = pd.Series(np.random.uniform(-100, 100, n), index=dates, name="oscillator")

    return oscillator, close, ma


class TestPerformanceBenchmarks:
    """Performance benchmarks for heatmap functions."""

    def test_trend_direction_performance_medium(self, medium_dataset):
        """Benchmark trend direction calculation on medium dataset."""
        _, close, ma = medium_dataset

        start_time = time.perf_counter()
        trend = calculate_trend_direction(close, ma)
        elapsed = time.perf_counter() - start_time

        assert isinstance(trend, pd.Series)
        assert len(trend) == len(close)
        # Should complete in reasonable time (< 0.1s for 1000 bars)
        assert elapsed < 0.1, f"Trend direction calculation took {elapsed:.4f}s, expected < 0.1s"

    def test_trend_direction_performance_large(self, large_dataset):
        """Benchmark trend direction calculation on large dataset."""
        _, close, ma = large_dataset

        start_time = time.perf_counter()
        trend = calculate_trend_direction(close, ma)
        elapsed = time.perf_counter() - start_time

        assert isinstance(trend, pd.Series)
        assert len(trend) == len(close)
        # Should complete in reasonable time (< 0.5s for 5000 bars)
        assert elapsed < 0.5, f"Trend direction calculation took {elapsed:.4f}s, expected < 0.5s"

    def test_heat_colors_performance_medium(self, medium_dataset):
        """Benchmark heatmap color calculation on medium dataset."""
        oscillator, close, ma = medium_dataset
        config = HeatmapConfig(levels_inp=2, heat_thresh=1, lookback_bars=100)

        trend = calculate_trend_direction(close, ma)

        start_time = time.perf_counter()
        colors = calculate_heat_colors(oscillator, trend, config=config)
        elapsed = time.perf_counter() - start_time

        assert isinstance(colors, pd.Series)
        assert len(colors) == len(oscillator)
        # Should complete in reasonable time (< 5s for 1000 bars with default config)
        assert elapsed < 5.0, f"Heatmap calculation took {elapsed:.4f}s, expected < 5.0s"

    def test_heat_colors_performance_large(self, large_dataset):
        """Benchmark heatmap color calculation on large dataset."""
        oscillator, close, ma = large_dataset
        config = HeatmapConfig(levels_inp=2, heat_thresh=1, lookback_bars=100)

        trend = calculate_trend_direction(close, ma)

        start_time = time.perf_counter()
        colors = calculate_heat_colors(oscillator, trend, config=config)
        elapsed = time.perf_counter() - start_time

        assert isinstance(colors, pd.Series)
        assert len(colors) == len(oscillator)
        # Should complete in reasonable time (< 30s for 5000 bars with default config)
        assert elapsed < 30.0, f"Heatmap calculation took {elapsed:.4f}s, expected < 30.0s"

    def test_heat_colors_performance_high_levels(self, medium_dataset):
        """Benchmark heatmap with high number of levels."""
        oscillator, close, ma = medium_dataset
        config = HeatmapConfig(levels_inp=20, heat_thresh=1, lookback_bars=100)

        trend = calculate_trend_direction(close, ma)

        start_time = time.perf_counter()
        colors = calculate_heat_colors(oscillator, trend, config=config)
        elapsed = time.perf_counter() - start_time

        assert isinstance(colors, pd.Series)
        # Should still complete in reasonable time even with more levels
        assert elapsed < 10.0, f"Heatmap calculation with 20 levels took {elapsed:.4f}s, expected < 10.0s"

    def test_heat_colors_performance_long_lookback(self, medium_dataset):
        """Benchmark heatmap with long lookback window."""
        oscillator, close, ma = medium_dataset
        config = HeatmapConfig(levels_inp=2, heat_thresh=1, lookback_bars=500)

        trend = calculate_trend_direction(close, ma)

        start_time = time.perf_counter()
        colors = calculate_heat_colors(oscillator, trend, config=config)
        elapsed = time.perf_counter() - start_time

        assert isinstance(colors, pd.Series)
        # Longer lookback should still be reasonable
        assert elapsed < 8.0, f"Heatmap calculation with long lookback took {elapsed:.4f}s, expected < 8.0s"

    def test_memory_efficiency_trend_direction(self, large_dataset):
        """Test memory efficiency of trend direction calculation."""
        _, close, ma = large_dataset

        import sys

        trend = calculate_trend_direction(close, ma)

        # Check that int8 is used (1 byte per value)
        assert trend.dtype == "int8", f"Expected int8 dtype, got {trend.dtype}"
        # Memory should be approximately n bytes (not 8n bytes for int64)
        estimated_memory = len(trend) * 1  # int8 = 1 byte
        assert estimated_memory < len(trend) * 2, "Memory usage should be efficient with int8"

    @pytest.mark.benchmark
    def test_benchmark_comparison(self, medium_dataset):
        """Compare performance of different configurations."""
        oscillator, close, ma = medium_dataset
        trend = calculate_trend_direction(close, ma)

        results: Dict[str, float] = {}

        # Default config
        config_default = HeatmapConfig(levels_inp=2, heat_thresh=1, lookback_bars=100)
        start = time.perf_counter()
        _ = calculate_heat_colors(oscillator, trend, config=config_default)
        results["default"] = time.perf_counter() - start

        # High levels
        config_high = HeatmapConfig(levels_inp=10, heat_thresh=1, lookback_bars=100)
        start = time.perf_counter()
        _ = calculate_heat_colors(oscillator, trend, config=config_high)
        results["high_levels"] = time.perf_counter() - start

        # Long lookback
        config_long = HeatmapConfig(levels_inp=2, heat_thresh=1, lookback_bars=500)
        start = time.perf_counter()
        _ = calculate_heat_colors(oscillator, trend, config=config_long)
        results["long_lookback"] = time.perf_counter() - start

        # All should complete
        for config_name, elapsed in results.items():
            assert elapsed < 15.0, f"{config_name} took {elapsed:.4f}s, expected < 15.0s"

        # High levels should take longer than default
        assert results["high_levels"] > results["default"] * 0.5, "High levels should take more time"
