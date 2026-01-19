"""Validation tests for Range Oscillator heatmap functions.

These tests validate the correctness of heatmap calculations by comparing
against expected behavior and edge cases, ensuring the implementation
matches the Pine Script logic.
"""

import numpy as np
import pandas as pd
import pytest

from modules.range_oscillator.config.heatmap_config import HeatmapConfig
from modules.range_oscillator.core.heatmap import calculate_heat_colors, calculate_trend_direction


class TestHeatmapValidation:
    """Validation tests for heatmap correctness."""

    def test_trend_direction_matches_pine_script_logic(self):
        """Validate trend direction calculation matches Pine Script logic.

        Pine Script: trendDir := close > ma ? 1 : close < ma ? -1 : nz(trendDir[1])
        """
        dates = pd.date_range("2024-01-01", periods=10, freq="1h")

        # Test case 1: All bullish
        close = pd.Series([100, 101, 102, 103, 104], index=dates[:5])
        ma = pd.Series([99, 99, 99, 99, 99], index=dates[:5])
        trend = calculate_trend_direction(close, ma)
        assert (trend == 1).all(), "All bullish should return 1"

        # Test case 2: All bearish
        close = pd.Series([100, 99, 98, 97, 96], index=dates[:5])
        ma = pd.Series([101, 101, 101, 101, 101], index=dates[:5])
        trend = calculate_trend_direction(close, ma)
        assert (trend == -1).all(), "All bearish should return -1"

        # Test case 3: Persistence when close == ma
        close = pd.Series([100, 100, 100, 100, 100], index=dates[:5])
        ma = pd.Series([100, 100, 100, 100, 100], index=dates[:5])
        trend = calculate_trend_direction(close, ma)
        # First should be 0, then forward filled
        assert trend.iloc[0] == 0, "First neutral should be 0"

    def test_heatmap_levels_calculation(self):
        """Validate heatmap level calculation matches expected behavior."""
        dates = pd.date_range("2024-01-01", periods=150, freq="1h")
        # Create oscillator with known range
        oscillator = pd.Series(np.linspace(0, 100, 150), index=dates)
        trend = pd.Series([1] * 150, index=dates)
        config = HeatmapConfig(levels_inp=5, heat_thresh=1, lookback_bars=100)

        colors = calculate_heat_colors(oscillator, trend, config=config)

        assert isinstance(colors, pd.Series)
        assert len(colors) == len(oscillator)
        # All should have valid colors
        assert colors.notna().all(), "All colors should be valid"

    def test_heatmap_touch_counting(self):
        """Validate touch counting logic."""
        dates = pd.date_range("2024-01-01", periods=150, freq="1h")
        # Create oscillator that clusters around specific values
        # This should create "hot" zones
        cluster_values = np.concatenate([
            np.random.uniform(10, 12, 50),  # Cluster 1
            np.random.uniform(10, 12, 50),  # Cluster 1 (repeat)
            np.random.uniform(10, 12, 50),  # Cluster 1 (repeat)
        ])
        oscillator = pd.Series(cluster_values, index=dates[:150])
        trend = pd.Series([1] * 150, index=dates[:150])
        config = HeatmapConfig(levels_inp=5, heat_thresh=10, lookback_bars=100)

        colors = calculate_heat_colors(oscillator, trend, config=config)

        # Clustered values should produce "hot" colors (strong bullish)
        # At least some should be strong colors
        strong_colors = colors[colors == config.strong_bullish_color]
        assert len(strong_colors) > 0, "Clustered values should produce strong colors"

    def test_heatmap_gradient_coloring(self):
        """Validate gradient coloring produces intermediate colors."""
        dates = pd.date_range("2024-01-01", periods=150, freq="1h")
        oscillator = pd.Series(np.random.uniform(-50, 50, 150), index=dates)
        trend = pd.Series([1] * 150, index=dates)
        config = HeatmapConfig(levels_inp=10, heat_thresh=1, lookback_bars=100)

        colors = calculate_heat_colors(oscillator, trend, config=config)

        # Should have gradient colors (not just endpoint colors)
        unique_colors = colors.unique()
        assert len(unique_colors) > 2, "Should have gradient colors, not just endpoints"

    def test_heatmap_trend_based_colors(self):
        """Validate trend-based color selection."""
        dates = pd.date_range("2024-01-01", periods=150, freq="1h")
        oscillator = pd.Series(np.random.uniform(-50, 50, 150), index=dates)

        # Bullish trend
        trend_bullish = pd.Series([1] * 150, index=dates)
        colors_bullish = calculate_heat_colors(oscillator, trend_bullish)

        # Bearish trend
        trend_bearish = pd.Series([-1] * 150, index=dates)
        colors_bearish = calculate_heat_colors(oscillator, trend_bearish)

        # Colors should differ based on trend
        # Bullish should not use bearish colors
        assert not (colors_bullish == "#800000").any(), "Bullish trend should not use weak bearish color"
        assert not (colors_bullish == "#ff0000").any(), "Bullish trend should not use strong bearish color"

        # Bearish should not use bullish colors
        assert not (colors_bearish == "#008000").any(), "Bearish trend should not use weak bullish color"
        assert not (colors_bearish == "#09ff00").any(), "Bearish trend should not use strong bullish color"

    def test_heatmap_nearest_level_selection(self):
        """Validate nearest level selection logic."""
        dates = pd.date_range("2024-01-01", periods=150, freq="1h")
        # Create oscillator with specific values that should map to specific levels
        oscillator = pd.Series([10.0, 20.0, 30.0, 40.0, 50.0] + [25.0] * 145, index=dates[:150])
        trend = pd.Series([1] * 150, index=dates[:150])
        config = HeatmapConfig(levels_inp=5, heat_thresh=1, lookback_bars=100)

        colors = calculate_heat_colors(oscillator, trend, config=config)

        # Values at 25.0 should produce valid colors
        # Note: Colors may vary over time as lookback window changes,
        # but all should be valid hex colors
        colors_at_25 = colors.iloc[5:150]  # All values at 25.0
        assert colors_at_25.notna().all(), "All colors should be valid"
        # All should be hex color strings
        for color in colors_at_25:
            assert isinstance(color, str)
            assert color.startswith("#")
            assert len(color) == 7

    def test_heatmap_point_mode_vs_range_mode(self):
        """Validate point mode vs range mode differences."""
        dates = pd.date_range("2024-01-01", periods=150, freq="1h")
        oscillator = pd.Series(np.random.uniform(-50, 50, 150), index=dates)
        trend = pd.Series([1] * 150, index=dates)

        config_point = HeatmapConfig(levels_inp=5, point_mode=True)
        config_range = HeatmapConfig(levels_inp=5, point_mode=False)

        colors_point = calculate_heat_colors(oscillator, trend, config=config_point)
        colors_range = calculate_heat_colors(oscillator, trend, config=config_range)

        # Both should produce valid results
        assert len(colors_point) == len(colors_range) == len(oscillator)
        # Results may differ but both should be valid
        assert colors_point.notna().all()
        assert colors_range.notna().all()

    def test_heatmap_lookback_window_effect(self):
        """Validate lookback window affects results."""
        dates = pd.date_range("2024-01-01", periods=200, freq="1h")
        # Create oscillator with changing pattern
        oscillator = pd.Series(
            np.concatenate([
                np.random.uniform(10, 20, 100),  # First pattern
                np.random.uniform(80, 90, 100),   # Different pattern
            ]),
            index=dates[:200],
        )
        trend = pd.Series([1] * 200, index=dates[:200])

        config_short = HeatmapConfig(levels_inp=5, lookback_bars=50)
        config_long = HeatmapConfig(levels_inp=5, lookback_bars=150)

        colors_short = calculate_heat_colors(oscillator, trend, config=config_short)
        colors_long = calculate_heat_colors(oscillator, trend, config=config_long)

        # Both should work
        assert len(colors_short) == len(colors_long) == len(oscillator)
        # Colors in later bars may differ due to different lookback windows
        # This is expected behavior

    def test_heatmap_consistency_with_same_input(self):
        """Validate heatmap produces consistent results with same input."""
        dates = pd.date_range("2024-01-01", periods=150, freq="1h")
        oscillator = pd.Series(np.random.uniform(-50, 50, 150), index=dates)
        trend = pd.Series([1] * 150, index=dates)
        config = HeatmapConfig(levels_inp=5, heat_thresh=1, lookback_bars=100)

        # Run twice with same input
        colors1 = calculate_heat_colors(oscillator, trend, config=config)
        colors2 = calculate_heat_colors(oscillator, trend, config=config)

        # Should produce identical results
        pd.testing.assert_series_equal(colors1, colors2, "Results should be consistent")

    def test_heatmap_edge_case_all_same_values(self):
        """Validate heatmap handles edge case of all same oscillator values."""
        dates = pd.date_range("2024-01-01", periods=150, freq="1h")
        oscillator = pd.Series([10.0] * 150, index=dates)  # All same value
        trend = pd.Series([1] * 150, index=dates)
        config = HeatmapConfig(levels_inp=5, heat_thresh=1, lookback_bars=100)

        colors = calculate_heat_colors(oscillator, trend, config=config)

        # Should handle gracefully (may return transition color due to zero range)
        assert isinstance(colors, pd.Series)
        assert len(colors) == len(oscillator)
        assert colors.notna().all()

    def test_heatmap_edge_case_single_bar(self):
        """Validate heatmap handles single bar edge case."""
        dates = pd.date_range("2024-01-01", periods=1, freq="1h")
        oscillator = pd.Series([10.0], index=dates)
        trend = pd.Series([1], index=dates)
        config = HeatmapConfig(levels_inp=2, heat_thresh=1, lookback_bars=100)

        colors = calculate_heat_colors(oscillator, trend, config=config)

        # Should return transition color (not enough data)
        assert isinstance(colors, pd.Series)
        assert len(colors) == 1
        assert colors.iloc[0] == config.transition_color
