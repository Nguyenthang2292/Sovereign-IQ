"""Tests for Range Oscillator heatmap calculation.

Tests heatmap color calculation functions:
- calculate_trend_direction
- calculate_heat_colors
"""

import numpy as np
import pandas as pd
import pytest

from modules.range_oscillator.config.heatmap_config import HeatmapConfig
from modules.range_oscillator.core.heatmap import calculate_heat_colors, calculate_trend_direction


@pytest.fixture
def sample_oscillator_data():
    """Create sample oscillator data for testing."""
    np.random.seed(42)
    n = 200
    dates = pd.date_range("2024-01-01", periods=n, freq="1h")

    # Generate oscillator values (typically -100 to +100)
    oscillator = pd.Series(np.random.uniform(-100, 100, n), index=dates, name="oscillator")

    # Generate close and MA for trend calculation
    close = pd.Series(50000 + np.cumsum(np.random.randn(n) * 100), index=dates, name="close")
    ma = close.rolling(window=50).mean()

    return oscillator, close, ma


@pytest.fixture
def default_config():
    """Create default heatmap config."""
    return HeatmapConfig()


class TestHeatmapConfig:
    """Tests for HeatmapConfig dataclass."""

    def test_default_config(self):
        """Test default configuration values."""
        config = HeatmapConfig()

        assert config.levels_inp == 2
        assert config.heat_thresh == 1
        assert config.lookback_bars == 100
        assert config.point_mode is True
        assert config.weak_bullish_color == "#008000"
        assert config.strong_bullish_color == "#09ff00"
        assert config.weak_bearish_color == "#800000"
        assert config.strong_bearish_color == "#ff0000"
        assert config.transition_color == "#0000ff"

    def test_custom_config(self):
        """Test custom configuration values."""
        config = HeatmapConfig(
            levels_inp=5,
            heat_thresh=3,
            lookback_bars=50,
            point_mode=False,
        )

        assert config.levels_inp == 5
        assert config.heat_thresh == 3
        assert config.lookback_bars == 50
        assert config.point_mode is False

    def test_config_validation_levels_inp(self):
        """Test validation of levels_inp parameter."""
        with pytest.raises(ValueError, match="levels_inp must be between 2 and 100"):
            HeatmapConfig(levels_inp=1)

        with pytest.raises(ValueError, match="levels_inp must be between 2 and 100"):
            HeatmapConfig(levels_inp=101)

    def test_config_validation_heat_thresh(self):
        """Test validation of heat_thresh parameter."""
        with pytest.raises(ValueError, match="heat_thresh must be >= 1"):
            HeatmapConfig(heat_thresh=0)

    def test_config_validation_lookback_bars(self):
        """Test validation of lookback_bars parameter."""
        with pytest.raises(ValueError, match="lookback_bars must be >= 1"):
            HeatmapConfig(lookback_bars=0)

    def test_config_validation_color_format(self):
        """Test validation of color format."""
        with pytest.raises(ValueError, match="must be a hex color string"):
            HeatmapConfig(weak_bullish_color="green")

        with pytest.raises(ValueError, match="must be a 7-character hex color"):
            HeatmapConfig(weak_bullish_color="#00800")


class TestCalculateTrendDirection:
    """Tests for calculate_trend_direction function."""

    def test_trend_direction_bullish(self):
        """Test trend direction calculation for bullish trend."""
        dates = pd.date_range("2024-01-01", periods=10, freq="1h")
        close = pd.Series([100, 101, 102, 103, 104], index=dates[:5])
        ma = pd.Series([99, 99, 99, 99, 99], index=dates[:5])

        trend = calculate_trend_direction(close, ma)

        assert isinstance(trend, pd.Series)
        assert len(trend) == len(close)
        assert (trend == 1).all()  # All bullish

    def test_trend_direction_bearish(self):
        """Test trend direction calculation for bearish trend."""
        dates = pd.date_range("2024-01-01", periods=10, freq="1h")
        close = pd.Series([100, 99, 98, 97, 96], index=dates[:5])
        ma = pd.Series([101, 101, 101, 101, 101], index=dates[:5])

        trend = calculate_trend_direction(close, ma)

        assert isinstance(trend, pd.Series)
        assert len(trend) == len(close)
        assert (trend == -1).all()  # All bearish

    def test_trend_direction_mixed(self):
        """Test trend direction with mixed bullish and bearish."""
        dates = pd.date_range("2024-01-01", periods=5, freq="1h")
        close = pd.Series([100, 102, 98, 104, 96], index=dates)
        ma = pd.Series([100, 100, 100, 100, 100], index=dates)

        trend = calculate_trend_direction(close, ma)

        assert isinstance(trend, pd.Series)
        assert trend.iloc[0] == 0  # close == ma
        assert trend.iloc[1] == 1  # bullish
        assert trend.iloc[2] == -1  # bearish
        assert trend.iloc[3] == 1  # bullish
        assert trend.iloc[4] == -1  # bearish

    def test_trend_direction_persistence(self):
        """Test that trend direction persists when close == ma."""
        dates = pd.date_range("2024-01-01", periods=5, freq="1h")
        close = pd.Series([100, 100, 100, 100, 100], index=dates)
        ma = pd.Series([100, 100, 100, 100, 100], index=dates)

        trend = calculate_trend_direction(close, ma)

        assert isinstance(trend, pd.Series)
        # First value should be 0, then forward filled
        assert trend.iloc[0] == 0
        assert (trend == 0).all()

    def test_trend_direction_index_mismatch(self):
        """Test error handling for index mismatch."""
        dates1 = pd.date_range("2024-01-01", periods=5, freq="1h")
        dates2 = pd.date_range("2024-01-02", periods=5, freq="1h")
        close = pd.Series([100] * 5, index=dates1)
        ma = pd.Series([100] * 5, index=dates2)

        with pytest.raises(ValueError, match="must have the same index"):
            calculate_trend_direction(close, ma)

    def test_trend_direction_type_error(self):
        """Test error handling for wrong types."""
        close = [100, 101, 102]
        ma = pd.Series([100, 100, 100])

        with pytest.raises(TypeError, match="must be pandas Series"):
            calculate_trend_direction(close, ma)


class TestCalculateHeatColors:
    """Tests for calculate_heat_colors function."""

    def test_heat_colors_basic(self, sample_oscillator_data, default_config):
        """Test basic heatmap color calculation."""
        oscillator, close, ma = sample_oscillator_data

        trend = calculate_trend_direction(close, ma)
        colors = calculate_heat_colors(oscillator, trend, config=default_config)

        assert isinstance(colors, pd.Series)
        assert len(colors) == len(oscillator)
        assert colors.index.equals(oscillator.index)
        assert colors.dtype == "object"

        # All values should be hex color strings
        for color in colors.dropna():
            assert isinstance(color, str)
            assert color.startswith("#")
            assert len(color) == 7

    def test_heat_colors_with_nan(self, default_config):
        """Test heatmap colors with NaN values in oscillator."""
        dates = pd.date_range("2024-01-01", periods=10, freq="1h")
        oscillator = pd.Series([10.0, np.nan, 20.0, np.nan, 30.0], index=dates[:5])
        trend = pd.Series([1, 1, 1, 1, 1], index=dates[:5])

        colors = calculate_heat_colors(oscillator, trend, config=default_config)

        assert isinstance(colors, pd.Series)
        assert len(colors) == len(oscillator)
        # NaN values should result in transition color
        assert colors.iloc[1] == default_config.transition_color
        assert colors.iloc[3] == default_config.transition_color

    def test_heat_colors_empty_series(self, default_config):
        """Test heatmap colors with empty series."""
        dates = pd.date_range("2024-01-01", periods=0, freq="1h")
        oscillator = pd.Series([], index=dates, dtype="float64")
        trend = pd.Series([], index=dates, dtype="int64")

        colors = calculate_heat_colors(oscillator, trend, config=default_config)

        assert isinstance(colors, pd.Series)
        assert len(colors) == 0

    def test_heat_colors_bullish_trend(self, default_config):
        """Test heatmap colors for bullish trend."""
        dates = pd.date_range("2024-01-01", periods=150, freq="1h")
        # Create oscillator values that cluster around certain levels
        oscillator = pd.Series(
            np.concatenate(
                [np.random.uniform(10, 20, 50), np.random.uniform(10, 20, 50), np.random.uniform(10, 20, 50)]
            ),
            index=dates[:150],
        )
        trend = pd.Series([1] * 150, index=dates[:150])  # All bullish

        colors = calculate_heat_colors(oscillator, trend, config=default_config)

        assert isinstance(colors, pd.Series)
        # Colors should be bullish colors (green variants) or gradient colors between them
        # Gradient interpolation produces intermediate colors, so we just check they're valid hex colors
        for color in colors.dropna():
            assert isinstance(color, str)
            assert color.startswith("#")
            assert len(color) == 7
            # Should not be bearish colors
            assert color != default_config.weak_bearish_color
            assert color != default_config.strong_bearish_color

    def test_heat_colors_bearish_trend(self, default_config):
        """Test heatmap colors for bearish trend."""
        dates = pd.date_range("2024-01-01", periods=150, freq="1h")
        # Create oscillator values that cluster around certain levels
        oscillator = pd.Series(
            np.concatenate(
                [np.random.uniform(-20, -10, 50), np.random.uniform(-20, -10, 50), np.random.uniform(-20, -10, 50)]
            ),
            index=dates[:150],
        )
        trend = pd.Series([-1] * 150, index=dates[:150])  # All bearish

        colors = calculate_heat_colors(oscillator, trend, config=default_config)

        assert isinstance(colors, pd.Series)
        # Colors should be bearish colors (red/maroon variants) or gradient colors between them
        # Gradient interpolation produces intermediate colors, so we just check they're valid hex colors
        for color in colors.dropna():
            assert isinstance(color, str)
            assert color.startswith("#")
            assert len(color) == 7
            # Should not be bullish colors
            assert color != default_config.weak_bullish_color
            assert color != default_config.strong_bullish_color

    def test_heat_colors_custom_config(self):
        """Test heatmap colors with custom configuration."""
        config = HeatmapConfig(levels_inp=5, heat_thresh=3, lookback_bars=50)
        dates = pd.date_range("2024-01-01", periods=100, freq="1h")
        oscillator = pd.Series(np.random.uniform(-50, 50, 100), index=dates)
        trend = pd.Series([1] * 100, index=dates)

        colors = calculate_heat_colors(oscillator, trend, config=config)

        assert isinstance(colors, pd.Series)
        assert len(colors) == len(oscillator)

    def test_heat_colors_index_mismatch(self, default_config):
        """Test error handling for index mismatch."""
        dates1 = pd.date_range("2024-01-01", periods=10, freq="1h")
        dates2 = pd.date_range("2024-01-02", periods=10, freq="1h")
        oscillator = pd.Series([10.0] * 10, index=dates1)
        trend = pd.Series([1] * 10, index=dates2)

        with pytest.raises(ValueError, match="must have the same index"):
            calculate_heat_colors(oscillator, trend, config=default_config)

    def test_heat_colors_type_error(self, default_config):
        """Test error handling for wrong types."""
        oscillator = [10.0, 20.0, 30.0]
        trend = pd.Series([1, 1, 1])

        with pytest.raises(TypeError, match="must be pandas Series"):
            calculate_heat_colors(oscillator, trend, config=default_config)

    def test_heat_colors_single_value(self, default_config):
        """Test heatmap colors with single value."""
        dates = pd.date_range("2024-01-01", periods=1, freq="1h")
        oscillator = pd.Series([10.0], index=dates)
        trend = pd.Series([1], index=dates)

        colors = calculate_heat_colors(oscillator, trend, config=default_config)

        assert isinstance(colors, pd.Series)
        assert len(colors) == 1
        # Should return transition color (not enough data for heatmap)
        assert colors.iloc[0] == default_config.transition_color

    def test_heat_colors_extreme_values(self, default_config):
        """Test heatmap colors with extreme oscillator values."""
        dates = pd.date_range("2024-01-01", periods=150, freq="1h")
        # Extreme values
        oscillator = pd.Series([-1000.0, 1000.0, -500.0, 500.0] + [0.0] * 146, index=dates[:150])
        trend = pd.Series([1] * 150, index=dates[:150])

        colors = calculate_heat_colors(oscillator, trend, config=default_config)

        assert isinstance(colors, pd.Series)
        assert len(colors) == len(oscillator)
        # All should be valid colors
        for color in colors:
            assert isinstance(color, str)
            assert color.startswith("#")

    def test_heat_colors_all_nan(self, default_config):
        """Test heatmap colors with all NaN oscillator values."""
        dates = pd.date_range("2024-01-01", periods=10, freq="1h")
        oscillator = pd.Series([np.nan] * 10, index=dates)
        trend = pd.Series([1] * 10, index=dates)

        colors = calculate_heat_colors(oscillator, trend, config=default_config)

        assert isinstance(colors, pd.Series)
        assert len(colors) == len(oscillator)
        # All should be transition color
        assert (colors == default_config.transition_color).all()

    def test_heat_colors_zero_range(self, default_config):
        """Test heatmap colors when oscillator has zero range (all same values)."""
        dates = pd.date_range("2024-01-01", periods=150, freq="1h")
        oscillator = pd.Series([10.0] * 150, index=dates)
        trend = pd.Series([1] * 150, index=dates)

        colors = calculate_heat_colors(oscillator, trend, config=default_config)

        assert isinstance(colors, pd.Series)
        assert len(colors) == len(oscillator)
        # Should handle gracefully (may return transition color or level color)

    def test_heat_colors_short_lookback(self):
        """Test heatmap colors with lookback larger than data length."""
        config = HeatmapConfig(lookback_bars=1000)  # Larger than data
        dates = pd.date_range("2024-01-01", periods=50, freq="1h")
        oscillator = pd.Series(np.random.uniform(-50, 50, 50), index=dates)
        trend = pd.Series([1] * 50, index=dates)

        colors = calculate_heat_colors(oscillator, trend, config=config)

        assert isinstance(colors, pd.Series)
        assert len(colors) == len(oscillator)

    def test_heat_colors_high_levels(self):
        """Test heatmap colors with high number of levels."""
        config = HeatmapConfig(levels_inp=20, heat_thresh=1)
        dates = pd.date_range("2024-01-01", periods=150, freq="1h")
        oscillator = pd.Series(np.random.uniform(-100, 100, 150), index=dates)
        trend = pd.Series([1] * 150, index=dates)

        colors = calculate_heat_colors(oscillator, trend, config=config)

        assert isinstance(colors, pd.Series)
        assert len(colors) == len(oscillator)

    def test_heat_colors_neutral_trend(self, default_config):
        """Test heatmap colors with neutral trend direction."""
        dates = pd.date_range("2024-01-01", periods=150, freq="1h")
        oscillator = pd.Series(np.random.uniform(-50, 50, 150), index=dates)
        trend = pd.Series([0] * 150, index=dates)  # All neutral

        colors = calculate_heat_colors(oscillator, trend, config=default_config)

        assert isinstance(colors, pd.Series)
        assert len(colors) == len(oscillator)
        # Neutral trend should use transition colors
        for color in colors.dropna():
            assert color == default_config.transition_color

    def test_heat_colors_mixed_trend(self, default_config):
        """Test heatmap colors with mixed trend directions."""
        dates = pd.date_range("2024-01-01", periods=150, freq="1h")
        oscillator = pd.Series(np.random.uniform(-50, 50, 150), index=dates)
        trend = pd.Series([1, -1] * 75, index=dates)  # Alternating

        colors = calculate_heat_colors(oscillator, trend, config=default_config)

        assert isinstance(colors, pd.Series)
        assert len(colors) == len(oscillator)

    def test_heat_colors_point_mode_vs_range_mode(self):
        """Test heatmap colors with point_mode True vs False."""
        dates = pd.date_range("2024-01-01", periods=150, freq="1h")
        oscillator = pd.Series(np.random.uniform(-50, 50, 150), index=dates)
        trend = pd.Series([1] * 150, index=dates)

        config_point = HeatmapConfig(point_mode=True)
        config_range = HeatmapConfig(point_mode=False)

        colors_point = calculate_heat_colors(oscillator, trend, config=config_point)
        colors_range = calculate_heat_colors(oscillator, trend, config=config_range)

        assert isinstance(colors_point, pd.Series)
        assert isinstance(colors_range, pd.Series)
        assert len(colors_point) == len(colors_range) == len(oscillator)


class TestIntegration:
    """Integration tests for heatmap with oscillator calculation."""

    def test_oscillator_with_heatmap_integration(self, sample_oscillator_data):
        """Test integration of oscillator calculation with heatmap."""
        from modules.range_oscillator.core.oscillator import calculate_range_oscillator_with_heatmap

        _, close, ma = sample_oscillator_data
        high = close + 100
        low = close - 100

        oscillator, ma_calc, range_atr, heat_colors, trend_direction, osc_colors = (
            calculate_range_oscillator_with_heatmap(
                high=high,
                low=low,
                close=close,
                length=50,
                mult=2.0,
            )
        )

        assert isinstance(oscillator, pd.Series)
        assert isinstance(ma_calc, pd.Series)
        assert isinstance(range_atr, pd.Series)
        assert isinstance(heat_colors, pd.Series)
        assert isinstance(trend_direction, pd.Series)
        assert isinstance(osc_colors, pd.Series)

        assert len(oscillator) == len(heat_colors) == len(trend_direction) == len(osc_colors)
        assert oscillator.index.equals(heat_colors.index)
        assert oscillator.index.equals(trend_direction.index)
        assert oscillator.index.equals(osc_colors.index)

    def test_oscillator_data_with_heatmap_integration(self, sample_oscillator_data):
        """Test integration of oscillator_data utility with heatmap."""
        from modules.range_oscillator.utils.oscillator_data import get_oscillator_data_with_heatmap

        _, close, _ = sample_oscillator_data
        high = close + 100
        low = close - 100

        oscillator, ma, range_atr, heat_colors, trend_direction, osc_colors = get_oscillator_data_with_heatmap(
            high=high,
            low=low,
            close=close,
            length=50,
            mult=2.0,
        )

        assert isinstance(oscillator, pd.Series)
        assert isinstance(ma, pd.Series)
        assert isinstance(range_atr, pd.Series)
        assert isinstance(heat_colors, pd.Series)
        assert isinstance(trend_direction, pd.Series)
        assert isinstance(osc_colors, pd.Series)

        assert len(oscillator) == len(heat_colors) == len(trend_direction) == len(osc_colors)
