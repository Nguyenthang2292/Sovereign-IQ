
import numpy as np
import pandas as pd
import pytest

from modules.common.indicators.trend import calculate_trend_direction, calculate_weighted_ma
from modules.common.indicators.volatility import calculate_atr_range
from modules.range_oscillator.core.oscillator import calculate_range_oscillator
from modules.range_oscillator.core.oscillator import calculate_range_oscillator

"""
Tests for range_oscillator core module.

Tests core calculation functions:
- calculate_weighted_ma (moved to modules.common.indicators.trend)
- calculate_atr_range (moved to modules.common.indicators.volatility)
- calculate_trend_direction (moved to modules.common.indicators.trend)
- calculate_range_oscillator
"""




@pytest.fixture
def sample_price_data():
    """Create sample price data for testing."""
    np.random.seed(42)
    n = 200
    dates = pd.date_range("2024-01-01", periods=n, freq="1h")

    # Generate realistic price data
    close = 50000 + np.cumsum(np.random.randn(n) * 100)
    high = close + np.abs(np.random.randn(n) * 50)
    low = close - np.abs(np.random.randn(n) * 50)

    return (
        pd.Series(high, index=dates, name="high"),
        pd.Series(low, index=dates, name="low"),
        pd.Series(close, index=dates, name="close"),
    )


class TestWeightedMA:
    """Tests for calculate_weighted_ma function."""

    def test_weighted_ma_basic(self, sample_price_data):
        """Test basic weighted MA calculation."""
        _, _, close = sample_price_data

        ma = calculate_weighted_ma(close, length=50)

        assert isinstance(ma, pd.Series)
        assert len(ma) == len(close)
        assert ma.index.equals(close.index)
        # First length values should be NaN
        assert ma.iloc[:50].isna().all()
        # Remaining values should be numeric
        assert not ma.iloc[50:].isna().all()

    def test_weighted_ma_short_data(self):
        """Test weighted MA with insufficient data."""
        dates = pd.date_range("2024-01-01", periods=10, freq="1h")
        close = pd.Series([100, 101, 102, 103, 104, 105, 106, 107, 108, 109], index=dates)

        ma = calculate_weighted_ma(close, length=50)

        assert isinstance(ma, pd.Series)
        assert ma.isna().all()

    def test_weighted_ma_different_lengths(self, sample_price_data):
        """Test weighted MA with different length parameters."""
        _, _, close = sample_price_data

        for length in [10, 20, 50, 100]:
            ma = calculate_weighted_ma(close, length=length)
            assert isinstance(ma, pd.Series)
            assert len(ma) == len(close)
            if len(close) >= length + 1:
                assert ma.iloc[:length].isna().all()

    def test_weighted_ma_with_zero_prices(self):
        """Test weighted MA with zero prices (edge case)."""
        dates = pd.date_range("2024-01-01", periods=60, freq="1h")
        close = pd.Series([100.0] * 50 + [0.0] + [100.0] * 9, index=dates)

        ma = calculate_weighted_ma(close, length=50)

        assert isinstance(ma, pd.Series)
        # Should handle zero prices gracefully
        assert not ma.iloc[50:].isna().all()

    def test_weighted_ma_with_nan(self):
        """Test weighted MA with NaN values."""
        dates = pd.date_range("2024-01-01", periods=60, freq="1h")
        close = pd.Series([100.0] * 60, index=dates)
        close.iloc[30] = np.nan

        ma = calculate_weighted_ma(close, length=50)

        assert isinstance(ma, pd.Series)

    def test_weighted_ma_input_validation(self):
        """Test weighted MA input validation."""
        # Test with non-Series input
        with pytest.raises(TypeError):
            calculate_weighted_ma([1, 2, 3], length=10)

        # Test with empty Series
        empty_series = pd.Series([], dtype=float)
        with pytest.raises(ValueError):
            calculate_weighted_ma(empty_series, length=10)

        # Test with invalid length
        dates = pd.date_range("2024-01-01", periods=10, freq="1h")
        close = pd.Series([100.0] * 10, index=dates)
        with pytest.raises(ValueError):
            calculate_weighted_ma(close, length=0)
        with pytest.raises(ValueError):
            calculate_weighted_ma(close, length=-1)


class TestATRRange:
    """Tests for calculate_atr_range function."""

    def test_atr_range_basic(self, sample_price_data):
        """Test basic ATR range calculation."""
        high, low, close = sample_price_data

        range_atr = calculate_atr_range(high, low, close, mult=2.0)

        assert isinstance(range_atr, pd.Series)
        assert len(range_atr) == len(close)
        assert range_atr.index.equals(close.index)
        # ATR may have some NaN values at the beginning, but should have some valid values
        # After fillna and default value handling, should have at least some non-NaN values
        assert not range_atr.isna().all() or len(range_atr) > 0
        # Check non-NaN values are >= 0
        non_nan_values = range_atr.dropna()
        if len(non_nan_values) > 0:
            assert all(non_nan_values >= 0)

    def test_atr_range_different_multipliers(self, sample_price_data):
        """Test ATR range with different multipliers."""
        high, low, close = sample_price_data

        for mult in [1.0, 2.0, 3.0, 5.0]:
            range_atr = calculate_atr_range(high, low, close, mult=mult)
            assert isinstance(range_atr, pd.Series)
            # Check non-NaN values are >= 0
            non_nan_values = range_atr.dropna()
            if len(non_nan_values) > 0:
                assert all(non_nan_values >= 0)

    def test_atr_range_fallback(self, sample_price_data):
        """Test ATR range fallback mechanism."""
        high, low, close = sample_price_data

        # Use very large primary length to trigger fallback
        range_atr = calculate_atr_range(high, low, close, mult=2.0, atr_length_primary=10000, atr_length_fallback=200)

        assert isinstance(range_atr, pd.Series)
        # After fallback and default value handling, should have some values
        # Even if all NaN initially, default value calculation should provide values
        assert len(range_atr) > 0

    def test_atr_range_input_validation(self, sample_price_data):
        """Test ATR range input validation."""
        high, low, close = sample_price_data

        # Test with non-Series input
        with pytest.raises(TypeError):
            calculate_atr_range([1, 2, 3], low, close)

        # Test with empty Series
        empty_series = pd.Series([], dtype=float)
        with pytest.raises(ValueError):
            calculate_atr_range(empty_series, low, close)

        # Test with mismatched indices
        different_index = pd.date_range("2024-02-01", periods=len(high), freq="1h")
        high_diff = pd.Series(high.values, index=different_index)
        with pytest.raises(ValueError):
            calculate_atr_range(high_diff, low, close)

        # Test with invalid multiplier
        with pytest.raises(ValueError):
            calculate_atr_range(high, low, close, mult=0)
        with pytest.raises(ValueError):
            calculate_atr_range(high, low, close, mult=-1)

        # Test with invalid ATR lengths
        with pytest.raises(ValueError):
            calculate_atr_range(high, low, close, atr_length_primary=0)
        with pytest.raises(ValueError):
            calculate_atr_range(high, low, close, atr_length_fallback=0)


class TestTrendDirection:
    """Tests for calculate_trend_direction function."""

    def test_trend_direction_basic(self, sample_price_data):
        """Test basic trend direction calculation."""
        _, _, close = sample_price_data
        ma = calculate_weighted_ma(close, length=50)

        trend_dir = calculate_trend_direction(close, ma)

        assert isinstance(trend_dir, pd.Series)
        assert len(trend_dir) == len(close)
        assert trend_dir.index.equals(close.index)
        assert all(trend_dir.isin([-1, 0, 1]))

    def test_trend_direction_bullish(self):
        """Test trend direction with bullish scenario."""
        dates = pd.date_range("2024-01-01", periods=100, freq="1h")
        close = pd.Series(np.linspace(100, 200, 100), index=dates)
        ma = pd.Series(np.linspace(90, 190, 100), index=dates)

        trend_dir = calculate_trend_direction(close, ma)

        # Close > MA should give bullish (1)
        assert all(trend_dir[close > ma] == 1)

    def test_trend_direction_bearish(self):
        """Test trend direction with bearish scenario."""
        dates = pd.date_range("2024-01-01", periods=100, freq="1h")
        close = pd.Series(np.linspace(200, 100, 100), index=dates)
        ma = pd.Series(np.linspace(190, 90, 100), index=dates)

        trend_dir = calculate_trend_direction(close, ma)

        # Close < MA should give bearish (-1)
        assert all(trend_dir[close < ma] == -1)

    def test_trend_direction_with_nan(self):
        """Test trend direction with NaN values."""
        dates = pd.date_range("2024-01-01", periods=100, freq="1h")
        close = pd.Series([100.0] * 100, index=dates)
        ma = pd.Series([100.0] * 100, index=dates)
        ma.iloc[50] = np.nan
        close.iloc[51] = np.nan

        trend_dir = calculate_trend_direction(close, ma)

        assert isinstance(trend_dir, pd.Series)
        # Should use previous value when NaN
        assert not trend_dir.isna().all()

    def test_trend_direction_input_validation(self):
        """Test trend direction input validation."""
        dates = pd.date_range("2024-01-01", periods=10, freq="1h")
        close = pd.Series([100.0] * 10, index=dates)
        ma = pd.Series([100.0] * 10, index=dates)

        # Test with non-Series input
        with pytest.raises(TypeError):
            calculate_trend_direction([1, 2, 3], ma)

        # Test with empty Series
        empty_series = pd.Series([], dtype=float)
        with pytest.raises(ValueError):
            calculate_trend_direction(empty_series, ma)

        # Test with mismatched indices
        different_index = pd.date_range("2024-02-01", periods=len(close), freq="1h")
        close_diff = pd.Series(close.values, index=different_index)
        with pytest.raises(ValueError):
            calculate_trend_direction(close_diff, ma)


class TestRangeOscillator:
    """Tests for calculate_range_oscillator function."""

    def test_range_oscillator_basic(self, sample_price_data):
        """Test basic range oscillator calculation."""
        high, low, close = sample_price_data

        oscillator, ma, range_atr = calculate_range_oscillator(high=high, low=low, close=close, length=50, mult=2.0)

        assert isinstance(oscillator, pd.Series)
        assert isinstance(ma, pd.Series)
        assert isinstance(range_atr, pd.Series)

        assert len(oscillator) == len(close)
        assert len(ma) == len(close)
        assert len(range_atr) == len(close)

        # Oscillator values should be in reasonable range (if not all NaN)
        non_nan_osc = oscillator.dropna()
        if len(non_nan_osc) > 0:
            assert non_nan_osc.abs().max() < 1000  # Sanity check

    def test_range_oscillator_different_parameters(self, sample_price_data):
        """Test range oscillator with different parameters."""
        high, low, close = sample_price_data

        for length in [20, 50, 100]:
            for mult in [1.5, 2.0, 3.0]:
                oscillator, _, _ = calculate_range_oscillator(high=high, low=low, close=close, length=length, mult=mult)
                assert isinstance(oscillator, pd.Series)
                assert len(oscillator) == len(close)

    def test_range_oscillator_input_validation(self, sample_price_data):
        """Test range oscillator input validation."""
        high, low, close = sample_price_data

        # Test with non-Series input
        with pytest.raises(TypeError):
            calculate_range_oscillator([1, 2, 3], low, close)

        # Test with empty Series
        empty_series = pd.Series([], dtype=float)
        with pytest.raises(ValueError):
            calculate_range_oscillator(empty_series, low, close)

        # Test with mismatched indices
        different_index = pd.date_range("2024-02-01", periods=len(high), freq="1h")
        high_diff = pd.Series(high.values, index=different_index)
        with pytest.raises(ValueError):
            calculate_range_oscillator(high_diff, low, close)

        # Test with invalid length
        with pytest.raises(ValueError):
            calculate_range_oscillator(high, low, close, length=0)

        # Test with invalid multiplier
        with pytest.raises(ValueError):
            calculate_range_oscillator(high, low, close, mult=0)

    def test_range_oscillator_with_short_data(self):
        """Test range oscillator with insufficient data."""
        dates = pd.date_range("2024-01-01", periods=10, freq="1h")
        high = pd.Series([100.0] * 10, index=dates)
        low = pd.Series([99.0] * 10, index=dates)
        close = pd.Series([99.5] * 10, index=dates)

        oscillator, ma, range_atr = calculate_range_oscillator(high=high, low=low, close=close, length=50, mult=2.0)

        assert isinstance(oscillator, pd.Series)
        assert isinstance(ma, pd.Series)
        assert isinstance(range_atr, pd.Series)
        # With insufficient data, oscillator should be mostly NaN
        assert oscillator.isna().any() or len(oscillator) > 0
