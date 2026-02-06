import pytest
import pandas as pd
import numpy as np
from modules.adaptive_trend_LTS_mini.core.process_layer1.layer1_signal import _layer1_signal_for_ma
from modules.adaptive_trend_LTS_mini.core.process_layer1.weighted_signal import weighted_signal
from modules.adaptive_trend_LTS_mini.core.process_layer1.cut_signal import cut_signal
from modules.adaptive_trend_LTS_mini.core.process_layer1.trend_sign import trend_sign

@pytest.fixture
def sample_data():
    """Generate sample data for layer processing tests."""
    np.random.seed(42)
    prices = pd.Series(100 + np.random.randn(100).cumsum(), index=pd.date_range("2023-01-01", periods=100, freq="h"))
    # Create 9 MA series
    ma_tuple = tuple(prices.rolling(window=i+5).mean().fillna(prices) for i in range(9))
    return prices, ma_tuple

class TestLayer1SignalForMa:
    """Tests for _layer1_signal_for_ma function."""

    def test_happy_path(self, sample_data):
        """Test with valid inputs."""
        prices, ma_tuple = sample_data
        signal, signals_tuple, equity_tuple = _layer1_signal_for_ma(
            prices, ma_tuple, lambda_val=0.02, decay_val=0.03
        )
        assert isinstance(signal, pd.Series)
        assert len(signals_tuple) == 9
        assert len(equity_tuple) == 9
        assert len(signal) == len(prices)

    def test_invalid_ma_count(self, sample_data):
        """Test with incorrect number of MAs."""
        prices, ma_tuple = sample_data
        with pytest.raises(ValueError, match="ma_tuple must contain exactly 9 MA series"):
            _layer1_signal_for_ma(prices, ma_tuple[:5], lambda_val=0.02, decay_val=0.03)

    def test_invalid_types(self):
        """Test with invalid input types."""
        with pytest.raises(TypeError, match="prices must be a pandas Series"):
            _layer1_signal_for_ma([1, 2, 3], (), lambda_val=0.02, decay_val=0.03)

    def test_invalid_parameters(self, sample_data):
        """Test with invalid L or De parameters."""
        prices, ma_tuple = sample_data
        with pytest.raises(ValueError, match="lambda_val must be a finite number"):
            _layer1_signal_for_ma(prices, ma_tuple, lambda_val=np.nan, decay_val=0.03)
        with pytest.raises(ValueError, match="decay_val must be between 0 and 1"):
            _layer1_signal_for_ma(prices, ma_tuple, lambda_val=0.02, decay_val=1.5)

    def test_with_precalculated_r(self, sample_data):
        """Test with pre-calculated rate of change."""
        prices, ma_tuple = sample_data
        from modules.adaptive_trend_LTS_mini.utils.rate_of_change import rate_of_change
        R = rate_of_change(prices)
        signal, _, _ = _layer1_signal_for_ma(prices, ma_tuple, lambda_val=0.02, decay_val=0.03, rate_of_change_series=R)
        assert len(signal) == len(prices)

class TestWeightedSignal:
    """Tests for weighted_signal function."""

    def test_weighted_signal_basic(self):
        """Test basic weighted average calculation."""
        index = pd.date_range("2023-01-01", periods=3)
        s1 = pd.Series([1.0, 0.0, -1.0], index=index)
        s2 = pd.Series([0.0, 1.0, 1.0], index=index)
        w1 = pd.Series([1.0, 1.0, 1.0], index=index)
        w2 = pd.Series([1.0, 1.0, 3.0], index=index)

        # Expected:
        # t0: (1*1 + 0*1) / (1+1) = 0.5
        # t1: (0*1 + 1*1) / (1+1) = 0.5
        # t2: (-1*1 + 1*3) / (1+3) = 2/4 = 0.5
        result = weighted_signal([s1, s2], [w1, w2])
        pd.testing.assert_series_equal(result, pd.Series([0.5, 0.5, 0.5], index=index).round(2))

    def test_zero_weights_handling(self):
        """Test handling of zero weights."""
        index = pd.date_range("2023-01-01", periods=1)
        s = pd.Series([1.0], index=index)
        w = pd.Series([0.0], index=index)
        result = weighted_signal([s], [w])
        assert result.iloc[0] == 0.0

    def test_mismatched_lengths(self):
        """Test error when signals and weights lengths differ."""
        with pytest.raises(ValueError, match="signals and weights must have the same length"):
            weighted_signal([pd.Series([1])], [])

    def test_weighted_signal_alignment(self):
        """Test that weighted_signal aligns series with different indices."""
        idx1 = pd.date_range("2023-01-01", periods=2)
        idx2 = pd.date_range("2023-01-01", periods=3)
        s1 = pd.Series([1.0, 1.0], index=idx1)
        s2 = pd.Series([0.0, 0.0, 0.0], index=idx2)
        w1 = pd.Series([1.0, 1.0], index=idx1)
        w2 = pd.Series([1.0, 1.0, 1.0], index=idx2)

        # Should align to idx1
        result = weighted_signal([s1, s2], [w1, w2])
        assert len(result) == 2
        assert result.iloc[0] == 0.5

    def test_weighted_signal_with_nans(self):
        """Test weighted_signal with NaN values."""
        index = pd.date_range("2023-01-01", periods=2)
        s1 = pd.Series([1.0, np.nan], index=index)
        w1 = pd.Series([1.0, 1.0], index=index)
        result = weighted_signal([s1], [w1])
        assert np.isnan(result.iloc[1])

    def test_weighted_signal_empty(self):
        """Test weighted_signal with empty inputs."""
        result = weighted_signal([], [])
        assert result.empty
        assert isinstance(result, pd.Series)

class TestCutSignal:
    """Tests for cut_signal function."""

    def test_cut_signal_basic(self):
        """Test discretization with default threshold."""
        s = pd.Series([0.6, 0.4, 0.0, -0.4, -0.6])
        result = cut_signal(s, threshold=0.5)
        expected = pd.Series([1, 0, 0, 0, -1], dtype="int8")
        pd.testing.assert_series_equal(result, expected)

    def test_custom_thresholds(self):
        """Test discretization with custom long/short thresholds."""
        s = pd.Series([0.6, 0.4, 0.0, -0.2, -0.4])
        result = cut_signal(s, long_threshold=0.5, short_threshold=-0.3)
        expected = pd.Series([1, 0, 0, 0, -1], dtype="int8")
        pd.testing.assert_series_equal(result, expected)

    def test_cutout(self):
        """Test cutout (initial bars zeroing)."""
        s = pd.Series([1.0, 1.0, 1.0, 1.0])
        result = cut_signal(s, threshold=0.5, cutout=2)
        expected = pd.Series([0, 0, 1, 1], dtype="int8")
        pd.testing.assert_series_equal(result, expected)

class TestTrendSign:
    """Tests for trend_sign function."""

    def test_trend_sign_basic(self):
        """Test basic trend sign calculation."""
        s = pd.Series([1.0, 0.0, -1.0])
        result = trend_sign(s)
        expected = pd.Series([1, 0, -1], dtype="int8")
        pd.testing.assert_series_equal(result, expected)

    def test_trend_sign_strategy_mode(self):
        """Test strategy mode (shift 1)."""
        s = pd.Series([1.0, -1.0, 0.0])
        result = trend_sign(s, strategy=True)
        # Expected: shift(1) -> [NaN, 1.0, -1.0] -> [0, 1, -1]
        expected = pd.Series([0, 1, -1], dtype="int8")
        pd.testing.assert_series_equal(result, expected)
