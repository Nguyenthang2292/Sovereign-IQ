"""Tests for price-derived features creation.

This test module verifies that add_price_derived_features() correctly creates
all required price-derived features that are in MODEL_FEATURES:
- returns_1
- returns_5
- log_volume
- high_low_range
- close_open_diff
"""

import sys
from pathlib import Path

# Add project root to Python path for imports
ROOT = Path(__file__).resolve().parents[2]
ROOT_STR = str(ROOT)
if ROOT_STR not in sys.path:
    sys.path.insert(0, ROOT_STR)

import numpy as np
import pandas as pd
import pytest

from modules.random_forest.utils.features import add_price_derived_features


class TestAddPriceDerivedFeatures:
    """Test suite for add_price_derived_features() function."""

    @pytest.fixture
    def sample_ohlcv_data(self):
        """Create sample OHLCV data for testing."""
        np.random.seed(42)
        n = 100
        close = 100 + np.cumsum(np.random.randn(n) * 0.5)
        high = close + np.random.uniform(0, 2, n)
        low = close - np.random.uniform(0, 2, n)
        open_price = close + np.random.uniform(-1, 1, n)
        volume = np.random.uniform(1000, 10000, n)

        df = pd.DataFrame(
            {
                "open": open_price,
                "high": high,
                "low": low,
                "close": close,
                "volume": volume,
            }
        )
        return df

    def test_creates_all_required_features(self, sample_ohlcv_data):
        """Test that all required price-derived features are created."""
        result = add_price_derived_features(sample_ohlcv_data)

        # Check all required features exist
        required_features = ["returns_1", "returns_5", "log_volume", "high_low_range", "close_open_diff"]
        for feature in required_features:
            assert feature in result.columns, f"Missing required feature: {feature}"

    def test_returns_1_calculation(self, sample_ohlcv_data):
        """Test that returns_1 is calculated correctly."""
        result = add_price_derived_features(sample_ohlcv_data)

        # First row should be NaN (no previous value)
        assert pd.isna(result["returns_1"].iloc[0])

        # Second row should be (close[1] - close[0]) / close[0]
        expected = (result["close"].iloc[1] - result["close"].iloc[0]) / result["close"].iloc[0]
        assert abs(result["returns_1"].iloc[1] - expected) < 1e-10

    def test_returns_5_calculation(self, sample_ohlcv_data):
        """Test that returns_5 is calculated correctly."""
        result = add_price_derived_features(sample_ohlcv_data)

        # First 5 rows should be NaN (no previous 5 values)
        assert pd.isna(result["returns_5"].iloc[0:5]).all()

        # 6th row should be (close[5] - close[0]) / close[0]
        expected = (result["close"].iloc[5] - result["close"].iloc[0]) / result["close"].iloc[0]
        assert abs(result["returns_5"].iloc[5] - expected) < 1e-10

    def test_log_volume_calculation(self, sample_ohlcv_data):
        """Test that log_volume is calculated correctly."""
        result = add_price_derived_features(sample_ohlcv_data)

        # log_volume should be log1p(volume)
        expected = np.log1p(sample_ohlcv_data["volume"])
        np.testing.assert_array_almost_equal(result["log_volume"], expected)

    def test_high_low_range_calculation(self, sample_ohlcv_data):
        """Test that high_low_range is calculated correctly."""
        result = add_price_derived_features(sample_ohlcv_data)

        # high_low_range should be (high - low) / close
        expected = (sample_ohlcv_data["high"] - sample_ohlcv_data["low"]) / sample_ohlcv_data["close"]
        np.testing.assert_array_almost_equal(result["high_low_range"], expected)

    def test_close_open_diff_calculation(self, sample_ohlcv_data):
        """Test that close_open_diff is calculated correctly."""
        result = add_price_derived_features(sample_ohlcv_data)

        # close_open_diff should be (close - open) / open
        expected = (sample_ohlcv_data["close"] - sample_ohlcv_data["open"]) / sample_ohlcv_data["open"]
        np.testing.assert_array_almost_equal(result["close_open_diff"], expected)

    def test_handles_zero_volume(self):
        """Test that log_volume handles zero volume gracefully."""
        df = pd.DataFrame(
            {
                "open": [100, 101, 102],
                "high": [105, 106, 107],
                "low": [95, 96, 97],
                "close": [100, 101, 102],
                "volume": [0, 1000, 0],  # Zero volumes
            }
        )

        result = add_price_derived_features(df)

        # log1p(0) = 0, should not raise error
        assert result["log_volume"].iloc[0] == 0.0
        assert result["log_volume"].iloc[2] == 0.0
        assert result["log_volume"].iloc[1] > 0

    def test_handles_zero_close_price(self):
        """Test that high_low_range handles zero close price gracefully."""
        df = pd.DataFrame(
            {
                "open": [100, 0, 102],  # Zero close would cause division by zero
                "high": [105, 1, 107],
                "low": [95, 0, 97],
                "close": [100, 0, 102],  # Zero close
                "volume": [1000, 1000, 1000],
            }
        )

        result = add_price_derived_features(df)

        # Should set to 0.0 when close is 0
        assert result["high_low_range"].iloc[1] == 0.0
        assert result["high_low_range"].iloc[0] > 0
        assert result["high_low_range"].iloc[2] > 0

    def test_handles_zero_open_price(self):
        """Test that close_open_diff handles zero open price gracefully."""
        df = pd.DataFrame(
            {
                "open": [100, 0, 100],  # Zero open in middle
                "high": [105, 1, 107],
                "low": [95, 0, 97],
                "close": [100, 1, 102],  # close > open for row 2
                "volume": [1000, 1000, 1000],
            }
        )

        result = add_price_derived_features(df)

        # Should set to 0.0 when open is 0
        assert result["close_open_diff"].iloc[1] == 0.0
        assert result["close_open_diff"].iloc[0] == 0.0  # (100-100)/100 = 0
        # Row 2: (102-100)/100 = 0.02 > 0
        assert result["close_open_diff"].iloc[2] > 0

    def test_does_not_overwrite_existing_features(self, sample_ohlcv_data):
        """Test that existing features are not overwritten."""
        # Pre-create returns_1 with custom values
        sample_ohlcv_data["returns_1"] = 999.0
        sample_ohlcv_data["log_volume"] = 888.0

        result = add_price_derived_features(sample_ohlcv_data)

        # Existing features should be preserved
        assert (result["returns_1"] == 999.0).all()
        assert (result["log_volume"] == 888.0).all()

        # Other features should still be created
        assert "returns_5" in result.columns
        assert "high_low_range" in result.columns
        assert "close_open_diff" in result.columns

    def test_missing_required_columns_raises_error(self):
        """Test that missing required columns raise ValueError."""
        # Missing 'close' column
        df = pd.DataFrame(
            {
                "open": [100, 101],
                "high": [105, 106],
                "low": [95, 96],
                "volume": [1000, 1000],
                # Missing 'close'
            }
        )

        with pytest.raises(ValueError, match="Missing required OHLCV columns"):
            add_price_derived_features(df)

    def test_returns_new_dataframe(self, sample_ohlcv_data):
        """Test that function returns a new DataFrame (doesn't modify input)."""
        original_cols = sample_ohlcv_data.columns.tolist()
        result = add_price_derived_features(sample_ohlcv_data)

        # Input DataFrame should not be modified
        assert sample_ohlcv_data.columns.tolist() == original_cols
        assert "returns_1" not in sample_ohlcv_data.columns

        # Result should have new columns
        assert "returns_1" in result.columns
        assert len(result.columns) > len(sample_ohlcv_data.columns)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
