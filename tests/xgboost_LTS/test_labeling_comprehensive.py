"""
Comprehensive tests for XGBoost labeling module.
Tests edge cases, error handling, and complex scenarios.
"""

import numpy as np
import pandas as pd
import pytest
from unittest.mock import patch, MagicMock

from modules.xgboost_LTS.core.labeling import (
    apply_directional_labels,
    _calculate_volatility_multiplier,
    _calculate_lookback_weights,
)
from modules.xgboost_LTS.utils.cache_manager import CacheManager


class TestApplyDirectionalLabels:
    """Test apply_directional_labels function comprehensively."""

    def test_empty_dataframe(self):
        """Test handling of empty DataFrame."""
        df = pd.DataFrame()
        result = apply_directional_labels(df, use_cache=False)

        assert "Target" in result.columns
        assert "TargetLabel" in result.columns
        assert "DynamicThreshold" in result.columns
        assert len(result) == 0

    def test_missing_required_columns(self):
        """Test handling of DataFrame without 'close' column."""
        df = pd.DataFrame({"invalid_col": [1, 2, 3]})

        with pytest.raises(KeyError):
            apply_directional_labels(df, use_cache=False)

    def test_minimum_valid_dataframe(self):
        """Test with minimum required columns."""
        df = pd.DataFrame({"close": [100.0, 101.0, 102.0, 103.0, 104.0]})

        result = apply_directional_labels(df, use_cache=False)

        assert "Target" in result.columns
        assert "TargetLabel" in result.columns
        assert "DynamicThreshold" in result.columns
        assert len(result) == len(df)

    def test_with_atr_column(self):
        """Test labeling with ATR column present."""
        df = pd.DataFrame(
            {
                "close": np.linspace(100, 110, 50),
                "ATR_14": np.random.uniform(0.5, 2.0, 50),
                "ATR_RATIO_14_50": np.random.uniform(0.8, 1.2, 50),
                "high": np.linspace(101, 111, 50),
                "low": np.linspace(99, 109, 50),
                "volume": np.ones(50) * 1000,
            }
        )

        result = apply_directional_labels(df, use_cache=False)

        assert "Target" in result.columns
        assert len(result) == len(df)
        # Check that all targets are valid values (0, 1, 2, or NaN)
        valid_targets = result["Target"].dropna()
        assert all(t in [0, 1, 2] for t in valid_targets)

    def test_label_values_range(self):
        """Test that labels are only 0, 1, 2 or NaN."""
        df = pd.DataFrame({"close": np.cumsum(np.random.randn(100)) + 100})

        result = apply_directional_labels(df, use_cache=False)

        valid_targets = result["Target"].dropna()
        assert all(t in [0, 1, 2] for t in valid_targets)

        # Check string labels
        valid_labels = result["TargetLabel"].dropna()
        assert all(l in ["UP", "DOWN", "NEUTRAL"] for l in valid_labels)

    def test_future_data_nan(self):
        """Test that last N rows have NaN targets (no future data)."""
        from config import TARGET_HORIZON

        df = pd.DataFrame({"close": np.linspace(100, 110, 100)})

        result = apply_directional_labels(df, use_cache=False)

        # Last TARGET_HORIZON rows should have NaN
        last_targets = result["Target"].tail(TARGET_HORIZON)
        assert all(pd.isna(t) for t in last_targets)

    def test_cache_integration(self, tmp_path):
        """Test caching functionality."""
        df = pd.DataFrame({"close": np.linspace(100, 110, 50)})

        # First call - compute and cache
        result1 = apply_directional_labels(df, use_cache=True)

        # Second call - should load from cache
        result2 = apply_directional_labels(df, use_cache=True)

        # Results should be identical
        pd.testing.assert_frame_equal(result1, result2)

    def test_threshold_calculation(self):
        """Test dynamic threshold calculation."""
        df = pd.DataFrame({"close": np.linspace(100, 110, 50)})

        result = apply_directional_labels(df, use_cache=False)

        assert "DynamicThreshold" in result.columns
        # Threshold should be positive
        assert all(t > 0 for t in result["DynamicThreshold"].dropna())

    def test_volatility_multiplier_calculation(self):
        """Test volatility multiplier bounds."""
        df = pd.DataFrame({"close": np.linspace(100, 110, 50), "ATR_14": np.random.uniform(0.5, 2.0, 50)})

        vol_mult = _calculate_volatility_multiplier(df)

        # Should be clipped to [1.5, 3.0]
        assert all(v >= 1.5 for v in vol_mult.dropna())
        assert all(v <= 3.0 for v in vol_mult.dropna())

    def test_lookback_weights_sum_to_one(self):
        """Test that lookback weights sum to 1."""
        n = 50
        vol_mult = pd.Series(np.random.uniform(1.5, 3.0, n))
        vol_low = pd.Series(np.random.uniform(1.5, 2.0, n))
        vol_high = pd.Series(np.random.uniform(2.5, 3.0, n))

        w_short, w_medium, w_long = _calculate_lookback_weights(vol_mult, vol_low, vol_high)

        total = w_short + w_medium + w_long
        np.testing.assert_array_almost_equal(total.values, np.ones(n), decimal=10)

    def test_rapid_price_changes(self):
        """Test with rapid price changes (high volatility)."""
        # Create alternating up/down pattern
        prices = [100]
        for i in range(99):
            if i % 2 == 0:
                prices.append(prices[-1] * 1.05)  # +5%
            else:
                prices.append(prices[-1] * 0.95)  # -5%

        df = pd.DataFrame({"close": prices})
        result = apply_directional_labels(df, use_cache=False)

        assert "Target" in result.columns
        assert len(result) == len(df)

    def test_constant_price(self):
        """Test with constant price (zero volatility)."""
        df = pd.DataFrame({"close": np.ones(50) * 100})

        result = apply_directional_labels(df, use_cache=False)

        # With constant price, all changes are 0, so should be NEUTRAL
        valid_targets = result["Target"].dropna()
        if len(valid_targets) > 0:
            assert all(t == 1 for t in valid_targets)  # NEUTRAL = 1

    def test_single_row_dataframe(self):
        """Test with single row (edge case)."""
        df = pd.DataFrame({"close": [100.0]})

        result = apply_directional_labels(df, use_cache=False)

        assert len(result) == 1
        assert "Target" in result.columns

    def test_nan_handling_in_close(self):
        """Test handling of NaN values in close price."""
        close = np.linspace(100, 110, 50)
        close[10] = np.nan
        close[20] = np.nan

        df = pd.DataFrame({"close": close})

        # Should handle NaN gracefully or raise appropriate error
        result = apply_directional_labels(df, use_cache=False)

        assert "Target" in result.columns
        assert len(result) == len(df)

    def test_extreme_values(self):
        """Test with extreme price values."""
        df = pd.DataFrame(
            {
                "close": np.concatenate(
                    [
                        [0.0001],  # Very small
                        np.linspace(1, 100, 48),
                        [1000000],  # Very large
                    ]
                )
            }
        )

        result = apply_directional_labels(df, use_cache=False)

        assert "Target" in result.columns
        assert len(result) == len(df)
        # Targets should still be valid
        valid_targets = result["Target"].dropna()
        assert all(t in [0, 1, 2] for t in valid_targets)


class TestVolatilityMultiplier:
    """Test volatility multiplier calculation."""

    def test_with_atr(self):
        """Test calculation with ATR column."""
        df = pd.DataFrame({"close": np.linspace(100, 110, 50), "ATR_14": np.ones(50) * 1.0})

        result = _calculate_volatility_multiplier(df)

        assert len(result) == len(df)
        assert all(1.5 <= v <= 3.0 for v in result)

    def test_without_atr(self):
        """Test calculation without ATR (fallback)."""
        df = pd.DataFrame({"close": np.cumsum(np.random.randn(50) * 0.01) + 100})

        result = _calculate_volatility_multiplier(df)

        assert len(result) == len(df)
        assert all(1.5 <= v <= 3.0 for v in result)

    def test_zero_close_handling(self):
        """Test handling of zero close prices."""
        close = np.linspace(100, 110, 50)
        close[10] = 0  # Insert zero

        df = pd.DataFrame({"close": close, "ATR_14": np.ones(50)})

        # Should not crash
        result = _calculate_volatility_multiplier(df)
        assert len(result) == len(df)


class TestLookbackWeights:
    """Test lookback weights calculation."""

    def test_low_volatility_weights(self):
        """Test weights in low volatility regime."""
        vol_mult = pd.Series([1.5] * 10)  # Low vol
        vol_low = pd.Series([2.0] * 10)
        vol_high = pd.Series([2.5] * 10)

        w_short, w_medium, w_long = _calculate_lookback_weights(vol_mult, vol_low, vol_high)

        # Low vol should favor short lookback
        assert all(w_short > w_long)

    def test_high_volatility_weights(self):
        """Test weights in high volatility regime."""
        vol_mult = pd.Series([3.0] * 10)  # High vol
        vol_low = pd.Series([1.5] * 10)
        vol_high = pd.Series([2.0] * 10)

        w_short, w_medium, w_long = _calculate_lookback_weights(vol_mult, vol_low, vol_high)

        # High vol should favor long lookback
        assert all(w_long > w_short)

    def test_zero_total_weight_handling(self):
        """Test handling when total weight is zero."""
        vol_mult = pd.Series([2.0] * 10)
        vol_low = pd.Series([2.0] * 10)
        vol_high = pd.Series([2.0] * 10)

        # This would make all weights equal
        w_short, w_medium, w_long = _calculate_lookback_weights(vol_mult, vol_low, vol_high)

        total = w_short + w_medium + w_long
        np.testing.assert_array_almost_equal(total.values, np.ones(10))


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
