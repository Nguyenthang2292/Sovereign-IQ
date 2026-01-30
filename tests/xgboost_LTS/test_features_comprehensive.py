"""
Comprehensive tests for XGBoost features module.
Tests Rust/Python parity, edge cases, and feature calculations.
"""

import numpy as np
import pandas as pd
import pytest
from unittest.mock import patch, MagicMock

from modules.xgboost_LTS.utils.features import (
    add_price_derived_features,
    add_advanced_features,
)

# Check Rust availability
try:
    from modules.xgboost_LTS.rust_extensions import (
        add_price_derived_features_rust,
        add_advanced_features_rust,
    )

    RUST_AVAILABLE = True
except ImportError:
    RUST_AVAILABLE = False


class TestAddPriceDerivedFeatures:
    """Test price-derived features calculation."""

    def test_basic_functionality(self):
        """Test basic feature generation."""
        df = pd.DataFrame(
            {
                "open": [100.0, 101.0, 102.0, 103.0, 104.0],
                "high": [101.0, 102.0, 103.0, 104.0, 105.0],
                "low": [99.0, 100.0, 101.0, 102.0, 103.0],
                "close": [100.0, 101.0, 102.0, 103.0, 104.0],
                "volume": [1000, 1100, 1200, 1300, 1400],
            }
        )

        result = add_price_derived_features(df)

        # Check all expected columns exist
        expected_cols = ["returns_1", "returns_5", "log_volume", "high_low_range", "close_open_diff"]
        for col in expected_cols:
            assert col in result.columns, f"Missing column: {col}"

    def test_returns_1_calculation(self):
        """Test 1-period return calculation."""
        df = pd.DataFrame(
            {
                "open": [100.0, 100.0, 100.0],
                "high": [101.0, 101.0, 101.0],
                "low": [99.0, 99.0, 99.0],
                "close": [100.0, 110.0, 121.0],  # +10%, +10%
                "volume": [1000, 1000, 1000],
            }
        )

        result = add_price_derived_features(df)

        # First row should have returns_1 = 0 (or very small)
        assert abs(result["returns_1"].iloc[0]) < 0.001
        # Second row: (110-100)/100 = 0.10
        assert abs(result["returns_1"].iloc[1] - 0.10) < 0.001
        # Third row: (121-110)/110 = 0.10
        assert abs(result["returns_1"].iloc[2] - 0.10) < 0.001

    def test_returns_5_calculation(self):
        """Test 5-period return calculation."""
        df = pd.DataFrame(
            {
                "open": [100.0] * 6,
                "high": [101.0] * 6,
                "low": [99.0] * 6,
                "close": [100.0, 100.0, 100.0, 100.0, 100.0, 110.0],
                "volume": [1000] * 6,
            }
        )

        result = add_price_derived_features(df)

        # Row 5 should have returns_5 = (110-100)/100 = 0.10
        assert abs(result["returns_5"].iloc[5] - 0.10) < 0.001

    def test_log_volume_calculation(self):
        """Test log volume calculation."""
        df = pd.DataFrame({"open": [100.0], "high": [101.0], "low": [99.0], "close": [100.0], "volume": [1000]})

        result = add_price_derived_features(df)

        # log_volume should be log(1000 + 1)
        expected = np.log1p(1000)
        assert abs(result["log_volume"].iloc[0] - expected) < 0.001

    def test_high_low_range_calculation(self):
        """Test high-low range calculation."""
        df = pd.DataFrame(
            {
                "open": [100.0],
                "high": [105.0],  # Range = 5
                "low": [95.0],  # Range = 5
                "close": [100.0],  # Normalized by close
                "volume": [1000],
            }
        )

        result = add_price_derived_features(df)

        # high_low_range = (105-95)/100 = 0.10
        assert abs(result["high_low_range"].iloc[0] - 0.10) < 0.001

    def test_close_open_diff_calculation(self):
        """Test close-open difference calculation."""
        df = pd.DataFrame(
            {
                "open": [100.0],
                "high": [105.0],
                "low": [95.0],
                "close": [102.0],  # +2 from open
                "volume": [1000],
            }
        )

        result = add_price_derived_features(df)

        # close_open_diff = (102-100)/100 = 0.02
        assert abs(result["close_open_diff"].iloc[0] - 0.02) < 0.001

    def test_missing_columns_error(self):
        """Test error when required columns are missing."""
        df = pd.DataFrame(
            {
                "close": [100.0],  # Missing other OHLCV columns
            }
        )

        with pytest.raises(ValueError) as exc:
            add_price_derived_features(df)

        assert "Missing required" in str(exc.value)

    def test_empty_dataframe(self):
        """Test with empty DataFrame."""
        df = pd.DataFrame({"open": [], "high": [], "low": [], "close": [], "volume": []})

        result = add_price_derived_features(df)

        assert len(result) == 0
        assert "returns_1" in result.columns

    def test_zero_volume_handling(self):
        """Test handling of zero volume."""
        df = pd.DataFrame(
            {
                "open": [100.0],
                "high": [101.0],
                "low": [99.0],
                "close": [100.0],
                "volume": [0],  # Zero volume
            }
        )

        result = add_price_derived_features(df)

        # log_volume(0) = log(1) = 0
        assert abs(result["log_volume"].iloc[0] - 0.0) < 0.001

    def test_zero_close_handling(self):
        """Test handling of zero close price."""
        df = pd.DataFrame(
            {
                "open": [100.0],
                "high": [101.0],
                "low": [99.0],
                "close": [0.0],  # Zero close - edge case
                "volume": [1000],
            }
        )

        result = add_price_derived_features(df)

        # Should handle gracefully
        assert "high_low_range" in result.columns
        assert "close_open_diff" in result.columns

    @pytest.mark.skipif(not RUST_AVAILABLE, reason="Rust extensions not available")
    def test_rust_python_parity(self):
        """Test that Rust and Python implementations give same results."""
        np.random.seed(42)
        n = 100

        df = pd.DataFrame(
            {
                "open": np.cumsum(np.random.randn(n) * 0.5) + 100,
                "high": np.cumsum(np.random.randn(n) * 0.5) + 101,
                "low": np.cumsum(np.random.randn(n) * 0.5) + 99,
                "close": np.cumsum(np.random.randn(n) * 0.5) + 100,
                "volume": np.random.randint(1000, 10000, n),
            }
        )

        # Force Python implementation by temporarily disabling Rust
        with patch("modules.xgboost_LTS.utils.features.RUST_AVAILABLE", False):
            result_python = add_price_derived_features(df)

        # Use Rust implementation
        result_rust = add_price_derived_features(df)

        # Compare results (allowing small floating point differences)
        for col in ["returns_1", "returns_5", "log_volume", "high_low_range", "close_open_diff"]:
            np.testing.assert_allclose(
                result_python[col].values,
                result_rust[col].values,
                rtol=1e-10,
                atol=1e-10,
                err_msg=f"Mismatch in column {col}",
            )


class TestAddAdvancedFeatures:
    """Test advanced features calculation."""

    def test_basic_functionality(self):
        """Test basic advanced feature generation."""
        df = pd.DataFrame(
            {
                "open": [100.0] * 30,
                "high": [101.0] * 30,
                "low": [99.0] * 30,
                "close": np.linspace(100, 110, 30),
                "volume": [1000] * 30,
                "returns_1": np.random.randn(30) * 0.01,
            }
        )

        result = add_advanced_features(df)

        # Check ROC features
        for period in [3, 5, 10, 20]:
            assert f"roc_{period}" in result.columns

    def test_with_optional_columns(self):
        """Test with all optional technical indicators."""
        n = 50
        df = pd.DataFrame(
            {
                "open": [100.0] * n,
                "high": [101.0] * n,
                "low": [99.0] * n,
                "close": np.linspace(100, 120, n),
                "volume": [1000] * n,
                "returns_1": np.random.randn(n) * 0.01,
                "ATR_14": np.random.uniform(0.5, 2.0, n),
                "RSI_14": np.random.uniform(30, 70, n),
                "SMA_20": np.linspace(100, 120, n),
                "SMA_50": np.linspace(100, 120, n),
                "SMA_200": np.linspace(100, 120, n),
            }
        )

        result = add_advanced_features(df)

        # Check optional features
        assert "atr_ratio" in result.columns
        assert "price_to_SMA_20" in result.columns
        assert "rolling_std_10" in result.columns
        assert "rolling_skew_10" in result.columns

        # Check lag features
        assert "returns_1_lag_1" in result.columns
        assert "returns_1_lag_2" in result.columns
        assert "returns_1_lag_3" in result.columns

    def test_roc_calculation(self):
        """Test ROC (Rate of Change) calculation."""
        df = pd.DataFrame(
            {
                "open": [100.0] * 25,
                "high": [101.0] * 25,
                "low": [99.0] * 25,
                "close": [100.0] + [100.0] * 4 + [110.0] * 20,  # Jump at index 5
                "volume": [1000] * 25,
                "returns_1": [0.0] * 25,
            }
        )

        result = add_advanced_features(df)

        # roc_5 at index 5 should be (110-100)/100 = 0.10
        assert abs(result["roc_5"].iloc[5] - 0.10) < 0.001

    def test_sma_ratio_calculation(self):
        """Test price to SMA ratio calculation."""
        n = 30
        df = pd.DataFrame(
            {
                "open": [100.0] * n,
                "high": [101.0] * n,
                "low": [99.0] * n,
                "close": np.ones(n) * 110.0,
                "volume": [1000] * n,
                "returns_1": [0.0] * n,
                "SMA_20": np.ones(n) * 100.0,
            }
        )

        result = add_advanced_features(df)

        # price_to_SMA_20 = 110/100 = 1.1
        assert abs(result["price_to_SMA_20"].iloc[0] - 1.1) < 0.001

    def test_datetime_index_features(self):
        """Test time-based features with DatetimeIndex."""
        n = 100
        dates = pd.date_range("2024-01-01", periods=n, freq="H")

        df = pd.DataFrame(
            {
                "open": [100.0] * n,
                "high": [101.0] * n,
                "low": [99.0] * n,
                "close": np.linspace(100, 110, n),
                "volume": [1000] * n,
                "returns_1": np.random.randn(n) * 0.01,
            },
            index=dates,
        )

        result = add_advanced_features(df)

        # Check time-based features
        assert "hour" in result.columns
        assert "dayofweek" in result.columns
        assert "month" in result.columns

        # Verify hour extraction
        assert result["hour"].iloc[0] == dates[0].hour

    def test_rolling_std_skew_calculation(self):
        """Test rolling std and skew calculation."""
        n = 30
        # Create data with known volatility pattern
        returns = np.concatenate(
            [
                np.zeros(10),  # Low volatility
                np.random.randn(10) * 0.05,  # High volatility
                np.zeros(10),  # Low volatility
            ]
        )

        df = pd.DataFrame(
            {
                "open": [100.0] * n,
                "high": [101.0] * n,
                "low": [99.0] * n,
                "close": np.cumsum(returns) + 100,
                "volume": [1000] * n,
                "returns_1": returns,
            }
        )

        result = add_advanced_features(df)

        # Check that rolling_std increases during high volatility period
        low_vol_std = result["rolling_std_10"].iloc[9]
        high_vol_std = result["rolling_std_10"].iloc[19]

        if not pd.isna(low_vol_std) and not pd.isna(high_vol_std):
            assert high_vol_std > low_vol_std

    def test_lag_features_calculation(self):
        """Test lag feature calculation."""
        n = 10
        df = pd.DataFrame(
            {
                "open": [100.0] * n,
                "high": [101.0] * n,
                "low": [99.0] * n,
                "close": np.linspace(100, 110, n),
                "volume": [1000] * n,
                "returns_1": np.arange(n) * 0.01,  # 0, 0.01, 0.02, ...
            }
        )

        result = add_advanced_features(df)

        # Check lag features shift correctly
        assert result["returns_1_lag_1"].iloc[2] == result["returns_1"].iloc[1]
        assert result["returns_1_lag_2"].iloc[3] == result["returns_1"].iloc[1]

    def test_empty_dataframe(self):
        """Test with empty DataFrame."""
        df = pd.DataFrame({"open": [], "high": [], "low": [], "close": [], "volume": []})

        result = add_advanced_features(df)

        assert len(result) == 0

    @pytest.mark.skipif(not RUST_AVAILABLE, reason="Rust extensions not available")
    def test_rust_fallback_on_error(self):
        """Test that Python fallback is used when Rust fails."""
        df = pd.DataFrame(
            {
                "open": [100.0] * 10,
                "high": [101.0] * 10,
                "low": [99.0] * 10,
                "close": np.linspace(100, 110, 10),
                "volume": [1000] * 10,
                "returns_1": np.random.randn(10) * 0.01,
            }
        )

        # Mock Rust function to raise error
        with patch("modules.xgboost_LTS.utils.features.add_advanced_features_rust") as mock_rust:
            mock_rust.side_effect = Exception("Rust error")

            # Should fallback to Python and complete successfully
            result = add_advanced_features(df)

            assert len(result) == 10
            assert "roc_3" in result.columns


class TestFeatureEngineeringEdgeCases:
    """Test edge cases in feature engineering."""

    def test_extreme_price_values(self):
        """Test with extreme price values."""
        df = pd.DataFrame(
            {
                "open": [1e-10, 1e10],
                "high": [2e-10, 2e10],
                "low": [0.5e-10, 0.5e10],
                "close": [1e-10, 1e10],
                "volume": [1, 1000000],
            }
        )

        result = add_price_derived_features(df)

        # Should complete without error
        assert len(result) == 2
        assert all(np.isfinite(result["log_volume"]))

    def test_constant_prices(self):
        """Test with constant prices (no change)."""
        df = pd.DataFrame(
            {
                "open": [100.0] * 10,
                "high": [101.0] * 10,
                "low": [99.0] * 10,
                "close": [100.0] * 10,
                "volume": [1000] * 10,
            }
        )

        result = add_price_derived_features(df)

        # returns should be 0 or NaN
        assert all(abs(r) < 0.001 or np.isnan(r) for r in result["returns_1"])

    def test_single_row(self):
        """Test with single row."""
        df = pd.DataFrame({"open": [100.0], "high": [101.0], "low": [99.0], "close": [100.0], "volume": [1000]})

        result = add_price_derived_features(df)

        assert len(result) == 1
        assert "returns_1" in result.columns

    def test_nan_in_data(self):
        """Test handling of NaN values."""
        df = pd.DataFrame(
            {
                "open": [100.0, 101.0, np.nan],
                "high": [101.0, 102.0, 103.0],
                "low": [99.0, 100.0, 101.0],
                "close": [100.0, 101.0, 102.0],
                "volume": [1000, 1100, 1200],
            }
        )

        # May raise error or handle gracefully
        try:
            result = add_price_derived_features(df)
            assert len(result) == 3
        except (ValueError, FloatingPointError):
            pass  # Also acceptable


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
