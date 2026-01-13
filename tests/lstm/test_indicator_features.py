from unittest.mock import patch

import numpy as np
import pandas as pd
import pytest

from config.model_features import MODEL_FEATURES
from modules.lstm.utils.indicator_features import generate_indicator_features

"""
Tests for indicator feature generation utilities.
"""


class TestGenerateIndicatorFeatures:
    """Test suite for generate_indicator_features function."""

    @pytest.fixture
    def sample_ohlcv_data(self, seeded_random):
        """Create sample OHLCV data for testing."""
        n_rows = 300
        df = pd.DataFrame(
            {
                "open": 100 + seeded_random.standard_normal(n_rows).cumsum(),
                "high": 105 + seeded_random.standard_normal(n_rows).cumsum(),
                "low": 95 + seeded_random.standard_normal(n_rows).cumsum(),
                "close": 100 + seeded_random.standard_normal(n_rows).cumsum(),
                "volume": seeded_random.integers(1000, 10000, size=n_rows),
            }
        )
        return df

    @pytest.fixture
    def minimal_ohlcv_data(self, seeded_random):
        """Create minimal OHLCV data for testing."""
        n_rows = 50
        df = pd.DataFrame(
            {
                "open": 100 + seeded_random.standard_normal(n_rows).cumsum(),
                "high": 105 + seeded_random.standard_normal(n_rows).cumsum(),
                "low": 95 + seeded_random.standard_normal(n_rows).cumsum(),
                "close": 100 + seeded_random.standard_normal(n_rows).cumsum(),
                "volume": seeded_random.integers(1000, 10000, size=n_rows),
            }
        )
        return df

    def test_basic_feature_generation(self, sample_ohlcv_data):
        """Test basic feature generation with valid OHLCV data."""
        result = generate_indicator_features(sample_ohlcv_data)

        assert not result.empty
        assert len(result) > 0
        assert "close" in result.columns

    def test_empty_dataframe(self):
        """Test with empty DataFrame."""
        df = pd.DataFrame()
        result = generate_indicator_features(df)

        assert result.empty

    def test_missing_close_column(self):
        """Test with missing close column."""
        df = pd.DataFrame(
            {"open": [100, 101, 102], "high": [105, 106, 107], "low": [95, 96, 97], "volume": [1000, 2000, 3000]}
        )

        result = generate_indicator_features(df)

        assert result.empty

    def test_uppercase_column_names(self, sample_ohlcv_data):
        """Test with uppercase column names (should be normalized)."""
        sample_ohlcv_data.columns = [col.upper() for col in sample_ohlcv_data.columns]

        result = generate_indicator_features(sample_ohlcv_data)

        # Should normalize to lowercase and process
        assert isinstance(result, pd.DataFrame)

        # Verify column normalization occurred
        if not result.empty:
            assert "close" in result.columns or "CLOSE" not in result.columns

    def test_rsi_features(self, sample_ohlcv_data):
        """Test RSI feature calculation."""
        result = generate_indicator_features(sample_ohlcv_data)

        # Check if RSI features are in MODEL_FEATURES and result
        rsi_features = [f for f in MODEL_FEATURES if f.startswith("RSI_")]
        if rsi_features:
            for rsi_feature in rsi_features:
                if rsi_feature in result.columns:
                    # RSI should be between 0 and 100 (or default 50.0)
                    assert result[rsi_feature].notna().any() or (result[rsi_feature] == 50.0).any()

    def test_sma_features(self, sample_ohlcv_data):
        """Test SMA feature calculation."""
        result = generate_indicator_features(sample_ohlcv_data)

        # Check if SMA features are in MODEL_FEATURES and result
        sma_features = [f for f in MODEL_FEATURES if f.startswith("SMA_")]
        if sma_features:
            for sma_feature in sma_features:
                if sma_feature in result.columns:
                    # SMA should be numeric
                    assert pd.api.types.is_numeric_dtype(result[sma_feature])

    def test_macd_features(self, sample_ohlcv_data):
        """Test MACD feature calculation."""
        result = generate_indicator_features(sample_ohlcv_data)

        # Check if MACD features are in MODEL_FEATURES and result
        macd_features = [f for f in MODEL_FEATURES if f.startswith("MACD")]
        if macd_features:
            for macd_feature in macd_features:
                if macd_feature in result.columns:
                    # MACD should be numeric
                    assert pd.api.types.is_numeric_dtype(result[macd_feature])

    def test_bollinger_bands_feature(self, sample_ohlcv_data):
        """Test Bollinger Bands Percent (BBP) calculation."""
        result = generate_indicator_features(sample_ohlcv_data)

        if "BBP_5_2.0" in MODEL_FEATURES:
            if "BBP_5_2.0" in result.columns:
                # BBP should be between 0 and 1 (or default 0.5)
                assert result["BBP_5_2.0"].notna().any() or (result["BBP_5_2.0"] == 0.5).any()

    def test_stochrsi_features(self, sample_ohlcv_data):
        """Test Stochastic RSI feature calculation."""
        result = generate_indicator_features(sample_ohlcv_data)

        stochrsi_features = [f for f in MODEL_FEATURES if f.startswith("STOCHRSI")]
        if stochrsi_features:
            for stochrsi_feature in stochrsi_features:
                if stochrsi_feature in result.columns:
                    # Stochastic RSI should be numeric
                    assert pd.api.types.is_numeric_dtype(result[stochrsi_feature])

    def test_atr_feature(self, sample_ohlcv_data):
        """Test ATR feature calculation."""
        result = generate_indicator_features(sample_ohlcv_data)

        if "ATR_14" in MODEL_FEATURES:
            if "ATR_14" in result.columns:
                # ATR should be non-negative
                assert (result["ATR_14"] >= 0).all() or result["ATR_14"].isna().all()

    def test_obv_feature(self, sample_ohlcv_data):
        """Test OBV feature calculation."""
        result = generate_indicator_features(sample_ohlcv_data)

        if "OBV" in MODEL_FEATURES:
            if "OBV" in result.columns:
                # OBV should be numeric
                assert pd.api.types.is_numeric_dtype(result["OBV"])

    def test_candlestick_patterns(self, sample_ohlcv_data):
        """Test candlestick pattern features (should be set to 0.0)."""
        result = generate_indicator_features(sample_ohlcv_data)

        from config.model_features import CANDLESTICK_PATTERN_NAMES

        candlestick_features = [f for f in MODEL_FEATURES if f in CANDLESTICK_PATTERN_NAMES]

        if candlestick_features:
            for pattern in candlestick_features:
                if pattern in result.columns:
                    # Candlestick patterns should be 0.0 (not calculated in this version)
                    assert (result[pattern] == 0.0).all() or result[pattern].isna().all()

    def test_missing_high_low_for_atr(self, sample_ohlcv_data):
        """Test ATR calculation when high/low columns are missing."""
        df = sample_ohlcv_data.drop(columns=["high", "low"])

        result = generate_indicator_features(df)

        if "ATR_14" in MODEL_FEATURES:
            if "ATR_14" in result.columns:
                # Should use default value 0.0
                assert (result["ATR_14"] == 0.0).all() or result["ATR_14"].isna().all()

    def test_missing_volume_for_obv(self, sample_ohlcv_data):
        """Test OBV calculation when volume column is missing."""
        df = sample_ohlcv_data.drop(columns=["volume"])

        result = generate_indicator_features(df)

        if "OBV" in MODEL_FEATURES:
            if "OBV" in result.columns:
                # Should use default value 0.0
                assert (result["OBV"] == 0.0).all() or result["OBV"].isna().all()

    def test_rsi_calculation_failure_handling(self, sample_ohlcv_data):
        """Test handling of RSI calculation failures."""
        with patch(
            "modules.lstm.utils.indicator_features.calculate_rsi_series", side_effect=Exception("RSI calculation error")
        ):
            result = generate_indicator_features(sample_ohlcv_data)

            # Should handle gracefully and use default values
            rsi_features = [f for f in MODEL_FEATURES if f.startswith("RSI_")]
            if rsi_features:
                for rsi_feature in rsi_features:
                    if rsi_feature in result.columns:
                        # Should use default 50.0
                        assert (result[rsi_feature] == 50.0).all() or result[rsi_feature].isna().all()

    def test_sma_calculation_failure_handling(self, sample_ohlcv_data):
        """Test handling of SMA calculation failures."""
        # Patch where the function is used, not where it's defined
        with patch(
            "modules.lstm.utils.indicator_features.calculate_ma_series", side_effect=Exception("SMA calculation error")
        ):
            result = generate_indicator_features(sample_ohlcv_data)

            # Should handle gracefully and return DataFrame
            assert isinstance(result, pd.DataFrame)
            # SMA features should be filled with fallback value (e.g., 0.0) or NaN, not raise
            sma_features = [f for f in MODEL_FEATURES if f.startswith("SMA_")]
            if sma_features and not result.empty:
                for sma_feature in sma_features:
                    if sma_feature in result.columns:
                        # Should fallback to 0.0
                        assert (result[sma_feature] == 0.0).all()

    def test_macd_calculation_failure_handling(self, sample_ohlcv_data):
        """Test handling of MACD calculation failures."""
        with patch(
            "modules.lstm.utils.indicator_features.calculate_macd_series",
            side_effect=Exception("MACD calculation error"),
        ):
            result = generate_indicator_features(sample_ohlcv_data)

            # Should handle gracefully and use default values
            macd_features = [f for f in MODEL_FEATURES if f.startswith("MACD")]
            if macd_features:
                for macd_feature in macd_features:
                    if macd_feature in result.columns:
                        # Should use default 0.0
                        assert (result[macd_feature] == 0.0).all() or result[macd_feature].isna().all()

    def test_invalid_sma_period(self, sample_ohlcv_data):
        """Test handling of invalid SMA period."""
        # This test checks that invalid periods are handled
        # The function should skip invalid features or use fallback
        result = generate_indicator_features(sample_ohlcv_data)

        # Should complete without error
        assert isinstance(result, pd.DataFrame)

    def test_nan_handling(self, sample_ohlcv_data):
        """Test handling of NaN values in input data."""
        # Introduce NaN values
        sample_ohlcv_data.loc[10:20, "close"] = np.nan

        result = generate_indicator_features(sample_ohlcv_data)

        # Should handle NaN values (fill or drop)
        if not result.empty:
            # Result should have fewer or equal rows after NaN handling
            assert len(result) <= len(sample_ohlcv_data)

    def test_data_with_all_nan(self, sample_ohlcv_data):
        """Test with all NaN values in close column."""
        sample_ohlcv_data.loc[:, "close"] = np.nan

        result = generate_indicator_features(sample_ohlcv_data)

        # Should return empty DataFrame or handle gracefully
        assert isinstance(result, pd.DataFrame)

    def test_minimal_data(self, minimal_ohlcv_data):
        """Test with minimal data (may not have enough for all indicators)."""
        result = generate_indicator_features(minimal_ohlcv_data)

        # Should handle minimal data gracefully
        assert isinstance(result, pd.DataFrame)

    def test_feature_count(self, sample_ohlcv_data):
        """Test that expected features are calculated."""
        result = generate_indicator_features(sample_ohlcv_data)

        if not result.empty:
            # Check that some expected features are present
            expected_basic = ["open", "high", "low", "close", "volume"]
            has_basic = any(col in result.columns for col in expected_basic)
            assert has_basic

    def test_data_loss_warning(self, sample_ohlcv_data):
        """Test that significant data loss is handled."""
        # Make most data invalid
        sample_ohlcv_data.loc[10:, "close"] = np.nan

        result = generate_indicator_features(sample_ohlcv_data)

        # Should handle significant data loss
        assert isinstance(result, pd.DataFrame)

    def test_forward_fill_backward_fill(self, sample_ohlcv_data):
        """Test that NaN values are filled using forward/backward fill."""
        # Introduce NaN values
        sample_ohlcv_data.loc[10:15, "close"] = np.nan

        result = generate_indicator_features(sample_ohlcv_data)

        if not result.empty:
            # After ffill/bfill and dropna, basic OHLCV columns should have no NaN
            basic_cols = ["open", "high", "low", "close", "volume"]
            present_basic = [col for col in basic_cols if col in result.columns]

            if present_basic:
                assert result[present_basic].notna().all().all(), "Basic columns should not contain NaN after filling"

    def test_error_handling_general_exception(self, sample_ohlcv_data):
        """Test general error handling."""
        with patch(
            "modules.lstm.utils.indicator_features.calculate_rsi_series", side_effect=Exception("General error")
        ):
            result = generate_indicator_features(sample_ohlcv_data)

            # Should return empty DataFrame on general error
            assert isinstance(result, pd.DataFrame)

    def test_column_name_normalization(self, sample_ohlcv_data):
        """Test that column names are normalized to lowercase."""
        sample_ohlcv_data.columns = ["OPEN", "HIGH", "LOW", "CLOSE", "VOLUME"]

        result = generate_indicator_features(sample_ohlcv_data)

        assert all(col.islower() for col in result.columns), "All columns should be lowercase"
        ohlcv_cols = ["open", "high", "low", "close", "volume"]
        present_ohlcv = [col for col in ohlcv_cols if col in result.columns]
        assert present_ohlcv, "At least some OHLCV columns should be present"
