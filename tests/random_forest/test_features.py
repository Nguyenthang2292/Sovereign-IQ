import sys
from pathlib import Path

# Add project root to Python path for imports
ROOT = Path(__file__).resolve().parents[2]
ROOT_STR = str(ROOT)
if ROOT_STR not in sys.path:
    sys.path.insert(0, ROOT_STR)

import pandas as pd
import numpy as np
import pytest
from modules.random_forest.utils.features import add_advanced_features, get_enhanced_feature_names


class TestAdvancedFeatures:
    @pytest.fixture
    def basic_df(self):
        """Create a DataFrame with basic base features."""
        np.random.seed(42)
        dates = pd.date_range("2023-01-01", periods=50, freq="H")
        df = pd.DataFrame(index=dates)
        df["close"] = 100 + np.random.randn(50).cumsum()
        df["open"] = df["close"] + np.random.randn(50)
        df["high"] = df["close"] + 2
        df["low"] = df["close"] - 2
        df["volume"] = 1000 + np.random.rand(50) * 500

        # Add 'ATR_14', 'SMA_20', 'log_volume' as mock indicators
        df["ATR_14"] = np.random.rand(50)
        df["SMA_20"] = df["close"].rolling(20).mean()
        df["log_volume"] = np.log(df["volume"])

        return df

    def test_add_advanced_features_structure(self, basic_df):
        """Test that new features are added correctly."""
        enhanced_df = add_advanced_features(basic_df)

        # Check that original columns remain
        for col in basic_df.columns:
            assert col in enhanced_df.columns

        # Check specific enhanced features
        assert "roc_5" in enhanced_df.columns
        assert "atr_ratio" in enhanced_df.columns
        assert "price_to_SMA_20" in enhanced_df.columns
        assert "rolling_std_10" in enhanced_df.columns
        assert "log_volume_lag_1" in enhanced_df.columns
        assert "hour" in enhanced_df.columns  # From DatetimeIndex

    def test_get_enhanced_feature_names(self):
        """Test the logic for identifying enhanced feature names."""
        cols = ["roc_5", "atr_ratio", "open", "close", "SMA_50", "price_to_SMA_50"]
        enhanced = get_enhanced_feature_names(cols)

        assert "roc_5" in enhanced
        assert "atr_ratio" in enhanced
        assert "price_to_SMA_50" in enhanced
        assert "open" not in enhanced
        assert "SMA_50" not in enhanced

    def test_time_based_features(self, basic_df):
        """Test time-based feature extraction."""
        # Ensure index is datetime
        assert isinstance(basic_df.index, pd.DatetimeIndex)

        enhanced_df = add_advanced_features(basic_df)

        assert "hour" in enhanced_df.columns
        assert "dayofweek" in enhanced_df.columns

        # Check values
        assert enhanced_df["hour"].iloc[0] == 0  # 2023-01-01 00:00:00
        assert enhanced_df["dayofweek"].iloc[0] == 6  # Sunday
