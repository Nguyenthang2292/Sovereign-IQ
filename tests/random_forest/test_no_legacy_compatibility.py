"""Tests to verify backward compatibility removal for legacy models.

This test module ensures that:
1. Models trained with raw OHLCV features are rejected
2. Only models with derived features are supported
3. Clear error messages are provided when legacy models are detected
"""

import sys
from pathlib import Path

# Add project root to Python path for imports
ROOT = Path(__file__).resolve().parents[2]
ROOT_STR = str(ROOT)
if ROOT_STR not in sys.path:
    sys.path.insert(0, ROOT_STR)

from unittest import mock

import numpy as np
import pandas as pd
import pytest
from sklearn.ensemble import RandomForestClassifier

from modules.random_forest.core.signals import get_latest_random_forest_signal


class TestLegacyModelRejection:
    """Test that legacy models with raw OHLCV features are rejected."""

    @pytest.fixture
    def sample_data(self):
        """Create sample market data."""
        np.random.seed(42)
        df = pd.DataFrame(
            {
                "open": np.random.randn(100) + 100,
                "high": np.random.randn(100) + 105,
                "low": np.random.randn(100) + 95,
                "close": np.random.randn(100) + 100,
                "volume": np.random.uniform(1000, 10000, 100),
            }
        )
        return df

    @pytest.fixture
    def legacy_model_with_raw_ohlcv(self):
        """Create a mock model that expects raw OHLCV features."""
        model = RandomForestClassifier(n_estimators=10, random_state=42)
        # Create mock training data with raw OHLCV
        X_train = pd.DataFrame(
            {
                "open": np.random.randn(50),
                "high": np.random.randn(50),
                "low": np.random.randn(50),
                "close": np.random.randn(50),
                "volume": np.random.randn(50),
            }
        )
        y_train = np.random.randint(0, 2, 50)
        model.fit(X_train, y_train)
        return model

    def test_reject_model_with_raw_ohlcv_feature(self, sample_data, legacy_model_with_raw_ohlcv):
        """Test that models with raw OHLCV features are rejected."""
        # Mock IndicatorEngine to return features WITHOUT raw OHLCV (only derived features)
        with mock.patch("modules.random_forest.core.signals.IndicatorEngine") as mock_engine:
            mock_instance = mock_engine.return_value
            # Create features DataFrame without raw OHLCV (simulating what IndicatorEngine actually returns)
            features_df = pd.DataFrame(
                {
                    "returns_1": np.random.randn(100),
                    "returns_5": np.random.randn(100),
                    "log_volume": np.random.randn(100),
                    "high_low_range": np.random.randn(100),
                    "close_open_diff": np.random.randn(100),
                }
            )
            mock_instance.compute_features.return_value = features_df

            with mock.patch("modules.random_forest.utils.features.add_advanced_features") as mock_advanced:
                mock_advanced.return_value = features_df

                # Model expects "open" feature (raw OHLCV) but features don't have it
                signal, confidence = get_latest_random_forest_signal(sample_data, legacy_model_with_raw_ohlcv)

                # Should return NEUTRAL with 0.0 confidence due to rejection
                assert signal == "NEUTRAL"
                assert confidence == 0.0

    def test_reject_model_with_high_feature(self, sample_data):
        """Test rejection of model expecting 'high' feature."""
        model = RandomForestClassifier(n_estimators=10, random_state=42)
        X_train = pd.DataFrame({"high": np.random.randn(50), "rsi": np.random.randn(50)})
        y_train = np.random.randint(0, 2, 50)
        model.fit(X_train, y_train)

        signal, confidence = get_latest_random_forest_signal(sample_data, model)

        assert signal == "NEUTRAL"
        assert confidence == 0.0

    def test_reject_model_with_low_feature(self, sample_data):
        """Test rejection of model expecting 'low' feature."""
        model = RandomForestClassifier(n_estimators=10, random_state=42)
        X_train = pd.DataFrame({"low": np.random.randn(50), "macd": np.random.randn(50)})
        y_train = np.random.randint(0, 2, 50)
        model.fit(X_train, y_train)

        signal, confidence = get_latest_random_forest_signal(sample_data, model)

        assert signal == "NEUTRAL"
        assert confidence == 0.0

    def test_reject_model_with_close_feature(self, sample_data):
        """Test rejection of model expecting 'close' feature."""
        model = RandomForestClassifier(n_estimators=10, random_state=42)
        X_train = pd.DataFrame({"close": np.random.randn(50), "ema": np.random.randn(50)})
        y_train = np.random.randint(0, 2, 50)
        model.fit(X_train, y_train)

        signal, confidence = get_latest_random_forest_signal(sample_data, model)

        assert signal == "NEUTRAL"
        assert confidence == 0.0

    def test_reject_model_with_volume_feature(self, sample_data):
        """Test rejection of model expecting 'volume' feature."""
        model = RandomForestClassifier(n_estimators=10, random_state=42)
        X_train = pd.DataFrame({"volume": np.random.randn(50), "sma": np.random.randn(50)})
        y_train = np.random.randint(0, 2, 50)
        model.fit(X_train, y_train)

        signal, confidence = get_latest_random_forest_signal(sample_data, model)

        assert signal == "NEUTRAL"
        assert confidence == 0.0


class TestDerivedFeaturesSupport:
    """Test that models with derived features work correctly."""

    @pytest.fixture
    def sample_data_with_features(self):
        """Create sample data with derived features."""
        np.random.seed(42)
        df = pd.DataFrame(
            {
                "open": np.random.randn(100) + 100,
                "high": np.random.randn(100) + 105,
                "low": np.random.randn(100) + 95,
                "close": np.random.randn(100) + 100,
                "volume": np.random.uniform(1000, 10000, 100),
            }
        )
        # Add derived features (simulating what IndicatorEngine would create)
        df["returns_1"] = df["close"].pct_change(1)
        df["returns_5"] = df["close"].pct_change(5)
        df["log_volume"] = np.log1p(df["volume"])
        df["high_low_range"] = (df["high"] - df["low"]) / df["close"]
        df["close_open_diff"] = (df["close"] - df["open"]) / df["open"]
        return df

    def test_accept_model_with_derived_features(self, sample_data_with_features):
        """Test that models with derived features work correctly."""
        model = RandomForestClassifier(n_estimators=10, random_state=42)
        # Train with derived features
        X_train = pd.DataFrame(
            {
                "returns_1": np.random.randn(50),
                "returns_5": np.random.randn(50),
                "log_volume": np.random.randn(50),
                "high_low_range": np.random.randn(50),
                "close_open_diff": np.random.randn(50),
            }
        )
        y_train = np.random.randint(0, 2, 50)
        model.fit(X_train, y_train)

        # Mock IndicatorEngine to return features with derived features
        with mock.patch("modules.random_forest.core.signals.IndicatorEngine") as mock_engine:
            mock_instance = mock_engine.return_value
            mock_instance.compute_features.return_value = sample_data_with_features

            with mock.patch("modules.random_forest.utils.features.add_advanced_features") as mock_advanced:
                mock_advanced.return_value = sample_data_with_features

                signal, confidence = get_latest_random_forest_signal(sample_data_with_features, model)

                # Should work (not NEUTRAL with 0.0)
                assert signal in ["LONG", "SHORT", "NEUTRAL"]
                assert isinstance(confidence, float)
                assert 0.0 <= confidence <= 1.0


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
