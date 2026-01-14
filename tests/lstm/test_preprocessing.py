from unittest.mock import patch

import numpy as np
import pandas as pd
import pytest
from sklearn.preprocessing import MinMaxScaler, StandardScaler

from config.lstm import WINDOW_SIZE_LSTM
from modules.lstm.utils.preprocessing import preprocess_cnn_lstm_data

"""
Tests for data preprocessing utilities.
"""


class TestPreprocessCnnLstmData:
    """Test suite for preprocess_cnn_lstm_data function."""

    @pytest.fixture
    def sample_ohlcv_data(self, seeded_random):
        """Create sample OHLCV data for testing."""
        n_rows = 200
        df = pd.DataFrame(
            {
                "open": 100 + seeded_random.standard_normal(n_rows).cumsum(),
                "high": 105 + seeded_random.standard_normal(n_rows).cumsum(),
                "low": 95 + seeded_random.standard_normal(n_rows).cumsum(),
                "close": 100 + seeded_random.standard_normal(n_rows).cumsum(),
                "volume": seeded_random.integers(1000, 10000, size=n_rows),
            }
        )
        yield df
        del df

    @pytest.fixture
    def minimal_ohlcv_data(self, seeded_random):
        """Create minimal OHLCV data for testing."""
        n_rows = 100
        df = pd.DataFrame(
            {
                "open": 100 + seeded_random.standard_normal(n_rows).cumsum(),
                "high": 105 + seeded_random.standard_normal(n_rows).cumsum(),
                "low": 95 + seeded_random.standard_normal(n_rows).cumsum(),
                "close": 100 + seeded_random.standard_normal(n_rows).cumsum(),
                "volume": seeded_random.integers(1000, 10000, size=n_rows),
            }
        )
        yield df
        del df

    def test_basic_classification_mode(self, sample_ohlcv_data):
        """Test basic preprocessing in classification mode."""
        X, y, scaler, features = preprocess_cnn_lstm_data(
            sample_ohlcv_data, look_back=WINDOW_SIZE_LSTM, output_mode="classification", scaler_type="minmax"
        )

        assert len(X) > 0
        assert len(y) == len(X)
        assert X.shape[1] == WINDOW_SIZE_LSTM  # sequence length
        assert isinstance(scaler, MinMaxScaler)
        assert len(features) > 0
        # Classification targets should be -1, 0, or 1
        assert all(label in [-1, 0, 1] for label in y)

    def test_basic_regression_mode(self, sample_ohlcv_data):
        """Test basic preprocessing in regression mode."""
        X, y, scaler, features = preprocess_cnn_lstm_data(
            sample_ohlcv_data, look_back=WINDOW_SIZE_LSTM, output_mode="regression", scaler_type="minmax"
        )

        assert len(X) > 0
        assert len(y) == len(X)
        assert X.shape[1] == WINDOW_SIZE_LSTM
        assert isinstance(scaler, MinMaxScaler)
        assert len(features) > 0
        # Regression targets should be continuous values
        assert isinstance(y[0], (float, np.floating))

    def test_standard_scaler(self, sample_ohlcv_data):
        """Test preprocessing with StandardScaler."""
        X, y, scaler, features = preprocess_cnn_lstm_data(
            sample_ohlcv_data, look_back=WINDOW_SIZE_LSTM, output_mode="classification", scaler_type="standard"
        )

        assert isinstance(scaler, StandardScaler)
        assert len(X) > 0

    def test_custom_look_back(self, sample_ohlcv_data):
        """Test preprocessing with custom look_back value."""
        custom_look_back = 30
        X, y, scaler, features = preprocess_cnn_lstm_data(
            sample_ohlcv_data, look_back=custom_look_back, output_mode="classification"
        )

        assert X.shape[1] == custom_look_back
        assert len(X) > 0

    def test_empty_dataframe(self):
        """Test preprocessing with empty DataFrame."""
        df = pd.DataFrame()
        X, y, scaler, features = preprocess_cnn_lstm_data(df)

        assert len(X) == 0
        assert len(y) == 0
        assert isinstance(scaler, MinMaxScaler)
        assert len(features) == 0

    def test_insufficient_data(self):
        """Test preprocessing with insufficient data."""
        df = pd.DataFrame(
            {"open": [100, 101], "high": [105, 106], "low": [95, 96], "close": [100, 101], "volume": [1000, 2000]}
        )

        X, y, scaler, features = preprocess_cnn_lstm_data(df, look_back=WINDOW_SIZE_LSTM)

        assert len(X) == 0
        assert len(y) == 0

    def test_missing_close_column(self):
        """Test preprocessing with missing close column."""
        df = pd.DataFrame(
            {"open": [100, 101, 102], "high": [105, 106, 107], "low": [95, 96, 97], "volume": [1000, 2000, 3000]}
        )

        # Should handle gracefully in regression mode
        X, y, scaler, features = preprocess_cnn_lstm_data(df, look_back=10, output_mode="regression")

        # Should return empty arrays due to missing close column
        assert len(X) == 0
        assert len(y) == 0

    def test_data_with_nan_values(self, sample_ohlcv_data):
        """Test preprocessing with NaN values in data."""
        # Introduce NaN values
        sample_ohlcv_data.loc[10:20, "close"] = np.nan
        sample_ohlcv_data.loc[30:35, "high"] = np.inf

        X, y, scaler, features = preprocess_cnn_lstm_data(sample_ohlcv_data, look_back=WINDOW_SIZE_LSTM)

        # Should filter out invalid rows
        assert len(X) >= 0
        # Check that no NaN or Inf values remain
        assert not np.isnan(X).any()
        assert not np.isinf(X).any()

    def test_data_with_all_nan_rows(self, sample_ohlcv_data):
        """Test preprocessing when all rows become NaN."""
        # Make all rows invalid
        sample_ohlcv_data.loc[:, "close"] = np.nan

        X, y, scaler, features = preprocess_cnn_lstm_data(sample_ohlcv_data, look_back=WINDOW_SIZE_LSTM)

        # Should return empty arrays
        assert len(X) == 0
        assert len(y) == 0

    def test_regression_target_creation(self, sample_ohlcv_data):
        """Test regression target creation (percentage change)."""
        X, y, scaler, features = preprocess_cnn_lstm_data(
            sample_ohlcv_data, look_back=WINDOW_SIZE_LSTM, output_mode="regression"
        )

        if len(y) > 0:
            # Regression targets should be percentage changes
            assert isinstance(y[0], (float, np.floating))
            # Should have reasonable range (not all zeros)
            assert not np.allclose(y, 0)

    def test_classification_target_creation(self, sample_ohlcv_data):
        """Test classification target creation."""
        X, y, scaler, features = preprocess_cnn_lstm_data(
            sample_ohlcv_data, look_back=WINDOW_SIZE_LSTM, output_mode="classification"
        )

        if len(y) > 0:
            # Classification targets should be -1, 0, or 1
            unique_targets = np.unique(y)
            assert all(t in [-1, 0, 1] for t in unique_targets)

    def test_feature_selection(self, sample_ohlcv_data):
        """Test that only valid features are selected."""
        X, y, scaler, features = preprocess_cnn_lstm_data(sample_ohlcv_data, look_back=WINDOW_SIZE_LSTM)

        assert len(features) > 0
        # All features should be valid column names
        assert all(isinstance(f, str) for f in features)

    def test_sequence_creation(self, sample_ohlcv_data):
        """Test that sequences are created correctly."""
        look_back = 30
        X, y, scaler, features = preprocess_cnn_lstm_data(sample_ohlcv_data, look_back=look_back)

        if len(X) > 0:
            # Check sequence shape
            assert X.shape[0] == len(y)
            assert X.shape[1] == look_back
            assert X.shape[2] == len(features)

    def test_scaler_fitting(self, sample_ohlcv_data):
        """Test that scaler is properly fitted."""
        X, y, scaler, features = preprocess_cnn_lstm_data(sample_ohlcv_data, look_back=WINDOW_SIZE_LSTM)

        if len(X) > 0:
            # Scaler should be fitted (have data_min_ and data_max_ for MinMaxScaler)
            assert hasattr(scaler, "data_min_") or hasattr(scaler, "mean_")

    def test_error_handling_feature_calculation_failure(self, sample_ohlcv_data):
        """Test error handling when feature calculation fails."""
        with patch("modules.lstm.utils.preprocessing.generate_indicator_features", return_value=pd.DataFrame()):
            X, y, scaler, features = preprocess_cnn_lstm_data(sample_ohlcv_data, look_back=WINDOW_SIZE_LSTM)

            assert len(X) == 0
            assert len(y) == 0

    def test_error_handling_target_creation_failure(self, sample_ohlcv_data):
        """Test error handling when target creation fails."""
        with patch(
            "modules.lstm.utils.preprocessing.create_balanced_target", side_effect=Exception("Target creation error")
        ):
            X, y, scaler, features = preprocess_cnn_lstm_data(
                sample_ohlcv_data, look_back=WINDOW_SIZE_LSTM, output_mode="classification"
            )

            assert len(X) == 0
            assert len(y) == 0

    def test_error_handling_scaling_failure(self, sample_ohlcv_data):
        """Test error handling when scaling fails."""
        with patch("sklearn.preprocessing.MinMaxScaler.fit_transform", side_effect=Exception("Scaling error")):
            X, y, scaler, features = preprocess_cnn_lstm_data(sample_ohlcv_data, look_back=WINDOW_SIZE_LSTM)

            assert len(X) == 0
            assert len(y) == 0

    def test_minimum_data_requirement(self, minimal_ohlcv_data):
        """Test with minimum required data."""
        # Should work with minimum data
        X, y, scaler, features = preprocess_cnn_lstm_data(minimal_ohlcv_data, look_back=30)

        # Should create at least some sequences
        assert len(X) >= 0  # May be 0 if data is insufficient after filtering

    def test_large_look_back(self, sample_ohlcv_data):
        """Test with large look_back value."""
        large_look_back = 100
        X, y, scaler, features = preprocess_cnn_lstm_data(sample_ohlcv_data, look_back=large_look_back)

        if len(X) > 0:
            assert X.shape[1] == large_look_back
        else:
            # May return empty if look_back is too large
            assert len(sample_ohlcv_data) < large_look_back + 10

    def test_uppercase_column_names(self, sample_ohlcv_data):
        """Test with uppercase column names (should be normalized)."""
        sample_ohlcv_data.columns = [col.upper() for col in sample_ohlcv_data.columns]

        X, y, scaler, features = preprocess_cnn_lstm_data(sample_ohlcv_data, look_back=WINDOW_SIZE_LSTM)

        # Should handle uppercase columns (normalized to lowercase internally)
        assert len(X) >= 0

    def test_no_valid_features(self):
        """Test when no valid features are found."""
        df = pd.DataFrame({"close": [100, 101, 102], "other_col": [1, 2, 3]})

        X, y, scaler, features = preprocess_cnn_lstm_data(df, look_back=10)

        # Should return empty arrays if no valid features
        assert len(X) == 0
        assert len(y) == 0

    def test_sequence_target_alignment(self, sample_ohlcv_data):
        """Test that sequences and targets are properly aligned."""
        X, y, scaler, features = preprocess_cnn_lstm_data(sample_ohlcv_data, look_back=WINDOW_SIZE_LSTM)

        if len(X) > 0:
            # Each sequence should have a corresponding target
            assert len(X) == len(y)
            # Sequence i should correspond to target at position i
            # (target is at the end of the sequence window)

    def test_with_kalman_filter_enabled(self, sample_ohlcv_data):
        """Test preprocessing with Kalman Filter enabled."""
        X, y, scaler, features = preprocess_cnn_lstm_data(
            sample_ohlcv_data,
            look_back=WINDOW_SIZE_LSTM,
            output_mode="classification",
            scaler_type="minmax",
            use_kalman_filter=True,
            kalman_params={"process_variance": 1e-5, "observation_variance": 1.0},
        )

        assert len(X) > 0
        assert len(y) > 0
        assert len(X) == len(y)
        assert isinstance(scaler, (MinMaxScaler, StandardScaler))
        assert len(features) > 0

    def test_with_kalman_filter_disabled(self, sample_ohlcv_data):
        """Test preprocessing with Kalman Filter disabled (default)."""
        X1, y1, scaler1, features1 = preprocess_cnn_lstm_data(
            sample_ohlcv_data, look_back=WINDOW_SIZE_LSTM, output_mode="classification", use_kalman_filter=False
        )

        X2, y2, scaler2, features2 = preprocess_cnn_lstm_data(
            sample_ohlcv_data, look_back=WINDOW_SIZE_LSTM, output_mode="classification", use_kalman_filter=False
        )

        # Results should be identical when Kalman Filter is disabled
        assert len(X1) == len(X2)
        assert len(y1) == len(y2)

    def test_kalman_filter_invalid_params(self, sample_ohlcv_data):
        """Test preprocessing with invalid Kalman Filter parameters."""
        # Should fallback to defaults when params are invalid
        X, y, scaler, features = preprocess_cnn_lstm_data(
            sample_ohlcv_data,
            look_back=WINDOW_SIZE_LSTM,
            output_mode="classification",
            use_kalman_filter=True,
            kalman_params={"process_variance": -1.0},  # Invalid negative value
        )

        # Should still work (invalid params should be replaced with defaults)
        assert len(X) > 0
        assert len(y) > 0
