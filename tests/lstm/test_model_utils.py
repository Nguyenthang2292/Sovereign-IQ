from pathlib import Path
from unittest.mock import Mock, patch

import numpy as np
import pandas as pd
import pytest
import torch
import torch.nn as nn
from sklearn.preprocessing import MinMaxScaler

from config.lstm import WINDOW_SIZE_LSTM
from modules.lstm.models.model_factory import create_cnn_lstm_attention_model
from modules.lstm.models.model_utils import (
    get_latest_lstm_attention_signal,
    get_latest_signal,
    load_lstm_attention_model,
    load_lstm_model,
)

"""
Tests for model utilities module.
"""


class TestLoadLSTMModel:
    """Test suite for load_lstm_model function."""

    def test_load_model_nonexistent_file(self):
        """Test loading non-existent model file."""
        nonexistent_path = Path("/nonexistent/path/model.pth")
        result = load_lstm_model(nonexistent_path)
        assert result is None

    def test_load_model_default_path_nonexistent(self):
        """Test loading model with default path when file doesn't exist."""
        with patch("modules.lstm.models.model_utils.torch.load") as mock_load:
            mock_load.side_effect = FileNotFoundError("File not found")
            result = load_lstm_model()
            assert result is None

    def test_load_model_new_format_with_cnn_attention(self):
        """Test loading model with new format (CNN-Attention)."""
        checkpoint = {
            "model_config": {
                "input_size": 10,
                "use_attention": True,
                "use_cnn": True,
                "attention_heads": 4,
                "look_back": 20,
                "output_mode": "classification",
                "num_classes": 3,
            },
            "model_state_dict": {},
        }

        with patch("modules.lstm.models.model_utils.torch.load", return_value=checkpoint):
            with patch("modules.lstm.models.model_utils.create_cnn_lstm_attention_model") as mock_factory:
                with patch("pathlib.Path.exists", return_value=True):
                    mock_model = Mock()
                    mock_model.load_state_dict = Mock()
                    mock_factory.return_value = mock_model

                    result = load_lstm_model(Path("test.pth"))
                    assert result is not None
                    mock_model.load_state_dict.assert_called_once()
                    mock_model.eval.assert_called_once()

    def test_load_model_new_format_lstm_only(self):
        """Test loading model with new format (LSTM only)."""
        checkpoint = {
            "model_config": {
                "input_size": 10,
                "use_attention": False,
                "use_cnn": False,
                "look_back": 20,
                "output_mode": "classification",
                "num_classes": 3,
            },
            "model_state_dict": {},
        }

        with patch("modules.lstm.models.model_utils.torch.load", return_value=checkpoint):
            with patch("modules.lstm.models.model_utils.create_cnn_lstm_attention_model") as mock_factory:
                with patch("pathlib.Path.exists", return_value=True):
                    mock_model = Mock()
                    mock_model.load_state_dict = Mock()
                    mock_factory.return_value = mock_model

                    result = load_lstm_model(Path("test.pth"))
                    assert result is not None

    def test_load_model_old_format_with_attention(self):
        """Test loading model with old format (with attention)."""
        checkpoint = {"input_size": 10, "use_attention": True, "attention_heads": 4, "model_state_dict": {}}

        with patch("modules.lstm.models.model_utils.torch.load", return_value=checkpoint):
            with patch("modules.lstm.models.model_utils.LSTMAttentionModel") as mock_model_class:
                with patch("pathlib.Path.exists", return_value=True):
                    mock_model = Mock()
                    mock_model.load_state_dict = Mock()
                    mock_model_class.return_value = mock_model

                    result = load_lstm_model(Path("test.pth"))
                    assert result is not None

    def test_load_model_old_format_without_attention(self):
        """Test loading model with old format (without attention)."""
        checkpoint = {"input_size": 10, "use_attention": False, "model_state_dict": {}}

        with patch("modules.lstm.models.model_utils.torch.load", return_value=checkpoint):
            with patch("modules.lstm.models.model_utils.LSTMModel") as mock_model_class:
                with patch("pathlib.Path.exists", return_value=True):
                    mock_model = Mock()
                    mock_model.load_state_dict = Mock()
                    mock_model_class.return_value = mock_model

                    result = load_lstm_model(Path("test.pth"))
                    assert result is not None

    def test_load_model_old_format_missing_input_size(self):
        """Test loading model with old format missing input_size."""
        checkpoint = {"use_attention": False, "model_state_dict": {}}

        with patch("modules.lstm.models.model_utils.torch.load", return_value=checkpoint):
            result = load_lstm_model(Path("test.pth"))
            assert result is None

    def test_load_model_exception_handling(self):
        """Test exception handling during model loading."""
        with patch("modules.lstm.models.model_utils.torch.load") as mock_load:
            mock_load.side_effect = Exception("Unexpected error")
            result = load_lstm_model(Path("test.pth"))
            assert result is None

    def test_load_lstm_attention_model_backward_compat(self):
        """Test backward compatibility alias."""
        with patch("modules.lstm.models.model_utils.load_lstm_model") as mock_load:
            mock_model = Mock()
            mock_load.return_value = mock_model
            result = load_lstm_attention_model(Path("test.pth"))
            assert result == mock_model
            mock_load.assert_called_once_with(Path("test.pth"))


class TestGetLatestSignal:
    """Test suite for get_latest_signal function."""

    @pytest.fixture
    def sample_model(self):
        """Create a sample model for testing."""
        model = create_cnn_lstm_attention_model(input_size=10, use_attention=False, use_cnn=False)
        model.eval()
        return model

    @pytest.fixture
    def sample_dataframe(self):
        """Create sample market data DataFrame."""
        np.random.seed(42)
        n_samples = 100
        df = pd.DataFrame(
            {
                "open": np.random.randn(n_samples) * 100 + 50000,
                "high": np.random.randn(n_samples) * 100 + 50100,
                "low": np.random.randn(n_samples) * 100 + 49900,
                "close": np.random.randn(n_samples) * 100 + 50000,
            }
        )
        return df

    @pytest.fixture
    def sample_scaler(self):
        """Create a sample scaler."""
        scaler = MinMaxScaler()
        # Fit with dummy data
        dummy_data = np.random.randn(100, 10)
        scaler.fit(dummy_data)
        return scaler

    def test_get_signal_empty_dataframe(self, sample_model):
        """Test signal generation with empty DataFrame."""
        df_empty = pd.DataFrame()
        result = get_latest_signal(df_empty, sample_model)
        assert result == "NEUTRAL"

    def test_get_signal_missing_ohlc_columns(self, sample_model):
        """Test signal generation with missing OHLC columns."""
        df = pd.DataFrame({"open": [100, 101, 102]})
        result = get_latest_signal(df, sample_model)
        assert result == "NEUTRAL"

    def test_get_signal_insufficient_data(self, sample_model):
        """Test signal generation with insufficient data."""
        df = pd.DataFrame({"open": [100], "high": [101], "low": [99], "close": [100]})
        result = get_latest_signal(df, sample_model, look_back=WINDOW_SIZE_LSTM)
        assert result == "NEUTRAL"

    def test_get_signal_model_without_parameters(self):
        """Test signal generation with model that has no parameters."""

        class EmptyModel(nn.Module):
            def forward(self, x):
                return torch.randn(x.shape[0], 3)

        model = EmptyModel()
        df = pd.DataFrame({"open": [100] * 100, "high": [101] * 100, "low": [99] * 100, "close": [100] * 100})

        # Should handle gracefully
        result = get_latest_signal(df, model)
        assert result in ["BUY", "SELL", "NEUTRAL"]

    def test_get_signal_with_scaler(self, sample_model, sample_dataframe, sample_scaler):
        """Test signal generation with scaler."""
        with patch("modules.lstm.models.model_utils.generate_indicator_features") as mock_features:
            mock_features.return_value = sample_dataframe.copy()
            result = get_latest_signal(sample_dataframe, sample_model, sample_scaler)
            assert result in ["BUY", "SELL", "NEUTRAL"]

    def test_get_signal_scaler_transform_failure(self, sample_model, sample_dataframe):
        """Test signal generation when scaler transform fails."""
        scaler = Mock()
        scaler.transform.side_effect = Exception("Transform failed")

        with patch("modules.lstm.models.model_utils.generate_indicator_features") as mock_features:
            mock_features.return_value = sample_dataframe.copy()
            result = get_latest_signal(sample_dataframe, sample_model, scaler)
            assert result == "NEUTRAL"

    def test_get_signal_nan_in_features(self, sample_model, sample_dataframe):
        """Test signal generation with NaN values in features."""
        with patch("modules.lstm.models.model_utils.generate_indicator_features") as mock_features:
            df_with_nan = sample_dataframe.copy()
            df_with_nan.iloc[0, 0] = np.nan
            mock_features.return_value = df_with_nan

            scaler = Mock()
            scaler.transform.return_value = np.array([[np.nan] * 10] * len(df_with_nan))

            result = get_latest_signal(sample_dataframe, sample_model, scaler)
            assert result == "NEUTRAL"

    def test_get_signal_high_confidence_buy(self, sample_model):
        """Test signal generation with high confidence BUY."""
        df = pd.DataFrame({"open": [100] * 100, "high": [101] * 100, "low": [99] * 100, "close": [100] * 100})
        # Create features with enough columns to match model input_size
        feature_cols = ["open", "high", "low", "close", "SMA_20", "SMA_50", "SMA_200", "RSI_14", "ATR_14", "OBV"]
        df_features = pd.DataFrame({col: [1.0] * 100 for col in feature_cols})

        # Mock model to return high confidence BUY
        with patch.object(sample_model, "forward") as mock_forward:
            mock_output = torch.tensor([[0.1, 0.2, 0.9]])  # High confidence for class 2 (BUY)
            mock_forward.return_value = mock_output

            with patch("modules.lstm.models.model_utils.generate_indicator_features") as mock_features:
                mock_features.return_value = df_features

                scaler = Mock()
                scaler.transform.return_value = np.random.randn(100, 10)

                with patch("modules.lstm.models.model_utils.CONFIDENCE_THRESHOLD", 0.5):
                    result = get_latest_signal(df, sample_model, scaler)
                    assert result == "BUY"

    def test_get_signal_high_confidence_sell(self, sample_model):
        """Test signal generation with high confidence SELL."""
        df = pd.DataFrame({"open": [100] * 100, "high": [101] * 100, "low": [99] * 100, "close": [100] * 100})
        # Create features with enough columns to match model input_size
        feature_cols = ["open", "high", "low", "close", "SMA_20", "SMA_50", "SMA_200", "RSI_14", "ATR_14", "OBV"]
        df_features = pd.DataFrame({col: [1.0] * 100 for col in feature_cols})

        # Mock model to return high confidence SELL
        with patch.object(sample_model, "forward") as mock_forward:
            mock_output = torch.tensor([[0.9, 0.05, 0.05]])  # High confidence for class 0 (SELL)
            mock_forward.return_value = mock_output

            with patch("modules.lstm.models.model_utils.generate_indicator_features") as mock_features:
                mock_features.return_value = df_features

                scaler = Mock()
                scaler.transform.return_value = np.random.randn(100, 10)

                with patch("modules.lstm.models.model_utils.CONFIDENCE_THRESHOLD", 0.5):
                    result = get_latest_signal(df, sample_model, scaler)
                    assert result == "SELL"

    def test_get_signal_low_confidence(self, sample_model):
        """Test signal generation with low confidence."""
        df = pd.DataFrame({"open": [100] * 100, "high": [101] * 100, "low": [99] * 100, "close": [100] * 100})

        # Mock model to return low confidence
        with patch.object(sample_model, "forward") as mock_forward:
            mock_output = torch.tensor([[0.4, 0.3, 0.3]])  # Low confidence
            mock_forward.return_value = mock_output

            with patch("modules.lstm.models.model_utils.generate_indicator_features") as mock_features:
                mock_features.return_value = df.copy()

                scaler = Mock()
                scaler.transform.return_value = np.random.randn(100, 10)

                with patch("modules.lstm.models.model_utils.CONFIDENCE_THRESHOLD", 0.5):
                    result = get_latest_signal(df, sample_model, scaler)
                    assert result == "NEUTRAL"

    def test_get_signal_model_forward_exception(self, sample_model):
        """Test signal generation when model forward fails."""
        df = pd.DataFrame({"open": [100] * 100, "high": [101] * 100, "low": [99] * 100, "close": [100] * 100})

        with patch.object(sample_model, "forward") as mock_forward:
            mock_forward.side_effect = Exception("Model error")

            with patch("modules.lstm.models.model_utils.generate_indicator_features") as mock_features:
                mock_features.return_value = df.copy()

                scaler = Mock()
                scaler.transform.return_value = np.random.randn(100, 10)

                result = get_latest_signal(df, sample_model, scaler)
                assert result == "NEUTRAL"

    def test_get_signal_feature_count_mismatch(self, sample_model):
        """Test signal generation with feature count mismatch."""
        df = pd.DataFrame({"open": [100] * 100, "high": [101] * 100, "low": [99] * 100, "close": [100] * 100})

        # Mock model with different input size
        model_with_mismatch = create_cnn_lstm_attention_model(input_size=20, use_attention=False, use_cnn=False)

        with patch("modules.lstm.models.model_utils.generate_indicator_features") as mock_features:
            mock_features.return_value = df.copy()

            scaler = Mock()
            scaler.transform.return_value = np.random.randn(100, 10)  # 10 features but model expects 20

            result = get_latest_signal(df, model_with_mismatch, scaler)
            # Should return NEUTRAL due to mismatch
            assert result == "NEUTRAL"

    def test_get_latest_lstm_attention_signal_backward_compat(self, sample_model, sample_dataframe):
        """Test backward compatibility alias."""
        with patch("modules.lstm.models.model_utils.get_latest_signal") as mock_get:
            mock_get.return_value = "BUY"
            result = get_latest_lstm_attention_signal(sample_dataframe, sample_model)
            assert result == "BUY"
            mock_get.assert_called_once_with(sample_dataframe, sample_model, None)
