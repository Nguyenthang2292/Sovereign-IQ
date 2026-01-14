import numpy as np
import pandas as pd
import pytest
import torch

from config.lstm import GPU_MODEL_CONFIG
from modules.lstm.core.threshold_optimizer import GridSearchThresholdOptimizer
from modules.lstm.models.unified_trainer import LSTMTrainer

"""
Tests for unified LSTM trainer.
"""


class TestLSTMTrainer:
    """Test suite for LSTMTrainer class."""

    @pytest.fixture
    def sample_dataframe(self):
        """Create sample DataFrame for testing."""
        np.random.seed(42)
        n_samples = 200
        df = pd.DataFrame(
            {
                "open": np.random.randn(n_samples) * 100 + 50000,
                "high": np.random.randn(n_samples) * 100 + 50100,
                "low": np.random.randn(n_samples) * 100 + 49900,
                "close": np.random.randn(n_samples) * 100 + 50000,
                "volume": np.random.randn(n_samples) * 1000 + 10000,
            }
        )
        yield df
        del df

    def test_init_lstm_only(self):
        """Test initialization with LSTM only."""
        trainer = LSTMTrainer(use_cnn=False, use_attention=False)
        assert trainer.use_cnn is False
        assert trainer.use_attention is False
        assert trainer.early_stopping_patience == 10

    def test_init_lstm_attention(self):
        """Test initialization with LSTM-Attention."""
        trainer = LSTMTrainer(use_cnn=False, use_attention=True)
        assert trainer.use_cnn is False
        assert trainer.use_attention is True
        assert trainer.attention_heads == GPU_MODEL_CONFIG["nhead"]

    def test_init_cnn_lstm(self):
        """Test initialization with CNN-LSTM."""
        trainer = LSTMTrainer(use_cnn=True, use_attention=False)
        assert trainer.use_cnn is True
        assert trainer.use_attention is False

    def test_init_cnn_lstm_attention(self):
        """Test initialization with CNN-LSTM-Attention."""
        trainer = LSTMTrainer(use_cnn=True, use_attention=True)
        assert trainer.use_cnn is True
        assert trainer.use_attention is True

    def test_init_custom_parameters(self):
        """Test initialization with custom parameters."""
        trainer = LSTMTrainer(
            use_cnn=True,
            use_attention=True,
            look_back=30,
            output_mode="regression",
            attention_heads=4,
            use_early_stopping=False,
            early_stopping_patience=20,
            cnn_features=32,
            lstm_hidden=16,
            dropout=0.5,
        )
        assert trainer.look_back == 30
        assert trainer.output_mode == "regression"
        assert trainer.attention_heads == 4
        assert trainer.use_early_stopping is False
        assert trainer.early_stopping_patience == 20
        assert trainer.cnn_features == 32
        assert trainer.lstm_hidden == 16
        assert trainer.dropout == 0.5

    def test_get_model_type_name_lstm(self):
        """Test model type name for LSTM."""
        trainer = LSTMTrainer(use_cnn=False, use_attention=False)
        assert trainer._get_model_type_name() == "LSTM"

    def test_get_model_type_name_lstm_attention(self):
        """Test model type name for LSTM-Attention."""
        trainer = LSTMTrainer(use_cnn=False, use_attention=True)
        assert trainer._get_model_type_name() == "LSTM-Attention"

    def test_get_model_type_name_cnn_lstm(self):
        """Test model type name for CNN-LSTM."""
        trainer = LSTMTrainer(use_cnn=True, use_attention=False)
        assert trainer._get_model_type_name() == "CNN-LSTM"

    def test_get_model_type_name_cnn_lstm_attention(self):
        """Test model type name for CNN-LSTM-Attention."""
        trainer = LSTMTrainer(use_cnn=True, use_attention=True)
        assert trainer._get_model_type_name() == "CNN-LSTM-Attention"

    def test_create_model_lstm(self):
        """Test creating LSTM model."""
        trainer = LSTMTrainer(use_cnn=False, use_attention=False)
        model = trainer.create_model(input_size=10)
        assert model is not None
        assert hasattr(model, "forward")

    def test_create_model_lstm_attention(self):
        """Test creating LSTM-Attention model."""
        trainer = LSTMTrainer(use_cnn=False, use_attention=True)
        model = trainer.create_model(input_size=10)
        assert model is not None
        assert hasattr(model, "forward")

    def test_create_model_cnn_lstm(self):
        """Test creating CNN-LSTM model."""
        trainer = LSTMTrainer(use_cnn=True, use_attention=False)
        model = trainer.create_model(input_size=10)
        assert model is not None
        assert hasattr(model, "forward")

    def test_create_model_cnn_lstm_attention(self):
        """Test creating CNN-LSTM-Attention model."""
        trainer = LSTMTrainer(use_cnn=True, use_attention=True)
        model = trainer.create_model(input_size=10)
        assert model is not None
        assert hasattr(model, "forward")

    def test_create_model_classification(self):
        """Test creating model in classification mode."""
        trainer = LSTMTrainer(output_mode="classification")
        model = trainer.create_model(input_size=10)
        assert model is not None

    def test_create_model_regression(self):
        """Test creating model in regression mode."""
        trainer = LSTMTrainer(output_mode="regression")
        model = trainer.create_model(input_size=10)
        assert model is not None

    def test_create_model_sets_instance_model(self):
        """Test that create_model sets self.model."""
        trainer = LSTMTrainer()
        assert trainer.model is None
        model = trainer.create_model(input_size=10)
        assert trainer.model is not None
        assert trainer.model == model

    def test_validate_and_preprocess_data_lstm(self, sample_dataframe):
        """Test data validation for LSTM."""
        trainer = LSTMTrainer(use_cnn=False, use_attention=False)
        X, y = trainer._validate_and_preprocess_data(sample_dataframe)
        assert X is not None
        assert y is not None
        assert len(X) > 0
        assert len(y) > 0

    def test_validate_and_preprocess_data_cnn(self, sample_dataframe):
        """Test data validation for CNN-LSTM."""
        trainer = LSTMTrainer(use_cnn=True, use_attention=False)
        X, y = trainer._validate_and_preprocess_data(sample_dataframe)
        assert X is not None
        assert y is not None

    def test_validate_and_preprocess_data_insufficient_for_cnn(self):
        """Test data validation with insufficient data for CNN."""
        trainer = LSTMTrainer(use_cnn=True, use_attention=False)
        # Create minimal data that's insufficient for CNN
        df = pd.DataFrame(
            {
                "open": [100] * 50,
                "high": [101] * 50,
                "low": [99] * 50,
                "close": [100] * 50,
            }
        )

        with pytest.raises(ValueError, match="Insufficient data.*CNN"):
            trainer._validate_and_preprocess_data(df)

    def test_train_empty_dataframe(self):
        """Test training with empty DataFrame."""
        trainer = LSTMTrainer()
        df_empty = pd.DataFrame()

        model, optimizer, path = trainer.train(df_empty, epochs=1, save_model=False)
        assert model is None
        assert isinstance(optimizer, GridSearchThresholdOptimizer)
        assert path is None

    def test_train_insufficient_data(self):
        """Test training with insufficient data."""
        trainer = LSTMTrainer()
        df = pd.DataFrame(
            {
                "open": [100, 101],
                "high": [101, 102],
                "low": [99, 100],
                "close": [100, 101],
            }
        )

        model, optimizer, path = trainer.train(df, epochs=1, save_model=False)

        # Should handle gracefully and return None for insufficient data
        assert model is None
        assert isinstance(optimizer, GridSearchThresholdOptimizer)
        assert path is None

    def test_train_with_custom_epochs(self, sample_dataframe):
        """Test training with custom number of epochs."""
        trainer = LSTMTrainer(use_cnn=False, use_attention=False)

        # Use very small epochs for quick test
        model, optimizer, path = trainer.train(sample_dataframe, epochs=2, save_model=False)
        # Should successfully train with sufficient data
        assert model is not None, "Training should succeed with sufficient data"
        assert optimizer is not None, "Optimizer should be created"
        assert hasattr(model, "forward"), "Model should have forward method"

    def test_train_without_early_stopping(self, sample_dataframe):
        """Test training without early stopping."""
        trainer = LSTMTrainer(use_early_stopping=False, use_cnn=False, use_attention=False)

        model, optimizer, path = trainer.train(sample_dataframe, epochs=2, save_model=False)
        # Should successfully train with sufficient data
        assert model is not None, "Training should succeed with sufficient data"
        assert optimizer is not None, "Optimizer should be created"
        assert hasattr(model, "forward"), "Model should have forward method"

    def test_train_with_custom_patience(self, sample_dataframe):
        """Test training with custom patience."""
        trainer = LSTMTrainer(early_stopping_patience=5, use_cnn=False, use_attention=False)

        assert trainer.early_stopping_patience == 5

        model, optimizer, path = trainer.train(sample_dataframe, epochs=2, save_model=False)
        # Should successfully train with sufficient data
        assert model is not None, "Training should succeed with sufficient data"
        assert optimizer is not None, "Optimizer should be created"
        assert hasattr(model, "forward"), "Model should have forward method"

    def test_train_regression_mode(self, sample_dataframe):
        """Test training in regression mode."""
        trainer = LSTMTrainer(output_mode="regression", use_cnn=False, use_attention=False)

        model, optimizer, path = trainer.train(sample_dataframe, epochs=2, save_model=False)

        # Should successfully train with sufficient data
        assert model is not None, "Training should succeed with sufficient data"
        assert optimizer is not None, "Optimizer should be created"
        assert hasattr(model, "forward"), "Model should have forward method"

        # Verify model is in regression mode (output size should be 1 for regression)
        test_input = torch.randn(1, trainer.look_back, 5)  # batch=1, seq_len=look_back, features=5
        test_output = model(test_input)
        assert test_output.shape[-1] == 1, "Regression mode should output size 1"
