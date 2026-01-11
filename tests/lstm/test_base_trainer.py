
import numpy as np
import pandas as pd
import pytest
import torch

from config.lstm import WINDOW_SIZE_LSTM
from modules.lstm.models.trainer.base_trainer import BaseLSTMTrainer
from modules.lstm.models.trainer.base_trainer import BaseLSTMTrainer

"""
Tests for base LSTM trainer.
"""




class TestBaseLSTMTrainer:
    """Test suite for BaseLSTMTrainer class."""

    def test_init_default(self):
        """Test initialization with default parameters."""
        trainer = BaseLSTMTrainer()
        assert trainer.look_back == WINDOW_SIZE_LSTM
        assert trainer.output_mode == "classification"
        assert trainer.use_early_stopping is True
        assert trainer.device is not None
        assert trainer.model is None
        assert trainer.scaler is None

    def test_init_custom_parameters(self):
        """Test initialization with custom parameters."""
        trainer = BaseLSTMTrainer(look_back=30, output_mode="regression", use_early_stopping=False)
        assert trainer.look_back == 30
        assert trainer.output_mode == "regression"
        assert trainer.use_early_stopping is False

    def test_setup_device(self):
        """Test device setup."""
        trainer = BaseLSTMTrainer()
        assert trainer.device is not None
        assert trainer.device.type in ["cpu", "cuda"]
        assert isinstance(trainer.gpu_available, bool)
        assert isinstance(trainer.use_mixed_precision, bool)

    def test_validate_and_preprocess_data_valid(self):
        """Test data validation with valid data."""
        from unittest.mock import patch

        trainer = BaseLSTMTrainer()
        # Need at least look_back + 50 = 60 + 50 = 110 rows
        n_samples = 200
        np.random.seed(42)
        base_price = 50000
        prices = [base_price]
        for i in range(1, n_samples):
            # Generate realistic price movements
            change = np.random.randn() * 100
            new_price = prices[-1] + change
            prices.append(max(new_price, base_price * 0.9))  # Prevent negative prices

        df = pd.DataFrame(
            {
                "open": prices,
                "high": [p * 1.01 for p in prices],
                "low": [p * 0.99 for p in prices],
                "close": prices,
                "volume": np.random.randn(n_samples) * 1000 + 10000,
            }
        )

        # Mock preprocessing to return valid data for testing
        # This avoids dependency on actual feature generation which may fail with synthetic data
        mock_X = np.random.randn(50, trainer.look_back, 10)
        mock_y = np.random.choice([-1, 0, 1], 50)
        mock_scaler = None
        mock_feature_names = [f"feature_{i}" for i in range(10)]

        with patch("modules.lstm.models.trainer.base_trainer.preprocess_cnn_lstm_data") as mock_preprocess:
            mock_preprocess.return_value = (mock_X, mock_y, mock_scaler, mock_feature_names)

            X, y = trainer._validate_and_preprocess_data(df)
            assert X is not None
            assert y is not None
            assert len(X) > 0
            assert len(y) > 0
            assert len(X) == len(y)
            assert X.shape[0] == y.shape[0]

    def test_validate_and_preprocess_data_empty(self):
        """Test data validation with empty DataFrame."""
        trainer = BaseLSTMTrainer()
        df = pd.DataFrame()

        with pytest.raises((ValueError, KeyError)):
            trainer._validate_and_preprocess_data(df)

    def test_validate_and_preprocess_data_missing_close(self):
        """Test data validation with missing close column."""
        trainer = BaseLSTMTrainer()
        df = pd.DataFrame({"open": [100, 101, 102], "high": [105, 106, 107], "low": [95, 96, 97]})

        with pytest.raises((ValueError, KeyError)):
            trainer._validate_and_preprocess_data(df)

    def test_validate_and_preprocess_data_insufficient_raises_error(self):
        """Test that insufficient data raises ValueError."""
        trainer = BaseLSTMTrainer()
        # Very small dataset (2 rows) - insufficient for look_back + 50 requirement
        df = pd.DataFrame(
            {
                "open": [100, 101],
                "high": [101, 102],
                "low": [99, 100],
                "close": [100, 101],
            }
        )

        # Should raise ValueError when data is clearly insufficient
        min_required = trainer.look_back + 50
        with pytest.raises(ValueError, match=f"Insufficient data.*{min_required}"):
            trainer._validate_and_preprocess_data(df)

    def test_validate_and_preprocess_data_minimal_but_valid(self):
        """Test data validation with minimal but sufficient data."""
        from unittest.mock import patch

        trainer = BaseLSTMTrainer()
        # Create dataset that meets minimum requirement (look_back + 50)
        min_required = trainer.look_back + 50
        n_samples = min_required + 10  # Just above minimum

        np.random.seed(42)
        base_price = 50000
        prices = [base_price]
        for i in range(1, n_samples):
            change = np.random.randn() * 100
            new_price = prices[-1] + change
            prices.append(max(new_price, base_price * 0.9))

        df = pd.DataFrame(
            {
                "open": prices,
                "high": [p * 1.01 for p in prices],
                "low": [p * 0.99 for p in prices],
                "close": prices,
                "volume": np.random.randn(n_samples) * 1000 + 10000,
            }
        )

        # Mock preprocessing to return valid data
        mock_X = np.random.randn(10, trainer.look_back, 10)
        mock_y = np.random.choice([-1, 0, 1], 10)
        mock_scaler = None
        mock_feature_names = [f"feature_{i}" for i in range(10)]

        with patch("modules.lstm.models.trainer.base_trainer.preprocess_cnn_lstm_data") as mock_preprocess:
            mock_preprocess.return_value = (mock_X, mock_y, mock_scaler, mock_feature_names)

            X, y = trainer._validate_and_preprocess_data(df)
            assert X is not None
            assert y is not None
            assert len(X) > 0
            assert len(y) > 0
            assert len(X) == len(y)

    def test_prepare_tensors(self):
        """Test tensor preparation."""
        trainer = BaseLSTMTrainer()

        # Create dummy sequences and targets
        # Labels must be -1, 0, 1 for classification mode
        n_samples = 100
        n_features = 10
        X = np.random.randn(n_samples, WINDOW_SIZE_LSTM, n_features)
        y = np.random.choice([-1, 0, 1], n_samples)  # Use -1, 0, 1 instead of 0, 1, 2

        result = trainer._prepare_tensors(X, y)
        assert len(result) == 8  # Should return 8 values

        X_train, X_val, X_test, y_train, y_val, y_test, num_classes, test_indices = result

        assert X_train is not None
        assert X_val is not None
        assert X_test is not None
        assert isinstance(X_train, torch.Tensor)
        assert isinstance(X_val, torch.Tensor)
        assert isinstance(X_test, torch.Tensor)
        assert isinstance(y_train, torch.Tensor)
        assert isinstance(y_val, torch.Tensor)
        assert isinstance(y_test, torch.Tensor)
        assert num_classes == 3  # For labels -1, 0, 1
        assert test_indices is not None
        assert X_train.shape[0] + X_val.shape[0] + X_test.shape[0] == n_samples

    def test_get_batch_size(self):
        """Test batch size calculation."""
        trainer = BaseLSTMTrainer()
        input_size = 10

        batch_size = trainer._get_batch_size(input_size)
        assert batch_size > 0
        assert isinstance(batch_size, int)

    def test_get_batch_size_different_input_sizes(self):
        """Test batch size calculation with different input sizes."""
        trainer = BaseLSTMTrainer()

        for input_size in [5, 10, 20, 50, 100]:
            batch_size = trainer._get_batch_size(input_size)
            assert batch_size > 0

    def test_state_variables_initialization(self):
        """Test that state variables are properly initialized."""
        trainer = BaseLSTMTrainer()

        assert trainer.device is not None
        assert trainer.model is None
        assert trainer.scaler is None
        assert trainer.feature_names is None
        assert trainer.threshold_optimizer is not None
        assert trainer.training_history is None
        assert trainer.test_indices is None

    def test_output_mode_classification(self):
        """Test trainer with classification mode."""
        trainer = BaseLSTMTrainer(output_mode="classification")
        assert trainer.output_mode == "classification"

    def test_output_mode_regression(self):
        """Test trainer with regression mode."""
        trainer = BaseLSTMTrainer(output_mode="regression")
        assert trainer.output_mode == "regression"

    def test_early_stopping_enabled(self):
        """Test trainer with early stopping enabled."""
        trainer = BaseLSTMTrainer(use_early_stopping=True)
        assert trainer.use_early_stopping is True

    def test_early_stopping_disabled(self):
        """Test trainer with early stopping disabled."""
        trainer = BaseLSTMTrainer(use_early_stopping=False)
        assert trainer.use_early_stopping is False

    def test_look_back_custom(self):
        """Test trainer with custom look_back."""
        custom_look_back = 30
        trainer = BaseLSTMTrainer(look_back=custom_look_back)
        assert trainer.look_back == custom_look_back

    def test_threshold_optimizer_initialized(self):
        """Test that threshold optimizer is initialized."""
        trainer = BaseLSTMTrainer()
        assert trainer.threshold_optimizer is not None
        assert hasattr(trainer.threshold_optimizer, "optimize_regression_threshold")
        assert hasattr(trainer.threshold_optimizer, "optimize_classification_threshold")

    def test_hyperparameters_defaults(self):
        """Test that hyperparameters have correct default values."""
        trainer = BaseLSTMTrainer()
        assert trainer.learning_rate == 0.001
        assert trainer.weight_decay == 0.01
        assert trainer.adam_eps == 1e-8
        assert trainer.scheduler_T_0 == 10
        assert trainer.scheduler_T_mult == 2
        assert trainer.scheduler_eta_min == 1e-6

    def test_hyperparameters_custom(self):
        """Test that hyperparameters can be customized."""
        trainer = BaseLSTMTrainer(
            learning_rate=0.0005,
            weight_decay=0.02,
            adam_eps=1e-7,
            scheduler_T_0=20,
            scheduler_T_mult=3,
            scheduler_eta_min=1e-7,
        )
        assert trainer.learning_rate == 0.0005
        assert trainer.weight_decay == 0.02
        assert trainer.adam_eps == 1e-7
        assert trainer.scheduler_T_0 == 20
        assert trainer.scheduler_T_mult == 3
        assert trainer.scheduler_eta_min == 1e-7
