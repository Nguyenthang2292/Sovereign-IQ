"""Tests for model ensemble functionality."""

import sys
from pathlib import Path

# Add project root to Python path for imports
ROOT = Path(__file__).resolve().parents[2]
ROOT_STR = str(ROOT)
if ROOT_STR not in sys.path:
    sys.path.insert(0, ROOT_STR)

from unittest.mock import MagicMock, patch

import numpy as np
import pandas as pd
import pytest
from sklearn.ensemble import RandomForestClassifier, StackingClassifier, VotingClassifier

from modules.random_forest.core.ensemble import (
    LSTMWrapper,
    create_ensemble,
    load_lstm_model_wrapped,
    load_xgboost_model,
)


class TestLSTMWrapper:
    """Test LSTM wrapper for sklearn compatibility."""

    def test_lstm_wrapper_init(self):
        """Test LSTMWrapper initialization."""
        mock_model = MagicMock()
        wrapper = LSTMWrapper(mock_model, scaler=None, look_back=60)

        assert wrapper.lstm_model == mock_model
        assert wrapper.scaler is None
        assert wrapper.look_back == 60

    def test_lstm_wrapper_fit(self):
        """Test LSTMWrapper fit method."""
        mock_model = MagicMock()
        wrapper = LSTMWrapper(mock_model, look_back=60)

        X = pd.DataFrame(np.random.randn(100, 10))
        y = pd.Series([0, 1] * 50)

        result = wrapper.fit(X, y)

        assert result is wrapper  # Should return self
        assert wrapper._feature_size == 10

    def test_lstm_wrapper_predict_proba_fallback(self):
        """Test LSTMWrapper predict_proba fallback behavior when torch fails."""
        mock_model = MagicMock()
        wrapper = LSTMWrapper(mock_model, look_back=60)
        wrapper._feature_size = 10

        # Force exception to test fallback - patch import to raise
        X = pd.DataFrame(np.random.randn(5, 10))
        
        # Patch the import inside the method to raise ImportError
        with patch("builtins.__import__", side_effect=ImportError("No module named 'torch'")):
            # Should return uniform probabilities as fallback
            proba = wrapper.predict_proba(X)

            assert proba.shape[0] == 5
            assert proba.shape[1] == 3  # 3 classes (UP, NEUTRAL, DOWN)
            # Fallback should return uniform probabilities
            assert np.allclose(proba, 1.0 / 3.0, atol=0.01)

    def test_lstm_wrapper_predict(self):
        """Test LSTMWrapper predict method."""
        mock_model = MagicMock()
        wrapper = LSTMWrapper(mock_model, look_back=60)

        # Mock predict_proba
        mock_proba = np.array([[0.1, 0.2, 0.7], [0.8, 0.1, 0.1], [0.3, 0.4, 0.3]])
        wrapper.predict_proba = MagicMock(return_value=mock_proba)

        X = pd.DataFrame(np.random.randn(3, 10))
        predictions = wrapper.predict(X)

        assert len(predictions) == 3
        assert all(pred in [0, 1, 2] for pred in predictions)


class TestModelLoading:
    """Test model loading utilities."""

    def test_load_xgboost_model_success(self):
        """Test successful XGBoost model loading."""
        import joblib

        mock_model = MagicMock()
        with patch.object(joblib, "load", return_value=mock_model) as mock_joblib_load:
            from pathlib import Path

            mock_path_obj = MagicMock(spec=Path)
            mock_path_obj.exists.return_value = True

            model = load_xgboost_model(mock_path_obj)

            assert model == mock_model
            mock_joblib_load.assert_called_once()

    def test_load_lstm_model_wrapped_success(self):
        """Test successful LSTM model loading and wrapping."""
        from modules.lstm.models import model_utils

        mock_model = MagicMock()
        mock_scaler = MagicMock()

        with patch.object(model_utils, "load_model_and_scaler", return_value=(mock_model, mock_scaler, 60)):
            wrapper = load_lstm_model_wrapped()

            assert wrapper is not None
            assert isinstance(wrapper, LSTMWrapper)
            assert wrapper.lstm_model == mock_model
            assert wrapper.scaler == mock_scaler
            assert wrapper.look_back == 60

    def test_load_lstm_model_wrapped_failure(self):
        """Test LSTM model loading failure."""
        from modules.lstm.models import model_utils

        with patch.object(model_utils, "load_model_and_scaler", return_value=(None, None, None)):
            wrapper = load_lstm_model_wrapped()

            assert wrapper is None


class TestCreateEnsemble:
    """Test ensemble creation."""

    def test_create_ensemble_voting_only_rf(self):
        """Test ensemble with only RandomForest (should return RF, not ensemble)."""
        rf_model = RandomForestClassifier(n_estimators=10, random_state=42)

        with patch("modules.random_forest.core.ensemble.load_xgboost_model", return_value=None):
            with patch("modules.random_forest.core.ensemble.load_lstm_model_wrapped", return_value=None):
                ensemble = create_ensemble(
                    rf_model, include_xgboost=False, include_lstm=False, method="voting"
                )

                # Should return RF model directly (not an ensemble)
                assert ensemble == rf_model

    @patch("modules.random_forest.core.ensemble.load_xgboost_model")
    def test_create_ensemble_voting_with_xgboost(self, mock_load_xgb):
        """Test VotingClassifier ensemble with XGBoost."""
        rf_model = RandomForestClassifier(n_estimators=10, random_state=42)
        mock_xgb = MagicMock()
        mock_load_xgb.return_value = mock_xgb

        with patch("modules.random_forest.core.ensemble.load_lstm_model_wrapped", return_value=None):
            ensemble = create_ensemble(
                rf_model, include_xgboost=True, include_lstm=False, method="voting", voting="soft"
            )

            assert isinstance(ensemble, VotingClassifier)
            assert len(ensemble.estimators) == 2
            assert ensemble.voting == "soft"

    @patch("modules.random_forest.core.ensemble.load_xgboost_model")
    @patch("modules.random_forest.core.ensemble.load_lstm_model_wrapped")
    def test_create_ensemble_voting_all_models(self, mock_load_lstm, mock_load_xgb):
        """Test VotingClassifier with all models (RF, XGBoost, LSTM)."""
        rf_model = RandomForestClassifier(n_estimators=10, random_state=42)
        mock_xgb = MagicMock()
        mock_lstm_wrapper = MagicMock(spec=LSTMWrapper)
        mock_load_xgb.return_value = mock_xgb
        mock_load_lstm.return_value = mock_lstm_wrapper

        ensemble = create_ensemble(
            rf_model, include_xgboost=True, include_lstm=True, method="voting", voting="hard"
        )

        assert isinstance(ensemble, VotingClassifier)
        assert len(ensemble.estimators) == 3
        assert ensemble.voting == "hard"

    @patch("modules.random_forest.core.ensemble.load_xgboost_model")
    def test_create_ensemble_stacking(self, mock_load_xgb):
        """Test StackingClassifier ensemble."""
        rf_model = RandomForestClassifier(n_estimators=10, random_state=42)
        mock_xgb = MagicMock()
        mock_load_xgb.return_value = mock_xgb

        with patch("modules.random_forest.core.ensemble.load_lstm_model_wrapped", return_value=None):
            ensemble = create_ensemble(
                rf_model,
                include_xgboost=True,
                include_lstm=False,
                method="stacking",
                final_estimator="RandomForest",
            )

            assert isinstance(ensemble, StackingClassifier)
            assert len(ensemble.estimators) == 2

    def test_create_ensemble_unknown_method(self):
        """Test error handling for unknown ensemble method."""
        rf_model = RandomForestClassifier(n_estimators=10, random_state=42)

        with pytest.raises(ValueError, match="Unknown ensemble method"):
            create_ensemble(rf_model, method="unknown_method")

    @patch("modules.random_forest.core.ensemble.load_xgboost_model")
    def test_create_ensemble_xgboost_not_available(self, mock_load_xgb):
        """Test ensemble creation when XGBoost is not available."""
        rf_model = RandomForestClassifier(n_estimators=10, random_state=42)
        mock_load_xgb.return_value = None

        with patch("modules.random_forest.core.ensemble.load_lstm_model_wrapped", return_value=None):
            ensemble = create_ensemble(
                rf_model, include_xgboost=True, include_lstm=False, method="voting"
            )

            # Should fall back to RF only
            assert ensemble == rf_model

    @patch("modules.random_forest.core.ensemble.load_lstm_model_wrapped")
    def test_create_ensemble_lstm_not_available(self, mock_load_lstm):
        """Test ensemble creation when LSTM is not available."""
        rf_model = RandomForestClassifier(n_estimators=10, random_state=42)
        mock_load_lstm.return_value = None

        with patch("modules.random_forest.core.ensemble.load_xgboost_model", return_value=None):
            ensemble = create_ensemble(
                rf_model, include_xgboost=False, include_lstm=True, method="voting"
            )

            # Should fall back to RF only
            assert ensemble == rf_model


class TestEnsembleIntegration:
    """Test ensemble integration with training."""

    @patch("modules.random_forest.core.ensemble.create_ensemble")
    def test_train_with_ensemble_enabled(self, mock_create_ensemble):
        """Test training with ensemble enabled."""
        from modules.random_forest.core.model import train_random_forest_model

        # Create sample data
        np.random.seed(42)
        df = pd.DataFrame(
            {
                "open": np.random.randn(200) + 100,
                "high": np.random.randn(200) + 105,
                "low": np.random.randn(200) + 95,
                "close": np.random.randn(200) + 100,
                "volume": np.random.randn(200) * 1000 + 10000,
            }
        )

        mock_ensemble = MagicMock(spec=VotingClassifier)
        mock_create_ensemble.return_value = mock_ensemble

        with patch("config.random_forest.RANDOM_FOREST_USE_ENSEMBLE", True):
            with patch("modules.random_forest.core.model.prepare_training_data") as mock_prepare:
                # Mock prepared data
                features = pd.DataFrame(np.random.randn(150, 10))
                target = pd.Series([0, 1, -1] * 50)
                mock_prepare.return_value = (features, target)

                # Mock other dependencies
                with patch("modules.random_forest.core.model._time_series_split_with_gap") as mock_split:
                    mock_split.return_value = (
                        features.iloc[:120],
                        features.iloc[120:],
                        target.iloc[:120],
                        target.iloc[120:],
                    )

                    with patch("modules.random_forest.utils.training.apply_sampling") as mock_sampling:
                        mock_sampling.return_value = (features.iloc[:120], target.iloc[:120], False)

                        with patch("modules.random_forest.utils.training.create_model_and_weights") as mock_create:
                            mock_rf = MagicMock(spec=RandomForestClassifier)
                            mock_create.return_value = mock_rf

                            with patch("modules.random_forest.core.model.evaluate_model_with_confidence"):
                                with patch("modules.random_forest.core.model.joblib.dump"):
                                    result = train_random_forest_model(df, save_model=False)

                                    # Should have called create_ensemble
                                    mock_create_ensemble.assert_called_once()


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
