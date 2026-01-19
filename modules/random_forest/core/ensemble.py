"""Model Ensemble for Random Forest module.

This module provides ensemble functionality combining multiple models:
- RandomForestClassifier
- XGBoost (XGBClassifier)
- LSTM (PyTorch model wrapped for sklearn compatibility)

Supports both VotingClassifier and StackingClassifier from sklearn.
"""

from pathlib import Path
from typing import Any, List, Optional, Tuple, Union

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier, StackingClassifier, VotingClassifier
from sklearn.linear_model import LogisticRegression

from config.random_forest import (
    RANDOM_FOREST_ENSEMBLE_FINAL_ESTIMATOR,
    RANDOM_FOREST_ENSEMBLE_INCLUDE_LSTM,
    RANDOM_FOREST_ENSEMBLE_INCLUDE_XGBOOST,
    RANDOM_FOREST_ENSEMBLE_METHOD,
    RANDOM_FOREST_ENSEMBLE_VOTING,
)
from modules.common.ui.logging import log_error, log_info, log_progress, log_warn


class LSTMWrapper:
    """
    Wrapper for PyTorch LSTM model to make it compatible with sklearn ensemble.

    This wrapper adapts the LSTM model's interface to work with VotingClassifier
    and StackingClassifier by providing predict_proba and predict methods.
    """

    def __init__(self, lstm_model: Any, scaler: Optional[Any] = None, look_back: int = 60):
        """
        Initialize LSTM wrapper.

        Args:
            lstm_model: PyTorch LSTM model
            scaler: Optional scaler for preprocessing (MinMaxScaler)
            look_back: Sequence length for LSTM input
        """
        self.lstm_model = lstm_model
        self.scaler = scaler
        self.look_back = look_back
        self._feature_size = None

    def fit(self, X: pd.DataFrame, y: pd.Series) -> "LSTMWrapper":
        """
        Fit wrapper (LSTM model is pre-trained, so this is a no-op).

        Args:
            X: Feature DataFrame
            y: Target Series

        Returns:
            Self for method chaining
        """
        # LSTM is pre-trained, so we just store feature size
        if isinstance(X, pd.DataFrame):
            self._feature_size = X.shape[1]
        else:
            self._feature_size = X.shape[1] if len(X.shape) > 1 else 1
        return self

    def predict_proba(self, X: Union[pd.DataFrame, np.ndarray]) -> np.ndarray:
        """
        Predict class probabilities using LSTM model.

        Args:
            X: Feature array/DataFrame

        Returns:
            Array of shape (n_samples, n_classes) with class probabilities
        """
        try:
            import torch

            # Convert to numpy if needed
            if isinstance(X, pd.DataFrame):
                X_array = X.values
            else:
                X_array = np.asarray(X)

            # Reshape for LSTM: (batch, sequence, features)
            # If X is 2D, we need to create sequences
            if len(X_array.shape) == 2:
                # Create sequences from features
                # For simplicity, use last look_back rows as sequence
                if len(X_array) < self.look_back:
                    # Pad with first row if insufficient data
                    padding = np.tile(X_array[0:1], (self.look_back - len(X_array), 1))
                    X_array = np.vstack([padding, X_array])

                # Use sliding window to create sequences
                sequences = []
                for i in range(len(X_array) - self.look_back + 1):
                    sequences.append(X_array[i : i + self.look_back])
                X_sequences = np.array(sequences)

                # If we have fewer sequences than samples, use last sequence for all
                if len(X_sequences) < len(X_array):
                    last_sequence = X_sequences[-1:] if len(X_sequences) > 0 else X_array[-self.look_back :]
                    X_sequences = np.tile(last_sequence, (len(X_array), 1, 1))
            else:
                X_sequences = X_array

            # Apply scaler if available
            if self.scaler is not None:
                # Reshape for scaler: (batch * sequence, features)
                original_shape = X_sequences.shape
                X_flat = X_sequences.reshape(-1, original_shape[-1])
                X_scaled = self.scaler.transform(X_flat)
                X_sequences = X_scaled.reshape(original_shape)

            # Convert to torch tensor
            X_tensor = torch.FloatTensor(X_sequences)

            # Set model to eval mode
            self.lstm_model.eval()

            # Predict
            with torch.no_grad():
                outputs = self.lstm_model(X_tensor)

            # Handle different output formats
            if isinstance(outputs, dict):
                # Model returns dict with 'classification' key
                logits = outputs.get("classification", outputs.get("output"))
            elif isinstance(outputs, torch.Tensor):
                logits = outputs
            else:
                log_warn("Unexpected LSTM output format, using default probabilities")
                # Return uniform probabilities as fallback
                n_samples = len(X_array)
                n_classes = 3  # Default: UP, NEUTRAL, DOWN
                return np.ones((n_samples, n_classes)) / n_classes

            # Convert logits to probabilities
            if logits is not None:
                probs = torch.softmax(logits, dim=-1).numpy()
                # Ensure shape is (n_samples, n_classes)
                if len(probs.shape) > 2:
                    # Take last timestep if sequence output
                    probs = probs[:, -1, :]
                # Ensure we have probabilities for all samples
                if len(probs) < len(X_array):
                    # Repeat last probabilities
                    probs = np.vstack([probs, np.tile(probs[-1:], (len(X_array) - len(probs), 1))])
                return probs
            else:
                # Fallback: uniform probabilities
                n_samples = len(X_array)
                n_classes = 3
                return np.ones((n_samples, n_classes)) / n_classes

        except Exception as e:
            log_error(f"Error in LSTM prediction: {e}")
            # Fallback: return uniform probabilities
            n_samples = len(X_array) if "X_array" in locals() else len(X)
            n_classes = 3
            return np.ones((n_samples, n_classes)) / n_classes

    def predict(self, X: Union[pd.DataFrame, np.ndarray]) -> np.ndarray:
        """
        Predict class labels using LSTM model.

        Args:
            X: Feature array/DataFrame

        Returns:
            Array of predicted class labels
        """
        proba = self.predict_proba(X)
        return np.argmax(proba, axis=1)


def load_xgboost_model(model_path: Optional[Path] = None) -> Optional[Any]:
    """
    Load XGBoost model from file.

    Args:
        model_path: Path to XGBoost model file. If None, tries default path.

    Returns:
        Loaded XGBoost model or None if failed
    """
    try:
        import joblib

        if model_path is None:
            # Try default XGBoost model path
            from config.xgboost import MODELS_DIR

            model_files = list(MODELS_DIR.glob("*.joblib"))
            if not model_files:
                log_warn("No XGBoost model found in default directory")
                return None
            # Use latest model
            model_files.sort(key=lambda p: p.stat().st_mtime, reverse=True)
            model_path = model_files[0]

        if not model_path.exists():
            log_warn(f"XGBoost model not found at {model_path}")
            return None

        model = joblib.load(model_path)
        log_info(f"Loaded XGBoost model from {model_path}")
        return model

    except Exception as e:
        log_error(f"Error loading XGBoost model: {e}")
        return None


def load_lstm_model_wrapped(model_path: Optional[Path] = None) -> Optional[LSTMWrapper]:
    """
    Load LSTM model and wrap it for sklearn compatibility.

    Args:
        model_path: Path to LSTM model file. If None, uses default path.

    Returns:
        LSTMWrapper instance or None if failed
    """
    try:
        from modules.lstm.models.model_utils import load_model_and_scaler

        model, scaler, look_back = load_model_and_scaler(model_path)
        if model is None:
            log_warn("Failed to load LSTM model")
            return None

        wrapper = LSTMWrapper(model, scaler=scaler, look_back=look_back)
        log_info(f"Loaded and wrapped LSTM model (look_back={look_back})")
        return wrapper

    except Exception as e:
        log_error(f"Error loading LSTM model: {e}")
        return None


def create_ensemble(
    rf_model: RandomForestClassifier,
    include_xgboost: bool = RANDOM_FOREST_ENSEMBLE_INCLUDE_XGBOOST,
    include_lstm: bool = RANDOM_FOREST_ENSEMBLE_INCLUDE_LSTM,
    method: str = RANDOM_FOREST_ENSEMBLE_METHOD,
    voting: str = RANDOM_FOREST_ENSEMBLE_VOTING,
    final_estimator: str = RANDOM_FOREST_ENSEMBLE_FINAL_ESTIMATOR,
) -> Union[VotingClassifier, StackingClassifier]:
    """
    Create ensemble model combining RandomForest, XGBoost, and optionally LSTM.

    Args:
        rf_model: Trained RandomForestClassifier
        include_xgboost: Whether to include XGBoost in ensemble
        include_lstm: Whether to include LSTM in ensemble
        method: Ensemble method ("voting" or "stacking")
        voting: Voting type for VotingClassifier ("hard" or "soft")
        final_estimator: Final estimator for StackingClassifier

    Returns:
        Ensemble model (VotingClassifier or StackingClassifier)
    """
    estimators: List[Tuple[str, Any]] = [("rf", rf_model)]

    # Add XGBoost if enabled
    if include_xgboost:
        xgb_model = load_xgboost_model()
        if xgb_model is not None:
            estimators.append(("xgboost", xgb_model))
            log_progress("XGBoost model added to ensemble")
        else:
            log_warn("XGBoost model not available, skipping in ensemble")

    # Add LSTM if enabled
    if include_lstm:
        lstm_wrapper = load_lstm_model_wrapped()
        if lstm_wrapper is not None:
            estimators.append(("lstm", lstm_wrapper))
            log_progress("LSTM model added to ensemble")
        else:
            log_warn("LSTM model not available, skipping in ensemble")

    if len(estimators) == 1:
        log_warn("Only RandomForest available, returning single model (not an ensemble)")
        return rf_model

    log_progress(f"Creating {method} ensemble with {len(estimators)} models: {[name for name, _ in estimators]}")

    if method == "voting":
        ensemble = VotingClassifier(estimators=estimators, voting=voting)
        log_info(f"Created VotingClassifier with {voting} voting")
        return ensemble
    elif method == "stacking":
        # Create final estimator
        if final_estimator == "RandomForest":
            final_est = RandomForestClassifier(n_estimators=50, random_state=42, n_jobs=-1)
        elif final_estimator == "XGBoost":
            try:
                from xgboost import XGBClassifier

                final_est = XGBClassifier(n_estimators=50, random_state=42, n_jobs=-1)
            except ImportError:
                log_warn("XGBoost not available for final estimator, using RandomForest")
                final_est = RandomForestClassifier(n_estimators=50, random_state=42, n_jobs=-1)
        elif final_estimator == "LogisticRegression":
            final_est = LogisticRegression(random_state=42, max_iter=1000)
        else:
            log_warn(f"Unknown final estimator '{final_estimator}', using RandomForest")
            final_est = RandomForestClassifier(n_estimators=50, random_state=42, n_jobs=-1)

        ensemble = StackingClassifier(estimators=estimators, final_estimator=final_est, cv=3)
        log_info(f"Created StackingClassifier with {final_estimator} as final estimator")
        return ensemble
    else:
        raise ValueError(f"Unknown ensemble method: {method}. Use 'voting' or 'stacking'")


__all__ = [
    "LSTMWrapper",
    "create_ensemble",
    "load_xgboost_model",
    "load_lstm_model_wrapped",
]
