"""Walk-Forward Optimization for Random Forest models.

This module provides walk-forward validation, model drift detection,
and periodic retraining functionality.
"""

from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

from config.random_forest import (
    RANDOM_FOREST_DRIFT_DETECTION_ENABLED,
    RANDOM_FOREST_DRIFT_THRESHOLD,
    RANDOM_FOREST_DRIFT_WINDOW_SIZE,
    RANDOM_FOREST_RETRAIN_PERIOD_DAYS,
    RANDOM_FOREST_TOTAL_GAP,
    RANDOM_FOREST_WALK_FORWARD_EXPANDING_WINDOW,
    RANDOM_FOREST_WALK_FORWARD_N_SPLITS,
)
from modules.common.ui.logging import log_info, log_progress, log_warn


class WalkForwardValidator:
    """
    Walk-forward validation for time-series models.

    Supports both expanding and rolling window strategies.
    """

    def __init__(
        self,
        n_splits: int = RANDOM_FOREST_WALK_FORWARD_N_SPLITS,
        expanding_window: bool = RANDOM_FOREST_WALK_FORWARD_EXPANDING_WINDOW,
        gap: int = RANDOM_FOREST_TOTAL_GAP,
    ):
        """
        Initialize walk-forward validator.

        Args:
            n_splits: Number of validation splits
            expanding_window: If True, use expanding window; if False, use rolling window
            gap: Gap between train and test sets to prevent data leakage
        """
        self.n_splits = n_splits
        self.expanding_window = expanding_window
        self.gap = gap

    def split(
        self, features: pd.DataFrame, target: pd.Series
    ) -> List[Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]]:
        """
        Generate walk-forward train/test splits.

        Args:
            features: Feature DataFrame
            target: Target Series

        Returns:
            List of (features_train, features_test, target_train, target_test) tuples
        """
        total_size = len(features)
        splits = []

        # Calculate split sizes
        # Reserve last portion for final test (e.g., 20%)
        test_size = int(total_size * 0.2)
        available_size = total_size - test_size

        # Calculate step size for splits
        step_size = max(1, available_size // (self.n_splits + 1))

        for i in range(self.n_splits):
            # Calculate train end and test start
            if self.expanding_window:
                # Expanding window: train_end grows with each split
                train_end = step_size * (i + 1)
            else:
                # Rolling window: fixed window size, slides forward
                window_size = step_size * 2  # Use larger window for rolling
                train_start = step_size * i
                train_end = min(train_start + window_size, available_size)
                # Ensure train_end is at least window_size
                if train_end < window_size:
                    continue

            # Test set starts after gap
            test_start = train_end + self.gap
            test_end = min(test_start + step_size, total_size - test_size)

            # Validation
            if train_end <= 0 or test_start >= total_size:
                log_warn(f"Skipping split {i+1}: insufficient data (train_end={train_end}, test_start={test_start})")
                continue

            if test_end <= test_start:
                log_warn(f"Skipping split {i+1}: invalid test range (test_start={test_start}, test_end={test_end})")
                continue

            # Create splits
            features_train = features.iloc[:train_end].copy()
            features_test = features.iloc[test_start:test_end].copy()
            target_train = target.iloc[:train_end].copy()
            target_test = target.iloc[test_start:test_end].copy()

            splits.append((features_train, features_test, target_train, target_test))

        return splits


class ModelDriftDetector:
    """
    Detect model performance drift to trigger retraining.

    Monitors recent predictions and compares performance to baseline.
    """

    def __init__(
        self,
        enabled: bool = RANDOM_FOREST_DRIFT_DETECTION_ENABLED,
        threshold: float = RANDOM_FOREST_DRIFT_THRESHOLD,
        window_size: int = RANDOM_FOREST_DRIFT_WINDOW_SIZE,
    ):
        """
        Initialize drift detector.

        Args:
            enabled: Whether drift detection is enabled
            threshold: Performance degradation threshold (e.g., 0.05 = 5% drop)
            window_size: Number of recent predictions to monitor
        """
        self.enabled = enabled
        self.threshold = threshold
        self.window_size = window_size
        self.baseline_accuracy: Optional[float] = None
        self.recent_predictions: List[Tuple[np.ndarray, np.ndarray]] = []  # (y_true, y_pred)

    def set_baseline(self, y_true: np.ndarray, y_pred: np.ndarray) -> None:
        """
        Set baseline performance from initial model evaluation.

        Args:
            y_true: True labels
            y_pred: Predicted labels
        """
        if len(y_true) > 0 and len(y_pred) > 0:
            self.baseline_accuracy = accuracy_score(y_true, y_pred)
            log_info(f"Drift detector baseline accuracy set to: {self.baseline_accuracy:.4f}")

    def add_prediction(self, y_true: np.ndarray, y_pred: np.ndarray) -> None:
        """
        Add a new prediction to the monitoring window.

        Args:
            y_true: True labels
            y_pred: Predicted labels
        """
        if not self.enabled:
            return

        self.recent_predictions.append((y_true.copy(), y_pred.copy()))

        # Keep only recent predictions
        if len(self.recent_predictions) > self.window_size:
            self.recent_predictions.pop(0)

    def check_drift(self) -> Tuple[bool, Optional[float]]:
        """
        Check if model drift has occurred.

        Returns:
            Tuple of (drift_detected, current_accuracy)
        """
        if not self.enabled:
            return False, None

        if self.baseline_accuracy is None:
            log_warn("Drift detector baseline not set. Cannot detect drift.")
            return False, None

        if len(self.recent_predictions) < self.window_size:
            # Not enough data yet
            return False, None

        # Calculate current accuracy from recent predictions
        all_y_true = np.concatenate([y_true for y_true, _ in self.recent_predictions])
        all_y_pred = np.concatenate([y_pred for _, y_pred in self.recent_predictions])

        if len(all_y_true) == 0 or len(all_y_pred) == 0:
            return False, None

        current_accuracy = accuracy_score(all_y_true, all_y_pred)

        # Check if performance degraded beyond threshold
        performance_drop = self.baseline_accuracy - current_accuracy
        drift_detected = performance_drop >= self.threshold

        if drift_detected:
            log_warn(
                f"Model drift detected! Baseline: {self.baseline_accuracy:.4f}, "
                f"Current: {current_accuracy:.4f}, Drop: {performance_drop:.4f} (threshold: {self.threshold})"
            )

        return drift_detected, current_accuracy


class ModelVersionManager:
    """
    Manage model versioning with timestamps and version tracking.
    """

    def __init__(self, enabled: bool = True):
        """
        Initialize version manager.

        Args:
            enabled: Whether versioning is enabled
        """
        self.enabled = enabled
        self.version = 1

    def get_version_string(self) -> str:
        """
        Generate version string with timestamp.

        Returns:
            Version string in format: rf_v{version}_{timestamp}
        """
        if not self.enabled:
            return "rf_latest"

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        version_str = f"rf_v{self.version}_{timestamp}"
        return version_str

    def increment_version(self) -> None:
        """Increment model version."""
        self.version += 1

    def get_model_metadata(self) -> Dict[str, any]:
        """
        Get model metadata dictionary.

        Returns:
            Dictionary with version information
        """
        return {
            "version": self.version,
            "version_string": self.get_version_string(),
            "timestamp": datetime.now().isoformat(),
        }


def should_retrain_model(last_retrain_date: Optional[datetime]) -> bool:
    """
    Check if model should be retrained based on periodic schedule.

    Args:
        last_retrain_date: Date of last retraining (None if never trained)

    Returns:
        True if retraining is needed
    """
    if RANDOM_FOREST_RETRAIN_PERIOD_DAYS <= 0:
        return False

    if last_retrain_date is None:
        return True  # Never trained, should train

    days_since_retrain = (datetime.now() - last_retrain_date).days
    return days_since_retrain >= RANDOM_FOREST_RETRAIN_PERIOD_DAYS


__all__ = [
    "WalkForwardValidator",
    "ModelDriftDetector",
    "ModelVersionManager",
    "should_retrain_model",
]
