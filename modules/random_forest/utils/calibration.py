"""Probability Calibration for Random Forest models.

This module provides probability calibration functionality using
sklearn.calibration.CalibratedClassifierCV to improve the reliability
of predicted probabilities and confidence thresholds.
"""

from typing import Any, Dict, Optional, Union

import numpy as np
import pandas as pd
from sklearn.calibration import CalibratedClassifierCV
from sklearn.ensemble import RandomForestClassifier

from config.random_forest import (
    RANDOM_FOREST_CALIBRATION_CV,
    RANDOM_FOREST_CALIBRATION_METHOD,
)
from modules.common.ui.logging import log_info, log_progress, log_warn


def calibrate_model(
    model: Union[RandomForestClassifier, any],
    X_train: pd.DataFrame,
    y_train: pd.Series,
    method: str = RANDOM_FOREST_CALIBRATION_METHOD,
    cv: int = RANDOM_FOREST_CALIBRATION_CV,
) -> CalibratedClassifierCV:
    """
    Calibrate a trained model's probability predictions.

    Random Forest models often produce poorly calibrated probabilities.
    This function wraps the model with CalibratedClassifierCV to improve
    probability calibration using Platt scaling (sigmoid) or isotonic regression.

    Args:
        model: Trained base classifier (e.g., RandomForestClassifier)
        X_train: Training features
        y_train: Training target labels
        method: Calibration method ("sigmoid" for Platt scaling or "isotonic" for isotonic regression)
        cv: Number of cross-validation folds for calibration

    Returns:
        CalibratedClassifierCV wrapper around the base model

    Example:
        >>> model = RandomForestClassifier()
        >>> model.fit(X_train, y_train)
        >>> calibrated_model = calibrate_model(model, X_train, y_train, method='sigmoid', cv=5)
        >>> calibrated_proba = calibrated_model.predict_proba(X_test)
    """
    log_progress(f"Calibrating model probabilities using {method} method (CV={cv})...")

    if method not in ["sigmoid", "isotonic"]:
        log_warn(f"Unknown calibration method '{method}'. Using 'sigmoid' (Platt scaling).")
        method = "sigmoid"

    try:
        # Use 'estimator' parameter (sklearn >= 0.24) instead of 'base_estimator' (deprecated)
        calibrated_model = CalibratedClassifierCV(
            estimator=model,
            method=method,
            cv=cv,
        )

        # Fit the calibrator
        calibrated_model.fit(X_train, y_train)

        log_info(
            f"Model calibration completed successfully using {method} method. "
            f"Calibrated probabilities will be more reliable for confidence thresholds."
        )

        return calibrated_model

    except Exception as e:
        log_warn(f"Error during model calibration: {e}. Returning uncalibrated model.")
        # Return a wrapper that just passes through to the original model
        # This is a fallback - in practice, you might want to raise or handle differently
        return model


def evaluate_calibration(
    model: Union[CalibratedClassifierCV, RandomForestClassifier],
    X_test: pd.DataFrame,
    y_test: pd.Series,
) -> Optional[Dict[str, Any]]:
    """
    Evaluate the calibration quality of a model.

    Computes calibration metrics to assess how well the predicted probabilities
    match the true probabilities.

    Args:
        model: Trained model (calibrated or uncalibrated)
        X_test: Test features
        y_test: Test target labels

    Returns:
        Dictionary with calibration metrics (Brier score, ECE, etc.) or None if error
    """
    try:
        from sklearn.calibration import calibration_curve
        from sklearn.metrics import brier_score_loss

        # Get predicted probabilities
        y_proba = model.predict_proba(X_test)

        # For multi-class, use the positive class (class 1) for binary classification
        # or the max probability class for multi-class
        if y_proba.shape[1] > 2:
            # Multi-class: use the predicted class probabilities
            y_pred_class = model.predict(X_test)
            # Get probabilities for the predicted class
            y_proba_positive = y_proba[np.arange(len(y_proba)), y_pred_class]
            y_true_binary = (y_test == y_pred_class).astype(int)
        else:
            # Binary classification: use positive class probability
            y_proba_positive = y_proba[:, 1]
            y_true_binary = (y_test == 1).astype(int)

        # Compute Brier score (lower is better)
        brier_score = brier_score_loss(y_true_binary, y_proba_positive)

        # Compute calibration curve
        fraction_of_positives, mean_predicted_value = calibration_curve(
            y_true_binary, y_proba_positive, n_bins=10, strategy="uniform"
        )

        # Compute Expected Calibration Error (ECE)
        # ECE = mean(|fraction_of_positives - mean_predicted_value|)
        ece = np.mean(np.abs(fraction_of_positives - mean_predicted_value))

        metrics = {
            "brier_score": float(brier_score),
            "expected_calibration_error": float(ece),
            "calibration_curve": {
                "fraction_of_positives": fraction_of_positives.tolist(),
                "mean_predicted_value": mean_predicted_value.tolist(),
            },
        }

        log_info(
            f"Calibration metrics - Brier Score: {brier_score:.4f}, "
            f"Expected Calibration Error (ECE): {ece:.4f}"
        )

        return metrics

    except Exception as e:
        log_warn(f"Error evaluating calibration: {e}")
        return None


__all__ = [
    "calibrate_model",
    "evaluate_calibration",
]
