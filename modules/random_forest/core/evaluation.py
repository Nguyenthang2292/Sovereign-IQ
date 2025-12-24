"""Random Forest model evaluation.

This module provides functionality for evaluating Random Forest models with
various confidence thresholds and calculating performance metrics.
"""

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score

from config import (
    CONFIDENCE_THRESHOLDS,
)
from modules.common.ui.logging import log_error, log_model, log_warn


def evaluate_model_with_confidence(
    model: RandomForestClassifier, features_test: pd.DataFrame, target_test: pd.Series
) -> None:
    """Evaluate the model's performance at various confidence thresholds."""
    # Input validation
    if model is None:
        log_warn("Model is None, cannot evaluate.")
        return
    if features_test is None or features_test.empty:
        log_warn("Test features are None or empty, cannot evaluate.")
        return
    if target_test is None or target_test.empty:
        log_warn("Test target is None or empty, cannot evaluate.")
        return
    if len(features_test) != len(target_test):
        log_warn(f"Length mismatch: features ({len(features_test)}) != target ({len(target_test)})")
        return
    
    log_model("Evaluating model performance with different confidence thresholds...")
    try:
        y_proba = model.predict_proba(features_test)
    except (ValueError, AttributeError) as e:
        log_error(f"Error during prediction: {e}")
        return
    # Ensure y_proba is numpy.ndarray and model.classes_ is numpy.ndarray
    y_proba = np.asarray(y_proba)
    classes = np.asarray(model.classes_)
    for threshold in CONFIDENCE_THRESHOLDS:
        y_pred = apply_confidence_threshold(y_proba, threshold, classes)
        calculate_and_display_metrics(target_test, y_pred, threshold)


def apply_confidence_threshold(
    y_proba: np.ndarray, threshold: float, classes: np.ndarray
) -> np.ndarray:
    """Apply a confidence threshold to prediction probabilities.

    If the highest probability for a prediction is below the threshold, the
    prediction is set to neutral (0).

    Args:
        y_proba: The prediction probabilities from the model.
        threshold: The minimum confidence required to make a non-neutral prediction.
        classes: The class labels from the classifier.

    Returns:
        An array of predictions adjusted for the confidence threshold.
    """
    # Use numeric 0 for neutral instead of string "NEUTRAL"
    y_pred_confident = np.full(y_proba.shape[0], 0, dtype=int)
    max_proba = y_proba.max(axis=1)
    high_confidence_mask = max_proba >= threshold
    if np.any(high_confidence_mask):
        pred_indices = np.argmax(y_proba[high_confidence_mask], axis=1)
        y_pred_confident[high_confidence_mask] = classes[pred_indices]
    return y_pred_confident


def calculate_and_display_metrics(
    y_true: pd.Series, y_pred: np.ndarray, threshold: float
) -> None:
    """Calculate and logs performance metrics for a given confidence threshold.

    Args:
        y_true: The true labels.
        y_pred: The predicted labels.
        threshold: The confidence threshold used for the predictions.
    """
    # Handle empty arrays early to avoid RuntimeWarnings
    if len(y_true) == 0 or len(y_pred) == 0:
        log_warn(f"Empty arrays provided. y_true length: {len(y_true)}, y_pred length: {len(y_pred)}")
        return
    
    # Ensure both y_true and y_pred are numeric for consistent label types
    y_true_numeric = pd.to_numeric(y_true, errors='coerce').astype(int) # type: ignore
    y_pred_numeric = y_pred.astype(int)
    
    # Convert to numpy array if it's a Series
    y_true_array = y_true_numeric.to_numpy() if isinstance(y_true_numeric, pd.Series) else np.asarray(y_true_numeric)
    
    labels = np.unique(np.concatenate((y_true_array, y_pred_numeric)))
    precision = precision_score(
        y_true_numeric, y_pred_numeric, average='weighted', labels=labels, zero_division="warn"
    )
    recall = recall_score(
        y_true_numeric, y_pred_numeric, average='weighted', labels=labels, zero_division="warn"
    )
    f1 = f1_score(
        y_true_numeric, y_pred_numeric, average='weighted', labels=labels, zero_division="warn"
    )
    accuracy = accuracy_score(y_true_numeric, y_pred_numeric)
    log_model(
        f"Metrics @ {threshold:.2f} threshold | "
        f"Accuracy: {accuracy:.4f}, Precision: {precision:.4f}, "
        f"Recall: {recall:.4f}, F1: {f1:.4f}"
    )

