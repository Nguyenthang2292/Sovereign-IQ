"""
XGBoost model training and prediction functions.

This module provides functions for training XGBoost classification models
for cryptocurrency price direction prediction, including:
- Model training with proper time-series data splitting (gap prevention)
- Cross-validation with data leakage prevention
- Prediction probability calculation for next candle movement
"""

from typing import Any, Type, Union

import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.metrics import accuracy_score
from sklearn.model_selection import TimeSeriesSplit

from config import (
    ID_TO_LABEL,
    MODEL_FEATURES,
    TARGET_HORIZON,
    TARGET_LABELS,
    XGBOOST_MIN_TRAIN_FRACTION,
    XGBOOST_PARAMS,
    XGBOOST_TRAIN_TEST_SPLIT,
)
from config.position_sizing import (
    USE_GPU,
)
from modules.common.utils import (
    log_data,
    log_model,
    log_success,
    log_warn,
)

from .display import print_classification_report


class ClassDiversityError(ValueError):
    """
    Exception raised when training data lacks sufficient class diversity.

    This exception is raised when:
    - Training set has fewer than 2 classes (XGBoost requires at least 2)
    - Training set is missing required class 0 (XGBoost expects classes to start from 0)
    - XGBoost reports class mismatch errors during model fitting

    This allows callers to distinguish class diversity issues from other ValueError
    cases, enabling more precise error handling.
    """

    pass


def _resolve_xgb_classifier() -> Type:
    """
    Resolve XGBClassifier class with fallback support.

    Ensures XGBClassifier is available even with minimal xgboost installation.
    Falls back to sklearn's GradientBoostingClassifier if XGBoost is not available.

    Returns:
        XGBClassifier class (or fallback equivalent)

    Raises:
        AttributeError: If no suitable classifier can be found
    """
    if hasattr(xgb, "XGBClassifier"):
        return xgb.XGBClassifier
    try:
        from xgboost.sklearn import XGBClassifier as sklearn_classifier
    except Exception:  # pragma: no cover - only hit when package is broken
        try:
            from sklearn.ensemble import GradientBoostingClassifier
        except Exception as sklearn_exc:  # pragma: no cover - backup missing
            raise AttributeError(
                "XGBClassifier is not available in the installed xgboost distribution."
            ) from sklearn_exc

        class _GradientBoostingWrapper(GradientBoostingClassifier):
            """
            Fallback classifier mimicking the XGBoost sklearn API.

            Wraps sklearn's GradientBoostingClassifier to provide XGBoost-like interface.
            Only accepts parameters that are compatible with both XGBoost and sklearn.
            """

            XGB_PARAM_WHITELIST = {
                "learning_rate",
                "n_estimators",
                "subsample",
                "max_depth",
                "random_state",
            }

            def predict_proba(self, X: Any) -> np.ndarray:
                """Return probability estimates for each class."""
                return super().predict_proba(X)

        sklearn_classifier = _GradientBoostingWrapper
    # Cache the resolved classifier for subsequent calls
    xgb.XGBClassifier = sklearn_classifier
    return sklearn_classifier


def train_and_predict(df: pd.DataFrame) -> Any:
    """
    Train XGBoost model with proper time-series validation and return trained model.

    This function performs:
    1. Train/test split with gap to prevent data leakage
    2. Holdout set evaluation
    3. Time-series cross-validation with gap prevention
    4. Final model training on all available data

    Args:
        df: DataFrame containing features (MODEL_FEATURES) and target column ("Target")

    Returns:
        Trained XGBoost classifier model ready for prediction

    Raises:
        ClassDiversityError: If training data lacks sufficient class diversity:
            - Training set has fewer than 2 classes (XGBoost requires at least 2)
            - Training set is missing required class 0 (XGBoost expects classes to start from 0)
            - XGBoost reports class mismatch errors during model fitting

    Note:
        The gap between train and test sets equals TARGET_HORIZON to prevent
        using future prices when creating labels for training data.
    """
    X = df[MODEL_FEATURES]
    y = df["Target"].astype(int)

    def build_model():
        """
        Build XGBoost classifier instance with configuration parameters.

        Uses parameters from config, dynamically adds num_class based on TARGET_LABELS.
        Filters parameters through whitelist if classifier has one (for fallback compatibility).
        Adds GPU support if available.

        Returns:
            XGBoost classifier instance (or fallback equivalent)
        """
        classifier_cls = _resolve_xgb_classifier()
        params = XGBOOST_PARAMS.copy()
        params["num_class"] = len(TARGET_LABELS)

        # Add GPU support if available
        if USE_GPU:
            try:
                # Check if GPU is actually available
                # Try to detect CUDA availability
                import subprocess

                result = subprocess.run(["nvidia-smi"], capture_output=True, timeout=5)
                if result.returncode == 0:
                    # GPU is available, use GPU parameters
                    # In XGBoost 2.0+, use 'hist' with device='cuda' instead of 'gpu_hist'
                    params["tree_method"] = "hist"  # Changed from "gpu_hist" to "hist" for XGBoost 2.0+
                    params["device"] = "cuda"
                    # Remove n_jobs when using GPU (GPU handles parallelism)
                    if "n_jobs" in params:
                        del params["n_jobs"]
            except (subprocess.TimeoutExpired, FileNotFoundError, Exception):
                # GPU not available or nvidia-smi not found, fall back to CPU
                pass

        # Filter parameters through whitelist if classifier has one (for fallback compatibility)
        whitelist = getattr(classifier_cls, "XGB_PARAM_WHITELIST", None)
        if whitelist is not None:
            params = {k: v for k, v in params.items() if k in whitelist}

        try:
            return classifier_cls(**params)
        except Exception:
            # Try without device parameter if it fails
            # (XGBoost 3.x might not support device="cuda" with tree_method="hist")
            if "device" in params:
                params_without_device = params.copy()
                del params_without_device["device"]
                try:
                    return classifier_cls(**params_without_device)
                except Exception as e2:
                    raise e2
            else:
                raise

    # Train/Test Split with Gap Prevention
    # Strategy: 80/20 split with TARGET_HORIZON gap between train and test sets
    # IMPORTANT: The gap prevents data leakage because labels for the last TARGET_HORIZON
    # rows of the training set would require future prices from the test set.
    # Example: If TARGET_HORIZON=24, we predict 24 candles ahead, so we need a 24-candle gap.
    split = int(len(df) * XGBOOST_TRAIN_TEST_SPLIT)
    train_end = split - TARGET_HORIZON
    test_start = split

    # Adjust split if gap creation leaves insufficient training data
    if train_end < len(df) * XGBOOST_MIN_TRAIN_FRACTION:
        train_end = int(len(df) * XGBOOST_MIN_TRAIN_FRACTION)
        test_start = train_end + TARGET_HORIZON
        if test_start >= len(df):
            # Not enough data for proper train/test split with gap
            min_required_rows = int(len(df) * XGBOOST_MIN_TRAIN_FRACTION) + TARGET_HORIZON + 1
            raise ValueError(
                f"Insufficient data for train/test split with gap. "
                f"Need at least {min_required_rows} rows "
                f"({XGBOOST_MIN_TRAIN_FRACTION:.0%} train + {TARGET_HORIZON} gap + 1 test), "
                f"but only have {len(df)} rows."
            )

    X_train, X_test = X.iloc[:train_end], X.iloc[test_start:]
    y_train, y_test = y.iloc[:train_end], y.iloc[test_start:]

    gap_size = test_start - train_end
    if gap_size > 0:
        log_data(f"Train/Test split: {len(X_train)} train, {gap_size} gap (to prevent leakage), {len(X_test)} test")

    # Validate class diversity in training set before building model
    # XGBoost requires at least 2 classes, but model is configured for 3 classes
    unique_train_classes = sorted(y_train.unique())
    if len(unique_train_classes) < 2:
        raise ClassDiversityError(
            f"Insufficient class diversity in training set: "
            f"found {len(unique_train_classes)} class(es) {unique_train_classes}, "
            f"but XGBoost requires at least 2 classes. "
            f"Total training samples: {len(y_train)}"
        )

    # Check if we have all expected classes (0, 1, 2 for DOWN, NEUTRAL, UP)
    # If not, XGBoost may fail with "Invalid classes" error
    expected_classes = set(range(len(TARGET_LABELS)))  # {0, 1, 2}
    actual_classes = set(unique_train_classes)

    # If training set doesn't have class 0, XGBoost will fail because it expects classes to start from 0
    if 0 not in actual_classes:
        raise ClassDiversityError(
            f"Training set missing class 0 (DOWN). Found classes: {unique_train_classes}. "
            f"XGBoost expects classes to start from 0. Total training samples: {len(y_train)}"
        )

    if len(unique_train_classes) < len(TARGET_LABELS):
        # Model expects 3 classes but training set only has fewer - this leads to biased predictions
        missing_classes = expected_classes - actual_classes
        raise ClassDiversityError(
            f"Training set has {len(unique_train_classes)} class(es) {[ID_TO_LABEL[c] for c in unique_train_classes]}, "
            f"but model expects {len(TARGET_LABELS)} classes. Missing: {[ID_TO_LABEL[c] for c in missing_classes]}. "
            f"Training with missing classes produces biased predictions."
        )

    model = build_model()
    try:
        model.fit(X_train, y_train)
    except ValueError as e:
        error_msg = str(e)
        # Catch XGBoost class mismatch errors
        if "invalid classes" in error_msg.lower() or ("expected" in error_msg.lower() and "got" in error_msg.lower()):
            raise ClassDiversityError(
                f"XGBoost class mismatch: {error_msg}. "
                f"Training set has classes: {unique_train_classes}, expected classes: {list(expected_classes)}. "
                f"Total training samples: {len(y_train)}"
            ) from e
        raise

    if len(X_test) > 0:
        y_pred = model.predict(X_test)
        score = model.score(X_test, y_test)
        log_model(f"Holdout Accuracy: {score:.4f}")
        print_classification_report(y_test, y_pred, "Holdout Test Set Evaluation")
    else:
        log_warn("Skipping holdout evaluation (insufficient test data after gap).")

    # Time-Series Cross-Validation with Gap Prevention
    # Uses TimeSeriesSplit to respect temporal order, with gap between train/test in each fold
    max_splits = min(5, len(df) - 1)
    if max_splits >= 2:
        tscv = TimeSeriesSplit(n_splits=max_splits)
        cv_scores = []
        all_y_true = []
        all_y_pred = []

        for fold, (train_idx, test_idx) in enumerate(tscv.split(X), start=1):
            # Apply gap to prevent data leakage: remove last TARGET_HORIZON indices from train
            # This ensures labels for training data don't require future prices from test set
            train_idx_array = np.array(train_idx)
            if len(train_idx_array) > TARGET_HORIZON:
                train_idx_filtered = train_idx_array[:-TARGET_HORIZON]
            else:
                log_warn(f"CV Fold {fold}: Skipped (insufficient train data for gap)")
                continue

            # Ensure test set doesn't overlap with gap
            # Gap is sufficient when: test_start > train_end + TARGET_HORIZON
            test_idx_array = np.array(test_idx)
            if len(train_idx_filtered) > 0 and len(test_idx_array) > 0:
                min_test_start = train_idx_filtered[-1] + TARGET_HORIZON + 1
                if test_idx_array[0] < min_test_start:
                    # Adjust test start to create proper gap
                    test_idx_filtered = test_idx_array[test_idx_array >= min_test_start]
                    if len(test_idx_filtered) == 0:
                        log_warn(f"CV Fold {fold}: Skipped (no valid test data after gap)")
                        continue
                else:
                    test_idx_filtered = test_idx_array

            # Class Diversity Validation
            # XGBoost requires at least 2 classes, but we need all 3 for proper multi-class prediction
            y_train_fold = y.iloc[train_idx_filtered]
            unique_classes = sorted(y_train_fold.unique())

            if len(unique_classes) < 2:
                log_warn(f"CV Fold {fold}: Skipped (insufficient class diversity: {unique_classes})")
                continue

            # Require all target classes for consistency
            # Skipping folds with missing classes ensures consistent evaluation across folds
            if len(unique_classes) < len(TARGET_LABELS):
                class_list = [ID_TO_LABEL[c] for c in unique_classes]
                log_warn(f"CV Fold {fold}: Skipped (missing classes: expected {TARGET_LABELS}, got {class_list})")
                continue

            cv_model = build_model()
            cv_model.fit(X.iloc[train_idx_filtered], y.iloc[train_idx_filtered])
            if len(test_idx_filtered) > 0:
                y_test_fold = y.iloc[test_idx_filtered]
                preds = cv_model.predict(X.iloc[test_idx_filtered])
                acc = accuracy_score(y_test_fold, preds)
                cv_scores.append(acc)

                # Collect predictions for aggregated classification report across all folds
                all_y_true.extend(y_test_fold.tolist())
                all_y_pred.extend(preds.tolist())

                log_model(
                    f"CV Fold {fold} Accuracy: {acc:.4f} "
                    f"(train: {len(train_idx_filtered)}, "
                    f"gap: {TARGET_HORIZON}, test: {len(test_idx_array)})"
                )

        if len(cv_scores) > 0:
            mean_cv = sum(cv_scores) / len(cv_scores)
            log_success(f"CV Mean Accuracy ({len(cv_scores)} folds): {mean_cv:.4f}")

            # Generate aggregated classification report across all CV folds
            if len(all_y_true) > 0 and len(all_y_pred) > 0:
                print_classification_report(
                    np.array(all_y_true),
                    np.array(all_y_pred),
                    "Cross-Validation Aggregated Report (All Folds)",
                )
        else:
            log_warn("CV: No valid folds after applying gap. Consider increasing data limit.")
    else:
        log_warn("Not enough data for cross-validation (requires >=3 samples).")

    # Final Model Training
    # Train on all available data for production use
    model.fit(X, y)
    return model


def predict_next_move(model: Any, last_row: Union[pd.Series, pd.DataFrame]) -> np.ndarray:
    """
    Predict probability distribution for the next candle movement direction.

    Args:
        model: Trained XGBoost classifier model
        last_row: DataFrame row or Series containing current features (MODEL_FEATURES)

    Returns:
        numpy.ndarray: Probability array of shape (n_classes,) where:
            - Index 0: Probability of DOWN movement
            - Index 1: Probability of NEUTRAL movement
            - Index 2: Probability of UP movement

    Note:
        The probabilities sum to 1.0 and represent the model's confidence
        for each direction class.
    """
    X_new = last_row[MODEL_FEATURES]
    # Convert Series to DataFrame to preserve feature names and ensure proper shape
    if isinstance(X_new, pd.Series):
        X_new = X_new.to_frame().T

    # Get probability distribution for all classes
    proba = model.predict_proba(X_new)[0]

    return proba
