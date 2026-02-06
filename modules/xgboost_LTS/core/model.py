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
from sklearn.metrics import accuracy_score  # type: ignore[import-untyped]
from sklearn.model_selection import TimeSeriesSplit  # type: ignore[import-untyped]

from config import (
    ID_TO_LABEL,
    MODEL_FEATURES,
    TARGET_HORIZON,
    TARGET_LABELS,
    USE_GPU,
    XGBOOST_MIN_TRAIN_FRACTION,
    XGBOOST_PARAMS,
    XGBOOST_TRAIN_TEST_SPLIT,
    XGBOOST_USE_FLOAT32,
    XGBOOST_USE_PARALLEL_CV,
)
from modules.common.utils import (
    log_data,
    log_model,
    log_success,
    log_warn,
)
from modules.xgboost_LTS.utils.cache_manager import CacheManager
from modules.xgboost_LTS.utils.cv_parallel import run_parallel_cv
from modules.xgboost_LTS.utils.display import print_classification_report
from modules.xgboost_LTS.utils.gpu_utils import detect_cuda_available

# Optional CuPy for GPU prediction (avoids "input data is on cpu" warning when model is on cuda)
try:
    import cupy as cp  # type: ignore[import-untyped]  # noqa: F401

    _CUPY_AVAILABLE = True
except ImportError:
    cp = None
    _CUPY_AVAILABLE = False


def _use_gpu_for_predict() -> bool:
    """True when model may be on GPU and we should move input to GPU for predict/predict_proba."""
    return bool(USE_GPU and detect_cuda_available() and _CUPY_AVAILABLE)


def _to_gpu(X: Union[np.ndarray, pd.DataFrame]) -> Any:
    """Move array/DataFrame to GPU (CuPy) when USE_GPU and CuPy available; else return X unchanged."""
    if not _use_gpu_for_predict() or cp is None:
        return X
    if isinstance(X, pd.DataFrame):
        return cp.asarray(X.values, dtype=cp.float32)
    return cp.asarray(X, dtype=cp.float32)


def _ensure_numpy(a: Any) -> np.ndarray:
    """Convert CuPy array back to numpy; leave numpy/other unchanged."""
    if cp is not None and hasattr(a, "device"):
        return cp.asnumpy(a)  # type: ignore[union-attr]
    return np.asarray(a)


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
            from sklearn.ensemble import GradientBoostingClassifier  # type: ignore[import-untyped]
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

        sklearn_classifier = _GradientBoostingWrapper  # type: ignore[assignment,misc]
    # Return the resolved classifier without modifying global state
    # This prevents side effects on other modules that import xgboost
    return sklearn_classifier


def train_and_predict(df: pd.DataFrame, use_cache: bool = True) -> Any:
    """
    Train XGBoost model with proper time-series validation and return trained model.

    This function performs:
    1. Train/test split with gap to prevent data leakage
    2. Holdout set evaluation
    3. Time-series cross-validation with gap prevention
    4. Final model training on all available data

    Args:
        df: DataFrame containing features (MODEL_FEATURES) and target column ("Target")
        use_cache: Whether to use model caching (default: True)

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
    # Drop rows with non-finite target or features (labeling NaN for last TARGET_HORIZON rows,
    # indicator warmup NaN, or inf from divisions). Required before .astype(int) on Target.
    feature_cols = [c for c in MODEL_FEATURES if c in df.columns]
    if "Target" not in df.columns:
        raise ValueError("DataFrame must contain a 'Target' column from labeling.")
    check_cols = feature_cols + ["Target"]
    finite_mask = np.isfinite(df[check_cols].values).all(axis=1)
    if not finite_mask.all():
        n_dropped = (~finite_mask).sum()
        df = df.iloc[np.flatnonzero(finite_mask)].copy()
        log_data(f"Dropped {n_dropped} rows with non-finite target or features.")
    if df.empty:
        raise ValueError("No rows with finite target/features. Ensure labeling and indicators produced valid data.")

    X = df[MODEL_FEATURES]

    # Float32 Optimization (Task 2.3)
    # Note: float32 has ~7 decimal digits of precision. For features with very large
    # or very small values, this may cause precision loss. Use with caution.
    if XGBOOST_USE_FLOAT32:
        # Check for potential precision issues
        max_abs_val = X.abs().max().max()
        if max_abs_val > 1e6:
            log_warn(f"Float32 conversion may lose precision for large values (max: {max_abs_val:.2e})")
        X = X.astype(np.float32)

    y = df["Target"].astype(int)

    # Model Caching (Task 3.1)
    if use_cache:
        cache_manager = CacheManager()
        cached_model = cache_manager.load_model(df, XGBOOST_PARAMS)
        if cached_model is not None:
            log_model("Using cached model (skipping training)")
            return cached_model

    def build_model(seed_offset=0):
        """
        Build XGBoost classifier instance with configuration parameters.

        Uses parameters from config, dynamically adds num_class based on TARGET_LABELS.
        Filters parameters through whitelist if classifier has one (for fallback compatibility).
        Adds GPU support if available.

        Args:
            seed_offset: Offset to add to random_state for CV fold diversity (default: 0)

        Returns:
            XGBoost classifier instance (or fallback equivalent)
        """
        classifier_cls = _resolve_xgb_classifier()
        params = XGBOOST_PARAMS.copy()
        params["num_class"] = len(TARGET_LABELS)

        # Add seed offset for CV fold diversity
        if "random_state" in params:
            params["random_state"] = params["random_state"] + seed_offset

        # Add GPU support if available
        if USE_GPU and detect_cuda_available():
            params["tree_method"] = "hist"
            params["device"] = "cuda"
            # Remove n_jobs when using GPU (GPU handles parallelism)
            if "n_jobs" in params:
                del params["n_jobs"]

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
        model.fit(X_train, y_train, eval_set=[(X_test, y_test)], verbose=False)
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
        X_test_in = _to_gpu(X_test)
        y_pred = model.predict(X_test_in)
        y_pred = _ensure_numpy(y_pred)
        score = float((y_pred == np.asarray(y_test)).mean())
        log_model(f"Holdout Accuracy: {score:.4f}")
        print_classification_report(y_test, y_pred, "Holdout Test Set Evaluation")
    else:
        log_warn("Skipping holdout evaluation (insufficient test data after gap).")

    # Time-Series Cross-Validation with Gap Prevention
    # Uses TimeSeriesSplit to respect temporal order, with gap between train/test in each fold
    max_splits = min(5, len(df) - 1)

    # Configuration for Parallel CV (imported from config)

    if max_splits >= 2:
        tscv = TimeSeriesSplit(n_splits=max_splits)

        if XGBOOST_USE_PARALLEL_CV:
            # Parallel CV execution
            cv_scores, all_y_true, all_y_pred = run_parallel_cv(X, y, tscv, XGBOOST_PARAMS)
        else:
            # Sequential CV (original implementation)
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
                # Always filter test indices to prevent data leakage
                test_idx_array = np.array(test_idx)
                if len(train_idx_filtered) > 0 and len(test_idx_array) > 0:
                    min_test_start = train_idx_filtered[-1] + TARGET_HORIZON + 1
                    # Always filter, not just when first element is < min_test_start
                    test_idx_filtered = test_idx_array[test_idx_array >= min_test_start]
                    if len(test_idx_filtered) == 0:
                        log_warn(f"CV Fold {fold}: Skipped (no valid test data after gap)")
                        continue
                else:
                    log_warn(f"CV Fold {fold}: Skipped (insufficient data)")
                    continue

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

                # Prepare validation set for early stopping
                X_train_fold = X.iloc[train_idx_filtered]
                y_train_fold = y.iloc[train_idx_filtered]

                eval_set = []
                if len(test_idx_filtered) > 0:
                    X_test_fold = X.iloc[test_idx_filtered]
                    y_test_fold = y.iloc[test_idx_filtered]
                    eval_set = [(X_test_fold, y_test_fold)]

                cv_model = build_model(seed_offset=fold)
                cv_model.fit(X_train_fold, y_train_fold, eval_set=eval_set, verbose=False)

                if len(test_idx_filtered) > 0:
                    y_test_fold = y.iloc[test_idx_filtered]
                    X_fold_test = _to_gpu(X.iloc[test_idx_filtered])
                    preds = cv_model.predict(X_fold_test)
                    preds = _ensure_numpy(preds)
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
    # Note: Use a small part as eval_set if we want to keep early stopping,
    # or just use the whole set if we assume persistence of best_iteration from CV.
    # Here we follow the roadmap to add eval_set even to final fit.
    # For final fit on ALL data, we'll use the last 20% as eval_set.
    final_split = int(len(X) * 0.8)
    model.fit(X, y, eval_set=[(X.iloc[final_split:], y.iloc[final_split:])], verbose=False)

    # Save model to cache (Task 3.1)
    if use_cache:
        cache_manager = CacheManager()
        cache_manager.save_model(model, df, XGBOOST_PARAMS)

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

    Raises:
        ValueError: If required features are missing or contain invalid values
    """
    # Validate input features (ensure str for set difference; Index may be typed as int)
    if isinstance(last_row, pd.Series):
        available_features = {str(x) for x in last_row.index}
    else:
        available_features = {str(x) for x in last_row.columns}

    missing_features = set(MODEL_FEATURES) - available_features
    if missing_features:
        raise ValueError(f"Missing required features: {missing_features}")

    X_new = last_row[MODEL_FEATURES]

    # Check for NaN/Inf values
    if isinstance(X_new, pd.Series):
        if not np.isfinite(X_new.values).all():
            raise ValueError("Features contain NaN or Inf values")
    else:
        if not np.isfinite(X_new.values).all():
            raise ValueError("Features contain NaN or Inf values")

    # Convert Series to DataFrame to preserve feature names and ensure proper shape
    if isinstance(X_new, pd.Series):
        X_new = X_new.to_frame().T

    # Get probability distribution for all classes (move to GPU if model is on cuda)
    X_in = _to_gpu(X_new)
    proba = model.predict_proba(X_in)[0]
    proba = _ensure_numpy(proba)

    # Handle case where model was trained with fewer than 3 classes
    # Pad with zeros to ensure consistent shape (3 classes expected)
    if len(proba) < 3:
        log_warn(f"Model trained with {len(proba)} classes, expected 3. Padding with zeros.")
        padded_proba = np.zeros(3)
        padded_proba[: len(proba)] = proba
        return padded_proba

    return proba
