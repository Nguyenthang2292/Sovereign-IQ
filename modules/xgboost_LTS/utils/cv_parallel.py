"""
Parallel cross-validation utilities for XGBoost module.
"""

import multiprocessing as mp
from concurrent.futures import ProcessPoolExecutor, as_completed
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score  # type: ignore[import-untyped]

from config import ID_TO_LABEL, TARGET_HORIZON, TARGET_LABELS
from modules.xgboost_LTS.utils.cv_utils import apply_cv_gap


def _train_cv_fold(
    fold_num: int,
    train_idx: np.ndarray,
    test_idx: np.ndarray,
    X_values: np.ndarray,
    y_values: np.ndarray,
    feature_names: List[str],
    params: Dict[str, Any],
) -> Tuple[int, float, Optional[List[int]], Optional[List[int]], str]:
    """
    Train single CV fold (designed to run in separate process).

    Args:
        fold_num: Fold number for logging
        train_idx: Training indices
        test_idx: Test indices
        X_values: Feature values as numpy array (to avoid pickle issues)
        y_values: Target values as numpy array
        feature_names: List of feature names
        params: XGBoost parameters

    Returns:
        Tuple of (fold_num, accuracy, y_true_list, y_pred_list, message)
    """
    import xgboost as xgb

    train_idx_filtered, test_idx_filtered = apply_cv_gap(train_idx, test_idx, TARGET_HORIZON)
    if len(train_idx_filtered) == 0:
        return (fold_num, 0.0, None, None, "Skipped (insufficient train data for gap)")
    if len(test_idx_filtered) == 0:
        return (fold_num, 0.0, None, None, "Skipped (no valid test data after gap)")

    # Class diversity validation
    y_train_fold = y_values[train_idx_filtered]
    unique_classes = sorted(np.unique(y_train_fold))

    if len(unique_classes) < 2:
        return (fold_num, 0.0, None, None, f"Skipped (insufficient class diversity: {unique_classes})")

    if len(unique_classes) < len(TARGET_LABELS):
        class_list = [ID_TO_LABEL[c] for c in unique_classes]
        return (fold_num, 0.0, None, None, f"Skipped (missing classes: expected {TARGET_LABELS}, got {class_list})")

    # Train model
    try:
        # Add seed offset for CV fold diversity
        fold_params = params.copy()
        if "random_state" in fold_params:
            fold_params["random_state"] = fold_params["random_state"] + fold_num

        model = xgb.XGBClassifier(**fold_params)

        # Use DataFrame for training to preserve feature names
        X_train = pd.DataFrame(X_values[train_idx_filtered], columns=feature_names)
        y_train = y_train_fold

        # Prepare test data for early stopping
        X_test = pd.DataFrame(X_values[test_idx_filtered], columns=feature_names)
        y_test_fold = y_values[test_idx_filtered]

        model.fit(X_train, y_train, eval_set=[(X_test.values, y_test_fold)], verbose=False)

        # Evaluate (move X_test to GPU when model is on cuda to avoid device mismatch warning)
        try:
            import cupy as _cp  # type: ignore[import-untyped]

            _use_gpu = params.get("device") == "cuda"
        except ImportError:
            _cp = None
            _use_gpu = False
        if _use_gpu and _cp is not None:
            X_test_in = _cp.asarray(X_test.values, dtype=_cp.float32)
            preds = model.predict(X_test_in)
            preds = _cp.asnumpy(preds) if hasattr(preds, "device") else np.asarray(preds)
        else:
            preds = model.predict(X_test)
        acc = accuracy_score(y_test_fold, preds)

        message = f"Accuracy: {acc:.4f} (train: {len(train_idx_filtered)}, gap: {TARGET_HORIZON}, test: {len(test_idx_filtered)})"

        return (fold_num, acc, y_test_fold.tolist(), preds.tolist(), message)

    except Exception as e:
        return (fold_num, 0.0, None, None, f"Error: {str(e)}")


def run_parallel_cv(
    X: pd.DataFrame,
    y: pd.Series,
    tscv,
    params: Dict[str, Any],
    max_workers: Optional[int] = None,
) -> Tuple[List[float], List[int], List[int]]:
    """
    Run cross-validation folds in parallel.

    Args:
        X: Feature DataFrame
        y: Target Series
        tscv: TimeSeriesSplit object
        params: XGBoost parameters (will be filtered for pickle safety)
        max_workers: Maximum parallel workers (default: CPU count // 2)

    Returns:
        Tuple of (cv_scores, all_y_true, all_y_pred)
    """
    from modules.common.utils import log_model, log_warn

    # Prepare pickle-safe data
    X_values = X.values
    y_values = y.values
    feature_names = list(X.columns)

    # Filter params for pickle safety (remove non-serializable items)
    params_filtered = {k: v for k, v in params.items() if isinstance(v, (int, float, str, bool, type(None)))}

    # Prepare fold data
    fold_data = []
    for fold_num, (train_idx, test_idx) in enumerate(tscv.split(X), start=1):
        fold_data.append((fold_num, np.array(train_idx), np.array(test_idx)))

    # Determine workers
    if max_workers is None:
        max_workers = max(1, mp.cpu_count() // 2)

    # Run parallel CV
    cv_scores: list[float] = []
    all_y_true: list[int] = []
    all_y_pred: list[int] = []

    # Use ProcessPoolExecutor for true parallelism
    with ProcessPoolExecutor(max_workers=max_workers) as executor:
        futures = {
            executor.submit(
                _train_cv_fold,
                fold_num,
                train_idx,
                test_idx,
                X_values,
                np.asarray(y_values),
                feature_names,
                params_filtered,
            ): fold_num
            for fold_num, train_idx, test_idx in fold_data
        }

        # Collect results as they complete, then sort by fold number for consistent logging
        results = []
        for future in as_completed(futures):
            fold_num, acc, y_true, y_pred, message = future.result()
            results.append((fold_num, acc, y_true, y_pred, message))

        # Sort results by fold number for deterministic logging
        results.sort(key=lambda x: x[0])

        # Process results in order
        for fold_num, acc, y_true, y_pred, message in results:
            if acc > 0 and y_true is not None and y_pred is not None:
                cv_scores.append(acc)
                all_y_true.extend(y_true)
                all_y_pred.extend(y_pred)
                log_model(f"CV Fold {fold_num} {message}")
            else:
                log_warn(f"CV Fold {fold_num}: {message}")

    return cv_scores, all_y_true, all_y_pred
