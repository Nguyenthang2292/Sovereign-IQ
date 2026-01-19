"""Random Forest model training, loading, and saving.

This module provides functionality for training, loading, and saving Random Forest models
for trading signal prediction using sklearn's RandomForestClassifier.

⚠️ NOTE: This is the sklearn-based ML Random Forest module.
NOT to be confused with modules.decision_matrix.core.random_forest_core (Pine Script pattern matching).
"""

from datetime import datetime
from pathlib import Path
from typing import Any, Optional, Tuple, Union

import joblib
import pandas as pd
from sklearn.ensemble import RandomForestClassifier

from config import MAX_TRAINING_ROWS, MODEL_RANDOM_STATE, MODEL_TEST_SIZE, MODELS_DIR, RANDOM_FOREST_MODEL_FILENAME
from config.model_features import MODEL_FEATURES
from config.random_forest import RANDOM_FOREST_TOTAL_GAP
from modules.common.ui.logging import (
    log_error,
    log_model,
    log_progress,
    log_warn,
)
from modules.random_forest.core.evaluation import evaluate_model_with_confidence
from modules.random_forest.utils.data_preparation import prepare_training_data
from modules.random_forest.utils.training import create_model_and_weights


def _time_series_split_with_gap(
    features: pd.DataFrame, target: pd.Series, test_size: float, gap: int
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
    """
    Split time-series data with gap validation to prevent data leakage.

    Args:
        features: Feature DataFrame
        target: Target Series
        test_size: Proportion of data for test set
        gap: Total gap between train and test sets (target_horizon + safety_gap)
             - target_horizon accounts for lookahead in training labels
             - safety_gap ensures true independence between train/test windows

    Returns:
        Tuple of (features_train, features_test, target_train, target_test)
    """
    total_size = len(features)
    split_idx = int(total_size * (1 - test_size))

    # Apply gap: remove last 'gap' rows from training set
    train_end = split_idx - gap
    test_start = split_idx

    # Validation
    if train_end <= 0:
        raise ValueError(
            f"Insufficient data for split with gap: need at least {gap + int(total_size * test_size)} rows, "
            f"got {total_size}"
        )
    if test_start >= total_size:
        raise ValueError(f"Test set start index ({test_start}) exceeds data size ({total_size})")

    # Split data
    features_train = features.iloc[:train_end].copy()
    features_test = features.iloc[test_start:].copy()
    target_train = target.iloc[:train_end].copy()
    target_test = target.iloc[test_start:].copy()

    return features_train, features_test, target_train, target_test


def load_random_forest_model(model_path: Optional[Path] = None) -> Optional[RandomForestClassifier]:
    """Load a trained Random Forest model from a file.

    Args:
        model_path (Optional[Path]): Path to the saved model. If None, uses default path.

    Returns:
        Optional[RandomForestClassifier]: Loaded model or None if not found or error.
    """
    if model_path is None:
        model_path = MODELS_DIR / RANDOM_FOREST_MODEL_FILENAME
    if not model_path.exists():
        log_error(f"Model file not found at: {model_path}")
        return None
    try:
        model = joblib.load(model_path)
        log_model(f"Successfully loaded model from: {model_path}")
        return model
    except (OSError, IOError, ValueError) as e:
        log_error(f"Error loading model from {model_path}: {e}")
        return None


def train_random_forest_model(
    df_input: Optional[pd.DataFrame], save_model: bool = True
) -> Optional[Union[RandomForestClassifier, Any]]:
    """Train and save Random Forest model (or ensemble) for trading signal prediction.

    Args:
        df_input (Optional[pd.DataFrame]): Input OHLCV data.
        save_model (bool): If True, save model after training.

    Returns:
        Optional[Union[RandomForestClassifier, VotingClassifier, StackingClassifier]]:
        Trained model (single or ensemble) or None if error.
    """
    # Input validation
    if df_input is None:
        log_error("Input DataFrame for training is None.")
        return None
    if not isinstance(df_input, pd.DataFrame):
        log_error(f"Input must be a pandas DataFrame, got {type(df_input)}")
        return None
    if df_input.empty:
        log_error("Input DataFrame for training is empty.")
        return None
    # Check required columns
    required_cols = ["open", "high", "low", "close", "volume"]
    missing_cols = [col for col in required_cols if col not in df_input.columns]
    if missing_cols:
        log_error(f"Missing required columns: {missing_cols}")
        return None
    df_processed = df_input.copy()
    if len(df_processed) > MAX_TRAINING_ROWS:
        log_warn(f"Dataset too large ({len(df_processed)} rows), sampling down to {MAX_TRAINING_ROWS}.")
        df_processed = df_processed.sample(n=MAX_TRAINING_ROWS, random_state=MODEL_RANDOM_STATE)
    prepared_data = prepare_training_data(df_processed)
    if prepared_data is None:
        return None
    features, target = prepared_data
    if len(target.value_counts()) < 2:
        log_error("Cannot train model with only one class present in the target variable.")
        return None

    # Apply feature selection if enabled
    feature_selector = None
    from config.random_forest import (
        RANDOM_FOREST_FEATURE_SELECTION_METHOD,
        RANDOM_FOREST_USE_FEATURE_SELECTION,
    )
    from modules.random_forest.utils.feature_selection import select_features

    if RANDOM_FOREST_USE_FEATURE_SELECTION:
        log_progress(f"Applying feature selection (method: {RANDOM_FOREST_FEATURE_SELECTION_METHOD})...")
        features, selected_feature_names, feature_selector = select_features(
            features, target, method=RANDOM_FOREST_FEATURE_SELECTION_METHOD
        )
        log_progress(f"Feature selection completed. Using {len(selected_feature_names)} features.")

    # Split with gap validation BEFORE SMOTE to prevent data leakage
    # Total gap = target_horizon (for label lookahead) + safety_gap (for independence)
    # Training labels at index N use data from N+target_horizon, so we need:
    # - target_horizon periods to account for the lookahead in training labels
    # - safety_gap additional periods to ensure true independence between train/test windows
    log_progress(f"Splitting data with gap validation (total_gap={RANDOM_FOREST_TOTAL_GAP} periods)...")
    try:
        features_train, features_test, target_train, target_test = _time_series_split_with_gap(
            features, target, test_size=MODEL_TEST_SIZE, gap=RANDOM_FOREST_TOTAL_GAP
        )
    except ValueError as e:
        log_error(f"Data split failed: {e}")
        return None

    # Apply sampling strategy only on training set to avoid data leakage
    log_progress("Applying sampling strategy to training set only...")
    from modules.random_forest.utils.training import apply_sampling

    features_resampled, target_resampled, sampling_applied = apply_sampling(features_train, target_train)
    # Ensure test set is proper DataFrame/Series (should already be, but double-check)
    if not isinstance(features_test, pd.DataFrame):
        features_test = pd.DataFrame(
            features_test, columns=features_test.columns if hasattr(features_test, "columns") else MODEL_FEATURES
        )
    if not isinstance(target_test, pd.Series):
        target_test = pd.Series(target_test, name="target")

    log_progress(
        f"Data split into training ({len(features_resampled)} after sampling) "
        f"and testing ({len(features_test)}) sets with total gap of {RANDOM_FOREST_TOTAL_GAP} periods."
    )
    log_model("Creating model with appropriate class weights...")
    model = create_model_and_weights(target_resampled, sampling_applied=sampling_applied)
    log_model("Training the Random Forest model...")
    try:
        model.fit(features_resampled, target_resampled)
        log_model("Model training completed successfully.")
    except (ValueError, RuntimeError) as e:
        log_error(f"An error occurred during model.fit: {e}")
        return None
    # Create ensemble if enabled
    from config.random_forest import RANDOM_FOREST_USE_ENSEMBLE
    from modules.random_forest.core.ensemble import create_ensemble

    if RANDOM_FOREST_USE_ENSEMBLE:
        log_progress("Creating model ensemble...")
        try:
            ensemble_model = create_ensemble(model)
            if ensemble_model != model:  # Only if ensemble was actually created
                log_model("Training ensemble model...")
                ensemble_model.fit(features_resampled, target_resampled)
                log_model("Ensemble training completed successfully.")
                model = ensemble_model
            else:
                log_warn("Ensemble creation failed, using single RandomForest model")
        except Exception as e:
            log_error(f"Error creating ensemble: {e}. Using single RandomForest model.")

    # Apply probability calibration if enabled
    from config.random_forest import (
        RANDOM_FOREST_CALIBRATION_CV,
        RANDOM_FOREST_CALIBRATION_METHOD,
        RANDOM_FOREST_USE_PROBABILITY_CALIBRATION,
    )
    from modules.random_forest.utils.calibration import calibrate_model

    if RANDOM_FOREST_USE_PROBABILITY_CALIBRATION:
        log_progress("Applying probability calibration to improve confidence thresholds...")
        try:
            model = calibrate_model(
                model,
                features_resampled,
                target_resampled,
                method=RANDOM_FOREST_CALIBRATION_METHOD,
                cv=RANDOM_FOREST_CALIBRATION_CV,
            )
            log_model("Probability calibration completed successfully.")
        except Exception as e:
            log_error(f"Error during probability calibration: {e}. Using uncalibrated model.")

    evaluate_model_with_confidence(model, features_test, target_test)

    # Apply model versioning if enabled
    from config.random_forest import RANDOM_FOREST_MODEL_VERSIONING_ENABLED
    from modules.random_forest.utils.walk_forward import ModelVersionManager

    version_manager = ModelVersionManager(enabled=RANDOM_FOREST_MODEL_VERSIONING_ENABLED)
    model_metadata = version_manager.get_model_metadata()

    if save_model:
        try:
            # Ensure MODELS_DIR exists
            MODELS_DIR.mkdir(parents=True, exist_ok=True)

            # Use versioned filename if versioning is enabled
            if RANDOM_FOREST_MODEL_VERSIONING_ENABLED:
                model_filename = f"{version_manager.get_version_string()}.joblib"
            else:
                model_filename = RANDOM_FOREST_MODEL_FILENAME

            model_path = MODELS_DIR / model_filename
            joblib.dump(model, model_path)
            log_model(f"Model successfully saved to: {model_path}")

            # Save metadata if versioning is enabled
            if RANDOM_FOREST_MODEL_VERSIONING_ENABLED and feature_selector is not None:
                metadata_path = MODELS_DIR / f"{version_manager.get_version_string()}_metadata.json"
                import json

                metadata = {
                    **model_metadata,
                    "feature_selector_type": type(feature_selector).__name__ if feature_selector else None,
                    "n_features": len(features.columns) if hasattr(features, "columns") else None,
                }
                with open(metadata_path, "w") as f:
                    json.dump(metadata, f, indent=2)
                log_model(f"Model metadata saved to: {metadata_path}")
        except (OSError, IOError) as e:
            log_error(f"Error saving model to {model_path}: {e}")
    return model


def train_and_save_global_rf_model(
    combined_df: pd.DataFrame, model_filename: Optional[str] = None
) -> Tuple[Optional[RandomForestClassifier], str]:
    """Train a global Random Forest model on a combined dataset from multiple symbols.

    Args:
        combined_df: A DataFrame containing data from multiple trading symbols.
        model_filename: An optional filename for the saved model.

    Returns:
        A tuple containing the trained model and the path where it was saved.
        Returns (None, "") on failure.
    """
    if combined_df.empty:
        log_error("The combined DataFrame for global model training is empty.")
        return None, ""
    log_model("Starting training for the global Random Forest model...")
    model = train_random_forest_model(combined_df, save_model=False)
    if model is None:
        log_error("Global model training failed.")
        return None, ""
    filename = model_filename or f"rf_model_global_{datetime.now():%Y%m%d_%H%M}.joblib"
    # Ensure MODELS_DIR exists
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    model_path = MODELS_DIR / filename
    try:
        joblib.dump(model, model_path)
        log_model(f"Global Random Forest model successfully saved to: {model_path}")
        return model, str(model_path)
    except (OSError, IOError) as e:
        log_error(f"Error saving global model to {model_path}: {e}")
        return None, ""
