
from datetime import datetime
from pathlib import Path
from typing import Optional, Tuple

from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
import pandas as pd

from config import MAX_TRAINING_ROWS, MODEL_RANDOM_STATE, MODEL_TEST_SIZE, MODELS_DIR, RANDOM_FOREST_MODEL_FILENAME
from config.model_features import MODEL_FEATURES
from modules.common.ui.logging import (
import joblib
from modules.common.ui.logging import (
import joblib

"""Random Forest model training, loading, and saving.

This module provides functionality for training, loading, and saving Random Forest models
for trading signal prediction.
"""



    log_error,
    log_model,
    log_progress,
    log_warn,
)
from modules.random_forest.core.evaluation import evaluate_model_with_confidence
from modules.random_forest.utils.data_preparation import prepare_training_data
from modules.random_forest.utils.training import apply_smote, create_model_and_weights


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
) -> Optional[RandomForestClassifier]:
    """Train and save Random Forest model for trading signal prediction.

    Args:
        df_input (Optional[pd.DataFrame]): Input OHLCV data.
        save_model (bool): If True, save model after training.

    Returns:
        Optional[RandomForestClassifier]: Trained model or None if error.
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
    features_resampled, target_resampled = apply_smote(features, target)
    features_train, features_test, target_train, target_test = train_test_split(
        features_resampled, target_resampled, test_size=MODEL_TEST_SIZE, random_state=MODEL_RANDOM_STATE
    )
    if not isinstance(features_test, pd.DataFrame):
        features_test = pd.DataFrame(features_test, columns=pd.Index(MODEL_FEATURES))
    if not isinstance(target_test, pd.Series):
        target_test = pd.Series(target_test, name="target")
    log_progress(f"Data split into training ({len(features_train)}) and testing ({len(features_test)}) sets.")
    log_model("Computing class weights for model training...")
    model = create_model_and_weights(target_resampled)
    log_model("Training the Random Forest model...")
    try:
        model.fit(features_train, target_train)
        log_model("Model training completed successfully.")
    except (ValueError, RuntimeError) as e:
        log_error(f"An error occurred during model.fit: {e}")
        return None
    evaluate_model_with_confidence(model, features_test, target_test)
    if save_model:
        try:
            # Ensure MODELS_DIR exists
            MODELS_DIR.mkdir(parents=True, exist_ok=True)
            model_path = MODELS_DIR / RANDOM_FOREST_MODEL_FILENAME
            joblib.dump(model, model_path)
            log_model(f"Model successfully saved to: {model_path}")
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
