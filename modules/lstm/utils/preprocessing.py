
from typing import List, Optional, Tuple, Union

from sklearn.preprocessing import MinMaxScaler, StandardScaler
import numpy as np
import pandas as pd

from config.lstm import (

from config.lstm import (

"""
Data preprocessing utilities for CNN-LSTM models.
"""



    KALMAN_OBSERVATION_VARIANCE,
    KALMAN_PROCESS_VARIANCE,
    NEUTRAL_ZONE_LSTM,
    TARGET_THRESHOLD_LSTM,
    WINDOW_SIZE_LSTM,
)
from config.model_features import MODEL_FEATURES
from modules.common.ui.logging import log_error, log_model, log_warn
from modules.lstm.core.create_balanced_target import create_balanced_target
from modules.lstm.utils.indicator_features import generate_indicator_features
from modules.lstm.utils.kalman_filter import apply_kalman_to_ohlc, validate_kalman_params


def preprocess_cnn_lstm_data(
    df_input: pd.DataFrame,
    look_back: int = WINDOW_SIZE_LSTM,
    output_mode: str = "classification",
    scaler_type: str = "minmax",
    use_kalman_filter: bool = False,
    kalman_params: Optional[dict] = None,
) -> Tuple[np.ndarray, np.ndarray, Union[MinMaxScaler, StandardScaler], List[str]]:
    """
    Preprocess data for CNN-LSTM model with sliding window approach.

    Args:
        df_input: Input DataFrame containing price data
        look_back: Number of time steps to look back for sequence creation
        output_mode: 'classification' for signal prediction or 'regression' for return prediction
        scaler_type: Scaling method ('minmax' or 'standard')
        use_kalman_filter: Enable Kalman Filter preprocessing to smooth OHLC data before generating indicators
        kalman_params: Optional dictionary of Kalman Filter parameters. If None, uses config defaults.
                      Valid keys: 'process_variance', 'observation_variance', 'initial_state', 'initial_uncertainty'

    Returns:
        X_sequences: Feature sequences array
        y_targets: Target values array
        fitted_scaler: Fitted scaler for feature normalization
        feature_names: List of features used in model
    """

    def _create_empty_scaler(scaler_type: str) -> Union[MinMaxScaler, StandardScaler]:
        """
        Create an empty scaler based on the scaler_type parameter.

        Args:
            scaler_type: Scaling method ('minmax' or 'standard')

        Returns:
            MinMaxScaler if scaler_type == 'minmax', else StandardScaler
        """
        return MinMaxScaler() if scaler_type == "minmax" else StandardScaler()

    log_model(f"Starting CNN-LSTM preprocessing: {df_input.shape} rows, lookback={look_back}, mode={output_mode}")

    if df_input.empty or len(df_input) < look_back + 10:
        log_error(f"Insufficient data: {len(df_input)} rows, need at least {look_back + 10}")
        return np.array([]), np.array([]), _create_empty_scaler(scaler_type), []

    # Apply Kalman Filter if enabled
    df_for_indicators = df_input.copy()
    if use_kalman_filter:
        # Validate Kalman parameters
        if kalman_params is not None and not validate_kalman_params(kalman_params):
            log_warn("Invalid Kalman Filter parameters, using defaults")
            kalman_params = None

        # Use provided params or config defaults
        if kalman_params is None:
            kalman_params = {
                "process_variance": KALMAN_PROCESS_VARIANCE,
                "observation_variance": KALMAN_OBSERVATION_VARIANCE,
            }
        else:
            # Merge with defaults for missing keys
            kalman_params = {
                "process_variance": kalman_params.get("process_variance", KALMAN_PROCESS_VARIANCE),
                "observation_variance": kalman_params.get("observation_variance", KALMAN_OBSERVATION_VARIANCE),
            }

        try:
            df_for_indicators = apply_kalman_to_ohlc(df_input.copy(), **kalman_params)
            log_model("Kalman Filter applied to OHLC data before generating indicators")
        except Exception as e:
            log_error(f"Error applying Kalman Filter: {e}, using original data")
            df_for_indicators = df_input.copy()

    # Calculate technical features from (possibly smoothed) OHLC data
    df = generate_indicator_features(df_for_indicators)
    if df.empty:
        log_error("Feature calculation returned empty DataFrame")
        return np.array([]), np.array([]), _create_empty_scaler(scaler_type), []

    # Create targets based on mode
    if output_mode == "classification":
        try:
            df = create_balanced_target(df, threshold=TARGET_THRESHOLD_LSTM, neutral_zone=NEUTRAL_ZONE_LSTM)
        except Exception as e:
            log_error(f"Error in create_balanced_target: {e}")
            return np.array([]), np.array([]), _create_empty_scaler(scaler_type), []
        if "Target" not in df.columns:
            log_error("Classification target creation failed")
            return np.array([]), np.array([]), _create_empty_scaler(scaler_type), []
    else:
        try:
            if "close" not in df.columns:
                raise KeyError("'close' column is missing for regression mode")
            df["Target"] = df["close"].pct_change().shift(-1)
        except KeyError as ke:
            log_error(f"Regression mode requires 'close' column: {ke}")
            return np.array([]), np.array([]), _create_empty_scaler(scaler_type), []
        except Exception as e:
            log_error(f"Error creating target for regression: {e}")
            return np.array([]), np.array([]), _create_empty_scaler(scaler_type), []
        if "Target" not in df.columns:
            log_error("Regression target creation failed")
            return np.array([]), np.array([]), _create_empty_scaler(scaler_type), []

    # Prepare feature matrix
    available_features = [col for col in MODEL_FEATURES if col in df.columns]
    if not available_features:
        log_error(f"No valid features found from {MODEL_FEATURES}")
        return np.array([]), np.array([]), _create_empty_scaler(scaler_type), []

    # Check for invalid values before processing
    feature_df = df[available_features].copy()
    invalid_mask = feature_df.isna().any(axis=1) | feature_df.isin([np.inf, -np.inf]).any(axis=1)

    if invalid_mask.any():
        n_invalid = invalid_mask.sum()
        n_total = len(feature_df)
        invalid_pct = 100.0 * n_invalid / n_total

        # Identify which features have invalid values
        invalid_features = []
        for col in available_features:
            col_invalid = feature_df[col].isna().sum() + feature_df[col].isin([np.inf, -np.inf]).sum()
            if col_invalid > 0:
                invalid_features.append(f"{col}({col_invalid})")

        log_warn(f"Found {n_invalid}/{n_total} rows ({invalid_pct:.2f}%) with invalid values")
        if invalid_features:
            log_warn(f"Features with invalid values: {', '.join(invalid_features[:10])}")  # Show first 10

        # Drop rows with invalid values instead of replacing to avoid bias
        df = df[~invalid_mask].reset_index(drop=True)
        feature_df = feature_df[~invalid_mask]

        if len(df) < look_back + 10:
            log_error(f"After dropping invalid rows, insufficient data: {len(df)} rows, need at least {look_back + 10}")
            return np.array([]), np.array([]), _create_empty_scaler(scaler_type), []

        log_model(f"Dropped {n_invalid} rows with invalid values, {len(df)} rows remaining")

    features = feature_df.values

    # Scale features
    scaler = MinMaxScaler() if scaler_type == "minmax" else StandardScaler()
    try:
        scaled_features = scaler.fit_transform(features)
    except Exception as e:
        log_error(f"Feature scaling failed: {e}")
        return np.array([]), np.array([]), _create_empty_scaler(scaler_type), []

    # Create sliding window sequences (vectorized)
    target_values = df["Target"].values
    max_len = min(len(scaled_features), len(target_values))
    n_samples = max_len - look_back + 1

    if n_samples <= 0:
        log_error("No valid sequences can be created")
        return np.array([]), np.array([]), scaler, available_features

    # Vectorized sequence creation using sliding window view
    X_sequences = np.lib.stride_tricks.sliding_window_view(
        scaled_features[:max_len], window_shape=look_back, axis=0
    ).reshape(n_samples, look_back, scaled_features.shape[1])

    # Extract targets corresponding to the end of each sequence
    # Sequence i spans [i:i+look_back], so its last timestep is at i+look_back-1
    # For regression: target_values[i+look_back-1] is the return from i+look_back-1 to i+look_back
    # For classification: target_values[i+look_back-1] is the label for timestep i+look_back-1
    y_targets = target_values[look_back - 1 : max_len]

    # Final safety check: filter out any sequences with invalid values
    # (should be rare after dropping invalid rows, but can occur with edge cases in scaling)
    valid_mask = ~(np.isnan(X_sequences).any(axis=(1, 2)) | np.isinf(X_sequences).any(axis=(1, 2)))

    if not valid_mask.any():
        log_error("No valid sequences created after filtering")
        return np.array([]), np.array([]), scaler, available_features

    n_filtered = (~valid_mask).sum()
    if n_filtered > 0:
        log_warn(f"Filtered out {n_filtered} additional sequences with invalid values after scaling")

    X_sequences = X_sequences[valid_mask]
    y_targets = y_targets[valid_mask]

    # Log preprocessing results
    log_model(f"Preprocessing complete: {len(X_sequences)} sequences, shape {X_sequences.shape}")
    if output_mode == "classification":
        unique, counts = np.unique(y_targets, return_counts=True)
        log_model(f"Target distribution: {dict(zip(unique, counts))}")
    else:
        log_model(f"Target range: [{np.min(y_targets):.4f}, {np.max(y_targets):.4f}]")

    return X_sequences, y_targets, scaler, available_features
