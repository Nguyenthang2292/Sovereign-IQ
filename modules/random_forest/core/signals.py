"""Random Forest signal generation.

This module provides functionality for generating trading signals using trained
Random Forest models.
"""

from typing import Tuple
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier

from config import (
    CONFIDENCE_THRESHOLD,
    MODEL_FEATURES,
)
from modules.common.core.indicator_engine import IndicatorEngine, IndicatorConfig, IndicatorProfile
from modules.common.ui.logging import (
    log_info,
    log_model,
    log_error,
    log_warn,
    log_progress,
)


def get_latest_random_forest_signal(
    df_market_data: pd.DataFrame, model: RandomForestClassifier
) -> Tuple[str, float]:
    """Generate a trading signal for the most recent data point.

    Args:
        df_market_data: A DataFrame containing the latest market data.
        model: The trained RandomForestClassifier model.

    Returns:
        A tuple containing the signal string ('LONG', 'SHORT', 'NEUTRAL')
        and the confidence of the prediction.
    """
    # Input validation
    if df_market_data is None:
        log_error("Market data DataFrame is None.")
        return "NEUTRAL", 0.0
    if not isinstance(df_market_data, pd.DataFrame):
        log_error(f"Market data must be a pandas DataFrame, got {type(df_market_data)}")
        return "NEUTRAL", 0.0
    if df_market_data.empty:
        log_warn("Market data for signal generation is empty.")
        return "NEUTRAL", 0.0
    if model is None:
        log_error("Model is None.")
        return "NEUTRAL", 0.0
    if not hasattr(model, 'predict_proba'):
        log_error("Model does not have predict_proba method.")
        return "NEUTRAL", 0.0
    log_progress("Calculating features for the latest data point...")
    # Use IndicatorEngine to generate features
    try:
        engine = IndicatorEngine(IndicatorConfig.for_profile(IndicatorProfile.CORE))
        df_with_features = engine.compute_features(df_market_data)
    except (ValueError, KeyError) as e:
        log_warn(
            f"Error computing features: {e}. Returning NEUTRAL."
        )
        return "NEUTRAL", 0.0
    # Filter MODEL_FEATURES to only include features that actually exist in the DataFrame
    available_features = [f for f in MODEL_FEATURES if f in df_with_features.columns]
    if df_with_features.empty or not available_features:
        log_warn(
            "Could not generate features for the latest data. "
            "Returning NEUTRAL."
        )
        return "NEUTRAL", 0.0
    latest_features = df_with_features[available_features].iloc[-1:]
    
    # Check for NaN values in features before prediction
    if latest_features.isna().any().any():
        nan_cols = latest_features.columns[latest_features.isna().any()].tolist()
        log_warn(
            f"NaN values detected in features: {nan_cols[:5]}{'...' if len(nan_cols) > 5 else ''}. "
            "Returning NEUTRAL."
        )
        return "NEUTRAL", 0.0
    
    try:
        prediction_proba = model.predict_proba(latest_features)[0]
        confidence = max(prediction_proba)
        predicted_class = model.classes_[np.argmax(prediction_proba)]
        if confidence < CONFIDENCE_THRESHOLD:
            signal = "NEUTRAL"
        elif predicted_class == 1:
            signal = "LONG"
        elif predicted_class == -1:
            signal = "SHORT"
        else:
            signal = "NEUTRAL"
        log_model(
            f"Latest signal: {signal} with confidence {confidence:.4f} "
            f"(Class: {predicted_class})"
        )
        return signal, confidence
    except (ValueError, RuntimeError) as e:
        log_error(
            f"Error during signal generation: {e}"
        )
        return "NEUTRAL", 0.0

