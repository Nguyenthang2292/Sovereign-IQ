"""Random Forest signal generation.

This module provides functionality for generating trading signals using trained
Random Forest models.

⚠️ IMPORTANT: This module only supports models trained with derived features
(returns_1, returns_5, log_volume, high_low_range, close_open_diff).
Models trained with raw OHLCV features (open, high, low, close, volume) are
no longer supported and will be rejected with an error message.
"""

from typing import Tuple

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier

from config import CONFIDENCE_THRESHOLD
from config.model_features import MODEL_FEATURES
from modules.common.core.indicator_engine import IndicatorConfig, IndicatorEngine, IndicatorProfile
from modules.common.ui.logging import (
    log_error,
    log_model,
    log_progress,
    log_warn,
)


def get_latest_random_forest_signal(df_market_data: pd.DataFrame, model: RandomForestClassifier) -> Tuple[str, float]:
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
    if not hasattr(model, "predict_proba"):
        log_error("Model does not have predict_proba method.")
        return "NEUTRAL", 0.0
    log_progress("Calculating features for the latest data point...")
    # Use IndicatorEngine to generate features
    try:
        engine = IndicatorEngine(IndicatorConfig.for_profile(IndicatorProfile.CORE))
        df_with_features = engine.compute_features(df_market_data)

        # Apply Advanced Feature Engineering (must match training)
        from modules.random_forest.utils.features import add_advanced_features

        df_with_features = add_advanced_features(df_with_features)

    except (ValueError, KeyError) as e:
        log_warn(f"Error computing features: {e}. Returning NEUTRAL.")
        return "NEUTRAL", 0.0
    # Get features that model was trained with
    # Sklearn models store feature names in feature_names_in_ if trained with pandas DataFrame
    if hasattr(model, "feature_names_in_") and model.feature_names_in_ is not None:
        # Model was trained with specific feature names - use only those
        model_features = list(model.feature_names_in_)
        log_progress(
            f"Model expects {len(model_features)} features: "
            f"{model_features[:5]}{'...' if len(model_features) > 5 else ''}"
        )

        # Filter to only features that exist in DataFrame and match model's expected features
        available_features = [f for f in model_features if f in df_with_features.columns]

        # Check for missing features
        missing_features = [f for f in model_features if f not in df_with_features.columns]
        if missing_features:
            log_warn(
                f"Missing features expected by model: "
                f"{missing_features[:5]}{'...' if len(missing_features) > 5 else ''}. "
                "These features will be set to 0 or NaN."
            )

        # Check for extra features (not expected by model)
        extra_features = [f for f in df_with_features.columns if f in MODEL_FEATURES and f not in model_features]
        if extra_features:
            log_warn(
                f"Extra features in DataFrame not expected by model: "
                f"{extra_features[:5]}{'...' if len(extra_features) > 5 else ''}. "
                "These will be ignored."
            )
    else:
        # Fallback: use MODEL_FEATURES if model doesn't have feature_names_in_
        # This happens if model was trained with numpy array instead of pandas DataFrame
        log_warn("Model does not have feature_names_in_ attribute. Using MODEL_FEATURES as fallback.")
        available_features = [f for f in MODEL_FEATURES if f in df_with_features.columns]

        # Check if number of features matches model's expected number
        if hasattr(model, "n_features_in_") and model.n_features_in_ is not None:
            if len(available_features) != model.n_features_in_:
                log_warn(
                    f"Feature count mismatch: DataFrame has {len(available_features)} features, "
                    f"but model expects {model.n_features_in_} features. "
                    "This may cause prediction errors."
                )

    if df_with_features.empty or not available_features:
        log_warn("Could not generate required features for the latest data. Returning NEUTRAL.")
        return "NEUTRAL", 0.0

    # Select features in the exact order expected by model
    if hasattr(model, "feature_names_in_") and model.feature_names_in_ is not None:
        # First, check if model uses any deprecated raw OHLCV features
        raw_ohlcv_features = ["open", "high", "low", "close", "volume"]
        model_raw_ohlcv = [f for f in model.feature_names_in_ if f in raw_ohlcv_features]
        if model_raw_ohlcv:
            log_error(
                f"Model uses deprecated raw OHLCV features: {model_raw_ohlcv}. "
                f"Raw OHLCV features are no longer supported. "
                f"Please retrain model with derived features (returns_1, returns_5, log_volume, high_low_range, close_open_diff)."
            )
            return "NEUTRAL", 0.0

        # Reorder and fill missing features with 0
        latest_features = pd.DataFrame(index=df_with_features.index[-1:])
        for feature in model.feature_names_in_:
            if feature in df_with_features.columns:
                latest_features[feature] = df_with_features[feature].iloc[-1:]
            else:
                # Check for critical price-derived features (required for all models)
                critical_derived_features = [
                    "returns_1",
                    "returns_5",
                    "log_volume",
                    "high_low_range",
                    "close_open_diff",
                ]

                if feature in critical_derived_features:
                    log_error(
                        f"Critical: Missing price-derived feature '{feature}' in input data during inference. "
                        f"Model requires derived features. Please ensure feature engineering is applied."
                    )
                    return "NEUTRAL", 0.0
                else:
                    # Fill missing non-critical feature with 0
                    log_warn(f"Filling missing feature '{feature}' with 0")
                    latest_features[feature] = 0.0
        # Ensure column order matches model's expected order
        latest_features = latest_features[list(model.feature_names_in_)]
    else:
        # Fallback: use available features in MODEL_FEATURES order
        latest_features = df_with_features[available_features].iloc[-1:]

    # Check for NaN values in features before prediction
    if latest_features.isna().any().any():
        nan_cols = latest_features.columns[latest_features.isna().any()].tolist()
        log_warn(
            f"NaN values detected in features: {nan_cols[:5]}{'...' if len(nan_cols) > 5 else ''}. Returning NEUTRAL."
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
        log_model(f"Latest signal: {signal} with confidence {confidence:.4f} (Class: {predicted_class})")
        return signal, confidence
    except (ValueError, RuntimeError) as e:
        log_error(f"Error during signal generation: {e}")
        return "NEUTRAL", 0.0
