"""
Labeling functions for xgboost_prediction_main.py
"""

import numpy as np
import pandas as pd
from .config import (
    TARGET_HORIZON,
    TARGET_BASE_THRESHOLD,
    LABEL_TO_ID,
    DYNAMIC_LOOKBACK_SHORT_MULTIPLIER,
    DYNAMIC_LOOKBACK_MEDIUM_MULTIPLIER,
    DYNAMIC_LOOKBACK_LONG_MULTIPLIER,
    DYNAMIC_LOOKBACK_VOL_LOW_THRESHOLD,
    DYNAMIC_LOOKBACK_VOL_HIGH_THRESHOLD,
    DYNAMIC_LOOKBACK_WEIGHTS_LOW_VOL,
    DYNAMIC_LOOKBACK_WEIGHTS_MEDIUM_VOL,
    DYNAMIC_LOOKBACK_WEIGHTS_HIGH_VOL,
)


def apply_directional_labels(df: pd.DataFrame) -> pd.DataFrame:
    """
    Labels each row as UP/DOWN/NEUTRAL based on future price movement.
    
    Dynamic threshold is calculated using adaptive historical lookback period
    that scales with market volatility and recent price movements.
    """
    future_close = df["close"].shift(-TARGET_HORIZON)
    pct_change = (future_close - df["close"]) / df["close"]

    # Dynamic lookback calculation based on volatility and recent movements
    # Option 1: Volatility-based adaptive lookback (using ATR if available)
    if "ATR_14" in df.columns:
        # Normalize ATR relative to price to get volatility measure
        atr_pct = (df["ATR_14"] / df["close"]).fillna(0.01)
        # Scale lookback: higher volatility = longer lookback (1.5x to 3x TARGET_HORIZON)
        volatility_multiplier = (atr_pct / atr_pct.rolling(window=50, min_periods=1).median()).fillna(2.0)
        volatility_multiplier = volatility_multiplier.clip(lower=1.5, upper=3.0)
    else:
        # Fallback: Use rolling volatility of returns
        returns = df["close"].pct_change().fillna(0)
        rolling_vol = returns.rolling(window=20, min_periods=1).std().fillna(0.01)
        vol_median = rolling_vol.rolling(window=50, min_periods=1).median().fillna(0.01)
        volatility_multiplier = (rolling_vol / vol_median).fillna(2.0).clip(lower=1.5, upper=3.0)
    
    # Use multiple lookback periods and take weighted average based on volatility
    # This is more efficient than variable shift and captures different time horizons
    lookback_short = TARGET_HORIZON * DYNAMIC_LOOKBACK_SHORT_MULTIPLIER
    lookback_medium = TARGET_HORIZON * DYNAMIC_LOOKBACK_MEDIUM_MULTIPLIER
    lookback_long = TARGET_HORIZON * DYNAMIC_LOOKBACK_LONG_MULTIPLIER
    
    ref_short = df["close"].shift(int(lookback_short))
    ref_medium = df["close"].shift(int(lookback_medium))
    ref_long = df["close"].shift(int(lookback_long))
    
    # Weight based on volatility: higher volatility favors longer lookback
    # Use configurable weights for different volatility regimes
    weight_short = np.where(
        volatility_multiplier < DYNAMIC_LOOKBACK_VOL_LOW_THRESHOLD,
        DYNAMIC_LOOKBACK_WEIGHTS_LOW_VOL[0],
        np.where(
            volatility_multiplier > DYNAMIC_LOOKBACK_VOL_HIGH_THRESHOLD,
            DYNAMIC_LOOKBACK_WEIGHTS_HIGH_VOL[0],
            DYNAMIC_LOOKBACK_WEIGHTS_MEDIUM_VOL[0],
        ),
    )
    weight_medium = np.where(
        volatility_multiplier < DYNAMIC_LOOKBACK_VOL_LOW_THRESHOLD,
        DYNAMIC_LOOKBACK_WEIGHTS_LOW_VOL[1],
        np.where(
            volatility_multiplier > DYNAMIC_LOOKBACK_VOL_HIGH_THRESHOLD,
            DYNAMIC_LOOKBACK_WEIGHTS_HIGH_VOL[1],
            DYNAMIC_LOOKBACK_WEIGHTS_MEDIUM_VOL[1],
        ),
    )
    weight_long = np.where(
        volatility_multiplier < DYNAMIC_LOOKBACK_VOL_LOW_THRESHOLD,
        DYNAMIC_LOOKBACK_WEIGHTS_LOW_VOL[2],
        np.where(
            volatility_multiplier > DYNAMIC_LOOKBACK_VOL_HIGH_THRESHOLD,
            DYNAMIC_LOOKBACK_WEIGHTS_HIGH_VOL[2],
            DYNAMIC_LOOKBACK_WEIGHTS_MEDIUM_VOL[2],
        ),
    )
    
    # Normalize weights to ensure they sum to 1
    total_weight = weight_short + weight_medium + weight_long
    weight_short = weight_short / total_weight
    weight_medium = weight_medium / total_weight
    weight_long = weight_long / total_weight
    
    # Calculate weighted historical reference
    historical_ref = (
        ref_short * weight_short + 
        ref_medium * weight_medium + 
        ref_long * weight_long
    )
    historical_ref = historical_ref.fillna(ref_medium)  # Fallback to medium lookback
    
    historical_pct = (df["close"] - historical_ref) / historical_ref
    base_threshold = (
        historical_pct.abs()
        .fillna(TARGET_BASE_THRESHOLD)
        .clip(lower=TARGET_BASE_THRESHOLD)
    )
    atr_ratio = (
        df.get("ATR_RATIO_14_50", pd.Series(1.0, index=df.index))
        .replace([np.inf, -np.inf], np.nan)
        .fillna(1.0)
        .clip(lower=0.5, upper=2.0)
    )
    threshold_series = (base_threshold * atr_ratio).clip(lower=TARGET_BASE_THRESHOLD)
    df["DynamicThreshold"] = threshold_series

    df["TargetLabel"] = np.where(
        pct_change >= threshold_series,
        "UP",
        np.where(pct_change <= -threshold_series, "DOWN", "NEUTRAL"),
    )
    df.loc[future_close.isna(), "TargetLabel"] = np.nan
    df["Target"] = df["TargetLabel"].map(LABEL_TO_ID)
    return df
