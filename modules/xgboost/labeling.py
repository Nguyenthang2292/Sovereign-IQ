"""
Labeling functions for XGBoost prediction model.

This module provides functions for creating directional labels (UP/DOWN/NEUTRAL)
based on future price movements, using dynamic thresholds that adapt to market
volatility and historical price patterns.
"""

import numpy as np
import pandas as pd
from modules.config import (
    TARGET_HORIZON,
    TARGET_BASE_THRESHOLD,
    LABEL_TO_ID,
    DYNAMIC_LOOKBACK_SHORT_MULTIPLIER,
    DYNAMIC_LOOKBACK_MEDIUM_MULTIPLIER,
    DYNAMIC_LOOKBACK_LONG_MULTIPLIER,
    DYNAMIC_LOOKBACK_WEIGHTS_LOW_VOL,
    DYNAMIC_LOOKBACK_WEIGHTS_MEDIUM_VOL,
    DYNAMIC_LOOKBACK_WEIGHTS_HIGH_VOL,
)


def _calculate_lookback_weights(
    volatility_multiplier: pd.Series,
    vol_low_threshold: pd.Series,
    vol_high_threshold: pd.Series,
) -> tuple[pd.Series, pd.Series, pd.Series]:
    """
    Calculate weights for short, medium, and long lookback periods based on volatility regime.

    Classifies each row into low/medium/high volatility regime and assigns corresponding
    weights from configuration. Weights are normalized to sum to 1.0.

    Args:
        volatility_multiplier: Series of volatility multipliers (normalized ATR or rolling vol)
        vol_low_threshold: Series of rolling low volatility thresholds (33rd percentile)
        vol_high_threshold: Series of rolling high volatility thresholds (67th percentile)

    Returns:
        Tuple of (weight_short, weight_medium, weight_long) Series, all normalized to sum to 1.0
    """
    # Volatility Regime Classification
    # Classify each row into low/medium/high volatility based on rolling thresholds
    is_low_vol = volatility_multiplier < vol_low_threshold
    is_high_vol = volatility_multiplier > vol_high_threshold
    is_medium_vol = ~(is_low_vol | is_high_vol)

    # Weight Assignment
    # Assign weights based on volatility regime using vectorized operations
    weight_short = (
        is_low_vol * DYNAMIC_LOOKBACK_WEIGHTS_LOW_VOL[0] +
        is_medium_vol * DYNAMIC_LOOKBACK_WEIGHTS_MEDIUM_VOL[0] +
        is_high_vol * DYNAMIC_LOOKBACK_WEIGHTS_HIGH_VOL[0]
    )
    weight_medium = (
        is_low_vol * DYNAMIC_LOOKBACK_WEIGHTS_LOW_VOL[1] +
        is_medium_vol * DYNAMIC_LOOKBACK_WEIGHTS_MEDIUM_VOL[1] +
        is_high_vol * DYNAMIC_LOOKBACK_WEIGHTS_HIGH_VOL[1]
    )
    weight_long = (
        is_low_vol * DYNAMIC_LOOKBACK_WEIGHTS_LOW_VOL[2] +
        is_medium_vol * DYNAMIC_LOOKBACK_WEIGHTS_MEDIUM_VOL[2] +
        is_high_vol * DYNAMIC_LOOKBACK_WEIGHTS_HIGH_VOL[2]
    )
    
    # Weight Normalization
    # Ensure weights sum to 1.0 for proper weighted average calculation
    total_weight = weight_short + weight_medium + weight_long
    total_weight = total_weight.replace(0, 1.0)  # Avoid division by zero

    weight_short = weight_short / total_weight
    weight_medium = weight_medium / total_weight
    weight_long = weight_long / total_weight

    return weight_short, weight_medium, weight_long


def _calculate_volatility_multiplier(df: pd.DataFrame) -> pd.Series:
    """
    Calculate volatility multiplier based on ATR or rolling volatility of returns.

    The multiplier indicates how volatile the market is relative to recent history.
    Higher values (up to 3.0) indicate high volatility, lower values (down to 1.5) indicate low volatility.
    This multiplier is used to adjust lookback periods dynamically.

    Args:
        df: DataFrame with OHLCV data and technical indicators (must have "close" column)

    Returns:
        Series of volatility multipliers, clipped to range [1.5, 3.0]
    """
    if "ATR_14" in df.columns:
        # Primary Method: Use ATR (Average True Range) as volatility measure
        # Normalize ATR relative to price to get percentage-based volatility
        atr_pct = (df["ATR_14"] / df["close"]).fillna(0.01)
        # Compare current ATR to rolling median to get relative volatility
        atr_median = atr_pct.rolling(window=50, min_periods=1).median()
        volatility_multiplier = (atr_pct / atr_median).fillna(2.0)
        # Clip to reasonable range: 1.5x (low vol) to 3.0x (high vol) of base lookback
        volatility_multiplier = volatility_multiplier.clip(lower=1.5, upper=3.0)
    else:
        # Fallback Method: Use rolling volatility of returns
        # Calculate percentage returns and their rolling standard deviation
        returns = df["close"].pct_change(fill_method=None).fillna(0)
        rolling_vol = returns.rolling(window=20, min_periods=1).std().fillna(0.01)
        vol_median = rolling_vol.rolling(window=50, min_periods=1).median().fillna(0.01)
        volatility_multiplier = (rolling_vol / vol_median).fillna(2.0).clip(lower=1.5, upper=3.0)

    return volatility_multiplier


def apply_directional_labels(df: pd.DataFrame) -> pd.DataFrame:
    """
    Create directional labels (UP/DOWN/NEUTRAL) based on future price movement.

    Uses dynamic thresholds that adapt to market volatility and historical price patterns.
    The threshold calculation employs a multi-lookback approach:
    - Short lookback: TARGET_HORIZON * 1.5 (for low volatility)
    - Medium lookback: TARGET_HORIZON * 2.0 (baseline)
    - Long lookback: TARGET_HORIZON * 2.5 (for high volatility)

    The weights for combining these lookbacks are adjusted based on current volatility regime:
    - Low volatility: Favor short-medium lookbacks (more responsive)
    - Medium volatility: Balanced weights
    - High volatility: Favor medium-long lookbacks (more stable)

    Args:
        df: DataFrame with OHLCV data and technical indicators.
            Must contain "close" column. "ATR_14" and "ATR_RATIO_14_50" are optional.

    Returns:
        DataFrame with added columns:
        - TargetLabel: String labels ("UP", "DOWN", "NEUTRAL")
        - Target: Integer labels (0=DOWN, 1=NEUTRAL, 2=UP)
        - DynamicThreshold: Calculated threshold for each row

    Note:
        Rows without sufficient future data (last TARGET_HORIZON rows) will have NaN labels.
    """
    # Empty DataFrame Handling
    if len(df) == 0:
        df["TargetLabel"] = pd.Series(dtype=object)
        df["Target"] = pd.Series(dtype=float)
        df["DynamicThreshold"] = pd.Series(dtype=float)
        return df

    # Future Price Change Calculation
    # Shift close price forward by TARGET_HORIZON to get future price
    future_close = df["close"].shift(-TARGET_HORIZON)
    pct_change = (future_close - df["close"]) / df["close"]

    # Volatility Analysis
    # Calculate volatility multiplier to determine market regime
    volatility_multiplier = _calculate_volatility_multiplier(df)

    # Base Lookback Period Calculation
    # These are the base periods that will be adjusted by volatility multiplier
    base_short = TARGET_HORIZON * DYNAMIC_LOOKBACK_SHORT_MULTIPLIER
    base_medium = TARGET_HORIZON * DYNAMIC_LOOKBACK_MEDIUM_MULTIPLIER
    base_long = TARGET_HORIZON * DYNAMIC_LOOKBACK_LONG_MULTIPLIER
    
    # Rolling Volatility Thresholds
    # Calculate rolling quantiles to define volatility regimes without data leakage
    # Uses rolling window (max 500 periods) to compare current volatility to recent history
    # This prevents using future information when determining current volatility regime
    rolling_window = min(500, len(df))
    vol_low_rolling = volatility_multiplier.rolling(window=rolling_window, min_periods=1).quantile(0.33)
    vol_high_rolling = volatility_multiplier.rolling(window=rolling_window, min_periods=1).quantile(0.67)

    # Forward fill NaN values at the beginning (appropriate for time series)
    # Propagates first valid value forward to handle initial periods
    vol_low_rolling = vol_low_rolling.ffill().fillna(1.5)
    vol_high_rolling = vol_high_rolling.ffill().fillna(2.5)
    
    # Fixed Volatility Anchors for Vectorization
    # Use fixed anchors (1.5 and 3.0) to enable vectorized shift operations
    # Dynamic lookbacks per row would require loops, which is much slower
    # These anchors represent the typical range of volatility multipliers
    anchor_low = 1.5
    anchor_high = 3.0

    # Lookback Period Calculation
    # Calculate lookback periods for both low and high volatility scenarios
    # Cap maximum lookback to prevent excessive historical references
    max_lookback = min(len(df) - 1, int(TARGET_HORIZON * 5))
    max_lookback = max(1, max_lookback)
    
    lookback_short_low = max(1, min(int(base_short * anchor_low), max_lookback))
    lookback_short_high = max(1, min(int(base_short * anchor_high), max_lookback))
    lookback_medium_low = max(1, min(int(base_medium * anchor_low), max_lookback))
    lookback_medium_high = max(1, min(int(base_medium * anchor_high), max_lookback))
    lookback_long_low = max(1, min(int(base_long * anchor_low), max_lookback))
    lookback_long_high = max(1, min(int(base_long * anchor_high), max_lookback))
    
    # Historical Reference Price Calculation
    # Get reference prices for both low and high volatility scenarios
    # These will be interpolated based on current volatility
    ref_short_low = df["close"].shift(lookback_short_low)
    ref_short_high = df["close"].shift(lookback_short_high)
    ref_medium_low = df["close"].shift(lookback_medium_low)
    ref_medium_high = df["close"].shift(lookback_medium_high)
    ref_long_low = df["close"].shift(lookback_long_low)
    ref_long_high = df["close"].shift(lookback_long_high)

    # Volatility-Based Interpolation
    # Normalize current volatility to [0, 1] range based on fixed anchors
    # This allows smooth interpolation between low and high volatility scenarios
    vol_normalized = (volatility_multiplier - anchor_low) / (anchor_high - anchor_low + 1e-8)
    vol_normalized = vol_normalized.clip(0, 1)

    # Interpolate reference prices between low and high volatility scenarios
    # Use backward fill to handle NaN values at the beginning
    ref_short = (
        ref_short_low.bfill() * (1 - vol_normalized) +
        ref_short_high.bfill() * vol_normalized
    )
    ref_medium = (
        ref_medium_low.bfill() * (1 - vol_normalized) +
        ref_medium_high.bfill() * vol_normalized
    )
    ref_long = (
        ref_long_low.bfill() * (1 - vol_normalized) +
        ref_long_high.bfill() * vol_normalized
    )
    
    # Weighted Historical Reference
    # Calculate weights based on current volatility regime using rolling thresholds
    weight_short, weight_medium, weight_long = _calculate_lookback_weights(
        volatility_multiplier, vol_low_rolling, vol_high_rolling
    )

    # Combine reference prices using volatility-adjusted weights
    historical_ref = (
        ref_short * weight_short +
        ref_medium * weight_medium +
        ref_long * weight_long
    )
    historical_ref = historical_ref.fillna(ref_medium)  # Fallback to medium lookback

    # Dynamic Threshold Calculation
    # Base threshold is the absolute percentage deviation from historical reference
    historical_pct = (df["close"] - historical_ref) / historical_ref
    base_threshold = (
        historical_pct.abs()
        .fillna(TARGET_BASE_THRESHOLD)
        .clip(lower=TARGET_BASE_THRESHOLD)
    )

    # ATR Ratio Adjustment
    # Adjust threshold based on current volatility (ATR ratio)
    # Higher ATR ratio = higher volatility = larger threshold needed
    atr_ratio = (
        df.get("ATR_RATIO_14_50", pd.Series(1.0, index=df.index))
        .replace([np.inf, -np.inf], np.nan)
        .fillna(1.0)
        .clip(lower=0.5, upper=2.0)
    )
    threshold_series = (base_threshold * atr_ratio).clip(lower=TARGET_BASE_THRESHOLD)
    df["DynamicThreshold"] = threshold_series

    # Label Assignment
    # Assign UP if price change >= threshold, DOWN if <= -threshold, else NEUTRAL
    df["TargetLabel"] = np.where(
        pct_change >= threshold_series,
        "UP",
        np.where(pct_change <= -threshold_series, "DOWN", "NEUTRAL"),
    )
    # Set NaN for rows without sufficient future data
    df.loc[future_close.isna(), "TargetLabel"] = np.nan
    # Convert string labels to integer IDs
    df["Target"] = df["TargetLabel"].map(LABEL_TO_ID)
    return df
