"""
Labeling functions for XGBoost prediction model.

This module provides functions for creating directional labels (UP/DOWN/NEUTRAL)
based on future price movements, using dynamic thresholds that adapt to market
volatility and historical price patterns.
"""

import gc
import logging

import numpy as np
import pandas as pd

from config import (
    DYNAMIC_LOOKBACK_LONG_MULTIPLIER,
    DYNAMIC_LOOKBACK_MEDIUM_MULTIPLIER,
    DYNAMIC_LOOKBACK_SHORT_MULTIPLIER,
    DYNAMIC_LOOKBACK_WEIGHTS_HIGH_VOL,
    DYNAMIC_LOOKBACK_WEIGHTS_LOW_VOL,
    DYNAMIC_LOOKBACK_WEIGHTS_MEDIUM_VOL,
    ID_TO_LABEL,
    LABEL_TO_ID,
    TARGET_BASE_THRESHOLD,
    TARGET_HORIZON,
    XGBOOST_VOLATILITY_ROLLING_WINDOW,
)
from modules.xgboost_LTS.utils.cache_manager import CacheManager
from modules.xgboost_LTS.utils.numba_funcs import rolling_quantile_numba

try:
    from modules.xgboost_LTS.rust_extensions import (
        apply_directional_labels_rust,
        calculate_volatility_multiplier_rust,
        rolling_mean_rust,
        rolling_quantile_rust,
    )

    RUST_AVAILABLE = True
except ImportError:
    RUST_AVAILABLE = False


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

    # Weight Assignment
    # Assign weights based on volatility regime using np.select() for clearer code
    # np.select() is more readable than boolean masks with arithmetic operations
    weight_short = pd.Series(
        np.select(
            [is_low_vol, is_high_vol],
            [DYNAMIC_LOOKBACK_WEIGHTS_LOW_VOL[0], DYNAMIC_LOOKBACK_WEIGHTS_HIGH_VOL[0]],
            default=DYNAMIC_LOOKBACK_WEIGHTS_MEDIUM_VOL[0],
        ),
        index=volatility_multiplier.index,
    )
    weight_medium = pd.Series(
        np.select(
            [is_low_vol, is_high_vol],
            [DYNAMIC_LOOKBACK_WEIGHTS_LOW_VOL[1], DYNAMIC_LOOKBACK_WEIGHTS_HIGH_VOL[1]],
            default=DYNAMIC_LOOKBACK_WEIGHTS_MEDIUM_VOL[1],
        ),
        index=volatility_multiplier.index,
    )
    weight_long = pd.Series(
        np.select(
            [is_low_vol, is_high_vol],
            [DYNAMIC_LOOKBACK_WEIGHTS_LOW_VOL[2], DYNAMIC_LOOKBACK_WEIGHTS_HIGH_VOL[2]],
            default=DYNAMIC_LOOKBACK_WEIGHTS_MEDIUM_VOL[2],
        ),
        index=volatility_multiplier.index,
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
    """
    if RUST_AVAILABLE:
        try:
            close_vals = df["close"].values.astype(np.float64)
            atr_vals = df["ATR_14"].values.astype(np.float64) if "ATR_14" in df.columns else None

            vol_multiplier = calculate_volatility_multiplier_rust(close_vals, atr_vals)
            return pd.Series(vol_multiplier, index=df.index)
        except Exception as e:
            logging.warning(f"Rust calculate_volatility_multiplier failed, falling back to Python: {e}")

    if "ATR_14" in df.columns:
        atr_pct = (df["ATR_14"] / df["close"]).fillna(0.01)
        atr_median = atr_pct.rolling(window=50, min_periods=1).median().replace(0, 0.01)
        volatility_multiplier = (atr_pct / atr_median).fillna(2.0).clip(lower=1.5, upper=3.0)
    else:
        returns = df["close"].pct_change(fill_method=None).fillna(0)
        rolling_vol = returns.rolling(window=20, min_periods=1).std().fillna(0.01)
        vol_median = rolling_vol.rolling(window=50, min_periods=1).median().fillna(0.01)
        volatility_multiplier = (rolling_vol / vol_median).fillna(2.0).clip(lower=1.5, upper=3.0)

    return volatility_multiplier


def apply_directional_labels(df: pd.DataFrame, use_cache: bool = True) -> pd.DataFrame:
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
        use_cache: Whether to use label caching (default: True)

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

    # Label Caching (Task 3.2)
    cache_manager = None
    cache_config = None
    if use_cache:
        cache_manager = CacheManager()
        # Create a config dict that captures parameters affecting labeling
        cache_config = {
            "target_horizon": TARGET_HORIZON,
            "volatility_window": XGBOOST_VOLATILITY_ROLLING_WINDOW,
            "short_mult": DYNAMIC_LOOKBACK_SHORT_MULTIPLIER,
            "medium_mult": DYNAMIC_LOOKBACK_MEDIUM_MULTIPLIER,
            "long_mult": DYNAMIC_LOOKBACK_LONG_MULTIPLIER,
            "low_vol_weights": DYNAMIC_LOOKBACK_WEIGHTS_LOW_VOL,
            "med_vol_weights": DYNAMIC_LOOKBACK_WEIGHTS_MEDIUM_VOL,
            "high_vol_weights": DYNAMIC_LOOKBACK_WEIGHTS_HIGH_VOL,
            "base_threshold": TARGET_BASE_THRESHOLD,
        }

        cached_df = cache_manager.load_labels(df, cache_config)
        if cached_df is not None:
            return cached_df

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
    # Uses rolling window to compare current volatility to recent history
    # IMPORTANT: The rolling window is BACKWARD-LOOKING ONLY. At position i, it looks at
    # positions [i-window+1, i] (past and current data only), NOT future data.
    # This prevents using future information when determining current volatility regime.
    # The .rolling() method with default parameters is backward-looking by design.
    rolling_window = min(XGBOOST_VOLATILITY_ROLLING_WINDOW, len(df))

    # Use Rust if available, else Numba (approx 3-5x faster)
    try:
        vol_values = volatility_multiplier.values.astype(np.float64)
        if RUST_AVAILABLE:
            vol_low_rolling = pd.Series(
                rolling_quantile_rust(vol_values, rolling_window, 0.33), index=volatility_multiplier.index
            )
            vol_high_rolling = pd.Series(
                rolling_quantile_rust(vol_values, rolling_window, 0.67), index=volatility_multiplier.index
            )
        else:
            vol_low_rolling = pd.Series(
                rolling_quantile_numba(vol_values, rolling_window, 0.33), index=volatility_multiplier.index
            )
            vol_high_rolling = pd.Series(
                rolling_quantile_numba(vol_values, rolling_window, 0.67), index=volatility_multiplier.index
            )
    except Exception as e:
        logging.warning(f"Optimized rolling quantile failed, falling back to pandas: {e}")
        # Fallback to pandas if optimized versions fail
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
    # Rationale for capping:
    # 1. `len(df) - 1`: Ensures we never look back beyond available data (avoids index errors)
    #    For a 1000-row dataframe, max lookback is 999 (one less than total rows)
    # 2. `TARGET_HORIZON * 5`: Limits lookback to 5x the prediction horizon
    #    If TARGET_HORIZON=24, max lookback is 120 candles (~5 days for hourly data)
    #    This prevents using too-distant historical data that may not be relevant
    #    for current market conditions, while still providing sufficient history
    #    for volatility regime classification
    # 3. `max(1, max_lookback)`: Ensures minimum lookback of 1 period (safety check)
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
    ref_short = ref_short_low.bfill() * (1 - vol_normalized) + ref_short_high.bfill() * vol_normalized
    ref_medium = ref_medium_low.bfill() * (1 - vol_normalized) + ref_medium_high.bfill() * vol_normalized
    ref_long = ref_long_low.bfill() * (1 - vol_normalized) + ref_long_high.bfill() * vol_normalized

    # Weighted Historical Reference
    # Calculate weights based on current volatility regime using rolling thresholds
    weight_short, weight_medium, weight_long = _calculate_lookback_weights(
        volatility_multiplier, vol_low_rolling, vol_high_rolling
    )

    # Combine reference prices using volatility-adjusted weights
    historical_ref = ref_short * weight_short + ref_medium * weight_medium + ref_long * weight_long
    historical_ref = historical_ref.fillna(ref_medium)  # Fallback to medium lookback

    # Dynamic Threshold Calculation
    # Base threshold is the absolute percentage deviation from historical reference
    # Safeguard: Replace zeros in historical_ref to prevent division by zero
    # (unlikely with real price data, but possible with synthetic data or edge cases)
    historical_ref = historical_ref.replace(0, np.nan)
    historical_pct = (df["close"] - historical_ref) / historical_ref
    # Add upper bound to prevent extreme thresholds from pump/dump events
    # Lower: TARGET_BASE_THRESHOLD (default ~1%), Upper: 10% to keep labels meaningful
    base_threshold = historical_pct.abs().fillna(TARGET_BASE_THRESHOLD).clip(lower=TARGET_BASE_THRESHOLD, upper=0.1)

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
    df.loc[:, "DynamicThreshold"] = threshold_series

    # Label Assignment
    # Assign UP if price change >= threshold, DOWN if <= -threshold, else NEUTRAL
    # Use inplace operations and memory efficient constants
    # Define integer constants for direct mapping (faster than string -> map)
    UP_LABEL = "UP"
    DOWN_LABEL = "DOWN"
    NEUTRAL_LABEL = "NEUTRAL"

    # Pre-calculate IDs
    up_id = LABEL_TO_ID.get(UP_LABEL, 2)
    down_id = LABEL_TO_ID.get(DOWN_LABEL, 0)
    neutral_id = LABEL_TO_ID.get(NEUTRAL_LABEL, 1)

    # Use numpy select for direct ID assignment (avoids intermediate string series)
    conditions = [pct_change.values >= threshold_series.values, pct_change.values <= -threshold_series.values]
    choices = [up_id, down_id]

    # Assign integer targets directly
    df.loc[:, "Target"] = np.select(conditions, choices, default=neutral_id)

    # Create string labels for display/debugging (optional, could be removed for max speed)
    df.loc[:, "TargetLabel"] = df["Target"].map(ID_TO_LABEL)

    # Set NaN for rows without sufficient future data
    no_future_mask = future_close.isna()
    if no_future_mask.any():
        df.loc[no_future_mask, "TargetLabel"] = np.nan
        df.loc[no_future_mask, "Target"] = np.nan

    # Explicit garbage collection for large intermediates
    del volatility_multiplier, vol_low_rolling, vol_high_rolling
    del ref_short, ref_medium, ref_long, historical_ref
    gc.collect()

    # Save to cache (Task 3.2)
    if use_cache and cache_manager is not None and cache_config is not None:
        cache_manager.save_labels(df, df, cache_config)

    return df
