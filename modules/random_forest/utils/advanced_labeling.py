"""Advanced target labeling strategies for Random Forest.

This module provides enhanced target labeling strategies:
1. Volatility-adjusted thresholds: Dynamic thresholds based on market volatility
2. Multi-horizon labeling: Targets for different timeframes
3. Trend-based labeling: Only label when trend is strong (avoid choppy markets)
"""

from typing import Dict, Optional, Tuple

import numpy as np
import pandas as pd

from config.random_forest import (
    BUY_THRESHOLD,
    RANDOM_FOREST_HORIZON_1D,
    RANDOM_FOREST_HORIZON_1H,
    RANDOM_FOREST_HORIZON_4H,
    RANDOM_FOREST_MIN_TREND_STRENGTH,
    RANDOM_FOREST_MULTI_HORIZON_ENABLED,
    RANDOM_FOREST_TARGET_HORIZON,
    RANDOM_FOREST_TREND_BASED_LABELING_ENABLED,
    RANDOM_FOREST_TREND_STRENGTH_WINDOW,
    RANDOM_FOREST_USE_VOLATILITY_ADJUSTED_THRESHOLDS,
    RANDOM_FOREST_VOLATILITY_MULTIPLIER,
    RANDOM_FOREST_VOLATILITY_WINDOW,
    SELL_THRESHOLD,
)
from modules.common.ui.logging import log_progress, log_warn


def calculate_volatility_adjusted_thresholds(
    df: pd.DataFrame, volatility_window: int = RANDOM_FOREST_VOLATILITY_WINDOW, multiplier: float = RANDOM_FOREST_VOLATILITY_MULTIPLIER
) -> Tuple[pd.Series, pd.Series]:
    """
    Calculate dynamic buy/sell thresholds based on rolling volatility.

    Args:
        df: DataFrame with 'close' column
        volatility_window: Rolling window for volatility calculation
        multiplier: Multiplier for dynamic thresholds (e.g., 0.5 means threshold = 0.5 * volatility)

    Returns:
        Tuple of (buy_threshold_series, sell_threshold_series)
    """
    if "close" not in df.columns:
        log_warn("'close' column not found, using fixed thresholds")
        return pd.Series(BUY_THRESHOLD, index=df.index), pd.Series(SELL_THRESHOLD, index=df.index)

    # Calculate rolling volatility: std of percentage returns
    returns = df["close"].pct_change()
    volatility = returns.rolling(window=volatility_window, min_periods=1).std().fillna(0.01)

    # Dynamic thresholds: multiplier * volatility
    buy_threshold = multiplier * volatility
    sell_threshold = -multiplier * volatility

    # Ensure minimum thresholds (fallback to fixed if volatility is too low)
    buy_threshold = buy_threshold.clip(lower=BUY_THRESHOLD * 0.5, upper=BUY_THRESHOLD * 2.0)
    sell_threshold = sell_threshold.clip(lower=SELL_THRESHOLD * 2.0, upper=SELL_THRESHOLD * 0.5)

    return buy_threshold, sell_threshold


def calculate_trend_strength(df: pd.DataFrame, window: int = RANDOM_FOREST_TREND_STRENGTH_WINDOW) -> pd.Series:
    """
    Calculate trend strength using SMA slope.

    Args:
        df: DataFrame with 'close' column
        window: Window for SMA calculation

    Returns:
        Series of trend strength values (0-1, where 1 = strong trend)
    """
    if "close" not in df.columns:
        log_warn("'close' column not found, returning zero trend strength")
        return pd.Series(0.0, index=df.index)

    # Calculate SMA
    sma = df["close"].rolling(window=window, min_periods=1).mean()

    # Calculate slope (rate of change) of SMA
    sma_slope = sma.diff() / sma.shift(1)

    # Normalize to 0-1 range using absolute value and rolling max
    abs_slope = sma_slope.abs()
    max_slope = abs_slope.rolling(window=window * 2, min_periods=1).max().fillna(0.01)

    # Trend strength = normalized slope (0 = no trend, 1 = strong trend)
    trend_strength = (abs_slope / max_slope).fillna(0.0).clip(lower=0.0, upper=1.0)

    return trend_strength


def create_multi_horizon_targets(
    df: pd.DataFrame,
    horizon_1h: int = RANDOM_FOREST_HORIZON_1H,
    horizon_4h: int = RANDOM_FOREST_HORIZON_4H,
    horizon_1d: int = RANDOM_FOREST_HORIZON_1D,
) -> Dict[str, pd.Series]:
    """
    Create target variables for multiple timeframes.

    Args:
        df: DataFrame with 'close' column
        horizon_1h: Periods for 1-hour equivalent horizon
        horizon_4h: Periods for 4-hour equivalent horizon
        horizon_1d: Periods for 1-day equivalent horizon

    Returns:
        Dictionary with keys 'target_1h', 'target_4h', 'target_1d' containing target Series
    """
    if "close" not in df.columns:
        log_warn("'close' column not found, returning empty targets")
        return {}

    targets = {}

    for horizon_name, horizon_periods in [("1h", horizon_1h), ("4h", horizon_4h), ("1d", horizon_1d)]:
        future_return = df["close"].shift(-horizon_periods) / df["close"] - 1
        # Use fixed thresholds for multi-horizon (can be enhanced later)
        target = np.select(
            [future_return > BUY_THRESHOLD, future_return < SELL_THRESHOLD], [1, -1], default=0
        )
        targets[f"target_{horizon_name}"] = pd.Series(target, index=df.index, name=f"target_{horizon_name}")

    return targets


def create_advanced_target(
    df: pd.DataFrame,
    use_volatility_adjusted: bool = RANDOM_FOREST_USE_VOLATILITY_ADJUSTED_THRESHOLDS,
    use_trend_based: bool = RANDOM_FOREST_TREND_BASED_LABELING_ENABLED,
    use_multi_horizon: bool = RANDOM_FOREST_MULTI_HORIZON_ENABLED,
    target_horizon: int = RANDOM_FOREST_TARGET_HORIZON,
) -> Tuple[pd.Series, Optional[Dict[str, pd.Series]]]:
    """
    Create advanced target variable with optional enhancements.

    Args:
        df: DataFrame with 'close' column and features
        use_volatility_adjusted: Enable volatility-adjusted thresholds
        use_trend_based: Enable trend-based filtering
        use_multi_horizon: Enable multi-horizon targets
        target_horizon: Primary target horizon (default from config)

    Returns:
        Tuple of (primary_target_series, optional_multi_horizon_targets_dict)
    """
    if "close" not in df.columns:
        log_warn("'close' column not found, cannot create targets")
        return pd.Series(0, index=df.index, name="target"), None

    # Calculate future return for primary target
    future_return = df["close"].shift(-target_horizon) / df["close"] - 1

    # 1. Volatility-adjusted thresholds
    if use_volatility_adjusted:
        log_progress("Calculating volatility-adjusted thresholds...")
        buy_threshold, sell_threshold = calculate_volatility_adjusted_thresholds(df)
    else:
        buy_threshold = pd.Series(BUY_THRESHOLD, index=df.index)
        sell_threshold = pd.Series(SELL_THRESHOLD, index=df.index)

    # 2. Create base target with volatility-adjusted thresholds
    target = np.select(
        [future_return > buy_threshold, future_return < sell_threshold], [1, -1], default=0
    )

    # 3. Trend-based filtering (only label when trend is strong)
    if use_trend_based:
        log_progress("Applying trend-based filtering...")
        trend_strength = calculate_trend_strength(df)
        # Only keep signals where trend strength > minimum threshold
        # Neutral (0) signals are always kept
        trend_mask = (trend_strength >= RANDOM_FOREST_MIN_TREND_STRENGTH) | (target == 0)
        target = np.where(trend_mask, target, 0)

    target_series = pd.Series(target, index=df.index, name="target")

    # 4. Multi-horizon targets (optional)
    multi_horizon_targets = None
    if use_multi_horizon:
        log_progress("Creating multi-horizon targets...")
        multi_horizon_targets = create_multi_horizon_targets(df)

    return target_series, multi_horizon_targets


__all__ = [
    "calculate_volatility_adjusted_thresholds",
    "calculate_trend_strength",
    "create_multi_horizon_targets",
    "create_advanced_target",
]
