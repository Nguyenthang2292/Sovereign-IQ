"""
Model Features Configuration.

Shared model features list for XGBoost and Random Forest models.
This file contains the complete list of features used for model training.
"""

from typing import Tuple

# Model Features List
# Complete list of features including technical indicators and candlestick patterns
# Define candlestick pattern names once for reuse
CANDLESTICK_PATTERN_NAMES: Tuple[str, ...] = (
    "DOJI",
    "HAMMER",
    "INVERTED_HAMMER",
    "SHOOTING_STAR",
    "MARUBOZU_BULL",
    "MARUBOZU_BEAR",
    "SPINNING_TOP",
    "DRAGONFLY_DOJI",
    "GRAVESTONE_DOJI",
    "BULLISH_ENGULFING",
    "BEARISH_ENGULFING",
    "BULLISH_HARAMI",
    "BEARISH_HARAMI",
    "HARAMI_CROSS_BULL",
    "HARAMI_CROSS_BEAR",
    "MORNING_STAR",
    "EVENING_STAR",
    "PIERCING",
    "DARK_CLOUD",
    "THREE_WHITE_SOLDIERS",
    "THREE_BLACK_CROWS",
    "THREE_INSIDE_UP",
    "THREE_INSIDE_DOWN",
    "TWEEZER_TOP",
    "TWEEZER_BOTTOM",
    "RISING_WINDOW",
    "FALLING_WINDOW",
    "TASUKI_GAP_BULL",
    "TASUKI_GAP_BEAR",
    "MAT_HOLD_BULL",
    "MAT_HOLD_BEAR",
    "ADVANCE_BLOCK",
    "STALLED_PATTERN",
    "BELT_HOLD_BULL",
    "BELT_HOLD_BEAR",
    "KICKER_BULL",
    "KICKER_BEAR",
    "HANGING_MAN",
)

MODEL_FEATURES = [
    # Price-derived features (normalized, scale-invariant)
    # These features generalize across different assets and timeframes
    "returns_1",  # 1-period return (pct_change)
    "returns_5",  # 5-period return (pct_change)
    "log_volume",  # Log-normalized volume
    "high_low_range",  # (high - low) / close (normalized range)
    "close_open_diff",  # (close - open) / open (normalized price change)
    # Technical Indicators
    "SMA_20",
    "SMA_50",
    "SMA_200",
    "RSI_9",
    "RSI_14",
    "RSI_25",
    "ATR_14",
    "MACD_12_26_9",
    "MACDh_12_26_9",
    "MACDs_12_26_9",
    "BBP_5_2.0",
    "STOCHRSIk_14_14_3_3",
    "STOCHRSId_14_14_3_3",
    "OBV",
    # Candlestick patterns; reuse list
    *CANDLESTICK_PATTERN_NAMES,
    # Advanced Features
    # Price Momentum (Rate of Change)
    "roc_3",
    "roc_5",
    "roc_10",
    "roc_20",
    # Volatility Ratios
    "atr_ratio",
    # Relative Strength (Price vs Moving Averages)
    "price_to_SMA_20",
    "price_to_SMA_50",
    "price_to_SMA_200",
    # Rolling Statistics on Returns
    "rolling_std_10",
    "rolling_std_20",
    "rolling_skew_10",
    "rolling_skew_20",
    # Lag Features (for returns_1)
    "returns_1_lag_1",
    "returns_1_lag_2",
    "returns_1_lag_3",
    # Lag Features (for RSI_14)
    "RSI_14_lag_1",
    "RSI_14_lag_2",
    "RSI_14_lag_3",
    # Lag Features (for log_volume)
    "log_volume_lag_1",
    "log_volume_lag_2",
    "log_volume_lag_3",
    # Lag Features (for atr_ratio)
    "atr_ratio_lag_1",
    "atr_ratio_lag_2",
    "atr_ratio_lag_3",
    # Time-based features (if DatetimeIndex available)
    "hour",
    "dayofweek",
    "month",
]
