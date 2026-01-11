
from typing import Tuple

"""
Model Features Configuration.

Shared model features list for XGBoost and Random Forest models.
This file contains the complete list of features used for model training.
"""


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
    # Basic OHLCV data
    "open",
    "high",
    "low",
    "close",
    "volume",
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
]
