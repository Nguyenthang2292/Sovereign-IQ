from typing import List

import numpy as np
import pandas as pd


def add_price_derived_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add price-derived features that are required by MODEL_FEATURES.
    
    These features are scale-invariant and generalize across different assets:
    - returns_1: 1-period return (pct_change)
    - returns_5: 5-period return (pct_change)
    - log_volume: Log-normalized volume (handles 0 volume gracefully)
    - high_low_range: Normalized range (high - low) / close
    - close_open_diff: Normalized price change (close - open) / open
    
    Args:
        df: Input DataFrame with OHLCV columns (open, high, low, close, volume)
    
    Returns:
        DataFrame with price-derived features added
    """
    df = df.copy()
    
    # Validate required columns
    required_cols = ["open", "high", "low", "close", "volume"]
    missing_cols = [col for col in required_cols if col not in df.columns]
    if missing_cols:
        raise ValueError(
            f"Missing required OHLCV columns for price-derived features: {missing_cols}. "
            f"Required columns: {required_cols}"
        )
    
    # 1-period return (pct_change)
    if "returns_1" not in df.columns:
        df["returns_1"] = df["close"].pct_change(1)
    
    # 5-period return (pct_change)
    if "returns_5" not in df.columns:
        df["returns_5"] = df["close"].pct_change(5)
    
    # Log-normalized volume (handles 0 volume gracefully with log1p)
    if "log_volume" not in df.columns:
        df["log_volume"] = np.log1p(df["volume"])
    
    # Normalized range: (high - low) / close
    # Handles division by zero by using np.where
    if "high_low_range" not in df.columns:
        df["high_low_range"] = np.where(
            df["close"] != 0,
            (df["high"] - df["low"]) / df["close"],
            0.0
        )
    
    # Normalized price change: (close - open) / open
    # Handles division by zero by using np.where
    if "close_open_diff" not in df.columns:
        df["close_open_diff"] = np.where(
            df["open"] != 0,
            (df["close"] - df["open"]) / df["open"],
            0.0
        )
    
    return df


def add_advanced_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add advanced feature engineering for Random Forest models.

    Includes:
    - Price momentum (ROC)
    - Volatility ratios (ATR/Close)
    - Relative strength (Close / Moving Average)
    - Lag features
    - Rolling statistics
    - Time-based features

    Args:
        df: Input DataFrame with basic indicators already calculated

    Returns:
        DataFrame with additional features
    """
    df = df.copy()
    
    # FIRST: Add price-derived features (required by MODEL_FEATURES)
    # These must be created before advanced features that depend on them
    df = add_price_derived_features(df)

    # 1. Price Momentum (Rate of Change)
    for period in [3, 5, 10, 20]:
        df[f"roc_{period}"] = df["close"].pct_change(period)

    # 2. Volatility Ratios
    # Normalize ATR by price to make it scale-invariant
    if "ATR_14" in df.columns:
        df["atr_ratio"] = df["ATR_14"] / df["close"]

    # 3. Relative Strength (Price vs Moving Averages)
    for period in [20, 50, 200]:
        sma_col = f"SMA_{period}"
        if sma_col in df.columns:
            # Ratio of price to SMA ( > 1 means above MA)
            df[f"price_to_{sma_col}"] = df["close"] / df[sma_col]

    # 4. Rolling Statistics on Returns
    # Capture changing volatility and distribution shape
    # Note: returns_1 is already created by add_price_derived_features()
    for window in [10, 20]:
        df[f"rolling_std_{window}"] = df["returns_1"].rolling(window).std()
        df[f"rolling_skew_{window}"] = df["returns_1"].rolling(window).skew()

    # 5. Lag Features
    # Provide temporal context (what happened t-1, t-2...)
    features_to_lag = ["returns_1", "RSI_14", "log_volume", "atr_ratio"]
    for feat in features_to_lag:
        if feat in df.columns:
            for lag in [1, 2, 3]:
                df[f"{feat}_lag_{lag}"] = df[feat].shift(lag)

    # 6. Time-based features
    # Capture seasonality and intraday patterns
    # Require DatetimeIndex
    if isinstance(df.index, pd.DatetimeIndex):
        df["hour"] = df.index.hour
        df["dayofweek"] = df.index.dayofweek
        df["month"] = df.index.month

    return df


def get_enhanced_feature_names(df_columns: List[str]) -> List[str]:
    """
    Identify which of the enhanced features are present in the columns.
    Useful for filtering features for training.
    """
    # Base prefixes/suffixes for enhanced features
    enhanced_patterns = [
        "roc_",
        "atr_ratio",
        "price_to_SMA_",
        "rolling_std_",
        "rolling_skew_",
        "_lag_",
        "hour",
        "dayofweek",
        "month",
    ]

    enhanced_features = []
    for col in df_columns:
        if any(pattern in col for pattern in enhanced_patterns):
            enhanced_features.append(col)

    return enhanced_features
