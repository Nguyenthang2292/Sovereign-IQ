"""
Feature engineering functions for XGBoost module with Rust acceleration.
"""

import logging
from typing import Any, Iterable, Optional

import numpy as np
import pandas as pd

from config import MODEL_FEATURES

try:
    from modules.xgboost_LTS.rust_extensions import (
        add_advanced_features_rust,
        calculate_all_features_rust,
        add_price_derived_features_rust,
    )

    RUST_AVAILABLE = True
except ImportError:
    RUST_AVAILABLE = False
    calculate_all_features_rust = None


def add_price_derived_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add price-derived features that are required by MODEL_FEATURES.

    Includes:
    - returns_1: 1-period return
    - returns_5: 5-period return
    - log_volume: Log-normalized volume
    - high_low_range: Normalized range (high - low) / close
    - close_open_diff: Normalized price change (close - open) / open
    """
    df = df.copy()

    required_cols = ["open", "high", "low", "close", "volume"]
    missing_cols = [col for col in required_cols if col not in df.columns]
    if missing_cols:
        raise ValueError(f"Missing required OHLCV columns: {missing_cols}")

    if RUST_AVAILABLE:
        try:
            features = add_price_derived_features_rust(
                df["open"].values.astype(np.float64),
                df["high"].values.astype(np.float64),
                df["low"].values.astype(np.float64),
                df["close"].values.astype(np.float64),
                df["volume"].values.astype(np.float64),
            )
            for key, val in features.items():
                if key not in df.columns:
                    df[key] = val
            if "close_open_diff" in df.columns:
                df["close_open_diff"] = np.where(
                    df["open"] != 0,
                    (df["close"] - df["open"]) / df["open"],
                    0.0,
                )
            if "returns_1" in df.columns:
                df["returns_1"] = pd.Series(df["returns_1"]).fillna(0.0)
            if "returns_5" in df.columns:
                df["returns_5"] = pd.Series(df["returns_5"]).fillna(0.0)
            return df
        except Exception as e:
            logging.warning(f"Rust price-derived features failed: {e}")

    # Fallback to Python
    if "returns_1" not in df.columns:
        df["returns_1"] = df["close"].pct_change(1).fillna(0.0)
    if "returns_5" not in df.columns:
        df["returns_5"] = df["close"].pct_change(5).fillna(0.0)
    if "log_volume" not in df.columns:
        df["log_volume"] = np.log1p(df["volume"])
    if "high_low_range" not in df.columns:
        df["high_low_range"] = np.where(df["close"] != 0, (df["high"] - df["low"]) / df["close"], 0.0)
    if "close_open_diff" not in df.columns:
        df["close_open_diff"] = np.where(df["open"] != 0, (df["close"] - df["open"]) / df["open"], 0.0)

    return df


def add_advanced_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add advanced feature engineering for XGBoost models.
    """
    df = df.copy()
    df = add_price_derived_features(df)

    if RUST_AVAILABLE and calculate_all_features_rust is not None:
        try:
            atr_14 = df["ATR_14"].values.astype(np.float64) if "ATR_14" in df.columns else None
            rsi_14 = df["RSI_14"].values.astype(np.float64) if "RSI_14" in df.columns else None
            sma_20 = df["SMA_20"].values.astype(np.float64) if "SMA_20" in df.columns else None
            sma_50 = df["SMA_50"].values.astype(np.float64) if "SMA_50" in df.columns else None
            sma_200 = df["SMA_200"].values.astype(np.float64) if "SMA_200" in df.columns else None

            features = calculate_all_features_rust(
                df["open"].values.astype(np.float64),
                df["high"].values.astype(np.float64),
                df["low"].values.astype(np.float64),
                df["close"].values.astype(np.float64),
                df["volume"].values.astype(np.float64),
                atr_14,
                rsi_14,
                sma_20,
                sma_50,
                sma_200,
            )

            for key, val in features.items():
                df[key] = val

            if isinstance(df.index, pd.DatetimeIndex):
                df["hour"] = df.index.hour
                df["dayofweek"] = df.index.dayofweek
                df["month"] = df.index.month

            return df
        except Exception as e:
            logging.warning(f"Rust batch feature calculation failed: {e}")

    if RUST_AVAILABLE:
        try:
            atr_14 = df["ATR_14"].values.astype(np.float64) if "ATR_14" in df.columns else None
            rsi_14 = df["RSI_14"].values.astype(np.float64) if "RSI_14" in df.columns else None
            sma_20 = df["SMA_20"].values.astype(np.float64) if "SMA_20" in df.columns else None
            sma_50 = df["SMA_50"].values.astype(np.float64) if "SMA_50" in df.columns else None
            sma_200 = df["SMA_200"].values.astype(np.float64) if "SMA_200" in df.columns else None

            features = add_advanced_features_rust(
                df["close"].values.astype(np.float64),
                df["volume"].values.astype(np.float64),
                df["returns_1"].values.astype(np.float64),
                atr_14,
                rsi_14,
                sma_20,
                sma_50,
                sma_200,
            )

            for key, val in features.items():
                df[key] = val

            if isinstance(df.index, pd.DatetimeIndex):
                df["hour"] = df.index.hour
                df["dayofweek"] = df.index.dayofweek
                df["month"] = df.index.month

            return df
        except Exception as e:
            logging.warning(f"Rust advanced features failed: {e}")

    # Fallback to Python
    for period in [3, 5, 10, 20]:
        df[f"roc_{period}"] = df["close"].pct_change(period)

    if "ATR_14" in df.columns:
        df["atr_ratio"] = df["ATR_14"] / df["close"]

    for period in [20, 50, 200]:
        sma_col = f"SMA_{period}"
        if sma_col in df.columns:
            df[f"price_to_{sma_col}"] = df["close"] / df[sma_col]

    for window in [10, 20]:
        df[f"rolling_std_{window}"] = df["returns_1"].rolling(window).std()
        df[f"rolling_skew_{window}"] = df["returns_1"].rolling(window).skew()

    features_to_lag = ["returns_1", "RSI_14", "log_volume", "atr_ratio"]
    for feat in features_to_lag:
        if feat in df.columns:
            for lag in [1, 2, 3]:
                df[f"{feat}_lag_{lag}"] = df[feat].shift(lag)

    if isinstance(df.index, pd.DatetimeIndex):
        df["hour"] = df.index.hour
        df["dayofweek"] = df.index.dayofweek
        df["month"] = df.index.month

    return df


def _infer_important_features_from_model(model: Any, importance_threshold: float) -> list[str]:
    importances = getattr(model, "feature_importances_", None)
    if importances is None:
        return list(MODEL_FEATURES)

    bounded_threshold = max(0.0, float(importance_threshold))
    selected = [
        feature
        for feature, importance in zip(MODEL_FEATURES, np.asarray(importances).tolist())
        if float(importance) >= bounded_threshold
    ]
    return selected or list(MODEL_FEATURES)


def compute_features_lazy(
    df: pd.DataFrame,
    model: Optional[Any] = None,
    selected_features: Optional[Iterable[str]] = None,
    importance_threshold: float = 0.01,
) -> pd.DataFrame:
    """Compute features lazily by selecting only required feature columns.

    Priority:
    1) `selected_features` if provided
    2) Features inferred from `model.feature_importances_`
    3) Fallback to full `MODEL_FEATURES`
    """
    working_df = df.copy()

    if selected_features is not None:
        selected = {str(feature) for feature in selected_features}
    elif model is not None:
        selected = set(_infer_important_features_from_model(model, importance_threshold))
    else:
        selected = set(MODEL_FEATURES)

    if not selected:
        selected = set(MODEL_FEATURES)

    core_features = {"returns_1", "returns_5", "log_volume", "high_low_range", "close_open_diff"}
    advanced_candidates = selected.union(core_features)

    if RUST_AVAILABLE and calculate_all_features_rust is not None:
        try:
            atr_14 = working_df["ATR_14"].values.astype(np.float64) if "ATR_14" in working_df.columns else None
            rsi_14 = working_df["RSI_14"].values.astype(np.float64) if "RSI_14" in working_df.columns else None
            sma_20 = working_df["SMA_20"].values.astype(np.float64) if "SMA_20" in working_df.columns else None
            sma_50 = working_df["SMA_50"].values.astype(np.float64) if "SMA_50" in working_df.columns else None
            sma_200 = working_df["SMA_200"].values.astype(np.float64) if "SMA_200" in working_df.columns else None

            features = calculate_all_features_rust(
                working_df["open"].values.astype(np.float64),
                working_df["high"].values.astype(np.float64),
                working_df["low"].values.astype(np.float64),
                working_df["close"].values.astype(np.float64),
                working_df["volume"].values.astype(np.float64),
                atr_14,
                rsi_14,
                sma_20,
                sma_50,
                sma_200,
            )

            for key, val in features.items():
                if key in advanced_candidates:
                    working_df[key] = val
        except Exception as exc:
            logging.warning(f"Rust lazy feature computation failed: {exc}")
            working_df = add_advanced_features(working_df)
    else:
        working_df = add_advanced_features(working_df)

    if isinstance(working_df.index, pd.DatetimeIndex):
        if "hour" in selected:
            working_df["hour"] = working_df.index.hour
        if "dayofweek" in selected:
            working_df["dayofweek"] = working_df.index.dayofweek
        if "month" in selected:
            working_df["month"] = working_df.index.month

    return working_df
