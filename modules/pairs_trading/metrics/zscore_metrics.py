"""
Z-score and related metrics for pairs trading analysis.
"""

import pandas as pd
import numpy as np
from typing import Dict, Optional

try:
    from sklearn.metrics import (
        accuracy_score,
        f1_score,
        precision_score,
        recall_score,
    )
except ImportError:
    f1_score = None
    precision_score = None
    recall_score = None
    accuracy_score = None

try:
    from modules.config import (
        PAIRS_TRADING_ZSCORE_LOOKBACK,
        PAIRS_TRADING_CLASSIFICATION_ZSCORE,
        PAIRS_TRADING_MIN_LAG,
        PAIRS_TRADING_MAX_LAG_DIVISOR,
        PAIRS_TRADING_HURST_EXPONENT_MULTIPLIER,
        PAIRS_TRADING_MIN_CLASSIFICATION_SAMPLES,
        PAIRS_TRADING_HURST_EXPONENT_MIN,
        PAIRS_TRADING_HURST_EXPONENT_MAX,
        PAIRS_TRADING_HURST_EXPONENT_MEAN_REVERTING_MAX,
    )
except ImportError:
    PAIRS_TRADING_ZSCORE_LOOKBACK = 60
    PAIRS_TRADING_CLASSIFICATION_ZSCORE = 0.5
    PAIRS_TRADING_MIN_LAG = 2
    PAIRS_TRADING_MAX_LAG_DIVISOR = 2
    PAIRS_TRADING_HURST_EXPONENT_MULTIPLIER = 2.0
    PAIRS_TRADING_MIN_CLASSIFICATION_SAMPLES = 20
    PAIRS_TRADING_HURST_EXPONENT_MIN = 0.0
    PAIRS_TRADING_HURST_EXPONENT_MAX = 2.0
    PAIRS_TRADING_HURST_EXPONENT_MEAN_REVERTING_MAX = 0.5

# Alias for backward compatibility and cleaner code
MIN_LAG = PAIRS_TRADING_MIN_LAG
MAX_LAG_DIVISOR = PAIRS_TRADING_MAX_LAG_DIVISOR
HURST_EXPONENT_MULTIPLIER = PAIRS_TRADING_HURST_EXPONENT_MULTIPLIER
MIN_CLASSIFICATION_SAMPLES = PAIRS_TRADING_MIN_CLASSIFICATION_SAMPLES
HURST_EXPONENT_MIN = PAIRS_TRADING_HURST_EXPONENT_MIN
HURST_EXPONENT_MAX = PAIRS_TRADING_HURST_EXPONENT_MAX
HURST_EXPONENT_MEAN_REVERTING_MAX = PAIRS_TRADING_HURST_EXPONENT_MEAN_REVERTING_MAX


def calculate_zscore_stats(
    spread: pd.Series, zscore_lookback: int = PAIRS_TRADING_ZSCORE_LOOKBACK
) -> Dict[str, Optional[float]]:
    """
    Calculate z-score statistics for the spread series.
    
    Args:
        spread: Spread series
        zscore_lookback: Number of periods for rolling window
        
    Returns:
        Dictionary with z-score statistics
    """
    result = {
        "mean_zscore": None,
        "std_zscore": None,
        "skewness": None,
        "kurtosis": None,
        "current_zscore": None,
    }

    if spread is None or len(spread) < zscore_lookback:
        return result

    rolling_mean = spread.rolling(zscore_lookback).mean()
    rolling_std = spread.rolling(zscore_lookback).std()
    zscore = (spread - rolling_mean) / rolling_std.replace(0, np.nan)

    zscore = zscore.dropna()
    if zscore.empty:
        return result

    result.update(
        {
            "mean_zscore": float(zscore.mean()),
            "std_zscore": float(zscore.std()),
            "skewness": float(zscore.skew()) if hasattr(zscore, "skew") else None,
            "kurtosis": float(zscore.kurtosis())
            if hasattr(zscore, "kurtosis")
            else None,
            "current_zscore": float(zscore.iloc[-1]),
        }
    )
    return result


def calculate_hurst_exponent(
    spread: pd.Series,
    zscore_lookback: int = PAIRS_TRADING_ZSCORE_LOOKBACK,
    max_lag: int = 100,
) -> Optional[float]:
    """
    Calculate Hurst exponent using R/S (Rescaled Range) analysis.
    
    Hurst exponent measures long-term memory of a time series:
    - **H < 0.5**: Mean-reverting (spread returns to mean) → Good for pairs trading
    - **H ≈ 0.5**: Random walk (no predictable pattern)
    - **H > 0.5**: Trending (persistent trends) → Less suitable for mean reversion
    
    **Method**: Variance-based R/S analysis. Computes variance scaling across lags,
    fits log(tau) vs log(lag) regression, converts slope to Hurst exponent.
    
    Args:
        spread: Spread series (price1 - hedge_ratio * price2)
        zscore_lookback: Minimum data points required. Default: 60
        max_lag: Maximum lag for analysis (capped at series_length/2). Default: 100
        
    Returns:
        Hurst exponent in range [0, 2], or None if calculation fails.
        
    Example:
        >>> spread = pd.Series([0.1, -0.05, 0.15, -0.1, ...])
        >>> hurst = calculate_hurst_exponent(spread)
        >>> # hurst = 0.42 means mean-reverting (good for pairs trading)
    """
    if spread is None or len(spread.dropna()) < zscore_lookback:
        return None

    series = spread.dropna().values
    # Limit max_lag to half of series length for stability
    max_lag = min(max_lag, len(series) // MAX_LAG_DIVISOR)
    # Start from MIN_LAG (2) to ensure meaningful variance calculation
    lags_list = [lag for lag in range(MIN_LAG, max_lag) if lag < len(series)]
    if not lags_list:
        return None

    try:
        tau = []
        filtered_lags = []
        # Calculate variance-based scaling for each lag
        for lag in lags_list:
            # Difference between series at lag intervals
            diff = np.subtract(series[lag:], series[:-lag])
            # Square root of standard deviation approximates scaling behavior
            value = np.sqrt(np.std(diff))
            if value > 0:
                tau.append(value)
                filtered_lags.append(lag)
        if not tau:
            return None
        
        # Linear regression: log(tau) = H * log(lag) + constant
        # Using polyfit with degree 1 (linear)
        poly = np.polyfit(np.log(filtered_lags), np.log(tau), 1)
        # Convert slope to Hurst exponent (multiply by 2 based on R/S theory)
        hurst = poly[0] * HURST_EXPONENT_MULTIPLIER
        
        # Validate result
        if np.isnan(hurst) or np.isinf(hurst):
            return None
        
        # Clamp to theoretical bounds [0, 2]
        hurst = max(HURST_EXPONENT_MIN, min(HURST_EXPONENT_MAX, float(hurst)))
        
        return hurst
    except Exception:
        return None


def calculate_direction_metrics(
    spread: pd.Series,
    zscore_lookback: int = PAIRS_TRADING_ZSCORE_LOOKBACK,
    classification_zscore: float = PAIRS_TRADING_CLASSIFICATION_ZSCORE,
) -> Dict[str, Optional[float]]:
    """
    Calculate classification metrics for spread direction prediction using z-score.
    
    Evaluates how well z-score predicts future spread direction. Uses mean reversion logic:
    - When zscore < -threshold: spread is below mean → predicts UP (revert upward)
    - Actual labels: 1 if spread increases next period, 0 otherwise
    - Only predicts UP when zscore < -threshold (primary mean-reversion signal)
    
    Args:
        spread: Spread series (price1 - hedge_ratio * price2)
        zscore_lookback: Rolling window size for z-score calculation. Default: 60
        classification_zscore: Z-score threshold for prediction. Default: 0.5
        
    Returns:
        Dict with metrics (all in [0, 1] or None):
        - classification_f1: F1 score (weighted average)
        - classification_precision: Precision score
        - classification_recall: Recall score
        - classification_accuracy: Accuracy score
        
    Example:
        >>> spread = pd.Series([0.1, -0.05, 0.15, -0.1, ...])
        >>> metrics = calculate_direction_metrics(spread)
        >>> # Returns dict with f1, precision, recall, accuracy scores
    """
    result = {
        "classification_f1": None,
        "classification_precision": None,
        "classification_recall": None,
        "classification_accuracy": None,
    }

    if (
        f1_score is None
        or precision_score is None
        or recall_score is None
        or accuracy_score is None
    ):
        return result

    if spread is None or len(spread) < zscore_lookback:
        return result

    # Calculate rolling z-score
    rolling_mean = spread.rolling(zscore_lookback).mean()
    rolling_std = spread.rolling(zscore_lookback).std().replace(0, np.nan)
    zscore = ((spread - rolling_mean) / rolling_std).dropna()
    
    # Actual labels: 1 if spread increases next period, 0 otherwise
    future_return = spread.shift(-1) - spread
    actual = (future_return > 0).astype(int).dropna()

    # Align indices
    common_idx = zscore.index.intersection(actual.index)
    # Require minimum samples for reliable classification metrics
    if len(common_idx) < MIN_CLASSIFICATION_SAMPLES:
        return result

    zscore = zscore.loc[common_idx]
    actual = actual.loc[common_idx]

    # Prediction logic: Predict UP (1) when zscore is significantly below mean
    # This tests if extreme negative z-scores predict mean reversion upward
    threshold = classification_zscore
    predicted = pd.Series(
        np.where(zscore < -threshold, 1, 0), index=zscore.index, dtype=int
    )

    try:
        # Calculate classification metrics
        result["classification_f1"] = float(
            f1_score(actual, predicted, average="weighted")
        )
        result["classification_precision"] = float(
            precision_score(actual, predicted, average="weighted", zero_division=0)
        )
        result["classification_recall"] = float(
            recall_score(actual, predicted, average="weighted", zero_division=0)
        )
        result["classification_accuracy"] = float(accuracy_score(actual, predicted))
        
        # Validate return values are in [0, 1] range
        for key in result:
            if result[key] is not None:
                result[key] = max(0.0, min(1.0, result[key]))
    except Exception:
        return {
            "classification_f1": None,
            "classification_precision": None,
            "classification_recall": None,
            "classification_accuracy": None,
        }

    return result