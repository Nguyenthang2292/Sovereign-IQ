
from typing import Optional

import numpy as np
import pandas as pd

from config import (

from config import (

"""
HMM-KAMA Feature Engineering.

This module handles feature preparation and engineering for HMM-KAMA analysis.
"""



    HMM_FAST_KAMA_DEFAULT,
    HMM_SLOW_KAMA_DEFAULT,
    HMM_WINDOW_KAMA_DEFAULT,
)
from modules.common.indicators import calculate_kama
from modules.common.utils import log_analysis, log_data, log_error, log_warn


def prepare_observations(
    data: pd.DataFrame,
    window_kama: Optional[int] = None,
    fast_kama: Optional[int] = None,
    slow_kama: Optional[int] = None,
) -> Optional[np.ndarray]:
    """Generate crypto-optimized observation features.

    Uses price minus KAMA deviation to keep inputs closer to stationarity.

    Args:
        data: DataFrame with OHLCV data
        window_kama: KAMA window size (default: from config)
        fast_kama: Fast KAMA parameter (default: from config)
        slow_kama: Slow KAMA parameter (default: from config)
    """
    if data.empty or "close" not in data.columns or len(data) < 10:
        raise ValueError(f"Invalid data: empty={data.empty}, has close={'close' in data.columns}, len={len(data)}")

    close_prices = data["close"].replace([np.inf, -np.inf], np.nan).ffill().bfill()
    if close_prices.isna().any():  # type: ignore
        close_prices = close_prices.fillna(close_prices.median())

    # Normalize prices to 0-1000 scale for consistency (UPDATED)
    price_range = close_prices.max() - close_prices.min()
    unique_prices = close_prices.nunique()

    # FIX (2025-01-16): Handle case where all prices are the same (price_range = 0)
    # Problem: Pre-normalized data or constant values result in price_range = 0,
    #          causing zero variance in features and HMM training failures
    # Solution: Create small variation (0.1%) around mean to ensure variance for feature calculation
    if price_range == 0 or unique_prices < 3:
        log_data(f"Problematic price data: range={price_range}, unique_prices={unique_prices}")
        # If all prices are the same, create a small variation around the mean
        # This ensures we have some variance for feature calculation
        mean_price = close_prices.mean()
        # Create a small range (0.1% variation) to ensure variance
        close_prices = pd.Series(
            np.linspace(
                mean_price * 0.9995,
                mean_price * 1.0005,
                len(close_prices),
            )
        )
        price_range = close_prices.max() - close_prices.min()
        log_data(f"Fixed price data: new range={price_range}, new unique_prices={close_prices.nunique()}")

    close_prices_norm = (
        ((close_prices - close_prices.min()) / price_range * 1000)
        if price_range > 0
        else pd.Series(np.linspace(450, 550, len(close_prices)))
    )
    close_prices_array = close_prices_norm.values.astype(np.float64)

    # 1. Calculate KAMA
    try:
        window_param = int(window_kama if window_kama is not None else HMM_WINDOW_KAMA_DEFAULT)
        fast = int(fast_kama if fast_kama is not None else HMM_FAST_KAMA_DEFAULT)
        slow_raw = int(slow_kama if slow_kama is not None else HMM_SLOW_KAMA_DEFAULT)
        slow = max(slow_raw, fast + 5)

        window = max(2, min(window_param, len(close_prices_array) // 2))

        kama_values = calculate_kama(close_prices_array, window=window, fast=fast, slow=slow)

        if np.max(kama_values) - np.min(kama_values) < 1e-10:
            log_data("KAMA has zero variance. Adding gradient.")
            kama_values = np.linspace(kama_values[0] - 0.5, kama_values[0] + 0.5, len(kama_values))

    except Exception as e:
        log_error(f"KAMA calculation failed: {e}. Using EMA fallback.")
        kama_values = pd.Series(close_prices_array).ewm(alpha=2.0 / (window_param + 1), adjust=False).mean().values

    # 2. Calculate Features

    # Feature 1: Returns (Stationary)
    returns = np.diff(close_prices_array, prepend=close_prices_array[0])
    if np.std(returns) < 1e-10:
        log_warn("Returns have zero variance. Returning None (Neutral).")
        return None

    # Feature 2: Price Deviation from KAMA (Stationary-ish)
    kama_deviation = close_prices_array - kama_values

    # Feature 3: Volatility (Stationary)
    # Using change in KAMA as a proxy for trend strength/volatility
    volatility = np.abs(np.diff(np.array(kama_values), prepend=kama_values[0]))
    if np.std(volatility) < 1e-10:
        log_warn("Volatility has zero variance. Returning None (Neutral).")
        return None

    rolling_vol = pd.Series(returns).rolling(window=5, min_periods=1).std().fillna(0.01).values
    volatility = (volatility + np.asarray(rolling_vol)) / 2

    # Cleaning
    def _clean_crypto_array(arr, name="array", default_val=0.0):
        arr = np.where(np.isfinite(arr), arr, default_val)
        valid_values = arr[arr != default_val]
        q_range = (
            max(
                float(abs(np.percentile(valid_values, 95))),
                float(abs(np.percentile(valid_values, 5))),
            )
            * 1.5
            if np.any(valid_values)
            else 1000
        )
        return np.clip(arr, -q_range, q_range).astype(np.float64)

    returns = _clean_crypto_array(returns, "returns", 0.0)
    kama_deviation = _clean_crypto_array(kama_deviation, "kama_deviation", 0.0)
    volatility = _clean_crypto_array(volatility, "volatility", 0.01)

    # Final variance check
    if np.std(returns) == 0:
        returns[0] = 0.01
    if np.std(kama_deviation) == 0:
        kama_deviation[-1] += 0.01
    if np.std(volatility) == 0:
        volatility[0], volatility[-1] = 0.005, 0.015

    feature_matrix = np.column_stack([returns, kama_deviation, volatility])

    if not np.isfinite(feature_matrix).all():
        log_error("Feature matrix contains invalid values. Returning None (Neutral).")
        return None

    log_analysis(
        f"Crypto-optimized features - Shape: {feature_matrix.shape}, "
        f"Returns range: [{returns.min():.6f}, {returns.max():.6f}], "
        f"Deviation range: [{kama_deviation.min():.6f}, {kama_deviation.max():.6f}], "
        f"Volatility range: [{volatility.min():.6f}, {volatility.max():.6f}]"
    )

    return feature_matrix
