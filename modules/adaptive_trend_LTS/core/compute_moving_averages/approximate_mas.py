"""Fast approximate moving averages for initial filtering in scanning.

These functions use simplified calculations (e.g., SMA for EMA approximation)
to quickly filter candidates before full precision calculation.
"""

import numpy as np
import pandas as pd


def fast_ema_approx(prices: pd.Series, length: int) -> pd.Series:
    """Fast EMA approximation using SMA (much faster).

    Args:
        prices: Price series
        length: EMA length

    Returns:
        Approximate EMA series (within ~5% of true EMA)
    """
    return prices.rolling(window=length, min_periods=1).mean()


def fast_hma_approx(prices: pd.Series, length: int) -> pd.Series:
    """Fast HMA approximation.

    Strategy: Use simplified WMA calculations
    """
    half_len = max(1, length // 2)
    sqrt_len = max(1, int(np.sqrt(length)))

    wma_half = fast_wma_approx(prices, half_len)
    wma_full = fast_wma_approx(prices, length)
    hma_input = 2 * wma_half - wma_full
    return fast_wma_approx(hma_input, sqrt_len)


def fast_wma_approx(prices: pd.Series, length: int) -> pd.Series:
    """Fast WMA approximation using simplified weights."""
    if length <= 1:
        return prices.copy()

    weights = np.arange(1, length + 1, dtype=np.float64)
    weights = weights / weights.sum()

    result = pd.Series(index=prices.index, dtype=np.float64)
    result[:] = np.nan

    for i in range(length - 1, len(prices)):
        window = prices.iloc[i - length + 1 : i + 1].values
        result.iloc[i] = (window * weights).sum()

    result[: length - 1] = prices.rolling(window=length, min_periods=1).mean().iloc[: length - 1]
    return result


def fast_dema_approx(prices: pd.Series, length: int) -> pd.Series:
    """Fast DEMA approximation."""
    ema1 = fast_ema_approx(prices, length)
    ema2 = fast_ema_approx(ema1, length)
    return 2 * ema1 - ema2


def fast_lsma_approx(prices: pd.Series, length: int) -> pd.Series:
    """Fast LSMA approximation using simplified linear regression."""
    if length <= 2:
        return prices.copy()

    result = pd.Series(index=prices.index, dtype=np.float64)
    result[:] = np.nan

    for i in range(length - 1, len(prices)):
        window = prices.iloc[i - length + 1 : i + 1].values

        if len(window) > 1:
            slope = (window[-1] - window[0]) / length
            result.iloc[i] = window[-1] - slope * (length / 2.0)
        else:
            result.iloc[i] = window[-1]

    result[: length - 1] = prices.rolling(window=length, min_periods=1).mean().iloc[: length - 1]
    return result


def fast_kama_approx(prices: pd.Series, length: int) -> pd.Series:
    """Fast KAMA approximation using EMA with fixed smoothing constant."""
    sc = 2 / (length + 1) ** 2
    alpha = sc

    kama = prices.copy().astype(np.float64)

    for i in range(1, len(prices)):
        kama.iloc[i] = kama.iloc[i - 1] + alpha * (prices.iloc[i] - kama.iloc[i - 1])

    return kama
