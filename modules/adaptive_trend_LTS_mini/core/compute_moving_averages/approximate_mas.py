"""Fast approximate moving averages for initial filtering in scanning.

These functions use simplified calculations (e.g., SMA for EMA approximation)
to quickly filter candidates before full precision calculation.
"""

import numpy as np
import pandas as pd


def fast_ema_approx(prices: pd.Series, length: int, tolerance: float = 0.01) -> pd.Series:
    """Fast EMA approximation.

    If tolerance is strict (< 0.05), use pandas EWM (exact but still reasonably fast).
    Otherwise, use SMA approximation (fastest).

    Args:
        prices: Price series
        length: Window length
        tolerance: Maximum error tolerance (default: 0.01 = 1%)
                   Lower values increase accuracy but may reduce speed.
    """
    if len(prices) == 0:
        return pd.Series(dtype=np.float64, index=prices.index)
    if tolerance < 0.05:
        # Exact calculation for low tolerance
        return prices.ewm(span=length, adjust=False).mean()
    else:
        # Fast approximation using SMA
        return prices.rolling(window=length, min_periods=1).mean()


def fast_hma_approx(prices: pd.Series, length: int, tolerance: float = 0.01) -> pd.Series:
    """Fast HMA approximation.

    Strategy: Use simplified WMA calculations.

    Args:
        prices: Price series
        length: Window length
        tolerance: Maximum error tolerance (default: 0.01 = 1%)
                   Currently unused but kept for API consistency.
    """
    if len(prices) == 0:
        return pd.Series(dtype=np.float64, index=prices.index)
    half_len = max(1, length // 2)
    sqrt_len = max(1, int(np.sqrt(length)))

    # Pass tolerance recursively if we implement tolerance logic in WMA later
    wma_half = fast_wma_approx(prices, half_len, tolerance)
    wma_full = fast_wma_approx(prices, length, tolerance)
    hma_input = 2 * wma_half - wma_full
    return fast_wma_approx(hma_input, sqrt_len, tolerance)


def fast_wma_approx(prices: pd.Series, length: int, tolerance: float = 0.01) -> pd.Series:
    """Fast WMA approximation using simplified weights.

    Args:
        prices: Price series
        length: Window length
        tolerance: Maximum error tolerance (default: 0.01 = 1%)
                   Currently unused but kept for API consistency.
    """
    if len(prices) == 0:
        return pd.Series(dtype=np.float64, index=prices.index)
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


def fast_dema_approx(prices: pd.Series, length: int, tolerance: float = 0.01) -> pd.Series:
    """Fast DEMA approximation.

    Args:
        prices: Price series
        length: Window length
        tolerance: Maximum error tolerance (default: 0.01 = 1%)
                   Propagated to underlying EMA calculations.
    """
    if len(prices) == 0:
        return pd.Series(dtype=np.float64, index=prices.index)
    ema1 = fast_ema_approx(prices, length, tolerance)
    ema2 = fast_ema_approx(ema1, length, tolerance)
    return 2 * ema1 - ema2


def fast_lsma_approx(prices: pd.Series, length: int, tolerance: float = 0.01) -> pd.Series:
    """Fast LSMA (Least Squares Moving Average) approximation using proper linear regression.

    True LSMA fits a linear regression line to the window and returns the end-point projection.
    Formula: For window x=[0,1,2,...,n-1], y=prices:
        slope = (n*sum(x*y) - sum(x)*sum(y)) / (n*sum(x^2) - sum(x)^2)
        intercept = (sum(y) - slope*sum(x)) / n
        lsma = intercept + slope * (n - 1)  # Project to end of window

    Args:
        prices: Price series
        length: Window length
        tolerance: Maximum error tolerance (default: 0.01 = 1%)
                   Currently unused but kept for API consistency.
    """
    if len(prices) == 0:
        return pd.Series(dtype=np.float64, index=prices.index)
    if length <= 2:
        return prices.copy()

    result = pd.Series(index=prices.index, dtype=np.float64)
    result[:] = np.nan

    # Pre-compute x values and their sums (x = [0, 1, 2, ..., length-1])
    x = np.arange(length, dtype=np.float64)
    sum_x = x.sum()  # = length * (length - 1) / 2
    sum_x2 = (x**2).sum()  # = (length-1)*length*(2*length-1) / 6
    n = float(length)
    denominator = n * sum_x2 - sum_x**2

    for i in range(length - 1, len(prices)):
        window = np.asarray(prices.iloc[i - length + 1 : i + 1].values, dtype=np.float64)

        if len(window) == length:
            sum_y = float(window.sum())
            sum_xy = float((x * window).sum())

            # Calculate slope and intercept using least squares formula
            slope = (n * sum_xy - sum_x * sum_y) / denominator
            intercept = (sum_y - slope * sum_x) / n

            # LSMA is the projected value at the end of the window
            result.iloc[i] = intercept + slope * (length - 1)
        else:
            result.iloc[i] = window[-1]

    result[: length - 1] = prices.rolling(window=length, min_periods=1).mean().iloc[: length - 1]
    return result


def fast_kama_approx(prices: pd.Series, length: int, tolerance: float = 0.01) -> pd.Series:
    """Fast KAMA approximation using adaptive smoothing based on efficiency ratio.

    Real KAMA formula:
    - Change = |price - price[length bars ago]|
    - Volatility = sum of |price[i] - price[i-1]| over length bars
    - ER (Efficiency Ratio) = Change / Volatility (0 to 1)
    - SC (Smoothing Constant) = (ER * (fast_sc - slow_sc) + slow_sc)^2
    - KAMA = prev_KAMA + SC * (price - prev_KAMA)

    Where fast_sc = 2/(2+1) and slow_sc = 2/(30+1) per Kaufman's original formula.

    Args:
        prices: Price series
        length: Window length
        tolerance: Maximum error tolerance (default: 0.01 = 1%)
                   Currently unused but kept for API consistency.
    """
    if len(prices) == 0:
        return pd.Series(dtype=np.float64, index=prices.index)
    if length <= 1:
        return prices.copy()

    fast_sc = 2.0 / (2.0 + 1.0)  # Fast EMA constant (10-day equivalent)
    slow_sc = 2.0 / (30.0 + 1.0)  # Slow EMA constant (30-day equivalent)

    kama = prices.copy().astype(np.float64)

    for i in range(1, len(prices)):
        # Calculate efficiency ratio using rolling window
        if i >= length:
            window = prices.iloc[i - length + 1 : i + 1].values
            change = abs(window[-1] - window[0])
            volatility = sum(abs(window[j] - window[j - 1]) for j in range(1, len(window)))
            er = change / volatility if volatility != 0 else 0.0
        else:
            er = 0.0  # Before we have enough data, use minimum smoothing

        # Calculate adaptive smoothing constant
        sc = (er * (fast_sc - slow_sc) + slow_sc) ** 2

        # Update KAMA with adaptive smoothing
        kama.iloc[i] = kama.iloc[i - 1] + sc * (prices.iloc[i] - kama.iloc[i - 1])

    return kama
