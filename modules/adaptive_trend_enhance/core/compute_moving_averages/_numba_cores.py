from __future__ import annotations

import numpy as np

try:
    from numba import njit, prange

    _HAS_NUMBA = True
except ImportError:  # pragma: no cover
    _HAS_NUMBA = False

    def njit(*args, **kwargs):
        def decorator(func):
            return func

        return decorator

    prange = range


@njit(cache=True, fastmath=True)
def _calculate_wma_core(prices: np.ndarray, length: int) -> np.ndarray:
    n = len(prices)
    wma = np.full(n, np.nan, dtype=np.float64)

    if n < length:
        return wma

    denominator = length * (length + 1) / 2.0

    for i in range(length - 1, n):
        weighted_sum = 0.0
        for j in range(length):
            weight = length - j
            weighted_sum += prices[i - j] * weight

        wma[i] = weighted_sum / denominator

    return wma


@njit(cache=True, fastmath=True)
def _calculate_ema_core(prices: np.ndarray, length: int) -> np.ndarray:
    n = len(prices)
    ema = np.full(n, np.nan, dtype=np.float64)

    if n < 1:
        return ema

    alpha = 2.0 / (length + 1.0)
    ema[0] = prices[0]

    for i in range(1, n):
        ema[i] = alpha * prices[i] + (1.0 - alpha) * ema[i - 1]

    return ema


@njit(cache=True, fastmath=True)
def _calculate_dema_core(prices: np.ndarray, length: int) -> np.ndarray:
    ema1 = _calculate_ema_core(prices, length)
    ema2 = _calculate_ema_core(ema1, length)
    return 2.0 * ema1 - ema2


@njit(cache=True, fastmath=True)
def _calculate_lsma_core(prices: np.ndarray, length: int) -> np.ndarray:
    n = len(prices)
    lsma = np.full(n, np.nan, dtype=np.float64)

    if n < length:
        return lsma

    x_mean = (length - 1) / 2.0
    x_sq_sum = 0.0
    for i in range(length):
        x_sq_sum += (i - x_mean) ** 2

    for i in range(length - 1, n):
        y_sum = 0.0
        for j in range(length):
            y_sum += prices[i - length + 1 + j]
        y_mean = y_sum / length

        xy_sum = 0.0
        for j in range(length):
            x = float(j)
            y = prices[i - length + 1 + j]
            xy_sum += (x - x_mean) * (y - y_mean)

        b = xy_sum / x_sq_sum
        a = y_mean - b * x_mean
        lsma[i] = a + b * (length - 1)

    return lsma


@njit(cache=True)
def _calculate_kama_atc_core(prices_array: np.ndarray, length: int) -> np.ndarray:
    n = len(prices_array)
    kama = np.full(n, np.nan, dtype=np.float64)

    if n < 1:
        return kama

    fast = 0.666
    slow = 0.064

    for i in range(n):
        if i == 0:
            kama[i] = prices_array[i]
            continue

        if i < length:
            kama[i] = kama[i - 1]
            continue

        noise = 0.0
        for j in range(i - length + 1, i + 1):
            if j <= 0:
                continue
            noise += abs(prices_array[j] - prices_array[j - 1])

        signal = abs(prices_array[i] - prices_array[i - length])
        ratio = 0.0 if noise == 0 else signal / noise

        smooth = (ratio * (fast - slow) + slow) ** 2

        prev_kama = kama[i - 1]
        if np.isnan(prev_kama):
            prev_kama = prices_array[i]

        kama[i] = prev_kama + (smooth * (prices_array[i] - prev_kama))

    return kama


__all__ = [
    "_HAS_NUMBA",
    "njit",
    "prange",
    "_calculate_wma_core",
    "_calculate_ema_core",
    "_calculate_dema_core",
    "_calculate_lsma_core",
    "_calculate_kama_atc_core",
]
