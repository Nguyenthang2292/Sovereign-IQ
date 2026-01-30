"""
Numba-optimized functions for XGBoost module.
"""

import numpy as np
from numba import njit, prange


@njit(cache=True, parallel=True)
def rolling_quantile_numba(arr: np.ndarray, window: int, q: float) -> np.ndarray:
    """
    Calculate rolling quantile using Numba.

    Args:
        arr: Input array
        window: Rolling window size
        q: Quantile (0.0 to 1.0)

    Returns:
        Array of rolling quantiles
    """
    n = len(arr)
    result = np.full(n, np.nan)

    # Pre-calculate window indices to parallelize if possible
    # For simple rolling, parallelizing the outer loop works well
    for i in prange(n):
        if i >= window - 1:
            window_slice = arr[i - window + 1 : i + 1]
            result[i] = np.quantile(window_slice, q)

    return result


@njit(cache=True)
def rolling_mean_numba(arr: np.ndarray, window: int) -> np.ndarray:
    """
    Calculate rolling mean using Numba.
    """
    n = len(arr)
    result = np.full(n, np.nan)

    current_sum = 0.0

    for i in range(n):
        current_sum += arr[i]
        if i >= window:
            current_sum -= arr[i - window]

        if i >= window - 1:
            result[i] = current_sum / window

    return result
