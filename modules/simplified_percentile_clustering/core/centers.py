"""
Cluster center calculation using percentiles and running mean.

Builds sliding arrays of historical values for each feature, sorts them,
computes lower/upper percentiles and a center near the mean. Then derives
k centers as:
  k=2 -> [avg(low_pct, mean), avg(high_pct, mean)]
  k=3 -> [avg(low_pct, mean), mean, avg(high_pct, mean)]
"""

from __future__ import annotations

from collections import deque

import numpy as np
import pandas as pd

try:
    from numba import jit

    HAS_NUMBA = True
except ImportError:
    HAS_NUMBA = False

    # Dummy decorator if numba is missing
    def jit(nopython=True):
        def decorator(func):
            return func

        return decorator


class ClusterCenters:
    """Manages cluster centers calculation for a single feature."""

    def __init__(
        self, lookback: int, p_low: float = 5.0, p_high: float = 95.0, k: int = 2, volatility_adjustment: bool = False
    ):
        """
        Initialize cluster centers calculator.

        Args:
            lookback: Number of historical values to keep for percentile calculations.
            p_low: Lower percentile (default: 5.0).
            p_high: Upper percentile (default: 95.0).
            k: Number of cluster centers (2 or 3).
            volatility_adjustment: Whether to adjust percentiles based on volatility.
        """
        if k not in [2, 3]:
            raise ValueError("k must be 2 or 3")
        if not (0 <= p_low < p_high <= 100):
            raise ValueError("p_low must be < p_high and both in [0, 100]")

        # Gap 2: Extreme percentile values check
        # Recommend warning if p_high - p_low < 10 (too narrow range)
        if p_high - p_low < 10:
            import warnings

            warnings.warn(
                f"Percentile range (p_high - p_low = {p_high - p_low}) is very narrow (< 10). "
                "This may lead to poor clustering separation.",
                UserWarning,
                stacklevel=2,
            )

        self.lookback = lookback
        self.p_low = p_low
        self.p_high = p_high
        self.k = k
        self.volatility_adjustment = volatility_adjustment
        self._values = deque(maxlen=lookback)

        # Volatility tracking for adaptive mode
        self._vol_ema = 0.0
        self._vol_alpha = 0.01  # Slow decay for baseline

    @staticmethod
    def get_percentile(arr: list[float], pct: float) -> float:
        """
        Returns the value at `pct` percentile from a sorted array.

        Implementation: sorts a copy and chooses index floor((n-1)*pct/100).
        This is a deterministic and inexpensive approximation.
        """
        if len(arr) == 0:
            return np.nan

        sorted_arr = sorted(arr)
        # Ensure pct is within bounds
        pct = max(0.0, min(100.0, pct))
        idx = int(np.floor((len(sorted_arr) - 1) * pct / 100))
        return sorted_arr[idx]

    def update(self, value: float) -> list[float]:
        """
        Update with new value and return current cluster centers.

        Args:
            value: New feature value to add.

        Returns:
            List of k cluster centers.
        """
        if not np.isfinite(value):
            # If we have previous values, return last centers
            if len(self._values) > 0:
                return self._get_centers()
            return [np.nan] * self.k

        self._values.append(float(value))
        return self._get_centers()

    def _get_centers(self) -> list[float]:
        """Calculate centers from current values."""
        if len(self._values) == 0:
            return [np.nan] * self.k

        values_list = list(self._values)

        # Adaptive Logic
        cutoff_low = self.p_low
        cutoff_high = self.p_high

        if self.volatility_adjustment and len(values_list) > 1:
            vals = np.array(values_list)
            mean_val = np.mean(vals)
            if abs(mean_val) > 1e-9:
                std_val = np.std(vals)
                current_vol = std_val / abs(mean_val)

                # Update baseline volatility (EMA)
                if self._vol_ema == 0.0:
                    self._vol_ema = current_vol
                else:
                    self._vol_ema = (self._vol_alpha * current_vol) + ((1 - self._vol_alpha) * self._vol_ema)

                # Adjust percentiles
                # p_adj = p + (vol - vol_baseline) * 2
                # High vol > baseline => widen (subtract from low, add to high)
                vol_diff = current_vol - self._vol_ema
                adjustment = vol_diff * 2.0

                # Apply adjustment
                cutoff_low = max(1.0, min(20.0, self.p_low - adjustment))
                cutoff_high = max(80.0, min(99.0, self.p_high + adjustment))

        # Calculate percentiles and mean
        x_high = self.get_percentile(values_list, cutoff_high)
        x_low = self.get_percentile(values_list, cutoff_low)
        x_mid = np.mean(values_list)

        # Calculate centers
        x_k0_center = (x_low + x_mid) / 2
        x_k1_center = (x_high + x_mid) / 2

        if self.k == 2:
            return [x_k0_center, x_k1_center]
        else:  # k == 3
            return [x_k0_center, x_mid, x_k1_center]

    def get_current_centers(self) -> list[float]:
        """Get current cluster centers without updating."""
        return self._get_centers()


@jit(nopython=True)
def _calc_dynamic_quantiles_numba(values, lookback, p_low_arr, p_high_arr):
    """
    Numba-optimized helper for dynamic rolling quantiles.
    """
    n = len(values)
    out_low = np.full(n, np.nan)
    out_high = np.full(n, np.nan)

    # Pre-allocate buffer for sorting to avoid allocation in loop?
    # Numba handles small array allocations reasonably well.

    for i in range(n):
        if i < lookback - 1:
            continue

        # Get window
        window = values[i - lookback + 1 : i + 1]

        # Sort window (simple bubble sort or built-in if supported)
        # Numba supports np.sort
        sorted_window = np.sort(window)
        w_len = len(sorted_window)

        # Low quantile
        p_l = max(0.0, min(100.0, p_low_arr[i]))
        idx_l = int(np.floor((w_len - 1) * p_l / 100.0))
        out_low[i] = sorted_window[idx_l]

        # High quantile
        p_h = max(0.0, min(100.0, p_high_arr[i]))
        idx_h = int(np.floor((w_len - 1) * p_h / 100.0))
        out_high[i] = sorted_window[idx_h]

    return out_low, out_high


def compute_centers(
    values: pd.Series,
    lookback: int = 1000,
    p_low: float = 5.0,
    p_high: float = 95.0,
    k: int = 2,
    volatility_adjustment: bool = False,
) -> pd.DataFrame:
    """
    Compute cluster centers for a time series using vectorized operations.

    This function uses Pandas rolling window operations for better performance
    compared to the iterative approach. It computes percentiles and mean using
    vectorized operations.

    Args:
        values: Feature values time series.
        lookback: Number of historical values to keep.
        p_low: Lower percentile (0-100).
        p_high: Upper percentile (0-100).
        k: Number of cluster centers (2 or 3).
        volatility_adjustment: Whether to adjust percentiles based on volatility.

    Returns:
        DataFrame with columns 'k0', 'k1', and optionally 'k2' containing
        cluster centers for each timestamp.
    """
    if len(values) == 0:
        if k == 2:
            return pd.DataFrame(columns=["k0", "k1"], index=values.index)
        else:
            return pd.DataFrame(columns=["k0", "k1", "k2"], index=values.index)

    # Convert to numpy for numba/calculations
    vals_np = values.values.astype(np.float64)

    # 1. Calculate Rolling Mean (needed for center calc and volatility)
    # Using pandas rolling for mean as it's heavily optimized
    rolling_mean = values.rolling(window=lookback, min_periods=lookback).mean()

    # 2. Determine Percentiles (Static or Dynamic)
    if volatility_adjustment:
        # Calculate Rolling Volatility
        # Volatility = Rolling Std / Rolling Mean
        rolling_std = values.rolling(window=lookback, min_periods=lookback).std()

        # Avoid division by zero
        safe_mean = rolling_mean.replace(0, np.nan)
        volatility = rolling_std / safe_mean.abs()

        # Fill NaNs or handle initial period
        volatility = volatility.fillna(0)

        # Calculate Median Volatility (Global for batch, as requested)
        vol_median = volatility.median()

        # Calculate Adjustments
        # High vol -> widen percentiles (p_low decreases, p_high increases)
        # adjustment = (vol - median) * 2
        adjustment = (volatility - vol_median) * 2.0

        p_low_adj = (p_low - adjustment).clip(1, 20)
        p_high_adj = (p_high + adjustment).clip(80, 99)

        # Use Numba for dynamic quantiles if available, otherwise slow loop
        if HAS_NUMBA:
            x_low_np, x_high_np = _calc_dynamic_quantiles_numba(vals_np, lookback, p_low_adj.values, p_high_adj.values)
            x_low = pd.Series(x_low_np, index=values.index)
            x_high = pd.Series(x_high_np, index=values.index)
        else:
            # Fallback for no Numba (slow)
            import warnings

            warnings.warn("Numba not found/failed. Using slow loop for adaptive centers.", RuntimeWarning)

            x_low = pd.Series(index=values.index, dtype=float)
            x_high = pd.Series(index=values.index, dtype=float)

            p_low_vals = p_low_adj.values
            p_high_vals = p_high_adj.values

            for i in range(lookback - 1, len(values)):
                window = vals_np[i - lookback + 1 : i + 1]
                # Sort locally
                sorted_w = np.sort(window)
                w_len = len(sorted_w)

                idx_l = int(np.floor((w_len - 1) * p_low_vals[i] / 100.0))
                idx_h = int(np.floor((w_len - 1) * p_high_vals[i] / 100.0))

                x_low.iloc[i] = sorted_w[idx_l]
                x_high.iloc[i] = sorted_w[idx_h]

    else:
        # Standard Static Percentiles
        q_low = p_low / 100.0
        q_high = p_high / 100.0

        rolling = values.rolling(window=lookback, min_periods=lookback)
        x_low = rolling.quantile(q_low, interpolation="lower")
        x_high = rolling.quantile(q_high, interpolation="lower")

    # 3. Calculate Centers
    x_mid = rolling_mean

    # Calculate centers: k0 = (p_low + mean)/2, k1 = (p_high + mean)/2
    x_k0_center = (x_low + x_mid) / 2.0
    x_k1_center = (x_high + x_mid) / 2.0

    # Build result DataFrame
    if k == 2:
        result = pd.DataFrame(
            {
                "k0": x_k0_center,
                "k1": x_k1_center,
            },
            index=values.index,
        )
    else:  # k == 3
        result = pd.DataFrame(
            {
                "k0": x_k0_center,
                "k1": x_mid,  # For k=3, k1 is the mean
                "k2": x_k1_center,
            },
            index=values.index,
        )

    return result


__all__ = ["ClusterCenters", "compute_centers"]
