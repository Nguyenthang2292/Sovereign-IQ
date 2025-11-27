"""Shared indicator utilities."""

from __future__ import annotations

import numpy as np
import pandas as pd


def calculate_kama(
    prices,
    window: int = 10,
    fast: int = 2,
    slow: int = 30,
) -> np.ndarray:
    """Calculate Kaufman's Adaptive Moving Average (KAMA) with safeguards."""
    prices_array = np.asarray(prices, dtype=np.float64)

    if len(prices_array) < window:
        return (
            np.full_like(prices_array, float(prices_array.flat[0]))
            if len(prices_array) > 0
            else np.array([0.0])
        )

    kama = np.zeros_like(prices_array, dtype=np.float64)
    first_valid_idx = next(
        (i for i, price in enumerate(prices_array) if np.isfinite(price)), 0
    )
    initial_value = (
        float(prices_array[first_valid_idx])
        if first_valid_idx < len(prices_array)
        else float(np.nanmean(prices_array[:window]))
    )
    kama[:window] = initial_value

    fast_sc, slow_sc = 2 / (fast + 1), 2 / (slow + 1)

    try:
        price_series = pd.Series(prices)
        changes = price_series.diff(window).abs()
        volatility = (
            price_series.rolling(window)
            .apply(
                lambda values: (
                    np.sum(np.abs(np.diff(values))) if len(values) > 1 else 1e-10
                ),
                raw=False,
            )
            .fillna(1e-10)
        )

        volatility = np.where(
            np.logical_or(volatility == 0, np.isinf(volatility)), 1e-10, volatility
        )

        efficiency_ratio = np.clip(
            (changes / volatility).fillna(0).replace([np.inf, -np.inf], 0), 0, 1
        )

        for idx in range(window, len(prices_array)):
            if not np.isfinite(prices_array[idx]):
                kama[idx] = kama[idx - 1]
                continue

            ratio_value = float(
                efficiency_ratio.iloc[idx]
                if isinstance(efficiency_ratio, pd.Series)
                else efficiency_ratio[idx]
            )
            if not np.isfinite(ratio_value):
                kama[idx] = kama[idx - 1]
                continue

            smoothing_constant = np.clip(
                (ratio_value * (fast_sc - slow_sc) + slow_sc) ** 2, 1e-10, 1.0
            )
            price_diff = prices_array[idx] - kama[idx - 1]
            if abs(price_diff) > 1e10:
                price_diff = np.clip(price_diff, -1e10, 1e10)

            kama[idx] = kama[idx - 1] + smoothing_constant * price_diff

            if not np.isfinite(kama[idx]):
                kama[idx] = kama[idx - 1]

    except Exception as err:  # pragma: no cover - fallback path
        from modules.common.utils import log_warn
        log_warn(f"Error in KAMA calculation: {err}. Using simple moving average fallback.")
        kama = (
            pd.Series(prices)
            .rolling(window=window, min_periods=1)
            .mean()
            .ffill()
            .values
        )

    kama_array = np.asarray(kama, dtype=np.float64)
    return np.where(~np.isfinite(kama_array), initial_value, kama_array).astype(
        np.float64
    )
