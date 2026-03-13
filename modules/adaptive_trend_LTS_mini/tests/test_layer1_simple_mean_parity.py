"""Regression tests for Layer 1 simple-mean parity contract.

These tests lock the intended behavior:
- Layer 1 output must be simple mean across the 9 variation signals.
- Layer 1 must not use equity-weighted aggregation.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from modules.adaptive_trend_LTS_mini.core.process_layer1.layer1_signal import _layer1_signal_for_ma


def _make_price_series(n: int = 96) -> pd.Series:
    """Create deterministic price data with trend + oscillation."""
    idx = pd.date_range("2024-01-01", periods=n, freq="h")
    t = np.arange(n, dtype=np.float64)
    # Upward drift + cyclic movement to create non-trivial long/short equity divergence.
    prices = 100.0 + 0.35 * t + 2.5 * np.sin(t / 3.0)
    return pd.Series(prices, index=idx, dtype="float64")


def _build_balanced_ma_tuple(prices: pd.Series) -> tuple[pd.Series, ...]:
    """Build 9 MA-like series that yield balanced long/short/neutral signals.

    Expected per-bar signal mix from price-vs-MA:
    - 4 long (+1): offsets below price
    - 4 short (-1): offsets above price
    - 1 neutral (0): exactly equal to price
    => simple mean should be exactly 0.0 per bar.
    """
    offsets = [0.0, 0.3, 1.0, 2.0, 3.0, -0.3, -1.0, -2.0, -3.0]
    return tuple((prices + off).astype("float64") for off in offsets)


def test_layer1_uses_simple_mean_not_equity_weighted() -> None:
    """Regression: Layer 1 must remain simple mean across 9 variation signals."""
    prices = _make_price_series()
    ma_tuple = _build_balanced_ma_tuple(prices)

    layer1_series, signals_tuple, equity_tuple = _layer1_signal_for_ma(
        prices=prices,
        ma_tuple=ma_tuple,
        lambda_val=0.02,
        decay_val=0.0,
    )

    signal_matrix = np.stack([np.asarray(s.values, dtype=np.float64) for s in signals_tuple])
    expected_simple_mean = np.nanmean(signal_matrix, axis=0)

    np.testing.assert_allclose(
        np.asarray(layer1_series.values, dtype=np.float64),
        expected_simple_mean,
        rtol=0.0,
        atol=1e-12,
    )

    # Guard against accidental regression back to equity-weighted aggregation.
    equity_matrix = np.stack([np.asarray(e.values, dtype=np.float64) for e in equity_tuple])
    weighted_num = np.nansum(signal_matrix * equity_matrix, axis=0)
    weighted_den = np.nansum(equity_matrix, axis=0)
    weighted_variant = np.divide(
        weighted_num,
        weighted_den,
        out=np.zeros_like(weighted_num),
        where=np.abs(weighted_den) > 1e-12,
    )

    finite_mask = np.isfinite(weighted_variant) & np.isfinite(layer1_series.values)
    assert finite_mask.any(), "Expected finite bars for parity comparison"

    max_delta = np.max(np.abs(weighted_variant[finite_mask] - layer1_series.values[finite_mask]))
    assert max_delta > 1e-6, "Test setup invalid: weighted and simple-mean variants should diverge"
