import numpy as np
import pandas as pd
import pytest

from modules.adaptive_trend_enhance.core.compute_atc_signals.average_signal import calculate_average_signal
from modules.adaptive_trend_enhance.core.process_layer1.cut_signal import cut_signal
from modules.adaptive_trend_enhance.core.process_layer1.trend_sign import trend_sign
from modules.adaptive_trend_enhance.core.process_layer1.weighted_signal import weighted_signal


def test_weighted_signal_consistency():
    """Verify weighted_signal produces same result as manual loop."""
    n = 100
    np.random.seed(42)
    sigs = [pd.Series(np.random.randn(n)) for _ in range(5)]
    wgts = [pd.Series(np.random.rand(n)) for _ in range(5)]

    # Original logic (manual loop)
    num = np.zeros(n)
    den = np.zeros(n)
    for s, w in zip(sigs, wgts):
        num += s.values * w.values
        den += w.values
    expected = (num / den).round(2)

    # Vectorized logic
    result = weighted_signal(sigs, wgts)

    np.testing.assert_array_almost_equal(result.values, expected)


def test_cut_signal_consistency():
    """Verify cut_signal produces same result as iterative logic."""
    x = pd.Series([0.6, 0.4, 0.0, -0.4, -0.6, np.nan])

    # Expected: 1, 0, 0, 0, -1, 0 (threshold 0.49)
    res = cut_signal(x, threshold=0.49)
    expected = np.array([1, 0, 0, 0, -1, 0], dtype=np.int8)

    np.testing.assert_array_equal(res.values, expected)


def test_trend_sign_consistency():
    """Verify trend_sign produces same result as iterative logic."""
    s = pd.Series([1.5, 0.0, -0.5, np.nan, 2.0])

    # Current bar
    res = trend_sign(s, strategy=False)
    expected = np.array([1, 0, -1, 0, 1], dtype=np.int8)
    np.testing.assert_array_equal(res.values, expected)

    # Strategy mode (shift 1)
    res_strat = trend_sign(s, strategy=True)
    expected_strat = np.array([0, 1, 0, -1, 0], dtype=np.int8)  # Shifted
    np.testing.assert_array_equal(res_strat.values, expected_strat)


def test_average_signal_consistency():
    """Verify average_signal batch processing matches iterative accumulation."""
    n = 100
    np.random.seed(42)
    prices = pd.Series(np.random.randn(n))
    layer1_signals = {"EMA": pd.Series(np.random.randn(n)), "HMA": pd.Series(np.random.randn(n))}
    layer2_equities = {"EMA": pd.Series(np.random.rand(n)), "HMA": pd.Series(np.random.rand(n))}
    ma_configs = [("EMA", 20, 1.0), ("HMA", 20, 1.0)]

    # Iterative calculation (original)
    nom = np.zeros(n)
    den = np.zeros(n)
    for ma, _, _ in ma_configs:
        sv = layer1_signals[ma].values
        ev = layer2_equities[ma].values
        cv = np.where(sv > 0.49, 1.0, np.where(sv < -0.49, -1.0, 0.0))
        nom += cv * ev
        den += ev
    expected = np.divide(nom, den)
    expected = np.where(np.isfinite(expected), expected, 0.0)

    # Vectorized batch calculation
    result = calculate_average_signal(
        layer1_signals, layer2_equities, ma_configs, prices, long_threshold=0.49, short_threshold=-0.49
    )

    np.testing.assert_array_almost_equal(result.values, expected)


if __name__ == "__main__":
    pytest.main([__file__])
