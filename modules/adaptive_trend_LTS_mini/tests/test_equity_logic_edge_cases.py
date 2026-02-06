"""
Edge case and logic tests for Adaptive Trend Classification (ATC).
"""

import os
import numpy as np
import pandas as pd
import pytest
from modules.adaptive_trend_LTS_mini.core.compute_atc_signals import compute_atc_signals
from modules.adaptive_trend_LTS_mini.core.compute_equity.core import set_equity_floor, get_equity_floor


@pytest.fixture
def sample_data():
    """Generate sample price data."""
    np.random.seed(42)
    n = 100
    prices = pd.Series(
        100.0 * np.cumprod(1 + np.random.normal(0.001, 0.01, n)), index=pd.date_range("2023-01-01", periods=n, freq="H")
    )
    return prices


@pytest.fixture
def standard_config():
    """Standard ATC configuration for testing."""
    return {
        "ema_len": 10,
        "hma_len": 10,
        "wma_len": 10,
        "dema_len": 10,
        "lsma_len": 10,
        "kama_len": 10,
        "ema_w": 1.0,
        "hma_w": 1.0,
        "wma_w": 1.0,
        "dema_w": 1.0,
        "lsma_w": 1.0,
        "kama_w": 1.0,
        "robustness": "Medium",
        "La": 0.02,
        "De": 0.03,
        "use_rust_backend": False,  # Use Python/Numba for predictable testing
        "parallel_l2": False,
    }


def test_src_index_mismatch(sample_data, standard_config):
    """
    Test 1: 'src' index mismatch.
    Verifies how the system handles 'src' having a different index/length than 'prices'.
    """
    prices = sample_data
    # Create src with different index (shifted by 5 bars)
    src = prices.copy()
    src.index = src.index + pd.Timedelta(hours=5)

    # This might fail if the system doesn't align src and prices properly before MA calculation
    # or if L2 calculation assumes they are aligned by position.
    try:
        results = compute_atc_signals(prices, src=src, **standard_config)
        assert "Average_Signal" in results
        assert len(results["Average_Signal"]) == len(prices)
        # Check if index is preserved from prices
        pd.testing.assert_index_equal(results["Average_Signal"].index, prices.index)
    except Exception as e:
        pytest.fail(f"compute_atc_signals failed with mismatched src index: {e}")


def test_weight_zero_disables_signal(sample_data, standard_config):
    """
    Test 2: Weight = 0.
    Verifies that if initial_weight is 0, the MA is disabled effectively.
    """
    prices = sample_data

    # 2. Config with all 0 weights
    config_all_zero = standard_config.copy()
    for k in ["ema_w", "hma_w", "wma_w", "dema_w", "lsma_w", "kama_w"]:
        config_all_zero[k] = 0.0

    results_all_zero = compute_atc_signals(prices, **config_all_zero)

    # With the fix, starting_equity=0 should result in equity=0 throughout.
    # Consequently, the weighted signal sum should be 0.
    # However, depending on normalization, it might result in NaN or 0.
    # If equity sum is 0, output should be 0 (Neutral).

    avg_signal = results_all_zero["Average_Signal"]

    # Check if signals are all zero (ignoring NaNs at start due to calculation window)
    # We assert that effectively the signal is dead.
    assert (avg_signal.fillna(0) == 0).all(), "Average_Signal should be all zero when weights are 0"


def test_atc_equity_floor_dynamic_change(sample_data, standard_config):
    """
    Test 3: equity_floor parameter.
    Verifies if passing different equity floor values is reflected in calculations.
    """
    prices = sample_data

    # Force some extreme losses to trigger floor
    # Price drops 99%
    crash_prices = prices.copy()
    crash_prices.iloc[-10:] = crash_prices.iloc[-11] * 0.01

    # Default floor (0.25)
    config_default = standard_config.copy()
    config_default["equity_floor"] = 0.25
    results_default = compute_atc_signals(crash_prices, **config_default)

    # Lower floor (0.01)
    config_low = standard_config.copy()
    config_low["equity_floor"] = 0.01
    results_low = compute_atc_signals(crash_prices, **config_low)

    # Check EMA_S (Layer 2 equity)
    ema_s_default = results_default["EMA_S"]
    ema_s_low = results_low["EMA_S"]

    # In the crash region, low floor should result in lower equity values
    assert ema_s_low.iloc[-1] < ema_s_default.iloc[-1], (
        f"Lower floor should allow lower equity: {ema_s_low.iloc[-1]} vs {ema_s_default.iloc[-1]}"
    )
    assert (ema_s_low.iloc[-10:] < 0.2).any(), "Low floor should allow equity to drop below 0.25"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
