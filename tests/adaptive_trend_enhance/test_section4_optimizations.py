"""
Verification for Section 4 Optimizations: Average Signal & ROC Caching.
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

# Add project root
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from modules.adaptive_trend_enhance.core.compute_atc_signals.average_signal import calculate_average_signal
from modules.adaptive_trend_enhance.utils.cache_manager import get_cache_manager, reset_cache_manager
from modules.adaptive_trend_enhance.utils.rate_of_change import rate_of_change


@pytest.fixture(autouse=True)
def clean_cache():
    """Reset the cache manager before and after each test."""
    reset_cache_manager()
    yield
    reset_cache_manager()


def test_average_signal_correctness():
    """Test that vectorized average_signal produces correct weighted average."""
    # Setup data
    size = 100
    prices = pd.Series(range(size), dtype="float64")

    # Layer 1 signals:
    # EMA: all 1.0 (above threshold)
    # HMA: all -1.0 (below threshold)
    # WMA: all 0.5 (below long threshold 0.8, above short -0.8 -> 0.0)

    layer1_signals = {
        "EMA": pd.Series(np.ones(size) * 0.9, index=prices.index),
        "HMA": pd.Series(np.ones(size) * -0.9, index=prices.index),
        "WMA": pd.Series(np.ones(size) * 0.5, index=prices.index),  # Should become 0
    }

    # Layer 2 equities:
    # All equal weight 1.0
    layer2_equities = {
        "EMA": pd.Series(np.ones(size), index=prices.index),
        "HMA": pd.Series(np.ones(size), index=prices.index),
        "WMA": pd.Series(np.ones(size), index=prices.index),
    }

    ma_configs = [("EMA", 10, 1.0), ("HMA", 10, 1.0), ("WMA", 10, 1.0)]

    # Thresholds: 0.8 / -0.8
    # EMA (0.9 > 0.8) -> 1.0
    # HMA (-0.9 < -0.8) -> -1.0
    # WMA (0.5) -> 0.0

    # Weighted Sum: 1.0*1.0 + (-1.0)*1.0 + 0.0*1.0 = 0.0
    # Denom: 1.0 + 1.0 + 1.0 = 3.0
    # Expected Avg: 0.0

    avg = calculate_average_signal(
        layer1_signals, layer2_equities, ma_configs, prices, long_threshold=0.8, short_threshold=-0.8, cutout=0
    )

    assert np.allclose(avg.values, 0.0)

    # Test cutout
    # Cutout should force first 10 to 0. (Actually inputs are constant so output is 0 anyway)
    # Note: avg_cutout is not used as inputs are constant, but we verify cutout logic below

    # Let's change WMA to be valid long signal 0.9
    layer1_signals["WMA"] = pd.Series(np.ones(size) * 0.9, index=prices.index)
    # Sum: 1 + -1 + 1 = 1. Denom = 3. Avg = 1/3 ~ 0.333

    avg2 = calculate_average_signal(
        layer1_signals, layer2_equities, ma_configs, prices, long_threshold=0.8, short_threshold=-0.8, cutout=10
    )

    # After cutout (10), values should be 0.333.
    # Before cutout, signals are 0?
    # Wait, the logic is `cut_vals[:cutout] = 0`.
    # So for first 10: Sum = 0*1 + 0*1 + 0*1 = 0?
    # Yes.

    assert np.allclose(avg2.values[10:], 1.0 / 3.0)
    assert np.allclose(avg2.values[:10], 0.0)


def test_roc_caching(clean_cache):
    """Verify ROC caching works."""
    cache = get_cache_manager()

    s = pd.Series([100, 101, 102, 103], dtype="float64")

    r1 = rate_of_change(s)
    r2 = rate_of_change(s)

    stats = cache.get_stats()
    # Should have 1 hit, 1 miss (populate)
    # Or 1 entry

    assert stats["entries"] == 1
    # Check if hit count increased
    assert stats["hits"] >= 1

    # Verify results identical
    pd.testing.assert_series_equal(r1, r2)
