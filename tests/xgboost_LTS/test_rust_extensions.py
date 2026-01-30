"""
Test Rust extensions for XGBoost module.
"""

import numpy as np
import pandas as pd
import pytest

# Try to import Rust extensions
try:
    from modules.xgboost_LTS.rust_extensions import (
        apply_directional_labels_rust,
        calculate_volatility_multiplier_rust,
        rolling_mean_rust,
        rolling_quantile_rust,
    )

    RUST_AVAILABLE = True
except ImportError:
    RUST_AVAILABLE = False


@pytest.mark.skipif(not RUST_AVAILABLE, reason="Rust extensions not available")
def test_rolling_mean_rust():
    """Test rolling mean implementation."""
    data = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
    window = 3

    result = rolling_mean_rust(data, window)

    # Expected: [NaN, NaN, 2.0, 3.0, 4.0]
    assert np.isnan(result[0])
    assert np.isnan(result[1])
    assert abs(result[2] - 2.0) < 1e-10
    assert abs(result[3] - 3.0) < 1e-10
    assert abs(result[4] - 4.0) < 1e-10


@pytest.mark.skipif(not RUST_AVAILABLE, reason="Rust extensions not available")
def test_rolling_quantile_rust():
    """Test rolling quantile implementation."""
    data = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
    window = 3
    q = 0.5  # median

    result = rolling_quantile_rust(data, window, q)

    # Expected: [NaN, NaN, 2.0, 3.0, 4.0]
    assert np.isnan(result[0])
    assert np.isnan(result[1])
    assert abs(result[2] - 2.0) < 1e-10
    assert abs(result[3] - 3.0) < 1e-10
    assert abs(result[4] - 4.0) < 1e-10


@pytest.mark.skipif(not RUST_AVAILABLE, reason="Rust extensions not available")
def test_apply_directional_labels_rust():
    """Test directional labeling."""
    close = np.array([100.0, 101.0, 102.0, 103.0, 104.0, 105.0])
    target_horizon = 2
    base_threshold = 0.01  # 1%

    labels, thresholds = apply_directional_labels_rust(close, target_horizon, base_threshold)

    # Check shapes
    assert len(labels) == len(close)
    assert len(thresholds) == len(close)

    # Check that last target_horizon labels are invalid (-1)
    assert labels[-1] == -1
    assert labels[-2] == -1


@pytest.mark.skipif(not RUST_AVAILABLE, reason="Rust extensions not available")
def test_calculate_volatility_multiplier_rust():
    """Test volatility multiplier calculation."""
    close = np.random.randn(100) + 100

    # Test without ATR
    result = calculate_volatility_multiplier_rust(close, None)

    assert len(result) == len(close)
    # Should be clipped to [1.5, 3.0]
    assert np.all(result >= 1.5)
    assert np.all(result <= 3.0)


@pytest.mark.skipif(not RUST_AVAILABLE, reason="Rust extensions not available")
def test_rust_vs_numba_rolling_mean():
    """Compare Rust implementation with Numba."""
    from modules.xgboost_LTS.utils.numba_funcs import rolling_mean_numba

    data = np.random.randn(1000)
    window = 50

    result_rust = rolling_mean_rust(data, window)
    result_numba = rolling_mean_numba(data, window)

    # Results should be very close
    valid_mask = ~np.isnan(result_rust) & ~np.isnan(result_numba)
    np.testing.assert_allclose(result_rust[valid_mask], result_numba[valid_mask], rtol=1e-10, atol=1e-10)


if __name__ == "__main__":
    if RUST_AVAILABLE:
        print("✅ Rust extensions available")
        test_rolling_mean_rust()
        test_rolling_quantile_rust()
        test_apply_directional_labels_rust()
        test_calculate_volatility_multiplier_rust()
        test_rust_vs_numba_rolling_mean()
        print("✅ All tests passed!")
    else:
        print("❌ Rust extensions not available")
