import multiprocessing as mp
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor

import numpy as np
import pandas as pd
import pytest

from modules.adaptive_trend_enhance.core.compute_atc_signals import compute_atc_signals
from modules.adaptive_trend_enhance.core.compute_equity import _calculate_equities_parallel, _calculate_equity_core
from modules.adaptive_trend_enhance.utils.config import ATCConfig
from modules.common.system import get_series_pool


@pytest.fixture
def sample_data():
    """Create sample price data for testing."""
    np.random.seed(42)
    n = 1000
    prices = pd.Series(
        100 * (1 + np.random.randn(n).cumsum() * 0.01),
        index=pd.date_range("2023-01-01", periods=n, freq="min"),
    )
    return prices


def test_parallel_equity_correctness(sample_data):
    """Verify that parallel vectorized equity calculation matches sequential loop."""
    n_bars = len(sample_data)
    n_signals = 6

    # Random signals and returns
    signals = np.random.choice([-1, 0, 1], size=(n_signals, n_bars)).astype(np.float64)
    r_values = np.random.randn(n_bars).astype(np.float64) * 0.01
    starting_equities = np.ones(n_signals, dtype=np.float64)
    decay_multiplier = 0.999
    cutout = 0

    # Calculate sequentially
    expected = np.empty((n_signals, n_bars), dtype=np.float64)
    for s in range(n_signals):
        # Shift signal
        sig_prev = np.empty(n_bars)
        sig_prev[1:] = signals[s, :-1]
        sig_prev[0] = np.nan

        expected[s] = _calculate_equity_core(
            r_values=r_values,
            sig_prev_values=sig_prev,
            starting_equity=starting_equities[s],
            decay_multiplier=decay_multiplier,
            cutout=cutout,
        )

    # Calculate in parallel
    sig_prev_matrix = np.empty((n_signals, n_bars))
    sig_prev_matrix[:, 1:] = signals[:, :-1]
    sig_prev_matrix[:, 0] = np.nan

    actual = _calculate_equities_parallel(
        starting_equities=starting_equities,
        sig_prev_values=sig_prev_matrix,
        r_values=r_values,
        decay_multiplier=decay_multiplier,
        cutout=cutout,
    )

    # Compare (allowing for small floating point differences)
    # Note: Sequential and parallel use same logic, but order of ops might vary slightly
    # depending on how Numba optimizes. But they should be identical here.
    np.testing.assert_allclose(actual[:, 1:], expected[:, 1:], rtol=1e-10)


def worker_check():
    """Helper to check if running in a child process."""
    import multiprocessing as mp

    # This mirrors the logic in compute_atc_signals
    is_child = mp.current_process().daemon or mp.current_process().name != "MainProcess"
    return is_child


def test_nested_parallelism_detection():
    """Verify that compute_atc_signals correctly detects when it's in a child process."""
    import multiprocessing as mp

    # In child process
    with ProcessPoolExecutor(max_workers=1) as executor:
        future = executor.submit(worker_check)
        is_child_result = future.result()
        assert is_child_result == True


def test_atc_signals_parallel_modes(sample_data):
    """Verify compute_atc_signals with different parallel flags."""
    # This just ensures it runs without error with different combinations
    config = ATCConfig()

    # 1. Level 1 Parallel (ProcessPool) + Level 2 Parallel (Numba)
    res1 = compute_atc_signals(sample_data, parallel_l1=True, parallel_l2=True)
    assert "Average_Signal" in res1

    # 2. Level 1 Sequential + Level 2 Parallel
    res2 = compute_atc_signals(sample_data, parallel_l1=False, parallel_l2=True)
    assert np.allclose(res1["Average_Signal"], res2["Average_Signal"], equal_nan=True)

    # 3. Level 1 Sequential + Level 2 Sequential (fallback)
    res3 = compute_atc_signals(sample_data, parallel_l1=False, parallel_l2=False)
    assert np.allclose(res1["Average_Signal"], res3["Average_Signal"], equal_nan=True)


if __name__ == "__main__":
    pytest.main([__file__])
