import numpy as np
import pandas as pd
import pytest

from modules.adaptive_trend_enhance.core.compute_atc_signals.compute_atc_signals import compute_atc_signals
from modules.adaptive_trend_enhance.core.compute_moving_averages import set_of_moving_averages
from modules.adaptive_trend_enhance.core.process_layer1._parallel_layer1 import _layer1_parallel_atc_signals
from modules.adaptive_trend_enhance.core.process_layer1.layer1_signal import _layer1_signal_for_ma
from modules.adaptive_trend_enhance.utils.rate_of_change import rate_of_change
from modules.common.system.shared_memory_utils import (
    cleanup_shared_memory,
    reconstruct_series_from_shared_memory,
    setup_shared_memory_for_series,
)


@pytest.fixture
def sample_prices():
    """Create a sample price series for testing."""
    np.random.seed(42)
    n_bars = 1000
    prices = pd.Series(
        np.cumsum(np.random.randn(n_bars)) + 100, index=pd.date_range("2020-01-01", periods=n_bars, freq="h")
    )
    prices.name = "close"
    return prices


def test_shared_memory_series(sample_prices):
    """Test sharing a single Series via shared memory."""
    shm_info = setup_shared_memory_for_series(sample_prices)
    try:
        reconstructed = reconstruct_series_from_shared_memory(shm_info)
        pd.testing.assert_series_equal(sample_prices, reconstructed)
    finally:
        cleanup_shared_memory(shm_info)


def test_parallel_layer1_correctness(sample_prices):
    """Verify parallel Layer 1 results match sequential results."""
    R = rate_of_change(sample_prices)
    ma_configs = [
        ("EMA", 28, 1.0),
        ("HMA", 28, 1.0),
    ]

    ma_tuples = {}
    for ma_type, length, _ in ma_configs:
        ma_tuples[ma_type] = set_of_moving_averages(length, sample_prices, ma_type)

    # Calculate sequential
    sequential_signals = {}
    for ma_type, _, _ in ma_configs:
        sig, _, _ = _layer1_signal_for_ma(sample_prices, ma_tuples[ma_type], L=0.02, De=0.03, R=R)
        sequential_signals[ma_type] = sig

    # Calculate parallel
    parallel_signals = _layer1_parallel_atc_signals(
        prices=sample_prices,
        ma_tuples=ma_tuples,
        ma_configs=ma_configs,
        R=R,
        L=0.02,
        De=0.03,
        max_workers=2,
    )

    # Compare results
    assert len(parallel_signals) > 0
    for ma_type in sequential_signals:
        pd.testing.assert_series_equal(
            sequential_signals[ma_type], parallel_signals[ma_type], obj=f"Signal mismatch for {ma_type}"
        )


def test_full_atc_pipeline_integration(sample_prices):
    """Verify full ATC pipeline works with parallel Layer 1."""
    # Use a smaller dataset for integration test
    prices = sample_prices.iloc[:600]  # > 500 to trigger parallel

    result = compute_atc_signals(prices, ema_len=20, hull_len=20)

    assert "Average_Signal" in result
    assert "EMA_Signal" in result
    assert "HMA_Signal" in result
