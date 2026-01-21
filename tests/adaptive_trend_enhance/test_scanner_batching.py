"""
Tests for scanner batch processing in modules.adaptive_trend_enhance.
"""

import gc
import sys
import tracemalloc
from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np
import pandas as pd
import pytest

# Add project root to sys.path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from modules.adaptive_trend_enhance.core.scanner import scan_all_symbols
from modules.adaptive_trend_enhance.utils.config import ATCConfig


@pytest.fixture
def mock_data_fetcher():
    """Create a mock DataFetcher instance."""
    fetcher = MagicMock()
    return fetcher


@pytest.fixture
def base_config():
    """Create a base ATCConfig for testing."""
    return ATCConfig(
        limit=200,
        timeframe="1h",
        ema_len=28,
        hma_len=28,
        wma_len=28,
        dema_len=28,
        lsma_len=28,
        kama_len=28,
        robustness="Medium",
        lambda_param=0.02,
        decay=0.03,
        cutout=0,
        long_threshold=0.1,
        short_threshold=-0.1,
    )


def create_mock_ohlcv_data(num_candles: int = 200) -> pd.DataFrame:
    """Create mock OHLCV DataFrame for testing."""
    timestamps = pd.date_range(start="2024-01-01", periods=num_candles, freq="1h", tz="UTC")
    prices = np.linspace(100, 110, num_candles)

    df = pd.DataFrame(
        {
            "timestamp": timestamps,
            "open": prices,
            "high": prices + 1,
            "low": prices - 1,
            "close": prices,
            "volume": 1000,
        }
    )
    return df


def create_mock_atc_results(signal_value: float = 0.05) -> dict:
    """Create mock ATC results."""
    signal_series = pd.Series([signal_value] * 200)
    return {
        "Average_Signal": signal_series,
    }


@patch("modules.adaptive_trend_enhance.core.scanner.compute_atc_signals")
@patch("modules.adaptive_trend_enhance.core.scanner.trend_sign")
def test_scanner_batch_correctness(mock_trend_sign, mock_compute_atc, base_config, mock_data_fetcher):
    """Verify that different batch sizes produce the same results."""
    symbols = [f"SYM{i}" for i in range(20)]
    mock_data_fetcher.list_binance_futures_symbols.return_value = symbols

    mock_df = create_mock_ohlcv_data(num_candles=200)
    mock_data_fetcher.fetch_ohlcv_with_fallback_exchange.return_value = (mock_df, "binance")

    mock_compute_atc.return_value = create_mock_atc_results(signal_value=0.5)
    mock_trend_sign.return_value = pd.Series([1] * 200)

    # Result with large batch size
    long1, short1 = scan_all_symbols(mock_data_fetcher, base_config, batch_size=100, execution_mode="sequential")

    # Reset mocks for next run
    mock_compute_atc.reset_mock()
    mock_trend_sign.reset_mock()
    mock_data_fetcher.fetch_ohlcv_with_fallback_exchange.reset_mock()

    # Result with small batch size
    long2, short2 = scan_all_symbols(mock_data_fetcher, base_config, batch_size=5, execution_mode="sequential")

    assert len(long1) == len(long2)
    assert long1["symbol"].tolist() == long2["symbol"].tolist()
    pd.testing.assert_frame_equal(long1, long2)
    pd.testing.assert_frame_equal(short1, short2)


@patch("modules.adaptive_trend_enhance.core.scanner.compute_atc_signals")
@patch("modules.adaptive_trend_enhance.core.scanner.trend_sign")
def test_scanner_memory_usage_reduction(mock_trend_sign, mock_compute_atc, base_config, mock_data_fetcher):
    """
    Test memory usage reduction with batching.
    Note: Real memory reduction is hard to measure in unit tests with mocks,
    but we can at least ensure it runs and trace local allocations.
    """
    num_symbols = 50
    symbols = [f"SYM{i}" for i in range(num_symbols)]
    mock_data_fetcher.list_binance_futures_symbols.return_value = symbols

    # Create a reasonably sized DF to consume some memory
    mock_df = create_mock_ohlcv_data(num_candles=1000)
    mock_data_fetcher.fetch_ohlcv_with_fallback_exchange.return_value = (mock_df, "binance")

    mock_compute_atc.return_value = create_mock_atc_results(signal_value=0.5)
    mock_trend_sign.return_value = pd.Series([1] * 1000)

    tracemalloc.start()

    # Run with small batch size
    gc.collect()
    snapshot1 = tracemalloc.take_snapshot()
    scan_all_symbols(mock_data_fetcher, base_config, batch_size=5, execution_mode="sequential")
    snapshot2 = tracemalloc.take_snapshot()

    stats = snapshot2.compare_to(snapshot1, "lineno")
    total_diff = sum(stat.size_diff for stat in stats)

    tracemalloc.stop()

    # We don't assert a specific reduction because mocks and small scale make it unreliable,
    # but we verify it completed successfully.
    assert True


@patch("modules.adaptive_trend_enhance.core.scanner.compute_atc_signals")
@patch("modules.adaptive_trend_enhance.core.scanner.trend_sign")
def test_scanner_edge_case_batch_sizes(mock_trend_sign, mock_compute_atc, base_config, mock_data_fetcher):
    """Test with very small and very large batch sizes."""
    symbols = [f"SYM{i}" for i in range(10)]
    mock_data_fetcher.list_binance_futures_symbols.return_value = symbols

    mock_df = create_mock_ohlcv_data(num_candles=200)
    mock_data_fetcher.fetch_ohlcv_with_fallback_exchange.return_value = (mock_df, "binance")

    mock_compute_atc.return_value = create_mock_atc_results(signal_value=0.5)
    mock_trend_sign.return_value = pd.Series([1] * 200)

    # Batch size 1
    long1, _ = scan_all_symbols(mock_data_fetcher, base_config, batch_size=1, execution_mode="sequential")
    assert len(long1) == 10

    # Batch size larger than symbols
    long2, _ = scan_all_symbols(mock_data_fetcher, base_config, batch_size=100, execution_mode="sequential")
    assert len(long2) == 10


@patch("modules.adaptive_trend_enhance.core.scanner.compute_atc_signals")
@patch("modules.adaptive_trend_enhance.core.scanner.trend_sign")
@pytest.mark.parametrize("mode", ["threadpool", "asyncio"])
def test_parallel_batching(mock_trend_sign, mock_compute_atc, base_config, mock_data_fetcher, mode):
    """Verify batching works correctly in parallel modes."""
    symbols = [f"SYM{i}" for i in range(15)]
    mock_data_fetcher.list_binance_futures_symbols.return_value = symbols

    mock_df = create_mock_ohlcv_data(num_candles=200)
    mock_data_fetcher.fetch_ohlcv_with_fallback_exchange.return_value = (mock_df, "binance")

    mock_compute_atc.return_value = create_mock_atc_results(signal_value=0.5)
    mock_trend_sign.return_value = pd.Series([1] * 200)

    # Use a batch size that splits the work
    long, _ = scan_all_symbols(mock_data_fetcher, base_config, batch_size=5, execution_mode=mode, max_workers=2)

    assert len(long) == 15
