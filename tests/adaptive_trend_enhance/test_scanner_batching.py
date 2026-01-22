"""
Tests for scanner batch processing in modules.adaptive_trend_enhance.
"""

import sys
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
        limit=100,  # Match default mock data size
        timeframe="1h",
        ema_len=10,  # Reduced for faster computation
        hma_len=10,
        wma_len=10,
        dema_len=10,
        lsma_len=10,
        kama_len=10,
        robustness="Medium",
        lambda_param=0.02,
        decay=0.03,
        cutout=0,
        long_threshold=0.1,
        short_threshold=-0.1,
    )


def create_mock_ohlcv_data(num_candles: int = 100) -> pd.DataFrame:
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


def create_mock_atc_results(signal_value: float = 0.05, num_candles: int = 100) -> dict:
    """Create mock ATC results."""
    signal_series = pd.Series([signal_value] * num_candles)
    return {
        "Average_Signal": signal_series,
    }


def _make_mock_process():
    """Return a mock _process_symbol that yields one result per symbol."""

    def mock_process(symbol, data_fetcher, atc_config, min_signal):
        return {"symbol": symbol, "signal": 0.5, "trend": 1, "price": 100.0, "exchange": "binance"}

    return mock_process


@pytest.mark.timeout(30)  # 30 second timeout
@patch("modules.adaptive_trend_enhance.core.scanner.sequential._process_symbol")
def test_scanner_edge_case_batch_sizes(mock_process_symbol, base_config, mock_data_fetcher):
    """Test with very small and very large batch sizes."""
    symbols = [f"SYM{i}" for i in range(5)]
    mock_data_fetcher.list_binance_futures_symbols.return_value = symbols

    mock_process_symbol.side_effect = _make_mock_process()

    # Batch size 1
    long1, _ = scan_all_symbols(mock_data_fetcher, base_config, batch_size=1, execution_mode="sequential")
    assert len(long1) == 5

    # Reset mock
    mock_process_symbol.reset_mock()
    mock_process_symbol.side_effect = _make_mock_process()

    # Batch size larger than symbols
    long2, _ = scan_all_symbols(mock_data_fetcher, base_config, batch_size=100, execution_mode="sequential")
    assert len(long2) == 5


@pytest.mark.timeout(25)  # 25 second timeout
@patch("modules.adaptive_trend_enhance.core.scanner.asyncio_scan._process_symbol")
@patch("modules.adaptive_trend_enhance.core.scanner.threadpool._process_symbol")
@pytest.mark.parametrize("mode", ["threadpool", "asyncio"])
def test_parallel_batching(mock_tp, mock_asyncio, base_config, mock_data_fetcher, mode):
    """Verify batching works correctly in parallel modes."""
    symbols = [f"SYM{i}" for i in range(6)]
    mock_data_fetcher.list_binance_futures_symbols.return_value = symbols

    side_effect = _make_mock_process()
    mock_tp.side_effect = side_effect
    mock_asyncio.side_effect = side_effect

    # Use a batch size that splits the work
    long, _ = scan_all_symbols(mock_data_fetcher, base_config, batch_size=3, execution_mode=mode, max_workers=2)

    assert len(long) == 6
