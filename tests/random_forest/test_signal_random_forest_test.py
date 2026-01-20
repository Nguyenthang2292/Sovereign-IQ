"""
Tests for signal_random_forest_test.py using pytest.

Tests all functionality including:
- ResourceMonitor class
- Utility functions (calculate_confidence, get_signal_direction)
- Data fetching functions
- Batch creation and processing
- Multiple execution modes (sequential, threading, hybrid)
"""

from datetime import datetime
from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np
import pandas as pd
import pytest

# Add parent directory to path
import sys

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from config.random_forest import MODELS_DIR, RANDOM_FOREST_MODEL_FILENAME


# ============================================================================
# Pytest Fixtures
# ============================================================================


@pytest.fixture
def mock_data_fetcher():
    """Create a mock DataFetcher instance."""
    return MagicMock()


@pytest.fixture
def sample_ohlcv_data():
    """Create sample OHLCV DataFrame for testing."""
    timestamps = pd.date_range(start="2024-01-01", periods=200, freq="1h", tz="UTC")

    np.random.seed(42)
    returns = np.random.normal(0, 0.01, 200)
    prices = 100.0 * (1 + returns).cumprod()

    df = pd.DataFrame(
        {
            "timestamp": timestamps,
            "open": prices * (1 + np.random.normal(0, 0.001, 200)),
            "high": prices * (1 + abs(np.random.normal(0, 0.002, 200))),
            "low": prices * (1 - abs(np.random.normal(0, 0.002, 200))),
            "close": prices,
            "volume": np.random.uniform(10000, 100000, 200),
        }
    )
    df.set_index("timestamp", inplace=True)
    return df


@pytest.fixture
def mock_ohlcv_data():
    """Create mock OHLCV DataFrame for testing."""
    timestamps = pd.date_range(start="2024-01-01", periods=200, freq="1h", tz="UTC")

    np.random.seed(42)
    returns = np.random.normal(0, 0.01, 200)
    prices = 100.0 * (1 + returns).cumprod()

    df = pd.DataFrame(
        {
            "timestamp": timestamps,
            "open": prices * (1 + np.random.normal(0, 0.001, 200)),
            "high": prices * (1 + abs(np.random.normal(0, 0.002, 200))),
            "low": prices * (1 - abs(np.random.normal(0, 0.002, 200))),
            "close": prices,
            "volume": np.random.uniform(10000, 100000, 200),
        }
    )
    df.set_index("timestamp", inplace=True)
    return df


def get_test_ohlcv_data(num_candles: int = 200) -> pd.DataFrame:
    """Create test OHLCV data for use in tests."""
    timestamps = pd.date_range(start="2024-01-01", periods=num_candles, freq="1h", tz="UTC")

    np.random.seed(42)
    returns = np.random.normal(0, 0.01, num_candles)
    prices = 100.0 * (1 + returns).cumprod()

    df = pd.DataFrame(
        {
            "timestamp": timestamps,
            "open": prices * (1 + np.random.normal(0, 0.001, num_candles)),
            "high": prices * (1 + abs(np.random.normal(0, 0.002, num_candles))),
            "low": prices * (1 - abs(np.random.normal(0, 0.002, num_candles))),
            "close": prices,
            "volume": np.random.uniform(10000, 100000, num_candles),
        }
    )
    df.set_index("timestamp", inplace=True)
    return df


# ============================================================================
# Tests for Utility Functions
# ============================================================================


@pytest.mark.parametrize(
    "signal_value,expected_direction",
    [
        (1, "LONG"),
        (-1, "SHORT"),
        (0, "NEUTRAL"),
        (2, "NEUTRAL"),  # Invalid, should return NEUTRAL
        (-2, "NEUTRAL"),  # Invalid, should return NEUTRAL
    ],
)
def test_get_signal_direction(signal_value, expected_direction):
    """Test get_signal_direction returns correct direction."""
    from modules.random_forest.signal_random_forest_test import get_signal_direction

    assert get_signal_direction(signal_value) == expected_direction


@pytest.mark.parametrize(
    "confidence_value,expected_confidence",
    [
        (0.0, 0.0),
        (0.5, 0.5),
        (1.0, 1.0),
        (1.5, 1.0),  # Clamped to 1.0
        (-0.5, 0.0),  # Clamped to 0.0
    ],
)
def test_calculate_confidence(confidence_value, expected_confidence):
    """Test calculate_confidence clamps values to [0.0, 1.0]."""
    from modules.random_forest.signal_random_forest_test import calculate_confidence

    assert calculate_confidence(confidence_value) == expected_confidence


# ============================================================================
# Tests for ResourceMonitor
# ============================================================================


def test_resource_monitor_init():
    """Test ResourceMonitor initialization."""
    from modules.random_forest.signal_random_forest_test import ResourceMonitor

    monitor = ResourceMonitor(max_memory_pct=80.0, max_cpu_pct=80.0)

    assert monitor.max_memory_pct == 80.0
    assert monitor.max_cpu_pct == 80.0


def test_resource_monitor_get_memory_usage():
    """Test get_memory_usage returns tuple with expected structure."""
    from modules.random_forest.signal_random_forest_test import ResourceMonitor

    monitor = ResourceMonitor()
    process_mem, system_mem_pct, available_mem = monitor.get_memory_usage()

    assert isinstance(process_mem, float)
    assert isinstance(system_mem_pct, float)
    assert isinstance(available_mem, float)

    # If psutil is not available, all values should be 0
    import modules.random_forest.signal_random_forest_test as signal_module

    if not signal_module.PSUTIL_AVAILABLE:
        assert process_mem == 0.0
        assert system_mem_pct == 0.0
        assert available_mem == 0.0
    else:
        # Should have some values
        assert process_mem >= 0.0
        assert 0.0 <= system_mem_pct <= 100.0
        assert available_mem >= 0.0


def test_resource_monitor_get_cpu_usage():
    """Test get_cpu_usage returns tuple with expected structure."""
    from modules.random_forest.signal_random_forest_test import ResourceMonitor

    monitor = ResourceMonitor()
    process_cpu, system_cpu = monitor.get_cpu_usage()

    assert isinstance(process_cpu, float)
    assert isinstance(system_cpu, float)

    # If psutil is not available, all values should be 0
    import modules.random_forest.signal_random_forest_test as signal_module

    if not signal_module.PSUTIL_AVAILABLE:
        assert process_cpu == 0.0
        assert system_cpu == 0.0
    else:
        # Should have some values
        assert 0.0 <= process_cpu <= 100.0
        assert 0.0 <= system_cpu <= 100.0


# ============================================================================
# Tests for fetch_data_for_symbol
# ============================================================================


def test_fetch_data_for_symbol_success(mock_data_fetcher, sample_ohlcv_data):
    """Test fetch_data_for_symbol returns data on success."""
    from modules.random_forest.signal_random_forest_test import fetch_data_for_symbol

    mock_data_fetcher.fetch_ohlcv_with_fallback_exchange = MagicMock(return_value=(sample_ohlcv_data, "binance"))

    result = fetch_data_for_symbol("BTC/USDT", mock_data_fetcher, "1h", 200)

    assert result is not None
    df, exchange_id = result
    assert isinstance(df, pd.DataFrame)
    assert exchange_id == "binance"


def test_fetch_data_for_symbol_empty_dataframe(mock_data_fetcher):
    """Test fetch_data_for_symbol returns None for empty DataFrame."""
    from modules.random_forest.signal_random_forest_test import fetch_data_for_symbol

    mock_data_fetcher.fetch_ohlcv_with_fallback_exchange = MagicMock(return_value=(pd.DataFrame(), "binance"))

    result = fetch_data_for_symbol("BTC/USDT", mock_data_fetcher, "1h", 200)

    assert result is None


def test_fetch_data_for_symbol_none_result(mock_data_fetcher):
    """Test fetch_data_for_symbol returns None when fetch fails."""
    from modules.random_forest.signal_random_forest_test import fetch_data_for_symbol

    mock_data_fetcher.fetch_ohlcv_with_fallback_exchange = MagicMock(return_value=(None, None))

    result = fetch_data_for_symbol("BTC/USDT", mock_data_fetcher, "1h", 200)

    assert result is None


def test_fetch_data_for_symbol_exception(mock_data_fetcher):
    """Test fetch_data_for_symbol handles exceptions."""
    from modules.random_forest.signal_random_forest_test import fetch_data_for_symbol

    mock_data_fetcher.fetch_ohlcv_with_fallback_exchange = MagicMock(side_effect=Exception("Connection error"))

    result = fetch_data_for_symbol("BTC/USDT", mock_data_fetcher, "1h", 200)

    assert result is None


# ============================================================================
# Tests for compute_random_forest_signals_for_data
# ============================================================================


@patch("modules.random_forest.signal_random_forest_test.get_random_forest_signal")
def test_compute_random_forest_signals_for_data_success(
    mock_get_signal, sample_ohlcv_data, mock_data_fetcher
):
    """Test compute_random_forest_signals_for_data returns result dict."""
    from modules.random_forest.signal_random_forest_test import compute_random_forest_signals_for_data

    mock_get_signal.return_value = (1, 0.85)  # LONG signal with 0.85 confidence

    result = compute_random_forest_signals_for_data(
        "BTC/USDT",
        "1h",
        sample_ohlcv_data,
        "binance",
        mock_data_fetcher,
        200,
        model_path=None,
    )

    assert result is not None
    assert result["symbol"] == "BTC/USDT"
    assert result["timeframe"] == "1h"
    assert result["signal"] == 1
    assert result["signal_direction"] == "LONG"
    assert result["confidence"] == 0.85
    assert result["exchange"] == "binance"
    assert "timestamp" in result


@patch("modules.random_forest.signal_random_forest_test.get_random_forest_signal")
def test_compute_random_forest_signals_for_data_short_signal(
    mock_get_signal, sample_ohlcv_data, mock_data_fetcher
):
    """Test compute_random_forest_signals_for_data with SHORT signal."""
    from modules.random_forest.signal_random_forest_test import compute_random_forest_signals_for_data

    mock_get_signal.return_value = (-1, 0.75)  # SHORT signal with 0.75 confidence

    result = compute_random_forest_signals_for_data(
        "BTC/USDT",
        "1h",
        sample_ohlcv_data,
        "binance",
        mock_data_fetcher,
        200,
        model_path=None,
    )

    assert result is not None
    assert result["signal"] == -1
    assert result["signal_direction"] == "SHORT"
    assert result["confidence"] == 0.75


@patch("modules.random_forest.signal_random_forest_test.get_random_forest_signal")
def test_compute_random_forest_signals_for_data_neutral_signal(
    mock_get_signal, sample_ohlcv_data, mock_data_fetcher
):
    """Test compute_random_forest_signals_for_data with NEUTRAL signal."""
    from modules.random_forest.signal_random_forest_test import compute_random_forest_signals_for_data

    mock_get_signal.return_value = (0, 0.5)  # NEUTRAL signal with 0.5 confidence

    result = compute_random_forest_signals_for_data(
        "BTC/USDT",
        "1h",
        sample_ohlcv_data,
        "binance",
        mock_data_fetcher,
        200,
        model_path=None,
    )

    assert result is not None
    assert result["signal"] == 0
    assert result["signal_direction"] == "NEUTRAL"
    assert result["confidence"] == 0.5


@patch("modules.random_forest.signal_random_forest_test.get_random_forest_signal")
def test_compute_random_forest_signals_for_data_none_result(
    mock_get_signal, sample_ohlcv_data, mock_data_fetcher
):
    """Test compute_random_forest_signals_for_data returns None when signal is None."""
    from modules.random_forest.signal_random_forest_test import compute_random_forest_signals_for_data

    mock_get_signal.return_value = None

    result = compute_random_forest_signals_for_data(
        "BTC/USDT",
        "1h",
        sample_ohlcv_data,
        "binance",
        mock_data_fetcher,
        200,
        model_path=None,
    )

    assert result is None


@patch("modules.random_forest.signal_random_forest_test.get_random_forest_signal")
def test_compute_random_forest_signals_for_data_missing_close_column(
    mock_get_signal, sample_ohlcv_data, mock_data_fetcher
):
    """Test compute_random_forest_signals_for_data returns None when close column is missing."""
    from modules.random_forest.signal_random_forest_test import compute_random_forest_signals_for_data

    df_without_close = sample_ohlcv_data.drop(columns=["close"])

    result = compute_random_forest_signals_for_data(
        "BTC/USDT",
        "1h",
        df_without_close,
        "binance",
        mock_data_fetcher,
        200,
        model_path=None,
    )

    assert result is None


@patch("modules.random_forest.signal_random_forest_test.get_random_forest_signal")
def test_compute_random_forest_signals_for_data_exception(
    mock_get_signal, sample_ohlcv_data, mock_data_fetcher
):
    """Test compute_random_forest_signals_for_data handles exceptions."""
    from modules.random_forest.signal_random_forest_test import compute_random_forest_signals_for_data

    mock_get_signal.side_effect = Exception("Model error")

    result = compute_random_forest_signals_for_data(
        "BTC/USDT",
        "1h",
        sample_ohlcv_data,
        "binance",
        mock_data_fetcher,
        200,
        model_path=None,
    )

    assert result is None


# ============================================================================
# Tests for test_random_forest_for_symbol
# ============================================================================


@patch("modules.random_forest.signal_random_forest_test.compute_random_forest_signals_for_data")
@patch("modules.random_forest.signal_random_forest_test.fetch_data_for_symbol")
def test_test_random_forest_for_symbol_success(
    mock_fetch, mock_compute, mock_data_fetcher
):
    """Test test_random_forest_for_symbol returns result on success."""
    from modules.random_forest.signal_random_forest_test import test_random_forest_for_symbol

    mock_fetch.return_value = (mock_ohlcv_data(), "binance")
    mock_compute.return_value = {
        "symbol": "BTC/USDT",
        "timeframe": "1h",
        "signal": 1,
        "signal_direction": "LONG",
        "confidence": 0.85,
    }

    result = test_random_forest_for_symbol("BTC/USDT", mock_data_fetcher, "1h", 200, None)

    assert result is not None
    assert result["symbol"] == "BTC/USDT"


@patch("modules.random_forest.signal_random_forest_test.fetch_data_for_symbol")
def test_test_random_forest_for_symbol_fetch_failure(mock_fetch, mock_data_fetcher):
    """Test test_random_forest_for_symbol returns None on fetch failure."""
    from modules.random_forest.signal_random_forest_test import test_random_forest_for_symbol

    mock_fetch.return_value = None

    result = test_random_forest_for_symbol("BTC/USDT", mock_data_fetcher, "1h", 200, None)

    assert result is None


@patch("modules.random_forest.signal_random_forest_test.compute_random_forest_signals_for_data")
@patch("modules.random_forest.signal_random_forest_test.fetch_data_for_symbol")
def test_test_random_forest_for_symbol_compute_failure(
    mock_fetch, mock_compute, mock_data_fetcher
):
    """Test test_random_forest_for_symbol returns None on compute failure."""
    from modules.random_forest.signal_random_forest_test import test_random_forest_for_symbol

    mock_fetch.return_value = (mock_ohlcv_data(), "binance")
    mock_compute.return_value = None

    result = test_random_forest_for_symbol("BTC/USDT", mock_data_fetcher, "1h", 200, None)

    assert result is None


# ============================================================================
# Tests for create_batches
# ============================================================================


@pytest.mark.parametrize(
    "num_symbols,batch_size,expected_num_batches",
    [
        (10, 5, 2),
        (10, 3, 4),
        (10, 10, 1),
        (11, 5, 3),
        (0, 5, 0),
        (1, 5, 1),
    ],
)
def test_create_batches(num_symbols, batch_size, expected_num_batches):
    """Test create_batches creates correct number of batches."""
    from modules.random_forest.signal_random_forest_test import create_batches

    symbols = [f"SYMB{i}" for i in range(num_symbols)]
    batches = create_batches(symbols, batch_size)

    assert len(batches) == expected_num_batches

    # Verify all symbols are in batches
    all_symbols_in_batches = []
    for batch in batches:
        all_symbols_in_batches.extend(batch)

    assert len(all_symbols_in_batches) == num_symbols


def test_create_batches_preserves_order():
    """Test create_batches preserves symbol order."""
    from modules.random_forest.signal_random_forest_test import create_batches

    symbols = ["BTC/USDT", "ETH/USDT", "BNB/USDT", "SOL/USDT", "ADA/USDT"]
    batches = create_batches(symbols, 2)

    expected_batches = [["BTC/USDT", "ETH/USDT"], ["BNB/USDT", "SOL/USDT"], ["ADA/USDT"]]

    assert batches == expected_batches


def test_create_batches_empty_list():
    """Test create_batches returns empty list for empty input."""
    from modules.random_forest.signal_random_forest_test import create_batches

    batches = create_batches([], 10)

    assert batches == []


# ============================================================================
# Tests for calculate_optimal_batch_size
# ============================================================================


@patch("modules.random_forest.signal_random_forest_test.PSUTIL_AVAILABLE", False)
def test_calculate_optimal_batch_size_no_psutil():
    """Test calculate_optimal_batch_size returns default when psutil not available."""
    from modules.random_forest.signal_random_forest_test import calculate_optimal_batch_size

    batch_size = calculate_optimal_batch_size(100, 3)

    assert batch_size == 50  # Default value


@patch("modules.random_forest.signal_random_forest_test.PSUTIL_AVAILABLE", True)
@patch("psutil.virtual_memory")
def test_calculate_optimal_batch_size_boundaries(mock_vm):
    """Test calculate_optimal_batch_size boundary conditions."""
    from modules.random_forest.signal_random_forest_test import calculate_optimal_batch_size

    # Case 1: High memory available -> should be capped at 100
    mock_vm.return_value.available = 10 * 1024 * 1024 * 1024  # 10 GB
    batch_size = calculate_optimal_batch_size(1000, 3)
    assert batch_size == 100

    # Case 2: Very low memory available -> should be limited at 10
    mock_vm.return_value.available = 2 * 1024 * 1024  # 2 MB (usable ~1.4MB -> batch size ~4 -> limited to 10)
    batch_size = calculate_optimal_batch_size(1000, 3)
    assert batch_size == 10


# ============================================================================
# Tests for process_symbol_batch_sequential
# ============================================================================


@patch("modules.random_forest.signal_random_forest_test.test_random_forest_for_symbol")
def test_process_symbol_batch_sequential(mock_test_rf, mock_data_fetcher):
    """Test process_symbol_batch_sequential processes all symbols."""
    from modules.random_forest.signal_random_forest_test import process_symbol_batch_sequential

    symbols = ["BTC/USDT", "ETH/USDT", "BNB/USDT"]
    mock_test_rf.side_effect = [
        {"symbol": "BTC/USDT", "confidence": 0.5},
        {"symbol": "ETH/USDT", "confidence": 0.3},
        None,  # Failed for BNB
    ]

    results = process_symbol_batch_sequential(symbols, mock_data_fetcher, "1h", 200, None)

    assert len(results) == 2  # Only successful ones
    assert results[0]["symbol"] == "BTC/USDT"
    assert results[1]["symbol"] == "ETH/USDT"


# ============================================================================
# Tests for process_symbol_batch_threading
# ============================================================================


@patch("modules.random_forest.signal_random_forest_test.test_random_forest_for_symbol")
def test_process_symbol_batch_threading(mock_test_rf, mock_data_fetcher):
    """Test process_symbol_batch_threading processes with threads."""
    from modules.random_forest.signal_random_forest_test import process_symbol_batch_threading

    symbols = ["BTC/USDT", "ETH/USDT"]
    mock_test_rf.side_effect = [
        {"symbol": "BTC/USDT", "confidence": 0.5},
        {"symbol": "ETH/USDT", "confidence": 0.3},
    ]

    results = process_symbol_batch_threading(symbols, mock_data_fetcher, "1h", 200, 2, None)

    assert len(results) == 2
    assert results[0]["symbol"] == "BTC/USDT"
    assert results[1]["symbol"] == "ETH/USDT"


# ============================================================================
# Tests for process_symbol_batch_hybrid
# ============================================================================


@patch("modules.random_forest.signal_random_forest_test.compute_random_forest_signals_for_data")
@patch("modules.random_forest.signal_random_forest_test.fetch_data_for_symbol")
def test_process_symbol_batch_hybrid(mock_fetch, mock_compute, mock_data_fetcher):
    """Test process_symbol_batch_hybrid uses threading for fetch and compute."""
    from modules.random_forest.signal_random_forest_test import process_symbol_batch_hybrid

    symbols = ["BTC/USDT", "ETH/USDT"]

    # Mock fetch data
    mock_fetch.side_effect = [
        (mock_ohlcv_data(), "binance"),
        (mock_ohlcv_data(), "binance"),
    ]

    # Mock compute RF
    mock_compute.side_effect = [
        {"symbol": "BTC/USDT", "confidence": 0.5},
        {"symbol": "ETH/USDT", "confidence": 0.3},
    ]

    results = process_symbol_batch_hybrid(
        symbols, mock_data_fetcher, "1h", 200, 2, 2, None
    )

    assert len(results) == 2
    assert results[0]["symbol"] == "BTC/USDT"
    assert results[1]["symbol"] == "ETH/USDT"


@patch("modules.random_forest.signal_random_forest_test.compute_random_forest_signals_for_data")
@patch("modules.random_forest.signal_random_forest_test.fetch_data_for_symbol")
def test_process_symbol_batch_hybrid_fetch_failed(mock_fetch, mock_compute, mock_data_fetcher):
    """Test process_symbol_batch_hybrid when some fetches fail."""
    from modules.random_forest.signal_random_forest_test import process_symbol_batch_hybrid

    symbols = ["BTC/USDT", "ETH/USDT"]

    # One succeeds, one fails
    mock_fetch.side_effect = [
        (mock_ohlcv_data(), "binance"),
        None,
    ]

    mock_compute.return_value = {"symbol": "BTC/USDT", "confidence": 0.5}

    results = process_symbol_batch_hybrid(
        symbols, mock_data_fetcher, "1h", 200, 2, 2, None
    )

    assert len(results) == 1
    assert results[0]["symbol"] == "BTC/USDT"


# ============================================================================
# Integration tests
# ============================================================================


@patch("modules.random_forest.signal_random_forest_test.compute_random_forest_signals_for_data")
@patch("modules.random_forest.signal_random_forest_test.fetch_data_for_symbol")
def test_full_workflow_single_symbol(mock_fetch, mock_compute, mock_data_fetcher):
    """Test full workflow from fetch to compute."""
    from modules.random_forest.signal_random_forest_test import test_random_forest_for_symbol

    symbols = ["BTC/USDT"]
    mock_fetch.return_value = (mock_ohlcv_data(), "binance")

    mock_compute.return_value = {
        "symbol": "BTC/USDT",
        "timeframe": "1h",
        "current_price": 50000.0,
        "signal": 1,
        "signal_direction": "LONG",
        "confidence": 0.85,
        "exchange": "binance",
        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
    }

    result = test_random_forest_for_symbol("BTC/USDT", mock_data_fetcher, "1h", 200, None)

    assert result is not None
    assert result["symbol"] == "BTC/USDT"
    assert result["signal_direction"] == "LONG"
    assert result["confidence"] == 0.85


@patch("modules.random_forest.signal_random_forest_test.test_random_forest_for_symbol")
def test_multiple_timeframes(mock_test_rf, mock_data_fetcher):
    """Test processing across multiple timeframes."""
    from modules.random_forest.signal_random_forest_test import test_random_forest_for_symbol, create_batches

    timeframes = ["1h", "4h", "1d"]

    all_results = []
    for tf in timeframes:
        symbols = ["BTC/USDT", "ETH/USDT"]
        mock_test_rf.side_effect = [
            {"symbol": "BTC/USDT", "timeframe": tf, "confidence": 0.5},
            {"symbol": "ETH/USDT", "timeframe": tf, "confidence": 0.3},
        ]

        batches = create_batches(symbols, 2)
        for batch in batches:
            from modules.random_forest.signal_random_forest_test import process_symbol_batch_sequential

            results = process_symbol_batch_sequential(batch, mock_data_fetcher, tf, 200, None)
            all_results.extend(results)

    # Should have processed 2 symbols per timeframe
    assert len(all_results) == 6
    assert all(r["timeframe"] in timeframes for r in all_results)


# ============================================================================
# Error handling tests
# ============================================================================


def test_handle_missing_close_column(sample_ohlcv_data, mock_data_fetcher):
    """Test handling of missing close column in DataFrame."""
    from modules.random_forest.signal_random_forest_test import compute_random_forest_signals_for_data

    # Remove 'close' column
    df_without_close = sample_ohlcv_data.drop(columns=["close"])

    with patch("modules.random_forest.signal_random_forest_test.get_random_forest_signal") as mock_get_signal:
        mock_get_signal.return_value = (1, 0.85)

        result = compute_random_forest_signals_for_data(
            "BTC/USDT",
            "1h",
            df_without_close,
            "binance",
            mock_data_fetcher,
            200,
            model_path=None,
        )

        # Should return None because 'close' is missing
        assert result is None

