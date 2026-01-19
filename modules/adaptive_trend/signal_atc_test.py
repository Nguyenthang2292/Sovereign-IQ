"""
Tests for signal_atc.py using pytest.

Tests all functionality including:
- ResourceMonitor class
- Utility functions (calculate_confidence, get_signal_direction)
- Data fetching functions
- Batch creation and processing
- Multiple execution modes (sequential, threading, multiprocessing, hybrid)
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

from modules.adaptive_trend.utils.config import ATCConfig


# ============================================================================
# Pytest Fixtures
# ============================================================================


@pytest.fixture
def mock_data_fetcher():
    """Create a mock DataFetcher instance."""
    return MagicMock()


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
        calculation_source="close",
    )


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


# ============================================================================
# Mock data helper
# ============================================================================


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


# ============================================================================
# Tests for Utility Functions
# ============================================================================


@pytest.mark.parametrize(
    "signal_value,expected_confidence",
    [
        (0.0, 0.0),
        (0.5, 0.5),
        (1.0, 1.0),
        (-0.5, 0.5),
        (-1.0, 1.0),
        (0.25, 0.25),
        (-0.75, 0.75),
    ],
)
def test_calculate_confidence(signal_value, expected_confidence):
    """Test calculate_confidence returns absolute value of signal."""
    from modules.adaptive_trend.signal_atc import calculate_confidence

    assert calculate_confidence(signal_value) == expected_confidence


@pytest.mark.parametrize(
    "signal_value,expected_direction",
    [
        (0.1, "LONG"),
        (0.5, "LONG"),
        (1.0, "LONG"),
        (-0.1, "SHORT"),
        (-0.5, "SHORT"),
        (-1.0, "SHORT"),
        (0.0, "NEUTRAL"),
        (0.05, "NEUTRAL"),
        (-0.05, "NEUTRAL"),
        (0.049, "NEUTRAL"),
        (-0.049, "NEUTRAL"),
    ],
)
def test_get_signal_direction(signal_value, expected_direction):
    """Test get_signal_direction returns correct direction."""
    from modules.adaptive_trend.signal_atc import get_signal_direction

    assert get_signal_direction(signal_value) == expected_direction


# ============================================================================
# Tests for ResourceMonitor
# ============================================================================


def test_resource_monitor_init():
    """Test ResourceMonitor initialization."""
    from modules.adaptive_trend.signal_atc import ResourceMonitor

    monitor = ResourceMonitor(max_memory_pct=80.0, max_cpu_pct=80.0)

    assert monitor.max_memory_pct == 80.0
    assert monitor.max_cpu_pct == 80.0


@pytest.mark.parametrize(
    "max_memory_pct,max_cpu_pct",
    [
        (70.0, 70.0),
        (80.0, 90.0),
        (100.0, 100.0),
        (50.0, 50.0),
    ],
)
def test_resource_monitor_custom_limits(max_memory_pct, max_cpu_pct):
    """Test ResourceMonitor with custom limits."""
    from modules.adaptive_trend.signal_atc import ResourceMonitor

    monitor = ResourceMonitor(max_memory_pct=max_memory_pct, max_cpu_pct=max_cpu_pct)

    assert monitor.max_memory_pct == max_memory_pct
    assert monitor.max_cpu_pct == max_cpu_pct


def test_resource_monitor_get_memory_usage():
    """Test get_memory_usage returns tuple with expected structure."""
    from modules.adaptive_trend.signal_atc import ResourceMonitor

    monitor = ResourceMonitor()
    process_mem, system_mem_pct, available_mem = monitor.get_memory_usage()

    assert isinstance(process_mem, float)
    assert isinstance(system_mem_pct, float)
    assert isinstance(available_mem, float)

    # If psutil is not available, all values should be 0
    import modules.adaptive_trend.signal_atc as signal_module

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
    from modules.adaptive_trend.signal_atc import ResourceMonitor

    monitor = ResourceMonitor()
    process_cpu, system_cpu = monitor.get_cpu_usage()

    assert isinstance(process_cpu, float)
    assert isinstance(system_cpu, float)

    # If psutil is not available, all values should be 0
    import modules.adaptive_trend.signal_atc as signal_module

    if not signal_module.PSUTIL_AVAILABLE:
        assert process_cpu == 0.0
        assert system_cpu == 0.0
    else:
        # Should have some values
        assert 0.0 <= process_cpu <= 100.0
        assert 0.0 <= system_cpu <= 100.0


def test_resource_monitor_get_status_string():
    """Test get_status_string returns formatted string."""
    from modules.adaptive_trend.signal_atc import ResourceMonitor

    monitor = ResourceMonitor()
    status = monitor.get_status_string()

    assert isinstance(status, str)

    # If psutil is not available
    import modules.adaptive_trend.signal_atc as signal_module

    if not signal_module.PSUTIL_AVAILABLE:
        assert "unavailable" in status.lower()


@patch("modules.adaptive_trend.signal_atc.PSUTIL_AVAILABLE", True)
@patch("psutil.virtual_memory")
def test_resource_monitor_check_memory_limit(mock_vm):
    """Test check_memory_limit with mocked values."""
    from modules.adaptive_trend.signal_atc import ResourceMonitor
    
    # Mock system memory percent
    mock_vm.return_value.percent = 60.0
    monitor = ResourceMonitor(max_memory_pct=70.0)
    assert monitor.check_memory_limit() is True
    
    mock_vm.return_value.percent = 80.0
    assert monitor.check_memory_limit() is False


@patch("modules.adaptive_trend.signal_atc.PSUTIL_AVAILABLE", True)
@patch("psutil.cpu_percent")
def test_resource_monitor_check_cpu_limit(mock_cpu):
    """Test check_cpu_limit with mocked values."""
    from modules.adaptive_trend.signal_atc import ResourceMonitor
    
    mock_cpu.return_value = 60.0
    monitor = ResourceMonitor(max_cpu_pct=70.0)
    assert monitor.check_cpu_limit() is True
    
    mock_cpu.return_value = 80.0
    assert monitor.check_cpu_limit() is False


@patch("time.sleep", return_value=None)
@patch("modules.adaptive_trend.signal_atc.ResourceMonitor.check_memory_limit")
@patch("modules.adaptive_trend.signal_atc.ResourceMonitor.check_cpu_limit")
def test_resource_monitor_wait_if_over_limit_timeout(mock_cpu_limit, mock_mem_limit, mock_sleep):
    """Test wait_if_over_limit exits after timeout even if still over limit."""
    from modules.adaptive_trend.signal_atc import ResourceMonitor
    
    monitor = ResourceMonitor()
    
    # Always over limit
    mock_mem_limit.return_value = False
    mock_cpu_limit.return_value = False
    
    # Use a small wait interval to avoid long test time
    # This should call sleep multiple times and then exit
    start_time = datetime.now()
    monitor.wait_if_over_limit(max_wait_seconds=0.2)
    
    assert mock_sleep.called


# ============================================================================
# Tests for fetch_data_for_symbol
# ============================================================================


def test_fetch_data_for_symbol_success(mock_data_fetcher, sample_ohlcv_data):
    """Test fetch_data_for_symbol returns data on success."""
    from modules.adaptive_trend.signal_atc import fetch_data_for_symbol

    mock_data_fetcher.fetch_ohlcv_with_fallback_exchange = MagicMock(return_value=(sample_ohlcv_data, "binance"))

    result = fetch_data_for_symbol("BTC/USDT", mock_data_fetcher, "1h", 200)

    assert result is not None
    df, exchange_id = result
    assert isinstance(df, pd.DataFrame)
    assert exchange_id == "binance"


def test_fetch_data_for_symbol_empty_dataframe(mock_data_fetcher):
    """Test fetch_data_for_symbol returns None for empty DataFrame."""
    from modules.adaptive_trend.signal_atc import fetch_data_for_symbol

    mock_data_fetcher.fetch_ohlcv_with_fallback_exchange = MagicMock(return_value=(pd.DataFrame(), "binance"))

    result = fetch_data_for_symbol("BTC/USDT", mock_data_fetcher, "1h", 200)

    assert result is None


def test_fetch_data_for_symbol_none_result(mock_data_fetcher):
    """Test fetch_data_for_symbol returns None when fetch fails."""
    from modules.adaptive_trend.signal_atc import fetch_data_for_symbol

    mock_data_fetcher.fetch_ohlcv_with_fallback_exchange = MagicMock(return_value=(None, None))

    result = fetch_data_for_symbol("BTC/USDT", mock_data_fetcher, "1h", 200)

    assert result is None


def test_fetch_data_for_symbol_exception(mock_data_fetcher):
    """Test fetch_data_for_symbol handles exceptions."""
    from modules.adaptive_trend.signal_atc import fetch_data_for_symbol

    mock_data_fetcher.fetch_ohlcv_with_fallback_exchange = MagicMock(side_effect=Exception("Connection error"))

    result = fetch_data_for_symbol("BTC/USDT", mock_data_fetcher, "1h", 200)

    assert result is None


# ============================================================================
# Tests for compute_atc_signals_for_data
# ============================================================================


@patch("modules.adaptive_trend.signal_atc.compute_atc_signals")
def test_compute_atc_signals_for_data_success(mock_compute, sample_ohlcv_data, base_config):
    """Test compute_atc_signals_for_data returns result dict."""
    from modules.adaptive_trend.signal_atc import compute_atc_signals_for_data

    mock_compute.return_value = {
        "Average_Signal": pd.Series([0.0, 0.05, 0.1]),
        "EMA_Signal": pd.Series([0.0, 0.04, 0.08]),
        "EMA_S": pd.Series([1.0, 1.05, 1.1]),
    }

    result = compute_atc_signals_for_data(
        "BTC/USDT",
        "1h",
        sample_ohlcv_data,
        "binance",
        base_config,
    )

    assert result is not None
    assert result["symbol"] == "BTC/USDT"
    assert result["timeframe"] == "1h"
    assert "average_signal" in result
    assert "signal_direction" in result
    assert "confidence" in result


@patch("modules.adaptive_trend.signal_atc.compute_atc_signals")
def test_compute_atc_signals_for_data_empty_result(mock_compute, sample_ohlcv_data, base_config):
    """Test compute_atc_signals_for_data returns None for empty Average_Signal."""
    from modules.adaptive_trend.signal_atc import compute_atc_signals_for_data

    mock_compute.return_value = {
        "Average_Signal": pd.Series([]),
    }

    result = compute_atc_signals_for_data(
        "BTC/USDT",
        "1h",
        sample_ohlcv_data,
        "binance",
        base_config,
    )

    assert result is None


@patch("modules.adaptive_trend.signal_atc.compute_atc_signals")
def test_compute_atc_signals_for_data_nan_signal(mock_compute, sample_ohlcv_data, base_config):
    """Test compute_atc_signals_for_data handles NaN signals - returns result with valid fields."""
    from modules.adaptive_trend.signal_atc import compute_atc_signals_for_data

    mock_compute.return_value = {
        "Average_Signal": pd.Series([0.0, np.nan, 0.1]),
        "EMA_Signal": pd.Series([0.0, 0.05, 0.08]),
        "EMA_S": pd.Series([1.0, 1.05, 1.1]),
        "HMA_Signal": pd.Series([0.0, 0.05, 0.08]),
        "HMA_S": pd.Series([1.0, 1.05, 1.1]),
        "WMA_Signal": pd.Series([0.0, 0.05, 0.08]),
        "WMA_S": pd.Series([1.0, 1.05, 1.1]),
        "DEMA_Signal": pd.Series([0.0, 0.05, 0.08]),
        "DEMA_S": pd.Series([1.0, 1.05, 1.1]),
        "LSMA_Signal": pd.Series([0.0, 0.05, 0.08]),
        "LSMA_S": pd.Series([1.0, 1.05, 1.1]),
        "KAMA_Signal": pd.Series([0.0, 0.05, 0.08]),
        "KAMA_S": pd.Series([1.0, 1.05, 1.1]),
    }

    result = compute_atc_signals_for_data(
        "BTC/USDT",
        "1h",
        sample_ohlcv_data,
        "binance",
        base_config,
    )

    # Function doesn't return None - it processes the last signal which is 0.1 (not NaN)
    assert result is not None
    assert "symbol" in result
    assert "timeframe" in result


@patch("modules.adaptive_trend.signal_atc.compute_atc_signals")
@pytest.mark.parametrize("source", ["open", "high", "low"])
def test_compute_atc_signals_for_data_price_sources(mock_compute, sample_ohlcv_data, base_config, source):
    """Test compute_atc_signals_for_data with different price sources."""
    from modules.adaptive_trend.signal_atc import compute_atc_signals_for_data

    base_config.calculation_source = source
    mock_compute.return_value = {
        "Average_Signal": pd.Series([0.1, 0.2, 0.3]),
    }

    result = compute_atc_signals_for_data(
        "BTC/USDT",
        "1h",
        sample_ohlcv_data,
        "binance",
        base_config,
    )

    assert result is not None
    assert result["current_price"] == sample_ohlcv_data[source].iloc[-1]


@patch("modules.adaptive_trend.signal_atc.compute_atc_signals")
def test_compute_atc_signals_for_data_missing_ma_keys(mock_compute, sample_ohlcv_data, base_config):
    """Test compute_atc_signals_for_data handles missing optional MA keys."""
    from modules.adaptive_trend.signal_atc import compute_atc_signals_for_data

    # Only return the required Average_Signal
    mock_compute.return_value = {
        "Average_Signal": pd.Series([0.1, 0.2, 0.3]),
    }

    result = compute_atc_signals_for_data(
        "BTC/USDT",
        "1h",
        sample_ohlcv_data,
        "binance",
        base_config,
    )

    assert result is not None
    assert result["ema_signal"] == 0.0
    assert result["ema_equity"] == 0.0
    assert result["kama_signal"] == 0.0


@patch("modules.adaptive_trend.signal_atc.compute_atc_signals")
def test_compute_atc_signals_for_data_exception(mock_compute, sample_ohlcv_data, base_config):
    """Test compute_atc_signals_for_data handles exceptions."""
    from modules.adaptive_trend.signal_atc import compute_atc_signals_for_data

    mock_compute.side_effect = Exception("Compute error")

    result = compute_atc_signals_for_data(
        "BTC/USDT",
        "1h",
        sample_ohlcv_data,
        "binance",
        base_config,
    )

    assert result is None


@patch("modules.adaptive_trend.signal_atc.compute_atc_signals")
def test_compute_atc_signals_for_data_nan_signal_edge(mock_compute, sample_ohlcv_data, base_config):
    """Test compute_atc_signals_for_data returns None if latest signal is NaN."""
    from modules.adaptive_trend.signal_atc import compute_atc_signals_for_data

    mock_compute.return_value = {
        "Average_Signal": pd.Series([0.0, 0.1, np.nan]),  # Latest signal is NaN
    }

    result = compute_atc_signals_for_data(
        "BTC/USDT",
        "1h",
        sample_ohlcv_data,
        "binance",
        base_config,
    )

    # Should return None because latest signal is NaN
    assert result is None


# ============================================================================
# Tests for test_atc_for_symbol
# ============================================================================


@patch("modules.adaptive_trend.signal_atc.compute_atc_signals_for_data")
@patch("modules.adaptive_trend.signal_atc.fetch_data_for_symbol")
def test_test_atc_for_symbol_success(mock_fetch, mock_compute, mock_data_fetcher, base_config):
    """Test test_atc_for_symbol returns result on success."""
    from modules.adaptive_trend.signal_atc import test_atc_for_symbol

    mock_fetch.return_value = (mock_ohlcv_data(), "binance")
    mock_compute.return_value = {
        "symbol": "BTC/USDT",
        "timeframe": "1h",
        "average_signal": 0.1,
        "signal_direction": "LONG",
        "confidence": 0.1,
    }

    result = test_atc_for_symbol("BTC/USDT", mock_data_fetcher, "1h", base_config)

    assert result is not None
    assert result["symbol"] == "BTC/USDT"


@patch("modules.adaptive_trend.signal_atc.compute_atc_signals_for_data")
@patch("modules.adaptive_trend.signal_atc.fetch_data_for_symbol")
def test_test_atc_for_symbol_fetch_failure(mock_fetch, mock_data_fetcher, base_config):
    """Test test_atc_for_symbol returns None on fetch failure."""
    from modules.adaptive_trend.signal_atc import test_atc_for_symbol

    mock_fetch.return_value = None

    result = test_atc_for_symbol("BTC/USDT", mock_data_fetcher, "1h", base_config)

    assert result is None


@patch("modules.adaptive_trend.signal_atc.compute_atc_signals_for_data")
@patch("modules.adaptive_trend.signal_atc.fetch_data_for_symbol")
def test_test_atc_for_symbol_compute_failure(mock_fetch, mock_compute, mock_data_fetcher, base_config):
    """Test test_atc_for_symbol returns None on compute failure."""
    from modules.adaptive_trend.signal_atc import test_atc_for_symbol

    mock_fetch.return_value = (mock_ohlcv_data(), "binance")
    mock_compute.return_value = None

    result = test_atc_for_symbol("BTC/USDT", mock_data_fetcher, "1h", base_config)

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
    from modules.adaptive_trend.signal_atc import create_batches

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
    from modules.adaptive_trend.signal_atc import create_batches

    symbols = ["BTC/USDT", "ETH/USDT", "BNB/USDT", "SOL/USDT", "ADA/USDT"]
    batches = create_batches(symbols, 2)

    expected_batches = [["BTC/USDT", "ETH/USDT"], ["BNB/USDT", "SOL/USDT"], ["ADA/USDT"]]

    assert batches == expected_batches


def test_create_batches_empty_list():
    """Test create_batches returns empty list for empty input."""
    from modules.adaptive_trend.signal_atc import create_batches

    batches = create_batches([], 10)

    assert batches == []


# ============================================================================
# Tests for calculate_optimal_batch_size
# ============================================================================


@patch("modules.adaptive_trend.signal_atc.PSUTIL_AVAILABLE", False)
def test_calculate_optimal_batch_size_no_psutil():
    """Test calculate_optimal_batch_size returns default when psutil not available."""
    from modules.adaptive_trend.signal_atc import calculate_optimal_batch_size

    batch_size = calculate_optimal_batch_size(100, 3)

    assert batch_size == 50  # Default value


@patch("modules.adaptive_trend.signal_atc.PSUTIL_AVAILABLE", True)
@patch("psutil.virtual_memory")
def test_calculate_optimal_batch_size_boundaries(mock_vm):
    """Test calculate_optimal_batch_size boundary conditions."""
    from modules.adaptive_trend.signal_atc import calculate_optimal_batch_size
    
    # Case 1: High memory available -> should be capped at 100
    mock_vm.return_value.available = 10 * 1024 * 1024 * 1024 # 10 GB
    batch_size = calculate_optimal_batch_size(1000, 3)
    assert batch_size == 100
    
    # Case 2: Very low memory available -> should be limited at 10
    mock_vm.return_value.available = 2 * 1024 * 1024 # 2 MB (usable ~1.4MB -> batch size ~4 -> limited to 10)
    batch_size = calculate_optimal_batch_size(1000, 3)
    assert batch_size == 10
    
    # Case 3: Middle case
    # memory_per_symbol_mb * 3 timeframes = 0.33 MB
    # Usable: 16.5 MB / 0.33 = 50
    mock_vm.return_value.available = (16.5 / 0.7) * 1024 * 1024
    batch_size = calculate_optimal_batch_size(1000, 3, max_memory_pct=70.0)
    assert 45 <= batch_size <= 55


# ============================================================================
# Tests for process_symbol_batch_sequential
# ============================================================================


@patch("modules.adaptive_trend.signal_atc.test_atc_for_symbol")
def test_process_symbol_batch_sequential(mock_test_atc, mock_data_fetcher, base_config):
    """Test process_symbol_batch_sequential processes all symbols."""
    from modules.adaptive_trend.signal_atc import process_symbol_batch_sequential

    symbols = ["BTC/USDT", "ETH/USDT", "BNB/USDT"]
    mock_test_atc.side_effect = [
        {"symbol": "BTC/USDT", "confidence": 0.5},
        {"symbol": "ETH/USDT", "confidence": 0.3},
        None,  # Failed for BNB
    ]

    results = process_symbol_batch_sequential(symbols, mock_data_fetcher, "1h", base_config)

    assert len(results) == 2  # Only successful ones
    assert results[0]["symbol"] == "BTC/USDT"
    assert results[1]["symbol"] == "ETH/USDT"


# ============================================================================
# Tests for process_symbol_batch_threading
# ============================================================================


@patch("modules.adaptive_trend.signal_atc.test_atc_for_symbol")
def test_process_symbol_batch_threading(mock_test_atc, mock_data_fetcher, base_config):
    """Test process_symbol_batch_threading processes with threads."""
    from modules.adaptive_trend.signal_atc import process_symbol_batch_threading

    symbols = ["BTC/USDT", "ETH/USDT"]
    mock_test_atc.side_effect = [
        {"symbol": "BTC/USDT", "confidence": 0.5},
        {"symbol": "ETH/USDT", "confidence": 0.3},
    ]

    results = process_symbol_batch_threading(symbols, mock_data_fetcher, "1h", base_config, max_workers=2)

    assert len(results) == 2
    assert results[0]["symbol"] == "BTC/USDT"
    assert results[1]["symbol"] == "ETH/USDT"


# ============================================================================
# Tests for process_symbol_batch_multiprocessing
# ============================================================================


@patch("modules.adaptive_trend.signal_atc.process_symbol_batch_threading")
def test_process_symbol_batch_multiprocessing(mock_threading, mock_data_fetcher, base_config):
    """Test process_symbol_batch_multiprocessing falls back to threading."""
    from modules.adaptive_trend.signal_atc import process_symbol_batch_multiprocessing

    symbols = ["BTC/USDT", "ETH/USDT"]
    mock_threading.return_value = [
        {"symbol": "BTC/USDT", "confidence": 0.5},
    ]

    results = process_symbol_batch_multiprocessing(symbols, mock_data_fetcher, "1h", base_config, max_workers=2)

    # Should fallback to threading (DataFetcher can't be pickled)
    mock_threading.assert_called_once()


# ============================================================================
# Tests for process_symbol_batch_hybrid
# ============================================================================


@patch("modules.adaptive_trend.signal_atc.compute_atc_signals_for_data")
@patch("modules.adaptive_trend.signal_atc.fetch_data_for_symbol")
def test_process_symbol_batch_hybrid(mock_fetch, mock_compute, mock_data_fetcher, base_config):
    """Test process_symbol_batch_hybrid uses threading for fetch."""
    from modules.adaptive_trend.signal_atc import process_symbol_batch_hybrid

    symbols = ["BTC/USDT", "ETH/USDT"]

    # Mock fetch data
    mock_fetch.side_effect = [
        (mock_ohlcv_data(), "binance"),
        (mock_ohlcv_data(), "binance"),
    ]

    # Mock compute ATC
    mock_compute.side_effect = [
        {"symbol": "BTC/USDT", "confidence": 0.5},
        {"symbol": "ETH/USDT", "confidence": 0.3},
    ]

    results = process_symbol_batch_hybrid(
        symbols, mock_data_fetcher, "1h", base_config, max_workers_thread=2, max_workers_process=2
    )

    assert len(results) == 2
    assert results[0]["symbol"] == "BTC/USDT"
    assert results[1]["symbol"] == "ETH/USDT"


@patch("modules.adaptive_trend.signal_atc.compute_atc_signals_for_data")
@patch("modules.adaptive_trend.signal_atc.fetch_data_for_symbol")
def test_process_symbol_batch_hybrid_fetch_failed(mock_fetch, mock_compute, mock_data_fetcher, base_config):
    """Test process_symbol_batch_hybrid when some fetches fail."""
    from modules.adaptive_trend.signal_atc import process_symbol_batch_hybrid

    symbols = ["BTC/USDT", "ETH/USDT"]

    # One succeeds, one fails
    mock_fetch.side_effect = [
        (mock_ohlcv_data(), "binance"),
        None,
    ]

    mock_compute.return_value = {"symbol": "BTC/USDT", "confidence": 0.5}

    results = process_symbol_batch_hybrid(
        symbols, mock_data_fetcher, "1h", base_config, max_workers_thread=2, max_workers_process=2
    )

    assert len(results) == 1
    assert results[0]["symbol"] == "BTC/USDT"


# ============================================================================
# Tests for main function (partial)
# ============================================================================


def test_main_function_importable():
    """Test that main function can be imported."""
    from modules.adaptive_trend.signal_atc import main

    assert callable(main)


# ============================================================================
# Integration tests
# ============================================================================


@patch("modules.adaptive_trend.signal_atc.compute_atc_signals_for_data")
@patch("modules.adaptive_trend.signal_atc.fetch_data_for_symbol")
def test_full_workflow_single_symbol(mock_fetch, mock_compute, mock_data_fetcher, base_config):
    """Test full workflow from fetch to compute."""
    from modules.adaptive_trend.signal_atc import test_atc_for_symbol

    symbols = ["BTC/USDT"]
    mock_fetch.return_value = (mock_ohlcv_data(), "binance")

    mock_compute.return_value = {
        "symbol": "BTC/USDT",
        "timeframe": "1h",
        "current_price": 50000.0,
        "average_signal": 0.1,
        "signal_direction": "LONG",
        "confidence": 0.1,
        "exchange": "binance",
        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "ema_signal": 0.08,
        "hma_signal": 0.09,
        "wma_signal": 0.07,
        "dema_signal": 0.11,
        "lsma_signal": 0.1,
        "kama_signal": 0.12,
        "ema_equity": 1.05,
        "hma_equity": 1.08,
        "wma_equity": 1.03,
        "dema_equity": 1.11,
        "lsma_equity": 1.1,
        "kama_equity": 1.12,
    }

    result = test_atc_for_symbol("BTC/USDT", mock_data_fetcher, "1h", base_config)

    assert result is not None
    assert result["symbol"] == "BTC/USDT"
    assert result["signal_direction"] == "LONG"
    assert result["confidence"] == 0.1
    assert "ema_signal" in result
    assert "ema_equity" in result


@patch("modules.adaptive_trend.signal_atc.test_atc_for_symbol")
def test_multiple_timeframes(mock_test_atc, mock_data_fetcher, base_config):
    """Test processing across multiple timeframes."""
    from modules.adaptive_trend.signal_atc import test_atc_for_symbol, create_batches

    timeframes = ["15m", "30m", "1h"]

    all_results = []
    for tf in timeframes:
        base_config.timeframe = tf
        symbols = ["BTC/USDT", "ETH/USDT"]
        mock_test_atc.side_effect = [
            {"symbol": "BTC/USDT", "timeframe": tf, "confidence": 0.5},
            {"symbol": "ETH/USDT", "timeframe": tf, "confidence": 0.3},
        ]

        batches = create_batches(symbols, 2)
        for batch in batches:
            from modules.adaptive_trend.signal_atc import process_symbol_batch_sequential

            results = process_symbol_batch_sequential(batch, mock_data_fetcher, tf, base_config)
            all_results.extend(results)

    # Should have processed 2 symbols per timeframe
    assert len(all_results) == 6
    assert all(r["timeframe"] in timeframes for r in all_results)


# ============================================================================
# Error handling tests
# ============================================================================


def test_handle_missing_calculation_source(sample_ohlcv_data, base_config):
    """Test handling of missing calculation_source in DataFrame."""
    from modules.adaptive_trend.signal_atc import compute_atc_signals_for_data

    # Remove 'close' column
    df_without_close = sample_ohlcv_data.drop(columns=["close"])

    with patch("modules.adaptive_trend.signal_atc.compute_atc_signals") as mock_compute:
        mock_compute.return_value = {"Average_Signal": pd.Series([0.1, 0.2, 0.3])}

        result = compute_atc_signals_for_data(
            "BTC/USDT",
            "1h",
            df_without_close,
            "binance",
            base_config,
        )

        # Should return None because 'close' is missing
        assert result is None


def test_handle_pandas_series_empty(sample_ohlcv_data, base_config):
    """Test handling of empty pandas Series."""
    from modules.adaptive_trend.signal_atc import compute_atc_signals_for_data

    base_config.calculation_source = "close"

    with patch("modules.adaptive_trend.signal_atc.compute_atc_signals") as mock_compute:
        mock_compute.return_value = {
            "Average_Signal": pd.Series([]),
        }

        result = compute_atc_signals_for_data(
            "BTC/USDT",
            "1h",
            sample_ohlcv_data,
            "binance",
            base_config,
        )

        assert result is None


# ============================================================================
# Mock data helper
# ============================================================================


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
