"""
Tests for scanner module using pytest pure patterns.

Tests all scanner functionality including:
- Configuration validation
- Sequential execution mode
- ThreadPool execution mode
- AsyncIO execution mode
- Edge case handling
- Error handling
"""

from pathlib import Path
from typing import Any, Optional
from unittest.mock import MagicMock, patch

import numpy as np
import pandas as pd
import pytest

# Add parent directory to path
import sys

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from modules.adaptive_trend.core.scanner import scan_all_symbols
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
    )


def create_mock_ohlcv_data(
    start_price: float = 100.0,
    num_candles: int = 200,
) -> pd.DataFrame:
    """Create mock OHLCV DataFrame for testing."""
    timestamps = pd.date_range(start="2024-01-01", periods=num_candles, freq="1h", tz="UTC")

    np.random.seed(42)
    returns = np.random.normal(0, 0.01, num_candles)
    prices = start_price * (1 + returns).cumprod()

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

    return df


def create_mock_atc_results(signal_value: float = 0.05) -> dict:
    """Create mock ATC results."""
    signal_series = pd.Series([0.0, 0.01, signal_value, signal_value])
    return {
        "Average_Signal": signal_series,
        "EMA_Signal": signal_series,
        "HMA_Signal": signal_series,
    }


# ============================================================================
# Tests for ATCConfig validation - Parametrized
# ============================================================================


@pytest.mark.parametrize(
    "invalid_input,expected_error",
    [
        ("not_a_config", "atc_config must be an ATCConfig instance"),
    ],
)
def test_scan_all_symbols_invalid_input(invalid_input, expected_error):
    """Test that scan_all_symbols raises error for invalid config."""
    mock_fetcher = MagicMock()

    with pytest.raises(ValueError, match=expected_error):
        scan_all_symbols(mock_fetcher, invalid_input)


def test_scan_all_symbols_none_data_fetcher():
    """Test that scan_all_symbols raises error for None data_fetcher."""
    # type: ignore[arg-type] - Intentionally passing invalid type to test error handling
    from typing import TYPE_CHECKING

    mock_fetcher = MagicMock()
    if TYPE_CHECKING:
        # This branch is never executed at runtime, but satisfies type checker
        data_fetcher_arg: Optional[Any] = None  # type: ignore[arg-type]
    else:
        data_fetcher_arg = None  # type: ignore[arg-type]

    with pytest.raises(ValueError, match="data_fetcher cannot be None"):
        scan_all_symbols(data_fetcher_arg, ATCConfig())


@pytest.mark.parametrize(
    "config_kwargs,expected_error",
    [
        ({"timeframe": ""}, "timeframe must be a non-empty string"),
        ({"limit": 0}, "limit must be a positive integer"),
        ({"limit": -1}, "limit must be a positive integer"),
        ({"ema_len": 0}, "ema_len must be a positive integer"),
        ({"ema_len": -1}, "ema_len must be a positive integer"),
        ({"robustness": "Invalid"}, "robustness must be one of"),
    ],
)
def test_scan_all_symbols_invalid_config(config_kwargs, expected_error, mock_data_fetcher):
    """Test that scan_all_symbols validates config parameters."""
    from modules.adaptive_trend.core.scanner import scan_all_symbols
    from modules.adaptive_trend.utils.config import ATCConfig

    config = ATCConfig(**config_kwargs)
    mock_data_fetcher.list_binance_futures_symbols = MagicMock(return_value=[])

    with pytest.raises(ValueError, match=expected_error):
        scan_all_symbols(mock_data_fetcher, config)


@pytest.mark.parametrize(
    "execution_mode,expected_error",
    [
        ("invalid", "execution_mode must be one of"),
        (123, "execution_mode must be one of"),
        (None, "execution_mode must be one of"),
    ],
)
def test_scan_all_symbols_invalid_execution_mode(execution_mode, expected_error, base_config, mock_data_fetcher):
    """Test that scan_all_symbols validates execution_mode."""
    from modules.adaptive_trend.core.scanner import scan_all_symbols

    mock_data_fetcher.list_binance_futures_symbols = MagicMock(return_value=[])

    with pytest.raises(ValueError, match=expected_error):
        scan_all_symbols(
            mock_data_fetcher,
            base_config,
            execution_mode=execution_mode,
        )


@pytest.mark.parametrize(
    "max_workers,expected_error",
    [
        (0, "max_workers must be a positive integer"),
        (-1, "max_workers must be a positive integer"),
    ],
)
def test_scan_all_symbols_invalid_max_workers(max_workers, expected_error, base_config, mock_data_fetcher):
    """Test that scan_all_symbols validates max_workers."""
    from modules.adaptive_trend.core.scanner import scan_all_symbols

    mock_data_fetcher.list_binance_futures_symbols = MagicMock(return_value=[])

    with pytest.raises(ValueError, match=expected_error):
        scan_all_symbols(
            mock_data_fetcher,
            base_config,
            max_workers=max_workers,
        )


# ============================================================================
# Tests for missing methods
# ============================================================================


def test_scan_all_symbols_missing_list_method(mock_data_fetcher):
    """Test that scan_all_symbols raises error when list method is missing."""
    from modules.adaptive_trend.core.scanner import scan_all_symbols

    config = ATCConfig()
    # Remove the required method
    del mock_data_fetcher.list_binance_futures_symbols

    with pytest.raises(AttributeError, match="must have method"):
        scan_all_symbols(mock_data_fetcher, config)


# ============================================================================
# Tests for scan_all_symbols - No symbols
# ============================================================================


def test_scan_all_symbols_no_symbols_returns_empty(base_config, mock_data_fetcher):
    """Test that scan_all_symbols returns empty DataFrames when no symbols."""
    from modules.adaptive_trend.core.scanner import scan_all_symbols

    mock_data_fetcher.list_binance_futures_symbols = MagicMock(return_value=[])

    long_signals, short_signals = scan_all_symbols(mock_data_fetcher, base_config)

    assert long_signals.empty
    assert short_signals.empty
    assert isinstance(long_signals, pd.DataFrame)
    assert isinstance(short_signals, pd.DataFrame)


# ============================================================================
# Tests for scan_all_symbols - Sequential mode
# ============================================================================


@patch("modules.adaptive_trend.core.scanner.compute_atc_signals")
@patch("modules.adaptive_trend.core.scanner.trend_sign")
def test_scan_all_symbols_sequential_mode(mock_trend_sign, mock_compute_atc, base_config, mock_data_fetcher):
    """Test scan_all_symbols in sequential mode."""
    from modules.adaptive_trend.core.scanner import scan_all_symbols

    symbols = ["BTC/USDT", "ETH/USDT"]
    mock_data_fetcher.list_binance_futures_symbols = MagicMock(return_value=symbols)
    mock_df = create_mock_ohlcv_data(num_candles=200)
    mock_data_fetcher.fetch_ohlcv_with_fallback_exchange = MagicMock(return_value=(mock_df, "binance"))
    mock_compute_atc.return_value = create_mock_atc_results(signal_value=0.05)
    mock_trend_sign.return_value = pd.Series([0, 0, 1, 1])  # Bullish trend

    long_signals, short_signals = scan_all_symbols(
        mock_data_fetcher, base_config, execution_mode="sequential", max_symbols=2
    )

    assert isinstance(long_signals, pd.DataFrame)
    assert isinstance(short_signals, pd.DataFrame)
    assert len(long_signals) > 0  # Should find signals
    mock_data_fetcher.fetch_ohlcv_with_fallback_exchange.assert_called()


# ============================================================================
# Tests for scan_all_symbols - ThreadPool mode
# ============================================================================


@patch("modules.adaptive_trend.core.scanner.compute_atc_signals")
@patch("modules.adaptive_trend.core.scanner.trend_sign")
def test_scan_all_symbols_threadpool_mode(mock_trend_sign, mock_compute_atc, base_config, mock_data_fetcher):
    """Test scan_all_symbols in threadpool mode."""
    from modules.adaptive_trend.core.scanner import scan_all_symbols

    symbols = ["BTC/USDT", "ETH/USDT"]
    mock_data_fetcher.list_binance_futures_symbols = MagicMock(return_value=symbols)
    mock_df = create_mock_ohlcv_data(num_candles=200)
    mock_data_fetcher.fetch_ohlcv_with_fallback_exchange = MagicMock(return_value=(mock_df, "binance"))
    mock_compute_atc.return_value = create_mock_atc_results(signal_value=0.05)
    mock_trend_sign.return_value = pd.Series([0, 0, 1, 1])

    long_signals, short_signals = scan_all_symbols(
        mock_data_fetcher, base_config, execution_mode="threadpool", max_symbols=2, max_workers=2
    )

    assert isinstance(long_signals, pd.DataFrame)
    assert isinstance(short_signals, pd.DataFrame)


# ============================================================================
# Tests for scan_all_symbols - Asyncio mode
# ============================================================================


@patch("modules.adaptive_trend.core.scanner.compute_atc_signals")
@patch("modules.adaptive_trend.core.scanner.trend_sign")
def test_scan_all_symbols_asyncio_mode(mock_trend_sign, mock_compute_atc, base_config, mock_data_fetcher):
    """Test scan_all_symbols in asyncio mode."""
    from modules.adaptive_trend.core.scanner import scan_all_symbols

    symbols = ["BTC/USDT", "ETH/USDT"]
    mock_data_fetcher.list_binance_futures_symbols = MagicMock(return_value=symbols)
    mock_df = create_mock_ohlcv_data(num_candles=200)
    mock_data_fetcher.fetch_ohlcv_with_fallback_exchange = MagicMock(return_value=(mock_df, "binance"))
    mock_compute_atc.return_value = create_mock_atc_results(signal_value=0.05)
    mock_trend_sign.return_value = pd.Series([0, 0, 1, 1])

    long_signals, short_signals = scan_all_symbols(
        mock_data_fetcher, base_config, execution_mode="asyncio", max_symbols=2, max_workers=2
    )

    assert isinstance(long_signals, pd.DataFrame)
    assert isinstance(short_signals, pd.DataFrame)


# ============================================================================
# Tests for scan_all_symbols - Edge cases
# ============================================================================


@patch("modules.adaptive_trend.core.scanner.compute_atc_signals")
@patch("modules.adaptive_trend.core.scanner.trend_sign")
def test_scan_all_symbols_empty_dataframe(mock_trend_sign, mock_compute_atc, base_config, mock_data_fetcher):
    """Test scan_all_symbols handles empty DataFrame."""
    from modules.adaptive_trend.core.scanner import scan_all_symbols

    symbols = ["BTC/USDT"]
    mock_data_fetcher.list_binance_futures_symbols = MagicMock(return_value=symbols)
    mock_data_fetcher.fetch_ohlcv_with_fallback_exchange = MagicMock(return_value=(pd.DataFrame(), "binance"))

    long_signals, short_signals = scan_all_symbols(
        mock_data_fetcher, base_config, execution_mode="sequential", max_symbols=1
    )

    assert long_signals.empty
    assert short_signals.empty


@patch("modules.adaptive_trend.core.scanner.compute_atc_signals")
@patch("modules.adaptive_trend.core.scanner.trend_sign")
def test_scan_all_symbols_missing_close_column(mock_trend_sign, mock_compute_atc, base_config, mock_data_fetcher):
    """Test scan_all_symbols handles missing close column."""
    from modules.adaptive_trend.core.scanner import scan_all_symbols

    symbols = ["BTC/USDT"]
    mock_data_fetcher.list_binance_futures_symbols = MagicMock(return_value=symbols)
    mock_df = pd.DataFrame({"open": [100.0], "high": [101.0]})
    mock_data_fetcher.fetch_ohlcv_with_fallback_exchange = MagicMock(return_value=(mock_df, "binance"))

    long_signals, short_signals = scan_all_symbols(
        mock_data_fetcher, base_config, execution_mode="sequential", max_symbols=1
    )

    assert long_signals.empty
    assert short_signals.empty


@patch("modules.adaptive_trend.core.scanner.compute_atc_signals")
@patch("modules.adaptive_trend.core.scanner.trend_sign")
def test_scan_all_symbols_insufficient_data(mock_trend_sign, mock_compute_atc, base_config, mock_data_fetcher):
    """Test scan_all_symbols handles insufficient data."""
    from modules.adaptive_trend.core.scanner import scan_all_symbols

    symbols = ["BTC/USDT"]
    mock_data_fetcher.list_binance_futures_symbols = MagicMock(return_value=symbols)
    mock_df = create_mock_ohlcv_data(num_candles=50)  # Less than limit=200
    mock_data_fetcher.fetch_ohlcv_with_fallback_exchange = MagicMock(return_value=(mock_df, "binance"))

    long_signals, short_signals = scan_all_symbols(
        mock_data_fetcher, base_config, execution_mode="sequential", max_symbols=1
    )

    assert long_signals.empty
    assert short_signals.empty


@patch("modules.adaptive_trend.core.scanner.compute_atc_signals")
@patch("modules.adaptive_trend.core.scanner.trend_sign")
def test_scan_all_symbols_signal_below_threshold(mock_trend_sign, mock_compute_atc, mock_data_fetcher):
    """Test scan_all_symbols filters signals below threshold."""
    from modules.adaptive_trend.core.scanner import scan_all_symbols
    from modules.adaptive_trend.utils.config import ATCConfig

    config = ATCConfig(limit=200, timeframe="1h")
    symbols = ["BTC/USDT"]
    mock_data_fetcher.list_binance_futures_symbols = MagicMock(return_value=symbols)
    mock_df = create_mock_ohlcv_data(num_candles=200)
    mock_data_fetcher.fetch_ohlcv_with_fallback_exchange = MagicMock(return_value=(mock_df, "binance"))
    # Signal below threshold (0.005 < 0.01)
    mock_compute_atc.return_value = create_mock_atc_results(signal_value=0.005)
    mock_trend_sign.return_value = pd.Series([0, 0, 1, 1])

    long_signals, short_signals = scan_all_symbols(
        mock_data_fetcher, config, execution_mode="sequential", max_symbols=1, min_signal=0.01
    )

    assert long_signals.empty
    assert short_signals.empty


@patch("modules.adaptive_trend.core.scanner.compute_atc_signals")
@patch("modules.adaptive_trend.core.scanner.trend_sign")
def test_scan_all_symbols_long_and_short_signals(mock_trend_sign, mock_compute_atc, base_config, mock_data_fetcher):
    """Test scan_all_symbols separates LONG and SHORT signals."""
    symbols = ["BTC/USDT", "ETH/USDT"]
    mock_data_fetcher.list_binance_futures_symbols = MagicMock(return_value=symbols)
    mock_df = create_mock_ohlcv_data(num_candles=200)
    mock_data_fetcher.fetch_ohlcv_with_fallback_exchange = MagicMock(return_value=(mock_df, "binance"))

    # First symbol: bullish, Second symbol: bearish
    call_tracker = {"compute_atc": 0, "trend_sign": 0}

    def side_effect_compute(*args, **kwargs):
        call_tracker["compute_atc"] += 1

        if call_tracker["compute_atc"] == 1:
            return create_mock_atc_results(signal_value=0.05)  # Bullish
        else:
            return create_mock_atc_results(signal_value=-0.05)  # Bearish

    def side_effect_trend(*args, **kwargs):
        call_tracker["trend_sign"] += 1

        if call_tracker["trend_sign"] == 1:
            return pd.Series([0, 0, 1, 1])  # Bullish
        else:
            return pd.Series([0, 0, -1, -1])  # Bearish

    mock_compute_atc.side_effect = side_effect_compute
    mock_trend_sign.side_effect = side_effect_trend

    long_signals, short_signals = scan_all_symbols(
        mock_data_fetcher, base_config, execution_mode="sequential", max_symbols=2
    )

    assert len(long_signals) > 0
    assert len(short_signals) > 0
    assert all(long_signals["trend"] > 0)
    assert all(short_signals["trend"] < 0)


@patch("modules.adaptive_trend.core.scanner.compute_atc_signals")
@patch("modules.adaptive_trend.core.scanner.trend_sign")
def test_scan_all_symbols_max_symbols_limit(mock_trend_sign, mock_compute_atc, base_config, mock_data_fetcher):
    """Test scan_all_symbols respects max_symbols limit."""
    from modules.adaptive_trend.core.scanner import scan_all_symbols

    all_symbols = ["BTC/USDT", "ETH/USDT", "BNB/USDT", "SOL/USDT", "ADA/USDT"]
    mock_data_fetcher.list_binance_futures_symbols = MagicMock(return_value=all_symbols)
    mock_df = create_mock_ohlcv_data(num_candles=200)
    mock_data_fetcher.fetch_ohlcv_with_fallback_exchange = MagicMock(return_value=(mock_df, "binance"))
    mock_compute_atc.return_value = create_mock_atc_results(signal_value=0.05)
    mock_trend_sign.return_value = pd.Series([0, 0, 1, 1])

    long_signals, short_signals = scan_all_symbols(
        mock_data_fetcher, base_config, execution_mode="sequential", max_symbols=3
    )

    # Should only process 3 symbols
    assert mock_data_fetcher.fetch_ohlcv_with_fallback_exchange.call_count == 3


# ============================================================================
# Tests for error handling
# ============================================================================


@patch("modules.adaptive_trend.core.scanner.compute_atc_signals")
def test_scan_all_symbols_handles_exceptions(mock_compute_atc, base_config, mock_data_fetcher):
    """Test scan_all_symbols handles exceptions gracefully."""
    from modules.adaptive_trend.core.scanner import scan_all_symbols

    symbols = ["BTC/USDT"]
    mock_data_fetcher.list_binance_futures_symbols = MagicMock(return_value=symbols)
    mock_df = create_mock_ohlcv_data(num_candles=200)
    mock_data_fetcher.fetch_ohlcv_with_fallback_exchange = MagicMock(return_value=(mock_df, "binance"))

    # Make compute_atc_signals raise an exception
    mock_compute_atc.side_effect = Exception("Test error")

    long_signals, short_signals = scan_all_symbols(
        mock_data_fetcher, base_config, execution_mode="sequential", max_symbols=1
    )

    # Should return empty DataFrames and continue
    assert isinstance(long_signals, pd.DataFrame)
    assert isinstance(short_signals, pd.DataFrame)


# ============================================================================
# Integration tests
# ============================================================================


@patch("modules.adaptive_trend.core.scanner.compute_atc_signals")
@patch("modules.adaptive_trend.core.scanner.trend_sign")
def test_scan_all_symbols_full_workflow(mock_trend_sign, mock_compute_atc, base_config, mock_data_fetcher):
    """Test full workflow of scan_all_symbols."""
    from modules.adaptive_trend.core.scanner import scan_all_symbols

    symbols = ["BTC/USDT", "ETH/USDT", "BNB/USDT"]
    mock_data_fetcher.list_binance_futures_symbols = MagicMock(return_value=symbols)
    mock_df = create_mock_ohlcv_data(num_candles=200)
    mock_data_fetcher.fetch_ohlcv_with_fallback_exchange = MagicMock(return_value=(mock_df, "binance"))

    # Mock different results for each symbol
    call_count = [0]

    def side_effect_atc(*args, **kwargs):
        call_count[0] += 1
        if call_count[0] == 1:
            return create_mock_atc_results(signal_value=0.05)
        elif call_count[0] == 2:
            return create_mock_atc_results(signal_value=-0.03)
        else:
            return create_mock_atc_results(signal_value=0.02)

    def side_effect_trend(*args, **kwargs):
        # Return trend based on signal
        if call_count[0] == 1:
            return pd.Series([0, 0, 1, 1])  # Bullish
        elif call_count[0] == 2:
            return pd.Series([0, 0, -1, -1])  # Bearish
        else:
            return pd.Series([0, 0, 1, 1])  # Bullish

    mock_compute_atc.side_effect = side_effect_atc
    mock_trend_sign.side_effect = side_effect_trend

    long_signals, short_signals = scan_all_symbols(
        mock_data_fetcher, base_config, execution_mode="sequential", max_symbols=3
    )

    # Verify all 3 symbols were processed
    assert mock_compute_atc.call_count == 3
    assert mock_data_fetcher.fetch_ohlcv_with_fallback_exchange.call_count == 3

    # Verify results
    assert isinstance(long_signals, pd.DataFrame)
    assert isinstance(short_signals, pd.DataFrame)
    assert len(long_signals) > 0 or len(short_signals) > 0


# ============================================================================
# Tests for mock data helpers
# ============================================================================


def test_create_mock_ohlcv_data_creates_valid_dataframe():
    """Test that create_mock_ohlcv_data creates valid OHLCV data."""
    df = create_mock_ohlcv_data(start_price=100.0, num_candles=200)

    assert isinstance(df, pd.DataFrame)
    assert len(df) == 200
    assert "close" in df.columns
    assert "open" in df.columns
    assert "high" in df.columns
    assert "low" in df.columns
    assert "volume" in df.columns
    # Just verify data is reasonable (not exact due to random noise)
    assert df["close"].iloc[0] > 90.0


def test_create_mock_atc_results_creates_valid_dict():
    """Test that create_mock_atc_results creates valid results."""
    result = create_mock_atc_results(signal_value=0.05)

    assert isinstance(result, dict)
    assert "Average_Signal" in result
    assert "EMA_Signal" in result
    assert "HMA_Signal" in result
    assert isinstance(result["Average_Signal"], pd.Series)


# ============================================================================
# Tests for execution modes comparison
# ============================================================================


@patch("modules.adaptive_trend.core.scanner.compute_atc_signals")
@patch("modules.adaptive_trend.core.scanner.trend_sign")
@pytest.mark.parametrize(
    "execution_mode",
    ["sequential", "threadpool", "asyncio"],
)
def test_scan_all_symbols_all_execution_modes(
    mock_trend_sign, mock_compute_atc, execution_mode, base_config, mock_data_fetcher
):
    """Test scan_all_symbols works with all execution modes."""
    from modules.adaptive_trend.core.scanner import scan_all_symbols

    symbols = ["BTC/USDT", "ETH/USDT"]
    mock_data_fetcher.list_binance_futures_symbols = MagicMock(return_value=symbols)
    mock_df = create_mock_ohlcv_data(num_candles=200)
    mock_data_fetcher.fetch_ohlcv_with_fallback_exchange = MagicMock(return_value=(mock_df, "binance"))
    mock_compute_atc.return_value = create_mock_atc_results(signal_value=0.05)
    mock_trend_sign.return_value = pd.Series([0, 0, 1, 1])

    # Test each execution mode
    long_signals, short_signals = scan_all_symbols(
        mock_data_fetcher, base_config, execution_mode=execution_mode, max_symbols=2, max_workers=2
    )

    assert isinstance(long_signals, pd.DataFrame)
    assert isinstance(short_signals, pd.DataFrame)
    assert len(long_signals) > 0 or len(short_signals) > 0
