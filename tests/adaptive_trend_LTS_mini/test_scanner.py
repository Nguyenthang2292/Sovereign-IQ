"""
Comprehensive tests for adaptive_trend_LTS_mini scanner functionality.

Tests cover:
- Basic scan functionality with all execution modes
- Symbol filtering and pre-filtering
- Error handling and edge cases
- Parameter validation
- Memory management
- Result formatting and sorting
"""

import pytest
from unittest.mock import MagicMock, Mock, patch, call
import pandas as pd
import numpy as np
from argparse import Namespace

from modules.adaptive_trend_LTS_mini.core.scanner import scan_all_symbols
from modules.adaptive_trend_LTS_mini.utils.config import ATCConfig


# ==================== FIXTURES ====================


@pytest.fixture
def mock_atc_config():
    """Create a mock ATCConfig for testing."""
    return ATCConfig(
        timeframe="1h",
        limit=100,
        ema_len=10,
        hma_len=10,
        wma_len=10,
        dema_len=10,
        lsma_len=10,
        kama_len=10,
        robustness="Medium",
        lambda_param=0.1,
        decay=0.1,
        cutout=0,
        batch_size=50,
    )


@pytest.fixture
def mock_data_fetcher():
    """Create a mock DataFetcher for testing."""
    fetcher = MagicMock()

    # Mock OHLCV data
    periods = 100
    dates = pd.date_range("2023-01-01", periods=periods, freq="h")
    prices = 100 + np.cumsum(np.random.randn(periods) * 0.5)
    prices = np.maximum(prices, 10.0)

    df = pd.DataFrame(
        {
            "open": prices,
            "high": prices * 1.01,
            "low": prices * 0.99,
            "close": prices,
            "volume": np.random.uniform(1000, 10000, periods),
        },
        index=dates,
    )

    fetcher.fetch_ohlcv_with_fallback_exchange.return_value = (df, "binance")
    fetcher.list_binance_futures_symbols.return_value = [
        "BTC/USDT", "ETH/USDT", "SOL/USDT", "BNB/USDT", "XRP/USDT"
    ]

    return fetcher


@pytest.fixture
def sample_symbols():
    """Sample list of symbols for testing."""
    return ["BTC/USDT", "ETH/USDT", "SOL/USDT", "BNB/USDT", "XRP/USDT"]


# ==================== PARAMETER VALIDATION TESTS ====================


class TestParameterValidation:
    """Test parameter validation in scan_all_symbols."""

    def test_scan_rejects_none_data_fetcher(self, mock_atc_config):
        """Test that scan_all_symbols rejects None data_fetcher."""
        with pytest.raises(ValueError, match="data_fetcher cannot be None"):
            scan_all_symbols(None, mock_atc_config)

    def test_scan_rejects_invalid_atc_config_type(self, mock_data_fetcher):
        """Test that scan rejects invalid ATCConfig type."""
        with pytest.raises(ValueError, match="atc_config must be an ATCConfig instance"):
            scan_all_symbols(mock_data_fetcher, {"timeframe": "1h"})

    def test_scan_rejects_invalid_timeframe(self, mock_data_fetcher, mock_atc_config):
        """Test that scan rejects invalid timeframe."""
        mock_atc_config.timeframe = ""
        with pytest.raises(ValueError, match="timeframe must be a non-empty string"):
            scan_all_symbols(mock_data_fetcher, mock_atc_config)

    def test_scan_rejects_invalid_limit(self, mock_data_fetcher, mock_atc_config):
        """Test that scan rejects invalid limit."""
        mock_atc_config.limit = 0
        with pytest.raises(ValueError, match="limit must be a positive integer"):
            scan_all_symbols(mock_data_fetcher, mock_atc_config)

    def test_scan_rejects_invalid_ma_lengths(self, mock_data_fetcher, mock_atc_config):
        """Test that scan rejects invalid MA lengths."""
        mock_atc_config.ema_len = -5
        with pytest.raises(ValueError, match="ema_len must be a positive integer"):
            scan_all_symbols(mock_data_fetcher, mock_atc_config)

    def test_scan_rejects_invalid_robustness(self, mock_data_fetcher, mock_atc_config):
        """Test that scan rejects invalid robustness value."""
        mock_atc_config.robustness = "Invalid"
        with pytest.raises(ValueError, match="robustness must be one of"):
            scan_all_symbols(mock_data_fetcher, mock_atc_config)

    def test_scan_rejects_invalid_lambda_param(self, mock_data_fetcher, mock_atc_config):
        """Test that scan rejects invalid lambda_param."""
        mock_atc_config.lambda_param = float('nan')
        with pytest.raises(ValueError, match="lambda_param must be a finite number"):
            scan_all_symbols(mock_data_fetcher, mock_atc_config)

    def test_scan_rejects_invalid_decay(self, mock_data_fetcher, mock_atc_config):
        """Test that scan rejects decay outside [0, 1]."""
        mock_atc_config.decay = 1.5
        with pytest.raises(ValueError, match="decay must be between 0 and 1"):
            scan_all_symbols(mock_data_fetcher, mock_atc_config)

    def test_scan_rejects_invalid_min_signal(self, mock_data_fetcher, mock_atc_config):
        """Test that scan rejects negative min_signal."""
        with pytest.raises(ValueError, match="min_signal must be a non-negative number"):
            scan_all_symbols(mock_data_fetcher, mock_atc_config, min_signal=-0.1)

    def test_scan_rejects_invalid_execution_mode(self, mock_data_fetcher, mock_atc_config):
        """Test that scan rejects invalid execution mode."""
        with pytest.raises(ValueError, match="execution_mode must be one of"):
            scan_all_symbols(mock_data_fetcher, mock_atc_config, execution_mode="invalid_mode")

    def test_scan_rejects_invalid_max_workers(self, mock_data_fetcher, mock_atc_config):
        """Test that scan rejects invalid max_workers."""
        with pytest.raises(ValueError, match="max_workers must be a positive integer"):
            scan_all_symbols(mock_data_fetcher, mock_atc_config, max_workers=0)


# ==================== BASIC SCAN FUNCTIONALITY TESTS ====================


class TestBasicScanFunctionality:
    """Test basic scan functionality."""

    @patch("modules.adaptive_trend_LTS_mini.core.scanner.scan_all_symbols._scan_sequential")
    @patch("modules.adaptive_trend_LTS_mini.core.scanner.scan_all_symbols.get_hardware_manager")
    @patch("modules.adaptive_trend_LTS_mini.core.scanner.scan_all_symbols.get_memory_manager")
    def test_scan_discovers_symbols_when_no_filter(
        self, mock_mem_mgr, mock_hw_mgr, mock_scan_seq, mock_data_fetcher, mock_atc_config
    ):
        """Test that scan discovers symbols from exchange when no filter provided."""
        # Setup mocks
        mock_hw_mgr.return_value = Mock()
        mock_hw_mgr.return_value.get_optimal_workload_config.return_value = Mock(num_threads=4, num_processes=2)
        mock_mem_mgr.return_value = Mock()
        mock_mem_mgr.return_value.safe_memory_operation.return_value.__enter__ = Mock()
        mock_mem_mgr.return_value.safe_memory_operation.return_value.__exit__ = Mock()
        mock_mem_mgr.return_value.log_memory_stats = Mock()

        mock_scan_seq.return_value = ([], 0, 0, [])

        scan_all_symbols(mock_data_fetcher, mock_atc_config, execution_mode="sequential")

        # Should call list_binance_futures_symbols
        mock_data_fetcher.list_binance_futures_symbols.assert_called_once()

    @patch("modules.adaptive_trend_LTS_mini.core.scanner.scan_all_symbols._scan_sequential")
    @patch("modules.adaptive_trend_LTS_mini.core.scanner.scan_all_symbols.get_hardware_manager")
    @patch("modules.adaptive_trend_LTS_mini.core.scanner.scan_all_symbols.get_memory_manager")
    def test_scan_skips_discovery_with_symbol_filter(
        self, mock_mem_mgr, mock_hw_mgr, mock_scan_seq, mock_data_fetcher, mock_atc_config, sample_symbols
    ):
        """Test that scan skips discovery when symbols list provided."""
        # Setup mocks
        mock_hw_mgr.return_value = Mock()
        mock_hw_mgr.return_value.get_optimal_workload_config.return_value = Mock(num_threads=4, num_processes=2)
        mock_mem_mgr.return_value = Mock()
        mock_mem_mgr.return_value.safe_memory_operation.return_value.__enter__ = Mock()
        mock_mem_mgr.return_value.safe_memory_operation.return_value.__exit__ = Mock()
        mock_mem_mgr.return_value.log_memory_stats = Mock()

        mock_scan_seq.return_value = ([], 0, 0, [])

        scan_all_symbols(
            mock_data_fetcher, mock_atc_config,
            symbols=sample_symbols, execution_mode="sequential"
        )

        # Should NOT call list_binance_futures_symbols
        mock_data_fetcher.list_binance_futures_symbols.assert_not_called()

    @patch("modules.adaptive_trend_LTS_mini.core.scanner.scan_all_symbols._scan_sequential")
    @patch("modules.adaptive_trend_LTS_mini.core.scanner.scan_all_symbols.get_hardware_manager")
    @patch("modules.adaptive_trend_LTS_mini.core.scanner.scan_all_symbols.get_memory_manager")
    def test_scan_respects_max_symbols(
        self, mock_mem_mgr, mock_hw_mgr, mock_scan_seq, mock_data_fetcher, mock_atc_config
    ):
        """Test that scan respects max_symbols parameter."""
        # Setup mocks
        mock_hw_mgr.return_value = Mock()
        mock_hw_mgr.return_value.get_optimal_workload_config.return_value = Mock(num_threads=4, num_processes=2)
        mock_mem_mgr.return_value = Mock()
        mock_mem_mgr.return_value.safe_memory_operation.return_value.__enter__ = Mock()
        mock_mem_mgr.return_value.safe_memory_operation.return_value.__exit__ = Mock()
        mock_mem_mgr.return_value.log_memory_stats = Mock()

        mock_scan_seq.return_value = ([], 0, 0, [])

        scan_all_symbols(
            mock_data_fetcher, mock_atc_config,
            max_symbols=3, execution_mode="sequential"
        )

        # Check that scan was called with limited symbols
        call_args = mock_scan_seq.call_args[0]
        symbols_arg = call_args[0]
        assert len(symbols_arg) <= 3

    @patch("modules.adaptive_trend_LTS_mini.core.scanner.scan_all_symbols._scan_sequential")
    @patch("modules.adaptive_trend_LTS_mini.core.scanner.scan_all_symbols.get_hardware_manager")
    @patch("modules.adaptive_trend_LTS_mini.core.scanner.scan_all_symbols.get_memory_manager")
    def test_scan_returns_empty_dataframes_when_no_symbols(
        self, mock_mem_mgr, mock_hw_mgr, mock_scan_seq, mock_data_fetcher, mock_atc_config
    ):
        """Test that scan returns empty DataFrames when no symbols found."""
        # Setup mocks
        mock_hw_mgr.return_value = Mock()
        mock_hw_mgr.return_value.get_optimal_workload_config.return_value = Mock(num_threads=4, num_processes=2)
        mock_mem_mgr.return_value = Mock()
        mock_mem_mgr.return_value.safe_memory_operation.return_value.__enter__ = Mock()
        mock_mem_mgr.return_value.safe_memory_operation.return_value.__exit__ = Mock()
        mock_mem_mgr.return_value.log_memory_stats = Mock()

        mock_data_fetcher.list_binance_futures_symbols.return_value = []

        long_signals, short_signals = scan_all_symbols(
            mock_data_fetcher, mock_atc_config, execution_mode="sequential"
        )

        assert isinstance(long_signals, pd.DataFrame)
        assert isinstance(short_signals, pd.DataFrame)
        assert len(long_signals) == 0
        assert len(short_signals) == 0


# ==================== EXECUTION MODE TESTS ====================


class TestExecutionModes:
    """Test different execution modes."""

    @patch("modules.adaptive_trend_LTS_mini.core.scanner.scan_all_symbols._scan_sequential")
    @patch("modules.adaptive_trend_LTS_mini.core.scanner.scan_all_symbols.get_hardware_manager")
    @patch("modules.adaptive_trend_LTS_mini.core.scanner.scan_all_symbols.get_memory_manager")
    def test_scan_uses_sequential_mode(
        self, mock_mem_mgr, mock_hw_mgr, mock_scan_seq, mock_data_fetcher, mock_atc_config, sample_symbols
    ):
        """Test that scan uses sequential mode when specified."""
        # Setup mocks
        mock_hw_mgr.return_value = Mock()
        mock_hw_mgr.return_value.get_optimal_workload_config.return_value = Mock(num_threads=4, num_processes=2)
        mock_mem_mgr.return_value = Mock()
        mock_mem_mgr.return_value.safe_memory_operation.return_value.__enter__ = Mock()
        mock_mem_mgr.return_value.safe_memory_operation.return_value.__exit__ = Mock()
        mock_mem_mgr.return_value.log_memory_stats = Mock()

        mock_scan_seq.return_value = ([], 0, 0, [])

        scan_all_symbols(
            mock_data_fetcher, mock_atc_config,
            symbols=sample_symbols, execution_mode="sequential"
        )

        mock_scan_seq.assert_called_once()

    @patch("modules.adaptive_trend_LTS_mini.core.scanner.scan_all_symbols._scan_threadpool")
    @patch("modules.adaptive_trend_LTS_mini.core.scanner.scan_all_symbols.get_hardware_manager")
    @patch("modules.adaptive_trend_LTS_mini.core.scanner.scan_all_symbols.get_memory_manager")
    def test_scan_uses_threadpool_mode(
        self, mock_mem_mgr, mock_hw_mgr, mock_scan_tp, mock_data_fetcher, mock_atc_config, sample_symbols
    ):
        """Test that scan uses threadpool mode when specified."""
        # Setup mocks
        mock_hw_mgr.return_value = Mock()
        mock_hw_mgr.return_value.get_optimal_workload_config.return_value = Mock(num_threads=4, num_processes=2)
        mock_mem_mgr.return_value = Mock()
        mock_mem_mgr.return_value.safe_memory_operation.return_value.__enter__ = Mock()
        mock_mem_mgr.return_value.safe_memory_operation.return_value.__exit__ = Mock()
        mock_mem_mgr.return_value.log_memory_stats = Mock()

        mock_scan_tp.return_value = ([], 0, 0, [])

        scan_all_symbols(
            mock_data_fetcher, mock_atc_config,
            symbols=sample_symbols, execution_mode="threadpool"
        )

        mock_scan_tp.assert_called_once()

    @patch("modules.adaptive_trend_LTS_mini.core.scanner.scan_all_symbols._scan_asyncio")
    @patch("modules.adaptive_trend_LTS_mini.core.scanner.scan_all_symbols.get_hardware_manager")
    @patch("modules.adaptive_trend_LTS_mini.core.scanner.scan_all_symbols.get_memory_manager")
    def test_scan_uses_asyncio_mode(
        self, mock_mem_mgr, mock_hw_mgr, mock_scan_async, mock_data_fetcher, mock_atc_config, sample_symbols
    ):
        """Test that scan uses asyncio mode when specified."""
        # Setup mocks
        mock_hw_mgr.return_value = Mock()
        mock_hw_mgr.return_value.get_optimal_workload_config.return_value = Mock(num_threads=4, num_processes=2)
        mock_mem_mgr.return_value = Mock()
        mock_mem_mgr.return_value.safe_memory_operation.return_value.__enter__ = Mock()
        mock_mem_mgr.return_value.safe_memory_operation.return_value.__exit__ = Mock()
        mock_mem_mgr.return_value.log_memory_stats = Mock()

        mock_scan_async.return_value = ([], 0, 0, [])

        scan_all_symbols(
            mock_data_fetcher, mock_atc_config,
            symbols=sample_symbols, execution_mode="asyncio"
        )

        mock_scan_async.assert_called_once()

    @patch("modules.adaptive_trend_LTS_mini.core.scanner.scan_all_symbols._scan_processpool")
    @patch("modules.adaptive_trend_LTS_mini.core.scanner.scan_all_symbols.get_hardware_manager")
    @patch("modules.adaptive_trend_LTS_mini.core.scanner.scan_all_symbols.get_memory_manager")
    def test_scan_uses_processpool_mode(
        self, mock_mem_mgr, mock_hw_mgr, mock_scan_pp, mock_data_fetcher, mock_atc_config, sample_symbols
    ):
        """Test that scan uses processpool mode when specified."""
        # Setup mocks
        mock_hw_mgr.return_value = Mock()
        mock_hw_mgr.return_value.get_optimal_workload_config.return_value = Mock(num_threads=4, num_processes=2)
        mock_mem_mgr.return_value = Mock()
        mock_mem_mgr.return_value.safe_memory_operation.return_value.__enter__ = Mock()
        mock_mem_mgr.return_value.safe_memory_operation.return_value.__exit__ = Mock()
        mock_mem_mgr.return_value.log_memory_stats = Mock()

        mock_scan_pp.return_value = ([], 0, 0, [])

        scan_all_symbols(
            mock_data_fetcher, mock_atc_config,
            symbols=sample_symbols, execution_mode="processpool"
        )

        mock_scan_pp.assert_called_once()

    @patch("modules.adaptive_trend_LTS_mini.core.scanner.scan_all_symbols._scan_dask")
    @patch("modules.adaptive_trend_LTS_mini.core.scanner.scan_all_symbols.get_hardware_manager")
    @patch("modules.adaptive_trend_LTS_mini.core.scanner.scan_all_symbols.get_memory_manager")
    def test_scan_uses_dask_mode(
        self, mock_mem_mgr, mock_hw_mgr, mock_scan_dask, mock_data_fetcher, mock_atc_config, sample_symbols
    ):
        """Test that scan uses dask mode when specified."""
        # Setup mocks
        mock_hw_mgr.return_value = Mock()
        mock_hw_mgr.return_value.get_optimal_workload_config.return_value = Mock(num_threads=4, num_processes=2)
        mock_mem_mgr.return_value = Mock()
        mock_mem_mgr.return_value.safe_memory_operation.return_value.__enter__ = Mock()
        mock_mem_mgr.return_value.safe_memory_operation.return_value.__exit__ = Mock()
        mock_mem_mgr.return_value.log_memory_stats = Mock()

        mock_scan_dask.return_value = ([], 0, 0, [])

        scan_all_symbols(
            mock_data_fetcher, mock_atc_config,
            symbols=sample_symbols, execution_mode="dask"
        )

        mock_scan_dask.assert_called_once()

    @patch("modules.adaptive_trend_LTS_mini.core.scanner.scan_all_symbols._scan_sequential")
    @patch("modules.adaptive_trend_LTS_mini.core.scanner.scan_all_symbols.get_hardware_manager")
    @patch("modules.adaptive_trend_LTS_mini.core.scanner.scan_all_symbols.get_memory_manager")
    def test_scan_auto_mode_selects_optimal_execution(
        self, mock_mem_mgr, mock_hw_mgr, mock_scan_seq, mock_data_fetcher, mock_atc_config, sample_symbols
    ):
        """Test that auto mode selects optimal execution mode."""
        # Setup mocks
        mock_hw = Mock()
        mock_hw.get_optimal_execution_mode.return_value = "sequential"
        mock_hw.get_optimal_workload_config.return_value = Mock(num_threads=4, num_processes=2)
        mock_hw_mgr.return_value = mock_hw

        mock_mem_mgr.return_value = Mock()
        mock_mem_mgr.return_value.safe_memory_operation.return_value.__enter__ = Mock()
        mock_mem_mgr.return_value.safe_memory_operation.return_value.__exit__ = Mock()
        mock_mem_mgr.return_value.log_memory_stats = Mock()

        mock_scan_seq.return_value = ([], 0, 0, [])

        scan_all_symbols(
            mock_data_fetcher, mock_atc_config,
            symbols=sample_symbols, execution_mode="auto"
        )

        # Should call get_optimal_execution_mode
        mock_hw.get_optimal_execution_mode.assert_called_once()
        mock_scan_seq.assert_called_once()


# ==================== RESULT FORMATTING TESTS ====================


class TestResultFormatting:
    """Test result formatting and sorting."""

    @patch("modules.adaptive_trend_LTS_mini.core.scanner.scan_all_symbols._scan_sequential")
    @patch("modules.adaptive_trend_LTS_mini.core.scanner.scan_all_symbols.get_hardware_manager")
    @patch("modules.adaptive_trend_LTS_mini.core.scanner.scan_all_symbols.get_memory_manager")
    def test_scan_separates_long_short_signals(
        self, mock_mem_mgr, mock_hw_mgr, mock_scan_seq, mock_data_fetcher, mock_atc_config, sample_symbols
    ):
        """Test that scan correctly separates LONG and SHORT signals."""
        # Setup mocks
        mock_hw_mgr.return_value = Mock()
        mock_hw_mgr.return_value.get_optimal_workload_config.return_value = Mock(num_threads=4, num_processes=2)
        mock_mem_mgr.return_value = Mock()
        mock_mem_mgr.return_value.safe_memory_operation.return_value.__enter__ = Mock()
        mock_mem_mgr.return_value.safe_memory_operation.return_value.__exit__ = Mock()
        mock_mem_mgr.return_value.log_memory_stats = Mock()

        # Mock results with mixed signals
        mock_results = [
            {"symbol": "BTC/USDT", "signal": 0.5, "trend": 1, "price": 50000, "exchange": "binance"},
            {"symbol": "ETH/USDT", "signal": -0.3, "trend": -1, "price": 3000, "exchange": "binance"},
            {"symbol": "SOL/USDT", "signal": 0.2, "trend": 1, "price": 100, "exchange": "binance"},
        ]
        mock_scan_seq.return_value = (mock_results, 0, 0, [])

        long_signals, short_signals = scan_all_symbols(
            mock_data_fetcher, mock_atc_config,
            symbols=sample_symbols, execution_mode="sequential"
        )

        # Check LONG signals
        assert len(long_signals) == 2
        assert all(long_signals["trend"] > 0)

        # Check SHORT signals
        assert len(short_signals) == 1
        assert all(short_signals["trend"] < 0)

    @patch("modules.adaptive_trend_LTS_mini.core.scanner.scan_all_symbols._scan_sequential")
    @patch("modules.adaptive_trend_LTS_mini.core.scanner.scan_all_symbols.get_hardware_manager")
    @patch("modules.adaptive_trend_LTS_mini.core.scanner.scan_all_symbols.get_memory_manager")
    def test_scan_sorts_long_signals_descending(
        self, mock_mem_mgr, mock_hw_mgr, mock_scan_seq, mock_data_fetcher, mock_atc_config, sample_symbols
    ):
        """Test that LONG signals are sorted by signal strength descending."""
        # Setup mocks
        mock_hw_mgr.return_value = Mock()
        mock_hw_mgr.return_value.get_optimal_workload_config.return_value = Mock(num_threads=4, num_processes=2)
        mock_mem_mgr.return_value = Mock()
        mock_mem_mgr.return_value.safe_memory_operation.return_value.__enter__ = Mock()
        mock_mem_mgr.return_value.safe_memory_operation.return_value.__exit__ = Mock()
        mock_mem_mgr.return_value.log_memory_stats = Mock()

        mock_results = [
            {"symbol": "BTC/USDT", "signal": 0.2, "trend": 1, "price": 50000, "exchange": "binance"},
            {"symbol": "ETH/USDT", "signal": 0.5, "trend": 1, "price": 3000, "exchange": "binance"},
            {"symbol": "SOL/USDT", "signal": 0.3, "trend": 1, "price": 100, "exchange": "binance"},
        ]
        mock_scan_seq.return_value = (mock_results, 0, 0, [])

        long_signals, _ = scan_all_symbols(
            mock_data_fetcher, mock_atc_config,
            symbols=sample_symbols, execution_mode="sequential"
        )

        # Check sorting (descending)
        signals = long_signals["signal"].tolist()
        assert signals == sorted(signals, reverse=True)
        assert long_signals.iloc[0]["symbol"] == "ETH/USDT"  # Highest signal

    @patch("modules.adaptive_trend_LTS_mini.core.scanner.scan_all_symbols._scan_sequential")
    @patch("modules.adaptive_trend_LTS_mini.core.scanner.scan_all_symbols.get_hardware_manager")
    @patch("modules.adaptive_trend_LTS_mini.core.scanner.scan_all_symbols.get_memory_manager")
    def test_scan_sorts_short_signals_ascending(
        self, mock_mem_mgr, mock_hw_mgr, mock_scan_seq, mock_data_fetcher, mock_atc_config, sample_symbols
    ):
        """Test that SHORT signals are sorted by signal strength ascending (most negative first)."""
        # Setup mocks
        mock_hw_mgr.return_value = Mock()
        mock_hw_mgr.return_value.get_optimal_workload_config.return_value = Mock(num_threads=4, num_processes=2)
        mock_mem_mgr.return_value = Mock()
        mock_mem_mgr.return_value.safe_memory_operation.return_value.__enter__ = Mock()
        mock_mem_mgr.return_value.safe_memory_operation.return_value.__exit__ = Mock()
        mock_mem_mgr.return_value.log_memory_stats = Mock()

        mock_results = [
            {"symbol": "BTC/USDT", "signal": -0.2, "trend": -1, "price": 50000, "exchange": "binance"},
            {"symbol": "ETH/USDT", "signal": -0.5, "trend": -1, "price": 3000, "exchange": "binance"},
            {"symbol": "SOL/USDT", "signal": -0.3, "trend": -1, "price": 100, "exchange": "binance"},
        ]
        mock_scan_seq.return_value = (mock_results, 0, 0, [])

        _, short_signals = scan_all_symbols(
            mock_data_fetcher, mock_atc_config,
            symbols=sample_symbols, execution_mode="sequential"
        )

        # Check sorting (ascending, most negative first)
        signals = short_signals["signal"].tolist()
        assert signals == sorted(signals)
        assert short_signals.iloc[0]["symbol"] == "ETH/USDT"  # Most negative


# ==================== ERROR HANDLING TESTS ====================


class TestErrorHandling:
    """Test error handling and edge cases."""

    @patch("modules.adaptive_trend_LTS_mini.core.scanner.scan_all_symbols._scan_sequential")
    @patch("modules.adaptive_trend_LTS_mini.core.scanner.scan_all_symbols.get_hardware_manager")
    @patch("modules.adaptive_trend_LTS_mini.core.scanner.scan_all_symbols.get_memory_manager")
    def test_scan_handles_empty_results(
        self, mock_mem_mgr, mock_hw_mgr, mock_scan_seq, mock_data_fetcher, mock_atc_config, sample_symbols
    ):
        """Test that scan handles empty results gracefully."""
        # Setup mocks
        mock_hw_mgr.return_value = Mock()
        mock_hw_mgr.return_value.get_optimal_workload_config.return_value = Mock(num_threads=4, num_processes=2)
        mock_mem_mgr.return_value = Mock()
        mock_mem_mgr.return_value.safe_memory_operation.return_value.__enter__ = Mock()
        mock_mem_mgr.return_value.safe_memory_operation.return_value.__exit__ = Mock()
        mock_mem_mgr.return_value.log_memory_stats = Mock()

        mock_scan_seq.return_value = ([], 5, 0, ["BTC/USDT", "ETH/USDT"])

        long_signals, short_signals = scan_all_symbols(
            mock_data_fetcher, mock_atc_config,
            symbols=sample_symbols, execution_mode="sequential"
        )

        assert isinstance(long_signals, pd.DataFrame)
        assert isinstance(short_signals, pd.DataFrame)
        assert len(long_signals) == 0
        assert len(short_signals) == 0

    @patch("modules.adaptive_trend_LTS_mini.core.scanner.scan_all_symbols._scan_sequential")
    @patch("modules.adaptive_trend_LTS_mini.core.scanner.scan_all_symbols.get_hardware_manager")
    @patch("modules.adaptive_trend_LTS_mini.core.scanner.scan_all_symbols.get_memory_manager")
    def test_scan_handles_keyboard_interrupt(
        self, mock_mem_mgr, mock_hw_mgr, mock_scan_seq, mock_data_fetcher, mock_atc_config, sample_symbols
    ):
        """Test that scan handles KeyboardInterrupt gracefully."""
        # Setup mocks
        mock_hw_mgr.return_value = Mock()
        mock_hw_mgr.return_value.get_optimal_workload_config.return_value = Mock(num_threads=4, num_processes=2)

        # Properly mock context manager
        mock_mem = Mock()
        mock_context = Mock()
        mock_context.__enter__ = Mock(return_value=mock_context)
        mock_context.__exit__ = Mock(return_value=False)
        mock_mem.safe_memory_operation.return_value = mock_context
        mock_mem.log_memory_stats = Mock()
        mock_mem_mgr.return_value = mock_mem

        mock_scan_seq.side_effect = KeyboardInterrupt()

        long_signals, short_signals = scan_all_symbols(
            mock_data_fetcher, mock_atc_config,
            symbols=sample_symbols, execution_mode="sequential"
        )

        # Should return empty DataFrames
        assert isinstance(long_signals, pd.DataFrame)
        assert isinstance(short_signals, pd.DataFrame)
        assert len(long_signals) == 0
        assert len(short_signals) == 0

    @patch("modules.adaptive_trend_LTS_mini.core.scanner.scan_all_symbols._scan_sequential")
    @patch("modules.adaptive_trend_LTS_mini.core.scanner.scan_all_symbols.get_hardware_manager")
    @patch("modules.adaptive_trend_LTS_mini.core.scanner.scan_all_symbols.get_memory_manager")
    def test_scan_handles_generic_exception(
        self, mock_mem_mgr, mock_hw_mgr, mock_scan_seq, mock_data_fetcher, mock_atc_config, sample_symbols
    ):
        """Test that scan handles generic exceptions gracefully."""
        # Setup mocks
        mock_hw_mgr.return_value = Mock()
        mock_hw_mgr.return_value.get_optimal_workload_config.return_value = Mock(num_threads=4, num_processes=2)

        # Properly mock context manager
        mock_mem = Mock()
        mock_context = Mock()
        mock_context.__enter__ = Mock(return_value=mock_context)
        mock_context.__exit__ = Mock(return_value=False)
        mock_mem.safe_memory_operation.return_value = mock_context
        mock_mem.log_memory_stats = Mock()
        mock_mem_mgr.return_value = mock_mem

        mock_scan_seq.side_effect = RuntimeError("Test error")

        long_signals, short_signals = scan_all_symbols(
            mock_data_fetcher, mock_atc_config,
            symbols=sample_symbols, execution_mode="sequential"
        )

        # Should return empty DataFrames
        assert isinstance(long_signals, pd.DataFrame)
        assert isinstance(short_signals, pd.DataFrame)
        assert len(long_signals) == 0
        assert len(short_signals) == 0

    def test_scan_handles_empty_symbol_list(self, mock_data_fetcher, mock_atc_config):
        """Test that scan handles empty symbol list gracefully."""
        with patch("modules.adaptive_trend_LTS_mini.core.scanner.scan_all_symbols.get_hardware_manager"):
            with patch("modules.adaptive_trend_LTS_mini.core.scanner.scan_all_symbols.get_memory_manager"):
                long_signals, short_signals = scan_all_symbols(
                    mock_data_fetcher, mock_atc_config,
                    symbols=[], execution_mode="sequential"
                )

                assert len(long_signals) == 0
                assert len(short_signals) == 0

    def test_scan_validates_data_fetcher_has_required_methods(self, mock_atc_config):
        """Test that scan validates data_fetcher has required methods."""
        incomplete_fetcher = Mock(spec=[])  # No methods

        with pytest.raises(AttributeError, match="data_fetcher must have method"):
            scan_all_symbols(incomplete_fetcher, mock_atc_config)


# ==================== MEMORY MANAGEMENT TESTS ====================


class TestMemoryManagement:
    """Test memory management features."""

    @patch("modules.adaptive_trend_LTS_mini.core.scanner.scan_all_symbols._scan_sequential")
    @patch("modules.adaptive_trend_LTS_mini.core.scanner.scan_all_symbols.get_hardware_manager")
    @patch("modules.adaptive_trend_LTS_mini.core.scanner.scan_all_symbols.get_memory_manager")
    def test_scan_uses_memory_manager_context(
        self, mock_mem_mgr, mock_hw_mgr, mock_scan_seq, mock_data_fetcher, mock_atc_config, sample_symbols
    ):
        """Test that scan uses memory manager context correctly."""
        # Setup mocks
        mock_hw_mgr.return_value = Mock()
        mock_hw_mgr.return_value.get_optimal_workload_config.return_value = Mock(num_threads=4, num_processes=2)

        # Properly mock context manager
        mock_mem = Mock()
        mock_context = Mock()
        mock_context.__enter__ = Mock(return_value=mock_context)
        mock_context.__exit__ = Mock(return_value=False)
        mock_mem.safe_memory_operation.return_value = mock_context
        mock_mem.log_memory_stats = Mock()
        mock_mem_mgr.return_value = mock_mem

        # Return results with signals so log_memory_stats is called
        mock_results = [
            {"symbol": "BTC/USDT", "signal": 0.5, "trend": 1, "price": 50000, "exchange": "binance"},
        ]
        mock_scan_seq.return_value = (mock_results, 0, 0, [])

        scan_all_symbols(
            mock_data_fetcher, mock_atc_config,
            symbols=sample_symbols, execution_mode="sequential"
        )

        # Should call safe_memory_operation
        mock_mem.safe_memory_operation.assert_called_once()
        # log_memory_stats is called when there are results
        mock_mem.log_memory_stats.assert_called_once()

    @patch("modules.adaptive_trend_LTS_mini.core.scanner.scan_all_symbols._scan_sequential")
    @patch("modules.adaptive_trend_LTS_mini.core.scanner.scan_all_symbols.get_hardware_manager")
    @patch("modules.adaptive_trend_LTS_mini.core.scanner.scan_all_symbols.get_memory_manager")
    def test_scan_passes_batch_size_to_execution(
        self, mock_mem_mgr, mock_hw_mgr, mock_scan_seq, mock_data_fetcher, mock_atc_config, sample_symbols
    ):
        """Test that scan passes batch_size parameter to execution function."""
        # Setup mocks
        mock_hw_mgr.return_value = Mock()
        mock_hw_mgr.return_value.get_optimal_workload_config.return_value = Mock(num_threads=4, num_processes=2)
        mock_mem_mgr.return_value = Mock()
        mock_mem_mgr.return_value.safe_memory_operation.return_value.__enter__ = Mock()
        mock_mem_mgr.return_value.safe_memory_operation.return_value.__exit__ = Mock()
        mock_mem_mgr.return_value.log_memory_stats = Mock()

        mock_scan_seq.return_value = ([], 0, 0, [])

        custom_batch_size = 25
        scan_all_symbols(
            mock_data_fetcher, mock_atc_config,
            symbols=sample_symbols,
            execution_mode="sequential",
            batch_size=custom_batch_size
        )

        # Check batch_size was passed
        call_kwargs = mock_scan_seq.call_args[1] if mock_scan_seq.call_args[1] else {}
        call_args = mock_scan_seq.call_args[0] if len(mock_scan_seq.call_args[0]) > 4 else []

        # batch_size is 5th positional arg or in kwargs
        if len(call_args) > 4:
            assert call_args[4] == custom_batch_size
        else:
            assert call_kwargs.get("batch_size") == custom_batch_size
