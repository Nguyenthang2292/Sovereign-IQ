"""
Tests for MarketBatchScanner class.

Tests cover:
- Initialization
- Symbol fetching (delegation)
- Batch splitting
- Batch processing (delegation)
- Results saving (delegation)
- Cleanup (delegation)
"""

import json
import os
from unittest.mock import MagicMock, Mock, patch

import pytest

from modules.gemini_chart_analyzer.core.exceptions import DataFetchError
from modules.gemini_chart_analyzer.core.scanners.market_batch_scanner import MarketBatchScanner


@pytest.fixture
def sample_symbols():
    """Sample trading symbols."""
    return ["BTC/USDT", "ETH/USDT", "BNB/USDT", "SOL/USDT", "ADA/USDT"]


@pytest.fixture
def mock_scanner_dependencies():
    """Fixture to patch all MarketBatchScanner dependencies."""
    with (
        patch(
            "modules.gemini_chart_analyzer.core.scanners.market_batch_scanner.ExchangeManager"
        ) as mock_exchange_manager,
        patch("modules.gemini_chart_analyzer.core.scanners.market_batch_scanner.DataFetcher") as mock_data_fetcher,
        patch(
            "modules.gemini_chart_analyzer.core.scanners.market_batch_scanner.ChartBatchGenerator"
        ) as mock_batch_chart,
        patch(
            "modules.gemini_chart_analyzer.core.scanners.market_batch_scanner.GeminiBatchChartAnalyzer"
        ) as mock_batch_analyzer,
        patch("modules.gemini_chart_analyzer.core.scanners.market_batch_scanner.SymbolFetcher") as mock_symbol_fetcher,
        patch(
            "modules.gemini_chart_analyzer.core.scanners.market_batch_scanner.DataFetcherAdapter"
        ) as mock_data_fetcher_adapter,
        patch("modules.gemini_chart_analyzer.core.scanners.market_batch_scanner.ResultManager") as mock_result_manager,
        patch(
            "modules.gemini_chart_analyzer.core.scanners.market_batch_scanner.CleanupManager"
        ) as mock_cleanup_manager,
    ):
        yield {
            "exchange_manager": mock_exchange_manager,
            "data_fetcher": mock_data_fetcher,
            "batch_chart": mock_batch_chart,
            "batch_analyzer": mock_batch_analyzer,
            "symbol_fetcher": mock_symbol_fetcher,
            "data_fetcher_adapter": mock_data_fetcher_adapter,
            "result_manager": mock_result_manager,
            "cleanup_manager": mock_cleanup_manager,
        }


class TestMarketBatchScannerInit:
    """Test MarketBatchScanner initialization."""

    def test_init_default_params(self, mock_scanner_dependencies):
        """Test initialization with default parameters."""
        scanner = MarketBatchScanner()

        assert scanner.charts_per_batch == 100
        assert scanner.cooldown_seconds == 2.5
        assert scanner.quote_currency == "USDT"
        assert scanner.min_candles == MarketBatchScanner.MIN_CANDLES
        assert scanner.exchange_name == "binance"

        # Verify components initialized
        mock_scanner_dependencies["symbol_fetcher"].assert_called_with(exchange_name="binance", quote_currency="USDT")
        mock_scanner_dependencies["data_fetcher_adapter"].assert_called()

    def test_init_custom_params(self, mock_scanner_dependencies):
        """Test initialization with custom parameters."""
        scanner = MarketBatchScanner(
            charts_per_batch=50, cooldown_seconds=5.0, quote_currency="BTC", exchange_name="okx", min_candles=30
        )

        assert scanner.charts_per_batch == 50
        assert scanner.cooldown_seconds == 5.0
        assert scanner.quote_currency == "BTC"
        assert scanner.exchange_name == "okx"
        assert scanner.min_candles == 30

        mock_scanner_dependencies["symbol_fetcher"].assert_called_with(exchange_name="okx", quote_currency="BTC")

    def test_init_min_candles_validation(self, mock_scanner_dependencies):
        """Test min_candles validation."""
        with pytest.raises(ValueError, match="min_candles must be greater than 0"):
            MarketBatchScanner(min_candles=0)

        with pytest.raises(ValueError, match="min_candles must be greater than 0"):
            MarketBatchScanner(min_candles=-1)


class TestMarketBatchScannerGetSymbols:
    """Test symbol fetching delegation."""

    def test_get_all_symbols_delegation(self, mock_scanner_dependencies, sample_symbols):
        """Test get_all_symbols delegates to SymbolFetcher."""
        mock_fetcher_instance = mock_scanner_dependencies["symbol_fetcher"].return_value
        mock_fetcher_instance.get_all_symbols.return_value = sample_symbols

        scanner = MarketBatchScanner()
        symbols = scanner.get_all_symbols(max_retries=5, retry_delay=2.0)

        assert symbols == sample_symbols
        mock_fetcher_instance.get_all_symbols.assert_called_once_with(max_retries=5, retry_delay=2.0)


class TestMarketBatchScannerSplitBatches:
    """Test batch splitting."""

    def test_split_into_batches_exact(self, mock_scanner_dependencies):
        """Test splitting into exact batch size."""
        scanner = MarketBatchScanner(charts_per_batch=10)
        symbols = [f"SYM{i}/USDT" for i in range(10)]
        batches = scanner._split_into_batches(symbols)
        assert len(batches) == 1
        assert len(batches[0]) == 10

    def test_split_into_batches_multiple(self, mock_scanner_dependencies):
        """Test splitting into multiple batches."""
        scanner = MarketBatchScanner(charts_per_batch=10)
        symbols = [f"SYM{i}/USDT" for i in range(25)]
        batches = scanner._split_into_batches(symbols)
        assert len(batches) == 3
        assert len(batches[0]) == 10
        assert len(batches[1]) == 10
        assert len(batches[2]) == 5


class TestMarketBatchScannerProcessBatches:
    """Test batch processing."""

    def test_process_single_tf_batch_delegation(self, mock_scanner_dependencies):
        """Test _process_single_tf_batch delegates to components."""
        mock_adapter = mock_scanner_dependencies["data_fetcher_adapter"].return_value
        mock_generator = mock_scanner_dependencies["batch_chart"].return_value
        mock_analyzer = mock_scanner_dependencies["batch_analyzer"].return_value

        # Setup mocks
        mock_adapter.fetch_batch_data.return_value = [{"symbol": "BTC/USDT"}]
        mock_generator.create_batch_chart.return_value = ("path/to/chart.png", False)
        mock_analyzer.analyze_batch_chart.return_value = {"BTC/USDT": {"signal": "LONG"}}

        scanner = MarketBatchScanner()
        # Trigger lazy init of analyzer
        _ = scanner.batch_gemini_analyzer

        # Inject mock analyzer (lazy property creates a new one usually, but we mocked the class)
        scanner._gemini_analyzer = mock_analyzer

        result = scanner._process_single_tf_batch(["BTC/USDT"], "1h", 100, 1)

        assert result == {"BTC/USDT": {"signal": "LONG"}}
        mock_adapter.fetch_batch_data.assert_called_once_with(["BTC/USDT"], "1h", 100)
        mock_generator.create_batch_chart.assert_called_once()
        mock_analyzer.analyze_batch_chart.assert_called_once()


class TestMarketBatchScannerCleanup:
    """Test cleanup functionality."""

    def test_cleanup_delegation(self, mock_scanner_dependencies):
        """Test cleanup delegates to managers."""
        mock_symbol_fetcher = mock_scanner_dependencies["symbol_fetcher"].return_value
        mock_exchange_manager = mock_scanner_dependencies["exchange_manager"].return_value

        scanner = MarketBatchScanner()

        # We need to ensure exchange_manager is the mock
        scanner.exchange_manager = mock_exchange_manager

        with patch("gc.collect") as mock_gc:
            scanner.cleanup(force_gc=True)

            mock_symbol_fetcher.cleanup.assert_called_once()
            # Check if exchange_manager cleanup methods are called if they exist
            # The test assumes they might exist or raises Attribute error if not mocked properly
            # Since we use MagicMock by default with patch, methods exist.

            # The implementation checks hasattr, so MagicMock answers True.
            mock_exchange_manager.cleanup_unused_exchanges.assert_called()
            mock_exchange_manager.clear.assert_called()

            assert mock_gc.call_count == 2
