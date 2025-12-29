"""
Tests for MarketBatchScanner class.

Tests cover:
- Initialization
- Symbol fetching
- Batch splitting
- Data fetching
- Full scan workflow (mocked)
- Results saving
"""

import pytest
import json
import os
from unittest.mock import Mock, MagicMock, patch
from pathlib import Path
import pandas as pd
import numpy as np

try:
    import ccxt
    AuthenticationError = ccxt.AuthenticationError
except ImportError:
    # Fallback if ccxt is not available for testing
    class AuthenticationError(Exception):
        """Local AuthenticationError for tests if ccxt is not available."""
        pass

from modules.gemini_chart_analyzer.core.scanners.market_batch_scanner import (
    MarketBatchScanner,
    SymbolFetchError
)

@pytest.fixture
def sample_symbols():
    """Sample trading symbols."""
    return ["BTC/USDT", "ETH/USDT", "BNB/USDT", "SOL/USDT", "ADA/USDT"]


@pytest.fixture
def mock_scanner_dependencies():
    """Fixture to patch all MarketBatchScanner dependencies."""
    with patch('modules.gemini_chart_analyzer.core.scanners.market_batch_scanner.ExchangeManager') as mock_exchange, \
         patch('modules.gemini_chart_analyzer.core.scanners.market_batch_scanner.PublicExchangeManager') as mock_public_exchange, \
         patch('modules.gemini_chart_analyzer.core.scanners.market_batch_scanner.DataFetcher') as mock_data_fetcher, \
         patch('modules.gemini_chart_analyzer.core.scanners.market_batch_scanner.ChartBatchGenerator') as mock_batch_chart, \
         patch('modules.gemini_chart_analyzer.core.scanners.market_batch_scanner.GeminiBatchChartAnalyzer') as mock_batch_analyzer:
        yield {
            'exchange': mock_exchange,
            'public_exchange': mock_public_exchange,
            'data_fetcher': mock_data_fetcher,
            'batch_chart': mock_batch_chart,
            'batch_analyzer': mock_batch_analyzer
        }


class TestMarketBatchScannerInit:
    """Test MarketBatchScanner initialization."""
    
    def test_init_default_params(self, mock_scanner_dependencies):
        """Test initialization with default parameters."""
        scanner = MarketBatchScanner()
        
        assert scanner.charts_per_batch == 100
        assert scanner.cooldown_seconds == 2.5
        assert scanner.quote_currency == 'USDT'
        assert scanner.min_candles == MarketBatchScanner.MIN_CANDLES
        assert scanner.exchange_name == 'binance'  # Default value
    
    def test_init_custom_params(self, mock_scanner_dependencies):
        """Test initialization with custom parameters."""
        scanner = MarketBatchScanner(
            charts_per_batch=50,
            cooldown_seconds=5.0,
            quote_currency='BTC',
            min_candles=30
        )
        
        assert scanner.charts_per_batch == 50
        assert scanner.cooldown_seconds == 5.0
        assert scanner.quote_currency == 'BTC'
        assert scanner.min_candles == 30
        assert scanner.exchange_name == 'binance'  # Default value
    
    def test_init_min_candles_validation(self, mock_scanner_dependencies):
        """Test min_candles validation."""
        # Test invalid min_candles (<= 0)
        with pytest.raises(ValueError, match="min_candles must be greater than 0"):
            MarketBatchScanner(min_candles=0)
        
        with pytest.raises(ValueError, match="min_candles must be greater than 0"):
            MarketBatchScanner(min_candles=-1)
    
    def test_init_with_exchange_name(self, mock_scanner_dependencies):
        """Test initialization with custom exchange name."""
        scanner = MarketBatchScanner(exchange_name='okx')
        
        assert scanner.exchange_name == 'okx'


class TestMarketBatchScannerGetSymbols:
    """Test symbol fetching."""
    
    def test_get_all_symbols_success(self, mock_scanner_dependencies, sample_symbols):
        """Test successful symbol fetching."""
        scanner = MarketBatchScanner()
        
        # Mock exchange and markets
        mock_markets = {}
        for symbol in sample_symbols:
            mock_markets[symbol] = {
                'quote': 'USDT',
                'active': True,
                'type': 'spot'
            }
        
        mock_exchange = mock_scanner_dependencies['exchange']
        mock_exchange.load_markets.return_value = mock_markets
        mock_connect = Mock(return_value=mock_exchange)
        scanner.public_exchange_manager.connect_to_exchange_with_no_credentials = mock_connect
        
        symbols = scanner.get_all_symbols()
        
        # Verify that connect_to_exchange_with_no_credentials was called with correct exchange name
        mock_connect.assert_called_once_with('binance')
        
        assert len(symbols) == 5
        assert "BTC/USDT" in symbols
        assert all(s.endswith('/USDT') for s in symbols)
    
    def test_get_all_symbols_with_custom_exchange(self, mock_scanner_dependencies, sample_symbols):
        """Test symbol fetching with custom exchange name."""
        scanner = MarketBatchScanner(exchange_name='okx')
        
        # Mock exchange and markets
        mock_markets = {}
        for symbol in sample_symbols:
            mock_markets[symbol] = {
                'quote': 'USDT',
                'active': True,
                'type': 'spot'
            }
        
        mock_exchange = mock_scanner_dependencies['exchange']
        mock_exchange.load_markets.return_value = mock_markets
        mock_connect = Mock(return_value=mock_exchange)
        scanner.public_exchange_manager.connect_to_exchange_with_no_credentials = mock_connect
        
        symbols = scanner.get_all_symbols()
        
        # Verify that connect_to_exchange_with_no_credentials was called with custom exchange name
        mock_connect.assert_called_once_with('okx')
        
        assert len(symbols) == 5
        assert "BTC/USDT" in symbols
    
    def test_get_all_symbols_filters_inactive(self, mock_scanner_dependencies):
        """Test symbol fetching filters inactive markets."""
        scanner = MarketBatchScanner()
        
        mock_markets = {
            'BTC/USDT': {'quote': 'USDT', 'active': True, 'type': 'spot'},
            'ETH/USDT': {'quote': 'USDT', 'active': False, 'type': 'spot'},  # Inactive
            'BNB/USDT': {'quote': 'USDT', 'active': True, 'type': 'spot'}
        }
        
        mock_exchange = mock_scanner_dependencies['exchange']
        mock_exchange.load_markets.return_value = mock_markets
        scanner.public_exchange_manager.connect_to_exchange_with_no_credentials = Mock(
            return_value=mock_exchange
        )
        
        symbols = scanner.get_all_symbols()
        
        assert len(symbols) == 2
        assert "BTC/USDT" in symbols
        assert "BNB/USDT" in symbols
        assert "ETH/USDT" not in symbols
    
    def test_get_all_symbols_error_handling(self, mock_scanner_dependencies):
        """Test symbol fetching error handling raises SymbolFetchError."""
        scanner = MarketBatchScanner()
        
        scanner.public_exchange_manager.connect_to_exchange_with_no_credentials = Mock(
            side_effect=Exception("Connection error")
        )
        
        # Should raise SymbolFetchError instead of returning empty list
        with pytest.raises(SymbolFetchError) as exc_info:
            scanner.get_all_symbols()
        
        # Verify exception properties
        assert exc_info.value.is_retryable is True  # Connection errors are retryable
        assert exc_info.value.original_exception is not None
    
    def test_get_all_symbols_non_retryable_error(self, mock_scanner_dependencies):
        """Test symbol fetching with non-retryable error."""
        scanner = MarketBatchScanner()
        
        # Create a non-retryable error (e.g., authentication error)
        auth_error = AuthenticationError("Invalid API key")
        scanner.public_exchange_manager.connect_to_exchange_with_no_credentials = Mock(
            side_effect=auth_error
        )
        
        # Should raise SymbolFetchError with is_retryable=False
        with pytest.raises(SymbolFetchError) as exc_info:
            scanner.get_all_symbols()
        
        # Non-retryable errors should have is_retryable=False
        assert exc_info.value.is_retryable is False
        assert exc_info.value.original_exception is auth_error
    
    def test_get_all_symbols_retry_success(self, mock_scanner_dependencies, sample_symbols):
        """Test symbol fetching with retry logic - succeeds after retry."""
        scanner = MarketBatchScanner()
        
        # Mock markets
        mock_markets = {}
        for symbol in sample_symbols:
            mock_markets[symbol] = {
                'quote': 'USDT',
                'active': True,
                'type': 'spot'
            }
        
        # Mock exchange that fails once then succeeds
        mock_exchange = Mock()
        mock_exchange.load_markets = Mock(
            side_effect=[
                Exception("503 Service Unavailable"),  # First attempt fails
                mock_markets  # Second succeeds
            ]
        )
        
        mock_connect = Mock(return_value=mock_exchange)
        scanner.public_exchange_manager.connect_to_exchange_with_no_credentials = mock_connect
        
        # Should succeed after retry
        symbols = scanner.get_all_symbols(max_retries=3)
        assert symbols == sorted(sample_symbols)
        # Verify retry happened (connect called twice)
        assert mock_connect.call_count == 2


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
    
    def test_split_into_batches_empty(self, mock_scanner_dependencies):
        """Test splitting empty symbol list."""
        scanner = MarketBatchScanner()
        
        batches = scanner._split_into_batches([])
        
        assert len(batches) == 0


class TestMarketBatchScannerFetchData:
    """Test data fetching."""
    
    def test_fetch_batch_data_success(self, mock_scanner_dependencies):
        """Test successful batch data fetching."""
        
        scanner = MarketBatchScanner()
        
        # Create sample DataFrame
        dates = pd.date_range(start='2024-01-01', periods=50, freq='1h')
        sample_df = pd.DataFrame({
            'open': np.random.rand(50) * 50000,
            'high': np.random.rand(50) * 51000,
            'low': np.random.rand(50) * 49000,
            'close': np.random.rand(50) * 50000,
            'volume': np.random.randint(1000, 10000, 50)
        }, index=dates)
        
        symbols = ["BTC/USDT", "ETH/USDT"]
        scanner.data_fetcher.fetch_ohlcv_with_fallback_exchange = Mock(
            return_value=(sample_df, 'binance')
        )
        
        result = scanner._fetch_batch_data(symbols, '1h', 200)
        
        assert len(result) == 2
        assert result[0]['symbol'] == "BTC/USDT"
        assert result[1]['symbol'] == "ETH/USDT"
    
    def test_fetch_batch_data_insufficient_data(self, mock_scanner_dependencies):
        """Test fetching with insufficient data (less than minimum candles)."""
        
        
        scanner = MarketBatchScanner()
        
        # Create DataFrame with only 10 rows (less than minimum required)
        dates = pd.date_range(start='2024-01-01', periods=10, freq='1h')
        small_df = pd.DataFrame({
            'open': [1] * 10,
            'high': [2] * 10,
            'low': [0.5] * 10,
            'close': [1.5] * 10,
            'volume': [1000] * 10
        }, index=dates)
        
        symbols = ["BTC/USDT"]
        scanner.data_fetcher.fetch_ohlcv_with_fallback_exchange = Mock(
            return_value=(small_df, 'binance')
        )
        
        result = scanner._fetch_batch_data(symbols, '1h', 200)
        
        # Should skip symbols with insufficient data
        assert len(result) == 0
    
    def test_fetch_batch_data_custom_min_candles(self, mock_scanner_dependencies):
        """Test fetching with custom min_candles threshold."""
        
        # Create scanner with custom min_candles
        scanner = MarketBatchScanner(min_candles=30)
        
        # Create DataFrame with 25 rows (less than custom min_candles but more than default)
        dates = pd.date_range(start='2024-01-01', periods=25, freq='1h')
        medium_df = pd.DataFrame({
            'open': np.random.rand(25) * 50000,
            'high': np.random.rand(25) * 51000,
            'low': np.random.rand(25) * 49000,
            'close': np.random.rand(25) * 50000,
            'volume': np.random.randint(1000, 10000, 25)
        }, index=dates)
        
        symbols = ["BTC/USDT"]
        scanner.data_fetcher.fetch_ohlcv_with_fallback_exchange = Mock(
            return_value=(medium_df, 'binance')
        )
        
        result = scanner._fetch_batch_data(symbols, '1h', 200)
        
        # Should skip because 25 < 30 (custom min_candles)
        assert len(result) == 0
        
        # Now test with enough candles
        dates = pd.date_range(start='2024-01-01', periods=35, freq='1h')
        enough_df = pd.DataFrame({
            'open': np.random.rand(35) * 50000,
            'high': np.random.rand(35) * 51000,
            'low': np.random.rand(35) * 49000,
            'close': np.random.rand(35) * 50000,
            'volume': np.random.randint(1000, 10000, 35)
        }, index=dates)
        
        scanner.data_fetcher.fetch_ohlcv_with_fallback_exchange = Mock(
            return_value=(enough_df, 'binance')
        )
        
        result = scanner._fetch_batch_data(symbols, '1h', 200)
        
        # Should include because 35 >= 30 (custom min_candles)
        assert len(result) == 1
        assert result[0]['symbol'] == "BTC/USDT"
    
    def test_fetch_batch_data_error_handling(self, mock_scanner_dependencies):
        """Test error handling during data fetching."""
        scanner = MarketBatchScanner()
        
        symbols = ["BTC/USDT", "ETH/USDT"]
        scanner.data_fetcher.fetch_ohlcv_with_fallback_exchange = Mock(
            side_effect=Exception("Fetch error")
        )
        
        result = scanner._fetch_batch_data(symbols, '1h', 200)
        
        # Should return empty list on errors
        assert len(result) == 0


class TestMarketBatchScannerSaveResults:
    """Test results saving."""
    
    def test_save_results(self, mock_scanner_dependencies, tmp_path):
        """Test saving results to JSON file."""
        scanner = MarketBatchScanner()
        
        all_results = {
            "BTC/USDT": {"signal": "LONG", "confidence": 0.85},
            "ETH/USDT": {"signal": "SHORT", "confidence": 0.70},
            "BNB/USDT": {"signal": "NONE", "confidence": 0.50}
        }
        
        long_symbols = ["BTC/USDT"]
        short_symbols = ["ETH/USDT"]
        summary = {
            'total_symbols': 3,
            'scanned_symbols': 3,
            'long_count': 1,
            'short_count': 1,
            'none_count': 1
        }
        
        # Mock the get_analysis_results_dir function
        with patch('modules.gemini_chart_analyzer.core.utils.chart_paths.get_analysis_results_dir',
                   return_value=str(tmp_path / "analysis_results")):
            results_file = scanner._save_results(
                all_results=all_results,
                long_symbols=long_symbols,
                short_symbols=short_symbols,
                summary=summary,
                timeframe='1h'
            )
            
            assert os.path.exists(results_file)
            assert results_file.endswith('.json')
            
            # Verify JSON content
            with open(results_file, 'r', encoding='utf-8') as f:
                data = json.load(f)
            
            assert data['timeframe'] == '1h'
            assert len(data['long_symbols']) == 1
            assert len(data['short_symbols']) == 1
            assert data['summary']['long_count'] == 1


class TestMarketBatchScannerCleanup:
    """Test cleanup functionality in MarketBatchScanner."""
    
    def test_cleanup_resources(self, mock_scanner_dependencies):
        """Test that cleanup() frees exchange connections and triggers GC."""
        scanner = MarketBatchScanner()
        
        # Mock cleanup methods
        mock_auth_cleanup = MagicMock()
        mock_public_cleanup = MagicMock()
        scanner.exchange_manager.cleanup_unused_exchanges = mock_auth_cleanup
        scanner.public_exchange_manager.cleanup_unused_exchanges = mock_public_cleanup
        
        # Mock gc.collect
        with patch('gc.collect') as mock_gc:
            scanner.cleanup()
            
            # Verify cleanup was called on both exchange managers
            mock_auth_cleanup.assert_called_once()
            mock_public_cleanup.assert_called_once()
            
            # Verify GC was called
            mock_gc.assert_called_once()
    
    def test_cleanup_handles_errors(self, mock_scanner_dependencies):
        """Test that cleanup() handles errors gracefully."""
        scanner = MarketBatchScanner()
        
        # Mock cleanup to raise error
        scanner.exchange_manager.cleanup_unused_exchanges = MagicMock(
            side_effect=Exception("Cleanup error")
        )
        scanner.public_exchange_manager.cleanup_unused_exchanges = MagicMock()
        
        # Should not raise error, should continue
        with patch('gc.collect') as mock_gc:
            scanner.cleanup()
            
            # Verify public cleanup was still called
            scanner.public_exchange_manager.cleanup_unused_exchanges.assert_called_once()
            
            # Verify GC was still called
            mock_gc.assert_called_once()
    
    def test_cleanup_with_all_errors(self, mock_scanner_dependencies):
        """Test cleanup when all cleanup methods raise errors."""
        scanner = MarketBatchScanner()
        
        # Mock both cleanup methods to raise errors
        scanner.exchange_manager.cleanup_unused_exchanges = MagicMock(
            side_effect=Exception("Auth cleanup error")
        )
        scanner.public_exchange_manager.cleanup_unused_exchanges = MagicMock(
            side_effect=Exception("Public cleanup error")
        )
        
        # Should not raise error, should still call GC
        with patch('gc.collect') as mock_gc:
            scanner.cleanup()
            
            # Verify GC was still called even with errors
            mock_gc.assert_called_once()


