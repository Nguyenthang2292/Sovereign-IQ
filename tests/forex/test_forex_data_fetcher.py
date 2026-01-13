"""
Unit tests for ForexDataFetcher module.

Tests cover:
- Timeframe conversion
- Symbol conversion
- Rate limiting
- Fallback behavior
- Timeout handling
- Async methods
"""

import time
import unittest
from pathlib import Path
from unittest.mock import Mock, patch

import pandas as pd

# Add project root to path for imports
project_root = Path(__file__).parent.parent.parent
import sys

sys.path.insert(0, str(project_root))

from modules.common.core.forex_data_fetcher import ForexDataFetcher


class TestForexDataFetcher(unittest.TestCase):
    """Test ForexDataFetcher functionality."""

    def test_init_without_credentials(self):
        """Test initialization without hardcoded credentials."""
        with patch.dict("os.environ", {}, clear=True):
            fetcher = ForexDataFetcher()
            self.assertIsNotNone(fetcher)
            self.assertEqual(fetcher.request_pause, 1.0)

    def test_init_with_custom_rate_limit(self):
        """Test initialization with custom rate limit."""
        fetcher = ForexDataFetcher(request_pause=2.5)
        self.assertEqual(fetcher.request_pause, 2.5)

    def test_timeframe_conversion_tradingview(self):
        """Test timeframe conversion to TradingView format."""
        fetcher = ForexDataFetcher()

        test_cases = [
            ("1h", "60"),
            ("4h", "240"),
            ("1d", "D"),
            ("1w", "W"),
            ("15m", "15"),
            ("30m", "30"),
        ]

        for input_timeframe, expected in test_cases:
            with self.subTest(input=input_timeframe):
                result = fetcher._convert_timeframe(input_timeframe, "tradingview")
                self.assertEqual(result, expected)

    def test_symbol_conversion(self):
        """Test forex symbol conversion to TradingView format."""
        fetcher = ForexDataFetcher()

        test_cases = [
            ("EUR/USD", "OANDA", "OANDA:EURUSD"),
            ("GBP/USD", "FXCM", "FXCM:GBPUSD"),
            ("AUD-USD", "FOREXCOM", "FOREXCOM:AUDUSD"),
        ]

        for symbol, exchange, expected in test_cases:
            with self.subTest(symbol=symbol, exchange=exchange):
                result = fetcher._convert_forex_symbol_to_tradingview(symbol, exchange)
                self.assertEqual(result, expected)

    def test_symbol_conversion_for_tvdatafeed(self):
        """Test forex symbol conversion for tvDatafeed."""
        fetcher = ForexDataFetcher()

        test_cases = [
            ("EUR/USD", "OANDA", ("EURUSD", "OANDA")),
            ("GBP/USD", "FXCM", ("GBPUSD", "FXCM")),
            ("AUD/USD", "IC MARKETS", ("AUDUSD", "ICMARKETS")),
        ]

        for symbol, exchange, expected in test_cases:
            with self.subTest(symbol=symbol, exchange=exchange):
                result = fetcher._convert_forex_symbol_for_tvdatafeed(symbol, exchange)
                self.assertEqual(result, expected)

    def test_rate_limiting(self):
        """Test that rate limiting works correctly."""
        fetcher = ForexDataFetcher(request_pause=0.1)

        mock_func = Mock()
        mock_func.return_value = "result"

        # First call should not wait
        start = time.time()
        fetcher._throttled_call(mock_func)
        time.time() - start

        # Second call should wait at least request_pause
        start = time.time()
        fetcher._throttled_call(mock_func)
        elapsed2 = time.time() - start

        self.assertGreaterEqual(elapsed2, 0.1)
        mock_func.assert_called()

    def test_supported_exchanges_order(self):
        """Test that supported exchanges are in correct priority order."""
        fetcher = ForexDataFetcher()

        expected_order = ["OANDA", "FXCM", "FOREXCOM", "IC MARKETS", "PEPPERSTONE"]
        self.assertEqual(fetcher.SUPPORTED_FOREX_EXCHANGES, expected_order)

    def test_process_tvdatafeed_result_valid_df(self):
        """Test processing valid tvDatafeed result."""
        fetcher = ForexDataFetcher()

        data = {
            "Open": [1.0, 1.1, 1.2],
            "High": [1.1, 1.2, 1.3],
            "Low": [0.9, 1.0, 1.1],
            "Close": [1.05, 1.15, 1.25],
        }
        df = pd.DataFrame(data, index=pd.date_range("2024-01-01", periods=3, tz="UTC"))

        result_df, source = fetcher._process_tvdatafeed_result(df, "EUR/USD", "OANDA", 5)

        self.assertIsNotNone(result_df)
        self.assertEqual(source, "tvDatafeed-OANDA")
        self.assertIn("volume", result_df.columns)
        self.assertEqual(len(result_df), 3)

    def test_process_tvdatafeed_result_invalid_df(self):
        """Test processing invalid tvDatafeed result."""
        fetcher = ForexDataFetcher()

        data = {"Open": [1.0, 1.1], "High": [1.1, 1.2]}
        df = pd.DataFrame(data)

        result_df, source = fetcher._process_tvdatafeed_result(df, "EUR/USD", "OANDA", 5)

        self.assertIsNone(result_df)
        self.assertIsNone(source)

    def test_process_tvdatafeed_result_limit(self):
        """Test that result is limited to requested number of bars."""
        fetcher = ForexDataFetcher()

        data = {
            "Open": [1.0, 1.1, 1.2, 1.3, 1.4, 1.5],
            "High": [1.1, 1.2, 1.3, 1.4, 1.5, 1.6],
            "Low": [0.9, 1.0, 1.1, 1.2, 1.3, 1.4],
            "Close": [1.05, 1.15, 1.25, 1.35, 1.45, 1.55],
        }
        df = pd.DataFrame(data, index=pd.date_range("2024-01-01", periods=6, tz="UTC"))

        result_df, _ = fetcher._process_tvdatafeed_result(df, "EUR/USD", "OANDA", 3)

        self.assertEqual(len(result_df), 3)


class TestForexDataFetcherAsync(unittest.IsolatedAsyncioTestCase):
    """Test async methods of ForexDataFetcher."""

    async def test_fetch_ohlcv_async_timeout(self):
        """Test async fetch with timeout handling."""
        fetcher = ForexDataFetcher()
        fetcher._tvdatafeed_available = True

        mock_tvdatafeed = Mock()
        mock_tvdatafeed.get_hist = Mock(side_effect=lambda **kwargs: time.sleep(60))
        fetcher.tv_datafeed = mock_tvdatafeed

        df, source = await fetcher.fetch_ohlcv_async("EUR/USD", "1h", 10)

        self.assertIsNone(df)
        self.assertIsNone(source)

    async def test_fetch_ohlcv_async_success(self):
        """Test async fetch with successful response (requires tvDatafeed installed)."""
        # Skip test if tvDatafeed is not available
        try:
            import tvDatafeed  # noqa: F401
        except ImportError:
            self.skipTest("tvDatafeed not installed - skipping async success test")

        fetcher = ForexDataFetcher()
        fetcher._tvdatafeed_available = True

        mock_tvdatafeed = Mock()
        data = {
            "open": [1.0, 1.1],
            "high": [1.1, 1.2],
            "low": [0.9, 1.0],
            "close": [1.05, 1.15],
            "volume": [100, 200],
        }
        df = pd.DataFrame(data, index=pd.date_range("2024-01-01", periods=2, tz="UTC"))
        mock_tvdatafeed.get_hist = Mock(return_value=df)
        fetcher.tv_datafeed = mock_tvdatafeed

        result_df, source = await fetcher.fetch_ohlcv_async("EUR/USD", "1h", 10)

        self.assertIsNotNone(result_df)
        self.assertIsNotNone(source)
        self.assertEqual(len(result_df), 2)


if __name__ == "__main__":
    unittest.main()
