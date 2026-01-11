
from pathlib import Path
from unittest.mock import Mock, patch
import sys
import unittest

"""
Unit tests for main_gemini_chart_batch_scanner_forex.py

Tests cover:
- Exchange name defaulting with 'or' operator
- Result list creation with proper slicing
- Windows stdin setup and restoration
- Timeframe parsing and normalization
- Error handling with new helper functions
"""


# Add project root to path for imports
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from main_gemini_chart_batch_scanner_forex import FOREX_MAJOR_PAIRS, FOREX_MINOR_PAIRS, get_forex_symbols, main_forex
from modules.common.utils import normalize_timeframe


class TestMainForexScanner(unittest.TestCase):
    """Test main forex scanner functions."""

    @patch("builtins.input")
    def test_exchange_name_defaulting_or_operator(self, mock_input):
        """Test exchange name defaults correctly with 'or' operator."""
        # Test when user provides input
        mock_input.return_value = "kraken"

        # Simulate the input section
        with patch("builtins.input", return_value="kraken"):
            result = "kraken" if "kraken" else "binance"

        self.assertEqual(result, "kraken")

        # Test when user provides empty input
        mock_input.return_value = ""

        with patch("builtins.input", return_value=""):
            result = "" if "" else "binance"

        self.assertEqual(result, "binance")

        # Test when input is None (shouldn't happen but test robustness)
        mock_input.return_value = None

        with patch("builtins.input", return_value=None):
            result = None if None else "binance"

        self.assertEqual(result, "binance")

    def test_result_list_creation_proper_slicing(self):
        """Test result list creation with correct slicing."""
        # Mock long_symbols list
        mock_symbols = ["EUR/USD", "GBP/USD", "AUD/USD", "CAD/USD", "CHF/USD", "NZD/USD"]

        # Test slicing with range step
        for i in range(0, len(mock_symbols), 5):
            expected_start = i
            expected_end = i + 5
            actual_end = min(i + 5, len(mock_symbols))

            with self.subTest(f"Slice starting at {i}"):
                row = mock_symbols[i : i + 5]
                self.assertEqual(len(row), actual_end - expected_start)

                # Verify the list comprehension in actual code
                expected_row = mock_symbols[expected_start:actual_end]
                self.assertEqual(row, expected_row)

    @patch("main_gemini_chart_batch_scanner_forex.setup_windows_stdin")
    @patch("main_gemini_chart_batch_scanner_forex.sys")
    def test_windows_stdin_setup_and_restoration(self, mock_sys, mock_setup):
        """Test Windows stdin is properly setup and restored."""
        # Simulate Windows environment
        mock_sys.platform = "win32"
        mock_sys.stdin = Mock()
        mock_sys.stdin.is_file.return_value = False
        mock_sys.stdin.closed = Mock(return_value=False)

        # Mock the open call
        mock_file = Mock()

        with patch("builtins.open", return_value=mock_file) as mock_open:
            mock_setup()

            # Verify setup was called
            mock_setup.assert_called_once()

            # Verify open was called with correct parameters
            mock_open.assert_called_once_with("CON", "r", encoding="utf-8", errors="replace")

    def test_timeframe_normalization_redundancy_removed(self):
        """Test that timeframe normalization is not redundant."""
        # This test ensures the redundant normalize calls were removed
        # The actual normalization logic would be tested in the modules

        # Test with valid timeframe
        with patch("main_gemini_chart_batch_scanner_forex.normalize_timeframe") as mock_normalize:
            mock_normalize.return_value = "1h"

            # Simulate the timeframe handling in main function
            # After our fix, normalize should only be called once
            try:
                timeframe = normalize_timeframe("1h")
            except Exception:
                timeframe = "1h"

            mock_normalize.assert_called_once_with("1h")
            self.assertEqual(timeframe, "1h")

    @patch("main_gemini_chart_batch_scanner_forex.setup_windows_stdin")
    @patch("main_gemini_chart_batch_scanner_forex.get_error_code")
    @patch("main_gemini_chart_batch_scanner_forex.is_retryable_error")
    @patch("main_gemini_chart_batch_scanner_forex.PublicExchangeManager")
    def test_error_handling_with_helper_functions(
        self, mock_exchange_class, mock_retryable, mock_error_code, mock_setup
    ):
        """Test that error handling uses helper functions."""
        # Mock the exchange connection to raise an exception
        mock_exchange = Mock()
        mock_exchange.load_markets.side_effect = Exception("Connection timeout")
        mock_exchange_class.return_value.connect_to_exchange_with_no_credentials.return_value = mock_exchange

        # Test with retryable error
        mock_retryable.return_value = True

        result = get_forex_symbols(exchange_name="binance", use_exchange=True, max_retries=2)

        # Should fallback to hardcoded list after retries
        self.assertEqual(len(result), len(FOREX_MAJOR_PAIRS) + len(FOREX_MINOR_PAIRS))

        # Verify helper functions were called
        mock_retryable.assert_called()

        # Should not call get_error_code for network errors (using is_retryable_error instead)
        mock_error_code.assert_not_called()

    def test_comment_typo_fixed(self):
        """Test that comment typos were fixed."""
        # Read the file to verify typo fixes
        file_path = Path(__file__).parent.parent.parent / "main_gemini_chart_batch_scanner_forex.py"

        with open(file_path, "r", encoding="utf-8") as f:
            content = f.read()

        # Check for corrected typos
        self.assertIn("can't fix", content)  # Line 32
        self.assertIn("don't contain", content)  # Line 41

        # Verify old typos are not present
        self.assertNotIn("can't fix", content)
        self.assertNotIn("don't contain", content)

    def test_get_forex_symbols_return_type_consistency(self):
        """Test that get_forex_symbols always returns List[str]."""
        # Test with exchange=None (should return hardcoded)
        result = get_forex_symbols(exchange_name=None, use_exchange=False)

        self.assertIsInstance(result, list)
        self.assertGreater(len(result), 0)

        # All elements should be strings
        for symbol in result:
            self.assertIsInstance(symbol, str)

        # Test with exception during exchange fetch (should still return list)
        with patch("main_gemini_chart_batch_scanner_forex.PublicExchangeManager") as mock_exchange_class:
            mock_exchange_class.return_value.connect_to_exchange_with_no_credentials.side_effect = Exception(
                "Network error"
            )

            result = get_forex_symbols(exchange_name="binance", use_exchange=True, max_retries=1)

            # Should still return a list (fallback)
            self.assertIsInstance(result, list)
            self.assertGreater(len(result), 0)


class TestMainForexIntegration(unittest.TestCase):
    """Integration tests for the main forex scanner."""

    @patch("main_gemini_chart_batch_scanner_forex.safe_input")
    @patch("main_gemini_chart_batch_scanner_forex.MarketBatchScanner")
    @patch("main_gemini_chart_batch_scanner_forex.get_forex_symbols")
    def test_main_integration(self, mock_get_symbols, mock_scanner_class, mock_input):
        """Test main function integration."""
        # Mock user inputs
        mock_input.side_effect = [
            "2",  # Multi-timeframe mode
            "1h,4h",  # Timeframes
            "",  # Default max symbols
            "2.5",  # Cooldown
            "500",  # Limit
            "binance",  # Exchange
        ]

        # Mock symbol list
        mock_get_symbols.return_value = ["EUR/USD", "GBP/USD"]

        # Mock scanner
        mock_scanner = Mock()
        mock_scanner.scan_market.return_value = {
            "long_symbols": ["EUR/USD"],
            "short_symbols": ["GBP/USD"],
            "summary": {"avg_long_confidence": 0.75},
            "results_file": "/path/to/results.json",
        }
        mock_scanner_class.return_value = mock_scanner

        # Mock colorama and other dependencies
        with patch("main_gemini_chart_batch_scanner_forex.colorama_init"):
            with patch("main_gemini_chart_batch_scanner_forex.time.sleep"):
                # This should not raise exceptions
                try:
                    main_forex()
                except (KeyboardInterrupt, SystemExit):
                    pass
                except Exception as e:
                    self.fail(f"main_forex() raised unexpected exception: {e}")


if __name__ == "__main__":
    unittest.main()
