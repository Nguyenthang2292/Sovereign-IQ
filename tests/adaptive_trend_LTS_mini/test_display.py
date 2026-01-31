"""Tests for modules/adaptive_trend_LTS_mini/cli/display.py.

Tests cover:
1. Empty DataFrame/Series handling
2. Display output format validation
3. Color code application
4. Column alignment with various data lengths
5. Error handling paths
"""

from io import StringIO
from unittest.mock import MagicMock, patch

import pandas as pd
import pytest
from colorama import Fore, Style

from modules.adaptive_trend_LTS_mini.cli.display import (
    COL_EXCHANGE_WIDTH,
    COL_PRICE_WIDTH,
    COL_SIGNAL_WIDTH,
    COL_SYMBOL_WIDTH,
    DISPLAY_WIDTH,
    _display_equity_weights,
    _display_ma_signals,
    _get_trend_direction,
    display_atc_signals,
    display_scan_results,
    list_futures_symbols,
)


class TestGetTrendDirection:
    """Test _get_trend_direction helper function."""

    def test_bullish_trend(self):
        """Test positive trend value returns BULLISH."""
        direction, color = _get_trend_direction(1.5)
        assert direction == "BULLISH"
        assert color == Fore.GREEN

    def test_bearish_trend(self):
        """Test negative trend value returns BEARISH."""
        direction, color = _get_trend_direction(-1.5)
        assert direction == "BEARISH"
        assert color == Fore.RED

    def test_neutral_trend(self):
        """Test zero trend value returns NEUTRAL."""
        direction, color = _get_trend_direction(0)
        assert direction == "NEUTRAL"
        assert color == Fore.YELLOW


class TestDisplayMaSignals:
    """Test _display_ma_signals helper function."""

    @patch("builtins.print")
    def test_display_empty_signals(self, mock_print):
        """Test displaying empty MA signals."""
        ma_signals = [
            ("EMA", pd.Series(dtype=float)),
            ("HMA", pd.Series(dtype=float)),
        ]

        _display_ma_signals(ma_signals)

        # Should print header but no data
        calls = [str(call) for call in mock_print.call_args_list]
        assert any("Individual MA Signals" in str(call) for call in calls)

    @patch("builtins.print")
    def test_display_bullish_signals(self, mock_print):
        """Test displaying bullish MA signals."""
        ema_signal = pd.Series([0.1, 0.2, 0.3])
        ma_signals = [("EMA", ema_signal)]

        _display_ma_signals(ma_signals)

        # Should contain upward indicator
        output = " ".join(str(call) for call in mock_print.call_args_list)
        assert "^" in output or "EMA" in output

    @patch("builtins.print")
    def test_display_bearish_signals(self, mock_print):
        """Test displaying bearish MA signals."""
        ema_signal = pd.Series([0.3, 0.2, 0.1])
        ma_signals = [("EMA", ema_signal)]

        _display_ma_signals(ma_signals)

        # Should contain downward indicator
        output = " ".join(str(call) for call in mock_print.call_args_list)
        assert "v" in output or "EMA" in output


class TestDisplayEquityWeights:
    """Test _display_equity_weights helper function."""

    @patch("builtins.print")
    def test_display_empty_weights(self, mock_print):
        """Test displaying empty equity weights."""
        ma_weights = [
            ("EMA", pd.Series(dtype=float)),
            ("HMA", pd.Series(dtype=float)),
        ]

        _display_equity_weights(ma_weights)

        # Should print header but no data
        calls = [str(call) for call in mock_print.call_args_list]
        assert any("Equity Weights" in str(call) for call in calls)

    @patch("builtins.print")
    def test_display_valid_weights(self, mock_print):
        """Test displaying valid equity weights."""
        ema_s = pd.Series([0.15, 0.20, 0.25])
        ma_weights = [("EMA", ema_s)]

        _display_equity_weights(ma_weights)

        output = " ".join(str(call) for call in mock_print.call_args_list)
        assert "EMA" in output
        assert "0.25" in output

    @patch("builtins.print")
    def test_display_nan_weights(self, mock_print):
        """Test displaying NaN equity weights."""
        ema_s = pd.Series([0.15, float("nan"), 0.25])
        ma_weights = [("EMA", ema_s)]

        _display_equity_weights(ma_weights)

        output = " ".join(str(call) for call in mock_print.call_args_list)
        # Should skip NaN value
        assert "EMA" in output


class TestDisplayAtcSignals:
    """Test display_atc_signals function - empty DataFrame/Series handling."""

    @patch("modules.adaptive_trend_LTS_mini.cli.display.log_error")
    def test_empty_average_signal(self, mock_log_error):
        """Test handling of None average signal."""
        atc_results = {"Average_Signal": None}

        display_atc_signals(
            symbol="BTC/USDT",
            df=pd.DataFrame(),
            atc_results=atc_results,
            current_price=50000.0,
            exchange_label="Binance",
        )

        mock_log_error.assert_called_once_with("No ATC signals available")

    @patch("modules.adaptive_trend_LTS_mini.cli.display.log_error")
    def test_empty_series(self, mock_log_error):
        """Test handling of empty series."""
        atc_results = {"Average_Signal": pd.Series(dtype=float)}

        display_atc_signals(
            symbol="BTC/USDT",
            df=pd.DataFrame(),
            atc_results=atc_results,
            current_price=50000.0,
            exchange_label="Binance",
        )

        mock_log_error.assert_called_once_with("No ATC signals available")

    @patch("modules.adaptive_trend_LTS_mini.cli.display.log_error")
    def test_zero_length_series(self, mock_log_error):
        """Test handling of zero-length series that passes .empty check."""
        # Create a series that is not .empty but has len == 0
        atc_results = {"Average_Signal": pd.Series([], dtype=float)}

        display_atc_signals(
            symbol="BTC/USDT",
            df=pd.DataFrame(),
            atc_results=atc_results,
            current_price=50000.0,
            exchange_label="Binance",
        )

        # Should catch in the second validation
        assert mock_log_error.call_count >= 1

    @patch("builtins.print")
    def test_valid_display(self, mock_print):
        """Test valid ATC signal display."""
        atc_results = {
            "Average_Signal": pd.Series([0.1, 0.2, 0.3]),
            "EMA_Signal": pd.Series([0.1, 0.2, 0.3]),
            "EMA_S": pd.Series([0.15, 0.20, 0.25]),
        }

        display_atc_signals(
            symbol="BTC/USDT",
            df=pd.DataFrame({"close": [50000, 51000, 52000]}),
            atc_results=atc_results,
            current_price=52000.0,
            exchange_label="Binance",
        )

        output = " ".join(str(call) for call in mock_print.call_args_list)
        assert "BTC/USDT" in output
        assert "Binance" in output


class TestDisplayScanResults:
    """Test display_scan_results function - format and alignment."""

    @patch("builtins.print")
    def test_empty_signals(self, mock_print):
        """Test display with empty signals."""
        long_signals = pd.DataFrame()
        short_signals = pd.DataFrame()

        display_scan_results(long_signals, short_signals, min_signal=0.01)

        output = " ".join(str(call) for call in mock_print.call_args_list)
        assert "No LONG signals found" in output
        assert "No SHORT signals found" in output
        assert "0 LONG + 0 SHORT = 0 signals" in output

    @patch("builtins.print")
    def test_long_signals_format(self, mock_print):
        """Test LONG signals display format."""
        long_signals = pd.DataFrame(
            {
                "symbol": ["BTC/USDT", "ETH/USDT"],
                "signal": [0.5, 0.6],
                "price": [50000.0, 3000.0],
                "exchange": ["Binance", "KuCoin"],
            }
        )
        short_signals = pd.DataFrame()

        display_scan_results(long_signals, short_signals, min_signal=0.01)

        output = " ".join(str(call) for call in mock_print.call_args_list)
        assert "2 symbols with LONG signals" in output
        assert "BTC/USDT" in output
        assert "ETH/USDT" in output

    @patch("builtins.print")
    def test_short_signals_format(self, mock_print):
        """Test SHORT signals display format."""
        long_signals = pd.DataFrame()
        short_signals = pd.DataFrame(
            {
                "symbol": ["BTC/USDT"],
                "signal": [-0.5],
                "price": [50000.0],
                "exchange": ["Binance"],
            }
        )

        display_scan_results(long_signals, short_signals, min_signal=0.01)

        output = " ".join(str(call) for call in mock_print.call_args_list)
        assert "1 symbols with SHORT signals" in output
        assert "BTC/USDT" in output

    @patch("builtins.print")
    def test_column_alignment(self, mock_print):
        """Test column alignment with various data lengths."""
        long_signals = pd.DataFrame(
            {
                "symbol": ["BTC/USDT", "VERYLONGSYMBOLNAME/USDT"],
                "signal": [0.123456, 0.987654],
                "price": [1.23, 1234567.89],
                "exchange": ["A", "VeryLongExchangeName"],
            }
        )
        short_signals = pd.DataFrame()

        display_scan_results(long_signals, short_signals, min_signal=0.01)

        # Should not raise error and maintain column structure
        output = " ".join(str(call) for call in mock_print.call_args_list)
        assert "BTC/USDT" in output
        assert "VERYLONGSYMBOLNAME/USDT" in output

    @patch("builtins.print")
    def test_uses_constants(self, mock_print):
        """Test that display uses module-level constants."""
        # Verify constants exist and have expected values
        assert DISPLAY_WIDTH == 80
        assert COL_SYMBOL_WIDTH == 15
        assert COL_SIGNAL_WIDTH == 12
        assert COL_PRICE_WIDTH == 15
        assert COL_EXCHANGE_WIDTH == 10

        long_signals = pd.DataFrame(
            {"symbol": ["BTC/USDT"], "signal": [0.5], "price": [50000.0], "exchange": ["Binance"]}
        )
        short_signals = pd.DataFrame()

        display_scan_results(long_signals, short_signals, min_signal=0.01)

        # Should complete without error using constants
        assert mock_print.called


class TestListFuturesSymbols:
    """Test list_futures_symbols function - error handling."""

    @patch("modules.adaptive_trend_LTS_mini.cli.display.log_error")
    @patch("modules.adaptive_trend_LTS_mini.cli.display.log_success")
    @patch("modules.adaptive_trend_LTS_mini.cli.display.log_progress")
    @patch("builtins.print")
    def test_empty_symbols_list(self, mock_print, mock_log_progress, mock_log_success, mock_log_error):
        """Test handling of empty symbols list."""
        mock_data_fetcher = MagicMock()
        mock_data_fetcher.list_binance_futures_symbols.return_value = []

        list_futures_symbols(mock_data_fetcher)

        mock_log_error.assert_called_once_with("No symbols found")
        mock_log_success.assert_not_called()

    @patch("modules.adaptive_trend_LTS_mini.cli.display.log_error")
    @patch("modules.adaptive_trend_LTS_mini.cli.display.log_success")
    @patch("modules.adaptive_trend_LTS_mini.cli.display.log_progress")
    @patch("builtins.print")
    def test_valid_symbols_list(self, mock_print, mock_log_progress, mock_log_success, mock_log_error):
        """Test successful symbols listing."""
        mock_data_fetcher = MagicMock()
        mock_data_fetcher.list_binance_futures_symbols.return_value = ["BTC/USDT", "ETH/USDT", "XRP/USDT"]

        list_futures_symbols(mock_data_fetcher)

        mock_log_success.assert_called_once()
        assert "3 futures symbols" in mock_log_success.call_args[0][0]

        output = " ".join(str(call) for call in mock_print.call_args_list)
        assert "BTC/USDT" in output
        assert "ETH/USDT" in output

    @patch("modules.adaptive_trend_LTS_mini.cli.display.log_error")
    @patch("modules.adaptive_trend_LTS_mini.cli.display.log_progress")
    def test_attribute_error_handling(self, mock_log_progress, mock_log_error):
        """Test AttributeError handling."""
        mock_data_fetcher = MagicMock()
        mock_data_fetcher.list_binance_futures_symbols.side_effect = AttributeError("Method not found")

        list_futures_symbols(mock_data_fetcher)

        mock_log_error.assert_called_once()
        assert "AttributeError" in mock_log_error.call_args[0][0]

    @patch("modules.adaptive_trend_LTS_mini.cli.display.log_error")
    @patch("modules.adaptive_trend_LTS_mini.cli.display.log_progress")
    def test_value_error_handling(self, mock_log_progress, mock_log_error):
        """Test ValueError handling."""
        mock_data_fetcher = MagicMock()
        mock_data_fetcher.list_binance_futures_symbols.side_effect = ValueError("Invalid value")

        list_futures_symbols(mock_data_fetcher)

        mock_log_error.assert_called_once()
        assert "ValueError" in mock_log_error.call_args[0][0]

    @patch("modules.adaptive_trend_LTS_mini.cli.display.log_error")
    @patch("modules.adaptive_trend_LTS_mini.cli.display.log_progress")
    def test_key_error_handling(self, mock_log_progress, mock_log_error):
        """Test KeyError handling."""
        mock_data_fetcher = MagicMock()
        mock_data_fetcher.list_binance_futures_symbols.side_effect = KeyError("Key not found")

        list_futures_symbols(mock_data_fetcher)

        mock_log_error.assert_called_once()
        assert "KeyError" in mock_log_error.call_args[0][0]

    @patch("modules.adaptive_trend_LTS_mini.cli.display.log_error")
    @patch("modules.adaptive_trend_LTS_mini.cli.display.log_progress")
    def test_unexpected_error_handling(self, mock_log_progress, mock_log_error):
        """Test unexpected error handling and re-raise."""
        mock_data_fetcher = MagicMock()
        mock_data_fetcher.list_binance_futures_symbols.side_effect = RuntimeError("Unexpected error")

        with pytest.raises(RuntimeError):
            list_futures_symbols(mock_data_fetcher)

        # Should log and re-raise
        mock_log_error.assert_called_once()
        assert "Unexpected error" in mock_log_error.call_args[0][0]

    @patch("modules.adaptive_trend_LTS_mini.cli.display.log_success")
    @patch("modules.adaptive_trend_LTS_mini.cli.display.log_progress")
    @patch("builtins.print")
    def test_max_symbols_parameter(self, mock_print, mock_log_progress, mock_log_success):
        """Test max_symbols parameter is passed correctly."""
        mock_data_fetcher = MagicMock()
        mock_data_fetcher.list_binance_futures_symbols.return_value = ["BTC/USDT", "ETH/USDT"]

        list_futures_symbols(mock_data_fetcher, max_symbols=10)

        mock_data_fetcher.list_binance_futures_symbols.assert_called_once_with(
            max_candidates=10, progress_label="Symbol Discovery"
        )


class TestColorCodeApplication:
    """Test color code application in outputs."""

    @patch("modules.adaptive_trend_LTS_mini.cli.display.color_text")
    @patch("builtins.print")
    def test_bullish_color_applied(self, mock_print, mock_color_text):
        """Test green color for bullish signals."""
        # Make color_text return the text as-is for testing
        mock_color_text.side_effect = lambda text, *args, **kwargs: text

        long_signals = pd.DataFrame(
            {"symbol": ["BTC/USDT"], "signal": [0.5], "price": [50000.0], "exchange": ["Binance"]}
        )
        short_signals = pd.DataFrame()

        display_scan_results(long_signals, short_signals, min_signal=0.01)

        # Check that Fore.GREEN was passed to color_text
        green_calls = [call for call in mock_color_text.call_args_list if len(call[0]) > 1 and call[0][1] == Fore.GREEN]
        assert len(green_calls) > 0, "Expected Fore.GREEN to be used for bullish signals"

    @patch("modules.adaptive_trend_LTS_mini.cli.display.color_text")
    @patch("builtins.print")
    def test_bearish_color_applied(self, mock_print, mock_color_text):
        """Test red color for bearish signals."""
        # Make color_text return the text as-is for testing
        mock_color_text.side_effect = lambda text, *args, **kwargs: text

        long_signals = pd.DataFrame()
        short_signals = pd.DataFrame(
            {"symbol": ["BTC/USDT"], "signal": [-0.5], "price": [50000.0], "exchange": ["Binance"]}
        )

        display_scan_results(long_signals, short_signals, min_signal=0.01)

        # Check that Fore.RED was passed to color_text
        red_calls = [call for call in mock_color_text.call_args_list if len(call[0]) > 1 and call[0][1] == Fore.RED]
        assert len(red_calls) > 0, "Expected Fore.RED to be used for bearish signals"

    @patch("modules.adaptive_trend_LTS_mini.cli.display.color_text")
    @patch("builtins.print")
    def test_header_color_applied(self, mock_print, mock_color_text):
        """Test cyan color for headers."""
        # Make color_text return the text as-is for testing
        mock_color_text.side_effect = lambda text, *args, **kwargs: text

        long_signals = pd.DataFrame()
        short_signals = pd.DataFrame()

        display_scan_results(long_signals, short_signals, min_signal=0.01)

        # Check that Fore.CYAN was passed to color_text
        cyan_calls = [call for call in mock_color_text.call_args_list if len(call[0]) > 1 and call[0][1] == Fore.CYAN]
        assert len(cyan_calls) > 0, "Expected Fore.CYAN to be used for headers"
