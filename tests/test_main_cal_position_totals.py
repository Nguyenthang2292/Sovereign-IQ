"""
Test for main_cal_position_totals.py - Position Totals Calculator.

Tests the position calculation functionality with mocked data.
"""

import sys
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

# Add project root to path
ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from main_cal_position_totals import calculate_position_totals, display_positions, display_totals, main


class TestCalculatePositionTotals:
    """Test the calculate_position_totals function."""

    def test_calculate_position_totals_success(self):
        """Test successful calculation with mock positions."""
        mock_positions = [
            {"symbol": "BTCUSDT", "direction": "LONG", "size_usdt": 1000.0, "entry_price": 50000.0, "contracts": 0.02},
            {"symbol": "ETHUSDT", "direction": "SHORT", "size_usdt": 500.0, "entry_price": 3000.0, "contracts": 0.1667},
            {"symbol": "ADAUSDT", "direction": "LONG", "size_usdt": 200.0, "entry_price": 1.5, "contracts": 133.33},
        ]

        with patch("main_cal_position_totals.DataFetcher") as mock_data_fetcher_class:
            mock_data_fetcher = MagicMock()
            mock_data_fetcher_class.return_value = mock_data_fetcher
            mock_data_fetcher.fetch_binance_futures_positions.return_value = mock_positions

            long_total, short_total, combined_total, positions = calculate_position_totals()

            assert long_total == 1200.0  # 1000 + 200
            assert short_total == 500.0
            assert combined_total == 1700.0  # 1200 + 500
            assert positions == mock_positions

    def test_calculate_position_totals_empty_positions(self):
        """Test calculation with no open positions."""
        with patch("main_cal_position_totals.DataFetcher") as mock_data_fetcher_class:
            mock_data_fetcher = MagicMock()
            mock_data_fetcher_class.return_value = mock_data_fetcher
            mock_data_fetcher.fetch_binance_futures_positions.return_value = []

            long_total, short_total, combined_total, positions = calculate_position_totals()

            assert long_total == 0.0
            assert short_total == 0.0
            assert combined_total == 0.0
            assert positions == []

    def test_calculate_position_totals_fetch_error(self):
        """Test handling of fetch errors."""
        with patch("main_cal_position_totals.DataFetcher") as mock_data_fetcher_class:
            mock_data_fetcher = MagicMock()
            mock_data_fetcher_class.return_value = mock_data_fetcher
            mock_data_fetcher.fetch_binance_futures_positions.side_effect = ValueError("API Error")

            long_total, short_total, combined_total, positions = calculate_position_totals()

            assert long_total is None
            assert short_total is None
            assert combined_total is None
            assert positions == []

    def test_calculate_position_totals_exception_handling(self):
        """Test handling of unexpected exceptions."""
        with patch("main_cal_position_totals.DataFetcher") as mock_data_fetcher_class:
            mock_data_fetcher = MagicMock()
            mock_data_fetcher_class.return_value = mock_data_fetcher
            mock_data_fetcher.fetch_binance_futures_positions.side_effect = Exception("Unexpected error")

            long_total, short_total, combined_total, positions = calculate_position_totals()

            assert long_total is None
            assert short_total is None
            assert combined_total is None
            assert positions == []


class TestDisplayFunctions:
    """Test the display functions."""

    def test_display_positions_with_data(self, capsys):
        """Test display_positions with position data."""
        positions = [
            {"symbol": "BTCUSDT", "direction": "LONG", "size_usdt": 1000.0, "entry_price": 50000.0, "contracts": 0.02},
            {"symbol": "ETHUSDT", "direction": "SHORT", "size_usdt": 500.0, "entry_price": 3000.0, "contracts": 0.1667},
        ]

        display_positions(positions)

        captured = capsys.readouterr()
        assert "POSITION DETAILS" in captured.out
        assert "BTCUSDT" in captured.out
        assert "ETHUSDT" in captured.out
        assert "LONG" in captured.out
        assert "SHORT" in captured.out

    def test_display_positions_empty(self, capsys):
        """Test display_positions with empty list."""
        display_positions([])

        captured = capsys.readouterr()
        # Should not print position details header for empty list
        assert "POSITION DETAILS" not in captured.out

    def test_display_totals(self, capsys):
        """Test display_totals function."""
        display_totals(1200.0, 500.0, 1700.0)

        captured = capsys.readouterr()
        assert "TOTALS" in captured.out
        assert "1200.000000" in captured.out  # LONG
        assert "500.000000" in captured.out  # SHORT
        assert "1700.000000" in captured.out  # Combined


class TestMainFunction:
    """Test the main function."""

    @patch("main_cal_position_totals.calculate_position_totals")
    @patch("main_cal_position_totals.display_positions")
    @patch("main_cal_position_totals.display_totals")
    def test_main_success(self, mock_display_totals, mock_display_positions, mock_calc_totals, capsys):
        """Test main function with successful execution."""
        mock_calc_totals.return_value = (
            1200.0,
            500.0,
            1700.0,
            [{"symbol": "BTCUSDT", "direction": "LONG", "size_usdt": 1200.0}],
        )

        main()

        # Verify calculation was called
        mock_calc_totals.assert_called_once()

        # Verify displays were called
        mock_display_positions.assert_called_once()
        mock_display_totals.assert_called_once_with(1200.0, 500.0, 1700.0)

        # Check output
        captured = capsys.readouterr()
        assert "Calculate Position Totals" in captured.out

    @patch("main_cal_position_totals.calculate_position_totals")
    def test_main_calculation_failure(self, mock_calc_totals, capsys):
        """Test main function when calculation fails."""
        mock_calc_totals.return_value = (None, None, None, [])

        with pytest.raises(SystemExit) as exc_info:
            main()

        assert exc_info.value.code == 1
        captured = capsys.readouterr()
        assert "Failed to calculate position totals" in captured.out

    @patch("main_cal_position_totals.calculate_position_totals")
    def test_main_no_positions(self, mock_calc_totals, capsys):
        """Test main function with no positions."""
        mock_calc_totals.return_value = (0.0, 0.0, 0.0, [])

        with pytest.raises(SystemExit) as exc_info:
            main()

        assert exc_info.value.code == 0
        captured = capsys.readouterr()
        assert "No open positions found" in captured.out

    @patch("main_cal_position_totals.calculate_position_totals")
    def test_main_keyboard_interrupt(self, mock_calc_totals):
        """Test main function handles KeyboardInterrupt."""
        mock_calc_totals.side_effect = KeyboardInterrupt()

        with pytest.raises(SystemExit):  # main() calls sys.exit(0) on KeyboardInterrupt
            main()

    @patch("main_cal_position_totals.calculate_position_totals")
    def test_main_unexpected_exception(self, mock_calc_totals):
        """Test main function handles unexpected exceptions."""
        mock_calc_totals.side_effect = Exception("Test exception")

        with pytest.raises(SystemExit):  # main() calls sys.exit(1) on exceptions
            main()


class TestModuleIntegration:
    """Test module-level integration."""

    def test_module_imports(self):
        """Test that all required modules can be imported."""
        try:
            from main_cal_position_totals import calculate_position_totals, display_positions, display_totals, main

            # Verify functions exist
            assert callable(calculate_position_totals)
            assert callable(display_positions)
            assert callable(display_totals)
            assert callable(main)

        except ImportError as e:
            pytest.fail(f"Failed to import functions from main_cal_position_totals: {e}")

    def test_colorama_integration(self):
        """Test that colorama is properly integrated."""
        # This tests that the colorama imports work
        from main_cal_position_totals import color_text

        # Test basic color functionality
        colored_text = color_text("test", "red")
        assert "test" in colored_text
        assert colored_text != "test"  # Should have color codes

    def test_data_fetcher_integration(self):
        """Test that DataFetcher can be imported and instantiated."""
        import importlib.util

        data_fetcher_spec = importlib.util.find_spec("modules.common.core.data_fetcher")
        if data_fetcher_spec is None:
            pytest.fail("DataFetcher module not found")

        exchange_manager_spec = importlib.util.find_spec("modules.common.core.exchange_manager")
        if exchange_manager_spec is None:
            pytest.fail("ExchangeManager module not found")

        from modules.common.core.exchange_manager import ExchangeManager
        from modules.common.core.data_fetcher import DataFetcher

        exchange_manager = ExchangeManager()
        assert exchange_manager is not None

        data_fetcher = DataFetcher(exchange_manager)
        assert data_fetcher is not None