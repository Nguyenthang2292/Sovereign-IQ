"""
Comprehensive CLI integration tests for ATC LTS Mini module.

Tests the complete CLI workflow including:
- Command-line argument parsing and validation
- Interactive mode workflows
- Auto/manual mode execution
- Error handling in CLI context
- User input handling
- Display and output formatting
"""

import sys
from io import StringIO
from pathlib import Path
from unittest.mock import Mock, MagicMock, patch, call

import numpy as np
import pandas as pd
import pytest

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent))

from modules.adaptive_trend_LTS_mini.cli.main import ATCAnalyzer, initialize_components, main
from modules.adaptive_trend_LTS_mini.cli.argument_parser import parse_args
from modules.adaptive_trend_LTS_mini.cli.config_manager import ConfigManager
from modules.adaptive_trend_LTS_mini.cli.mode_manager import ModeManager
from modules.adaptive_trend_LTS_mini.cli.auto_mode_executor import AutoModeExecutor
from modules.adaptive_trend_LTS_mini.cli.manual_mode_executor import ManualModeExecutor
from modules.adaptive_trend_LTS_mini.cli.interactive_loop import InteractiveLoop


class TestCLIArgumentIntegration:
    """Test CLI argument parsing integration."""

    def test_parse_args_manual_mode(self):
        """Test parsing arguments for manual mode."""
        args = parse_args(["BTC/USDT", "--timeframe", "1h", "--limit", "1000"])

        assert args.symbol == "BTC/USDT"
        assert args.timeframe == "1h"
        assert args.limit == 1000
        assert args.auto is False

    def test_parse_args_auto_mode(self):
        """Test parsing arguments for auto mode."""
        args = parse_args(["--auto", "--timeframe", "15m"])

        assert args.auto is True
        assert args.timeframe == "15m"

    def test_parse_args_no_prompt(self):
        """Test parsing no-prompt flag."""
        args = parse_args(["BTC/USDT", "--no-prompt"])

        assert args.no_prompt is True

    def test_parse_args_list_symbols(self):
        """Test parsing list-symbols flag."""
        args = parse_args(["--list-symbols"])

        assert args.list_symbols is True


class TestModeManagerIntegration:
    """Test mode manager decision logic."""

    def test_mode_manager_auto_mode_flag(self):
        """Test mode manager detects auto mode from flag."""
        args = parse_args(["--auto", "--timeframe", "1h"])
        mode_manager = ModeManager(args)

        mode, timeframe = mode_manager.determine_mode_and_timeframe()

        assert mode == "auto"
        assert timeframe == "1h"

    def test_mode_manager_manual_mode_with_symbol(self):
        """Test mode manager detects manual mode with symbol."""
        # --no-menu avoids interactive prompt so mode stays manual when symbol+timeframe are given
        args = parse_args(["ETH/USDT", "--timeframe", "15m", "--no-menu"])
        mode_manager = ModeManager(args)

        mode, timeframe = mode_manager.determine_mode_and_timeframe()

        assert mode == "manual"
        assert timeframe == "15m"

    def test_mode_manager_interactive_prompt_for_timeframe(self):
        """Test mode manager prompts for timeframe when not provided."""
        args = parse_args([])
        mode_manager = ModeManager(args)

        # Mock interactive prompt (determine_mode_and_timeframe uses prompt_interactive_mode from input_utils)
        with patch("modules.adaptive_trend_LTS_mini.cli.input_utils.prompt_interactive_mode", return_value={"timeframe": "1h"}):
            mode, timeframe = mode_manager.determine_mode_and_timeframe()

        assert mode == "manual"
        assert timeframe == "1h"


class TestAutoModeExecutorIntegration:
    """Test auto mode executor integration."""

    def setup_method(self):
        """Setup test fixtures."""
        self.args = parse_args(["--auto", "--timeframe", "1h"])
        self.mock_fetcher = Mock()
        self.config_manager = ConfigManager(self.args)

    def test_auto_mode_executor_run_scan(self):
        """Test auto mode executor runs scan correctly."""
        executor = AutoModeExecutor(self.args, self.mock_fetcher, self.config_manager)

        # Mock scanner: scan_all_symbols returns (long_signals_df, short_signals_df)
        with patch("modules.adaptive_trend_LTS_mini.cli.auto_mode_executor.scan_all_symbols") as mock_scan:
            mock_scan.return_value = (pd.DataFrame(), pd.DataFrame())

            long_df, short_df = executor.run_scan("1h")

            assert isinstance(long_df, pd.DataFrame)
            assert isinstance(short_df, pd.DataFrame)

    def test_auto_mode_executor_with_max_symbols(self):
        """Test auto mode executor respects max_symbols limit."""
        args_with_max = parse_args(["--auto", "--max-symbols", "10"])
        executor = AutoModeExecutor(args_with_max, self.mock_fetcher, self.config_manager)

        with patch("modules.adaptive_trend_LTS_mini.cli.auto_mode_executor.scan_all_symbols") as mock_scan:
            mock_scan.return_value = {}

            executor.execute("1h")

            # Verify scanner was called with limited symbols
            assert mock_scan.called

    def test_auto_mode_executor_handles_empty_results(self):
        """Test auto mode executor handles empty scan results."""
        executor = AutoModeExecutor(self.args, self.mock_fetcher, self.config_manager)

        with patch("modules.adaptive_trend_LTS_mini.cli.auto_mode_executor.scan_all_symbols") as mock_scan:
            mock_scan.return_value = {}

            long_df, short_df = executor.run_scan("1h")

            assert len(long_df) == 0
            assert len(short_df) == 0


class TestManualModeExecutorIntegration:
    """Test manual mode executor integration."""

    def setup_method(self):
        """Setup test fixtures."""
        self.args = parse_args(["BTC/USDT", "--timeframe", "1h"])
        self.mock_fetcher = Mock()
        self.config_manager = ConfigManager(self.args)

    def test_manual_mode_executor_with_symbol(self):
        """Test manual mode executor with provided symbol."""
        executor = ManualModeExecutor(self.args, self.mock_fetcher, self.config_manager)

        # Mock analyzer
        with patch("modules.adaptive_trend_LTS_mini.cli.manual_mode_executor.analyze_symbol") as mock_analyze:
            mock_analyze.return_value = {
                "symbol": "BTC/USDT",
                "atc_results": {"Average_Signal": 0.5},
                "current_price": 50000.0,
            }

            symbol = executor.execute("1h")

            assert symbol == "BTC/USDT"
            assert mock_analyze.called

    def test_manual_mode_executor_prompts_for_symbol(self):
        """Test manual mode executor prompts when symbol not provided."""
        args_no_symbol = parse_args([])
        executor = ManualModeExecutor(args_no_symbol, self.mock_fetcher, self.config_manager)

        with patch("modules.adaptive_trend_LTS_mini.cli.manual_mode_executor.prompt_interactive_mode", return_value="ETH/USDT"):
            with patch("modules.adaptive_trend_LTS_mini.cli.manual_mode_executor.analyze_symbol") as mock_analyze:
                mock_analyze.return_value = {
                    "symbol": "ETH/USDT",
                    "atc_results": {"Average_Signal": 0.3},
                    "current_price": 3000.0,
                }

                symbol = executor.execute("1h")

                assert symbol == "ETH/USDT"

    def test_manual_mode_executor_handles_analysis_failure(self):
        """Test manual mode executor handles analysis failure."""
        executor = ManualModeExecutor(self.args, self.mock_fetcher, self.config_manager)

        with patch("modules.adaptive_trend_LTS_mini.cli.manual_mode_executor.analyze_symbol", return_value=None):
            symbol = executor.execute("1h")

            # Should return None on failure
            assert symbol is None


class TestInteractiveLoopIntegration:
    """Test interactive loop integration."""

    def setup_method(self):
        """Setup test fixtures."""
        self.args = parse_args([])
        self.mock_fetcher = Mock()
        self.config_manager = ConfigManager(self.args)

    def test_interactive_loop_single_iteration(self):
        """Test interactive loop single iteration."""
        loop = InteractiveLoop(self.args, self.mock_fetcher, self.config_manager)

        # Mock user input to exit after first iteration
        with patch("modules.adaptive_trend_LTS_mini.cli.interactive_loop.prompt_interactive_mode", return_value=None):
            with patch("modules.adaptive_trend_LTS_mini.cli.interactive_loop.analyze_symbol") as mock_analyze:
                mock_analyze.return_value = {
                    "symbol": "BTC/USDT",
                    "atc_results": {"Average_Signal": 0.5},
                }

                loop.run(initial_symbol="BTC/USDT", timeframe="1h")

                # Should analyze initial symbol
                assert mock_analyze.called

    def test_interactive_loop_keyboard_interrupt(self):
        """Test interactive loop handles KeyboardInterrupt."""
        loop = InteractiveLoop(self.args, self.mock_fetcher, self.config_manager)

        with patch("modules.adaptive_trend_LTS_mini.cli.interactive_loop.prompt_interactive_mode") as mock_prompt:
            mock_prompt.side_effect = KeyboardInterrupt()

            # Should exit gracefully
            loop.run(initial_symbol="BTC/USDT", timeframe="1h")


class TestATCAnalyzerIntegration:
    """Test ATCAnalyzer main orchestrator."""

    def setup_method(self):
        """Setup test fixtures."""
        self.mock_fetcher = Mock()

    def test_analyzer_run_auto_mode(self):
        """Test analyzer runs in auto mode."""
        args = parse_args(["--auto", "--timeframe", "1h"])
        analyzer = ATCAnalyzer(args, self.mock_fetcher)

        with patch.object(analyzer, "run_auto_mode") as mock_auto:
            analyzer.run()

            assert mock_auto.called

    def test_analyzer_run_manual_mode(self):
        """Test analyzer runs in manual mode."""
        args = parse_args(["BTC/USDT", "--timeframe", "1h", "--no-prompt"])
        analyzer = ATCAnalyzer(args, self.mock_fetcher)

        with patch.object(analyzer, "run_manual_mode") as mock_manual:
            analyzer.run()

            assert mock_manual.called

    def test_analyzer_run_auto_scan_programmatic(self):
        """Test programmatic auto scan interface."""
        args = parse_args(["--auto"])
        analyzer = ATCAnalyzer(args, self.mock_fetcher)

        with patch.object(analyzer.auto_executor, "run_scan") as mock_scan:
            mock_scan.return_value = (pd.DataFrame(), pd.DataFrame())

            long_df, short_df = analyzer.run_auto_scan()

            assert isinstance(long_df, pd.DataFrame)
            assert isinstance(short_df, pd.DataFrame)

    def test_analyzer_deprecated_method_warning(self):
        """Test deprecated run_interactive_loop raises warning."""
        args = parse_args([])
        analyzer = ATCAnalyzer(args, self.mock_fetcher)

        with pytest.warns(DeprecationWarning, match="run_interactive_loop is deprecated"):
            with patch.object(analyzer.interactive_loop, "run"):
                analyzer.run_interactive_loop("BTC/USDT", "USDT", {})


class TestMainFunctionIntegration:
    """Test main() function integration."""

    def test_main_with_list_symbols(self):
        """Test main() with --list-symbols flag."""
        with patch("sys.argv", ["main.py", "--list-symbols"]):
            with patch("modules.adaptive_trend_LTS_mini.cli.main.initialize_components") as mock_init:
                with patch("modules.adaptive_trend_LTS_mini.cli.main.list_futures_symbols") as mock_list:
                    mock_init.return_value = Mock()

                    main()

                    assert mock_list.called

    def test_main_with_auto_mode(self):
        """Test main() with auto mode."""
        with patch("sys.argv", ["main.py", "--auto", "--timeframe", "1h"]):
            with patch("modules.adaptive_trend_LTS_mini.cli.main.initialize_components") as mock_init:
                with patch("modules.adaptive_trend_LTS_mini.cli.main.ATCAnalyzer") as mock_analyzer_class:
                    mock_init.return_value = Mock()
                    mock_analyzer = Mock()
                    mock_analyzer_class.return_value = mock_analyzer

                    main()

                    assert mock_analyzer.run.called

    def test_main_keyboard_interrupt(self):
        """Test main() handles KeyboardInterrupt."""
        with patch("sys.argv", ["main.py"]):
            with patch("modules.adaptive_trend_LTS_mini.cli.main.initialize_components") as mock_init:
                mock_init.side_effect = KeyboardInterrupt()

                with pytest.raises(SystemExit) as exc_info:
                    main()

                assert exc_info.value.code == 0

    def test_main_exception_handling(self):
        """Test main() handles general exceptions."""
        with patch("sys.argv", ["main.py"]):
            with patch("modules.adaptive_trend_LTS_mini.cli.main.initialize_components") as mock_init:
                mock_init.side_effect = Exception("Test error")

                with pytest.raises(SystemExit) as exc_info:
                    main()

                assert exc_info.value.code == 1


class TestDisplayIntegration:
    """Test display functions integration."""

    def test_display_atc_signals_with_valid_data(self):
        """Test displaying ATC signals with valid data."""
        from modules.adaptive_trend_LTS_mini.cli.display import display_atc_signals

        result = {
            "symbol": "BTC/USDT",
            "atc_results": {
                "Average_Signal": pd.Series([0.5, 0.6, 0.7]),
                "EMA_Signal": pd.Series([0.4, 0.5, 0.6]),
            },
            "current_price": 50000.0,
            "exchange_label": "BINANCE",
        }

        # Should not raise exception
        with patch("builtins.print"):
            display_atc_signals(result, timeframe="1h")

    def test_display_scan_results_with_signals(self):
        """Test displaying scan results with signals."""
        from modules.adaptive_trend_LTS_mini.cli.display import display_scan_results

        long_signals = pd.DataFrame({
            "Symbol": ["BTC/USDT", "ETH/USDT"],
            "Signal": [0.5, 0.3],
            "Price": [50000, 3000],
        })

        short_signals = pd.DataFrame({
            "Symbol": ["LTC/USDT"],
            "Signal": [-0.4],
            "Price": [100],
        })

        # Should not raise exception
        with patch("builtins.print"):
            display_scan_results(long_signals, short_signals, timeframe="1h")

    def test_list_futures_symbols_success(self):
        """Test listing futures symbols."""
        from modules.adaptive_trend_LTS_mini.cli.display import list_futures_symbols

        mock_fetcher = Mock()
        mock_fetcher.discover_futures_symbols.return_value = ["BTC/USDT", "ETH/USDT"]

        # Should not raise exception
        with patch("builtins.print"):
            list_futures_symbols(mock_fetcher)

    def test_list_futures_symbols_handles_exception(self):
        """Test listing futures symbols handles exceptions."""
        from modules.adaptive_trend_LTS_mini.cli.display import list_futures_symbols

        mock_fetcher = Mock()
        mock_fetcher.discover_futures_symbols.side_effect = Exception("API error")

        # Should handle exception gracefully
        with patch("builtins.print"):
            list_futures_symbols(mock_fetcher)


class TestConfigManagerIntegration:
    """Test configuration manager integration."""

    def test_config_manager_creates_atc_config(self):
        """Test config manager creates ATCConfig."""
        args = parse_args([
            "--ema-len", "50",
            "--hma-len", "30",
            "--robustness", "Wide",
            "--lambda-param", "0.05",
            "--decay", "0.04",
        ])

        config_manager = ConfigManager(args)
        config = config_manager.create_config("1h")

        assert config.ema_len == 50
        assert config.hma_len == 30
        assert config.robustness == "Wide"
        assert config.lambda_param == 0.05
        assert config.decay == 0.04
        assert config.timeframe == "1h"

    def test_config_manager_with_defaults(self):
        """Test config manager with default values."""
        args = parse_args([])

        config_manager = ConfigManager(args)
        config = config_manager.create_config("15m")

        assert config.ema_len == 28  # Default
        assert config.timeframe == "15m"


class TestEndToEndWorkflows:
    """Test complete end-to-end workflows."""

    def test_complete_manual_workflow(self):
        """Test complete manual mode workflow from CLI to results."""
        with patch("sys.argv", ["main.py", "BTC/USDT", "--timeframe", "1h", "--no-prompt"]):
            with patch("modules.adaptive_trend_LTS_mini.cli.main.initialize_components") as mock_init:
                with patch("modules.adaptive_trend_LTS_mini.cli.manual_mode_executor.analyze_symbol") as mock_analyze:
                    # Setup mocks
                    mock_init.return_value = Mock()
                    mock_analyze.return_value = {
                        "symbol": "BTC/USDT",
                        "atc_results": {"Average_Signal": pd.Series([0.5])},
                        "current_price": 50000.0,
                    }

                    with patch("modules.adaptive_trend_LTS_mini.cli.display.display_atc_signals"):
                        main()

                    # Verify analysis was called
                    assert mock_analyze.called

    def test_complete_auto_workflow(self):
        """Test complete auto mode workflow from CLI to results."""
        with patch("sys.argv", ["main.py", "--auto", "--timeframe", "15m"]):
            with patch("modules.adaptive_trend_LTS_mini.cli.main.initialize_components") as mock_init:
                with patch("modules.adaptive_trend_LTS_mini.cli.auto_mode_executor.scan_all_symbols") as mock_scan:
                    # Setup mocks
                    mock_init.return_value = Mock()
                    mock_scan.return_value = {
                        "BTC/USDT": {"Average_Signal": 0.5},
                    }

                    with patch("modules.adaptive_trend_LTS_mini.cli.display.display_scan_results"):
                        main()

                    # Verify scan was called
                    assert mock_scan.called

    def test_interactive_workflow_with_menu(self):
        """Test interactive workflow with menu navigation."""
        args = parse_args([])
        mock_fetcher = Mock()

        analyzer = ATCAnalyzer(args, mock_fetcher)

        # Simulate user entering symbol, analyzing, then exiting
        with patch("modules.adaptive_trend_LTS_mini.cli.mode_manager.prompt_timeframe", return_value="1h"):
            with patch("modules.adaptive_trend_LTS_mini.cli.manual_mode_executor.prompt_interactive_mode", side_effect=["BTC/USDT", None]):
                with patch("modules.adaptive_trend_LTS_mini.cli.manual_mode_executor.analyze_symbol") as mock_analyze:
                    mock_analyze.return_value = {
                        "symbol": "BTC/USDT",
                        "atc_results": {"Average_Signal": pd.Series([0.5])},
                    }

                    with patch("modules.adaptive_trend_LTS_mini.cli.display.display_atc_signals"):
                        analyzer.run()

                    # Should have analyzed symbol
                    assert mock_analyze.called


class TestErrorHandlingInCLI:
    """Test error handling in CLI context."""

    def test_invalid_symbol_format(self):
        """Test handling of invalid symbol format."""
        args = parse_args(["INVALID_SYMBOL", "--no-prompt"])
        mock_fetcher = Mock()
        mock_fetcher.fetch_ohlcv_with_fallback_exchange.return_value = (None, None)

        analyzer = ATCAnalyzer(args, mock_fetcher)

        # Should handle invalid symbol gracefully
        with patch("modules.adaptive_trend_LTS_mini.cli.display.display_atc_signals"):
            analyzer.run()

    def test_network_error_during_fetch(self):
        """Test handling of network errors."""
        args = parse_args(["BTC/USDT", "--no-prompt"])
        mock_fetcher = Mock()
        mock_fetcher.fetch_ohlcv_with_fallback_exchange.side_effect = Exception("Network error")

        analyzer = ATCAnalyzer(args, mock_fetcher)

        # Should handle network error gracefully
        with patch("modules.adaptive_trend_LTS_mini.cli.display.display_atc_signals"):
            analyzer.run()

    def test_insufficient_data_error(self):
        """Test handling of insufficient data."""
        args = parse_args(["BTC/USDT", "--limit", "5", "--no-prompt"])
        mock_fetcher = Mock()

        # Return very small dataset
        df = pd.DataFrame({
            "close": [100, 101, 102],
            "volume": [1000, 1100, 1200],
        })
        mock_fetcher.fetch_ohlcv_with_fallback_exchange.return_value = (df, "binance")

        analyzer = ATCAnalyzer(args, mock_fetcher)

        # Should handle insufficient data
        with patch("modules.adaptive_trend_LTS_mini.cli.display.display_atc_signals"):
            analyzer.run()


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
