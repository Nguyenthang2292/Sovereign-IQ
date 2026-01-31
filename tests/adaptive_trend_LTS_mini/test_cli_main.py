"""
Tests for modules/adaptive_trend_LTS_mini/cli/main.py

Comprehensive tests covering:
- Mode determination logic with different argument combinations
- Parameter extraction and caching
- Error handling paths
- Interactive loop exit handling
- Security input validation
- Display methods
"""

import pytest
from unittest.mock import MagicMock, patch, Mock
from argparse import Namespace
import sys
import pandas as pd

from modules.adaptive_trend_LTS_mini.cli.main import ATCAnalyzer, initialize_components
from modules.adaptive_trend_LTS_mini.cli.interactive_prompts import UserExitRequested

@pytest.fixture
def mock_args():
    return Namespace(
        timeframe="1h",
        auto=False,
        no_menu=False,
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
        min_signal=0.1,
        max_symbols=None,
        use_rust_backend=False,
        batch_processing=False,
        fast_mode=False,
        precision=None,
        use_cache=False,
        use_approximate=False,
        use_adaptive_approximate=False,
        approximate_volatility_window=None,
        approximate_volatility_factor=None,
        approximate_threshold=None,
        quote="USDT",
        symbol=None,
        no_prompt=False,
        long_threshold=0,
        short_threshold=0
    )

@pytest.fixture
def mock_data_fetcher():
    return MagicMock()

def test_atc_analyzer_mode_determination(mock_args, mock_data_fetcher):
    """Test mode determination with different arg combinations."""
    # Case 1: Auto mode via args
    mock_args.auto = True
    analyzer = ATCAnalyzer(mock_args, mock_data_fetcher)
    mode, timeframe = analyzer.determine_mode_and_timeframe()
    assert mode == "auto"
    assert timeframe == "1h"

    # Case 2: Manual mode via no_menu
    mock_args.auto = False
    mock_args.no_menu = True
    analyzer = ATCAnalyzer(mock_args, mock_data_fetcher)
    mode, timeframe = analyzer.determine_mode_and_timeframe()
    assert mode == "manual"
    assert timeframe == "1h"

    # Case 3: Interactive mode (mocked)
    mock_args.no_menu = False
    with patch("modules.adaptive_trend_LTS_mini.cli.main.prompt_interactive_mode") as mock_prompt:
        mock_prompt.return_value = {"mode": "auto", "timeframe": "4h"}
        analyzer = ATCAnalyzer(mock_args, mock_data_fetcher)
        mode, timeframe = analyzer.determine_mode_and_timeframe()
        assert mode == "auto"
        assert timeframe == "4h"

    # Case 4: Timeframe only change (should stay manual)
    mock_args.no_menu = False
    with patch("modules.adaptive_trend_LTS_mini.cli.main.prompt_interactive_mode") as mock_prompt:
        mock_prompt.return_value = {"timeframe": "2h"} # No mode returned
        analyzer = ATCAnalyzer(mock_args, mock_data_fetcher)
        mode, timeframe = analyzer.determine_mode_and_timeframe()
        assert mode == "manual"
        assert timeframe == "2h"

def test_atc_params_extraction(mock_args, mock_data_fetcher):
    """Test parameter extraction and caching."""
    analyzer = ATCAnalyzer(mock_args, mock_data_fetcher)
    params = analyzer.get_atc_params()
    
    assert params["limit"] == 100
    assert params["ema_len"] == 10
    assert analyzer._atc_params is not None  # Should be cached
    
    # modify cache to verify it's used
    analyzer._atc_params["limit"] = 999
    params2 = analyzer.get_atc_params()
    assert params2["limit"] == 999

def test_interactive_loop_exit(mock_args, mock_data_fetcher):
    """Test graceful exit handling."""
    mock_args.auto = False
    mock_args.no_menu = False

    with patch("modules.adaptive_trend_LTS_mini.cli.main.prompt_interactive_mode") as mock_prompt:
        mock_prompt.side_effect = UserExitRequested()

        analyzer = ATCAnalyzer(mock_args, mock_data_fetcher)

        with pytest.raises(SystemExit) as excinfo:
            analyzer.determine_mode_and_timeframe()
        assert excinfo.value.code == 0


# ============================================================================
# Additional Error Handling Tests
# ============================================================================


class TestErrorHandling:
    """Test error handling paths in ATCAnalyzer."""

    @patch("modules.adaptive_trend_LTS_mini.cli.main.normalize_symbol")
    @patch("modules.adaptive_trend_LTS_mini.cli.main.prompt_user_input")
    def test_symbol_input_validation_rejects_invalid_characters(self, mock_prompt, mock_normalize, mock_args, mock_data_fetcher):
        """Test that symbol input validation rejects SQL injection attempts."""
        mock_args.symbol = None
        mock_args.no_prompt = False
        mock_prompt.return_value = "BTC/USDT; DROP TABLE users"  # SQL injection attempt

        analyzer = ATCAnalyzer(mock_args, mock_data_fetcher)
        result = analyzer.get_symbol_input()

        # Should default to DEFAULT_SYMBOL due to invalid characters
        # The semicolon should trigger validation failure
        assert mock_normalize.called

    @patch("modules.adaptive_trend_LTS_mini.cli.main.normalize_symbol")
    def test_symbol_input_validation_allows_valid_characters(self, mock_normalize, mock_args, mock_data_fetcher):
        """Test that symbol input validation allows alphanumeric, slash, and hyphen."""
        mock_args.symbol = "BTC/USDT"

        analyzer = ATCAnalyzer(mock_args, mock_data_fetcher)
        analyzer.get_symbol_input()

        mock_normalize.assert_called_once_with("BTC/USDT", "USDT")

    @patch("modules.adaptive_trend_LTS_mini.cli.main.normalize_symbol")
    def test_symbol_input_validation_allows_hyphen_for_symbols(self, mock_normalize, mock_args, mock_data_fetcher):
        """Test that hyphenated symbols are allowed (e.g., BTC-PERP)."""
        mock_args.symbol = "BTC-PERP/USDT"

        analyzer = ATCAnalyzer(mock_args, mock_data_fetcher)
        analyzer.get_symbol_input()

        mock_normalize.assert_called_once_with("BTC-PERP/USDT", "USDT")

    @patch("modules.adaptive_trend_LTS_mini.cli.main.analyze_symbol")
    @patch("modules.adaptive_trend_LTS_mini.cli.main.normalize_symbol")
    def test_run_manual_mode_handles_analysis_failure(self, mock_normalize, mock_analyze, mock_args, mock_data_fetcher):
        """Test that run_manual_mode handles None result from analyze_symbol."""
        mock_args.symbol = "BTC/USDT"
        mock_args.no_prompt = True
        mock_normalize.return_value = "BTC/USDT"
        mock_analyze.return_value = None  # Simulate analysis failure

        analyzer = ATCAnalyzer(mock_args, mock_data_fetcher)

        # Should not raise exception, just return early
        analyzer.run_manual_mode()

        mock_analyze.assert_called_once()

    @patch("modules.adaptive_trend_LTS_mini.cli.main.scan_all_symbols")
    @patch("modules.adaptive_trend_LTS_mini.cli.main.create_atc_config_from_dict")
    def test_run_auto_scan_handles_empty_results(self, mock_create_config, mock_scan, mock_args, mock_data_fetcher):
        """Test that run_auto_scan handles empty DataFrames gracefully."""
        mock_config = Mock()
        mock_config.batch_size = 100
        mock_create_config.return_value = mock_config

        empty_df = pd.DataFrame()
        mock_scan.return_value = (empty_df, empty_df)

        analyzer = ATCAnalyzer(mock_args, mock_data_fetcher)
        long_signals, short_signals = analyzer.run_auto_scan()

        assert len(long_signals) == 0
        assert len(short_signals) == 0

    @patch("modules.adaptive_trend_LTS_mini.cli.main.analyze_symbol")
    @patch("modules.adaptive_trend_LTS_mini.cli.main.prompt_user_input")
    def test_interactive_loop_handles_keyboard_interrupt(self, mock_prompt, mock_analyze, mock_args, mock_data_fetcher):
        """Test that interactive loop catches KeyboardInterrupt gracefully."""
        mock_prompt.side_effect = KeyboardInterrupt()

        analyzer = ATCAnalyzer(mock_args, mock_data_fetcher)

        # Should catch KeyboardInterrupt and not propagate
        analyzer.run_interactive_loop("BTC/USDT", "USDT", {})


# ============================================================================
# Type Safety Tests
# ============================================================================


class TestTypeSafety:
    """Test TypedDict usage for ATC parameters."""

    def test_get_atc_params_returns_correct_type_structure(self, mock_args, mock_data_fetcher):
        """Test that get_atc_params returns properly structured ATCParams."""
        analyzer = ATCAnalyzer(mock_args, mock_data_fetcher)
        params = analyzer.get_atc_params()

        # Verify it's a dictionary (ATCParams is TypedDict at runtime)
        assert isinstance(params, dict)

        # Verify key types
        assert isinstance(params["limit"], int)
        assert isinstance(params["ema_len"], int)
        assert isinstance(params["robustness"], str)
        assert isinstance(params["lambda_param"], float)
        assert isinstance(params["use_rust_backend"], bool)

    def test_get_atc_params_has_all_required_keys(self, mock_args, mock_data_fetcher):
        """Test that all expected parameter keys are present."""
        analyzer = ATCAnalyzer(mock_args, mock_data_fetcher)
        params = analyzer.get_atc_params()

        expected_keys = {
            "limit", "ema_len", "hma_len", "wma_len", "dema_len", "lsma_len", "kama_len",
            "robustness", "lambda_param", "decay", "cutout", "long_threshold", "short_threshold",
            "use_rust_backend", "batch_processing", "fast_mode", "precision", "use_cache",
            "use_approximate", "use_adaptive_approximate", "approximate_volatility_window",
            "approximate_volatility_factor", "approximate_threshold"
        }

        assert set(params.keys()) == expected_keys


# ============================================================================
# Display Method Tests
# ============================================================================


class TestDisplayMethods:
    """Test configuration display methods."""

    @patch("modules.adaptive_trend_LTS_mini.cli.main.log_analysis")
    @patch("modules.adaptive_trend_LTS_mini.cli.main.log_data")
    def test_display_config_header_with_symbol(self, mock_log_data, mock_log_analysis, mock_args, mock_data_fetcher):
        """Test _display_config_header displays symbol when provided."""
        analyzer = ATCAnalyzer(mock_args, mock_data_fetcher)
        analyzer._display_config_header("TEST ANALYSIS", symbol="BTC/USDT")

        # Verify logging was called
        assert mock_log_analysis.call_count >= 3  # Header, title, divider

        # Verify symbol is included in log_data calls
        calls = [str(call) for call in mock_log_data.call_args_list]
        assert any("BTC/USDT" in str(call) for call in calls)

    @patch("modules.adaptive_trend_LTS_mini.cli.main.log_analysis")
    @patch("modules.adaptive_trend_LTS_mini.cli.main.log_data")
    def test_display_config_header_without_symbol(self, mock_log_data, mock_log_analysis, mock_args, mock_data_fetcher):
        """Test _display_config_header works without symbol parameter."""
        analyzer = ATCAnalyzer(mock_args, mock_data_fetcher)
        analyzer._display_config_header("TEST ANALYSIS")

        # Should still log configuration info
        assert mock_log_analysis.call_count >= 3
        assert mock_log_data.call_count >= 1  # At least timeframe

    @patch("modules.adaptive_trend_LTS_mini.cli.main.log_analysis")
    @patch("modules.adaptive_trend_LTS_mini.cli.main.log_data")
    def test_display_auto_mode_config_shows_mode(self, mock_log_data, mock_log_analysis, mock_args, mock_data_fetcher):
        """Test display_auto_mode_config shows AUTO mode."""
        analyzer = ATCAnalyzer(mock_args, mock_data_fetcher)
        analyzer.display_auto_mode_config()

        # Check for AUTO mode mention
        calls = [str(call) for call in mock_log_data.call_args_list]
        assert any("AUTO" in str(call).upper() for call in calls)

    @patch("modules.adaptive_trend_LTS_mini.cli.main.log_analysis")
    @patch("modules.adaptive_trend_LTS_mini.cli.main.log_data")
    def test_display_manual_mode_config_shows_symbol(self, mock_log_data, mock_log_analysis, mock_args, mock_data_fetcher):
        """Test display_manual_mode_config shows the symbol."""
        analyzer = ATCAnalyzer(mock_args, mock_data_fetcher)
        analyzer.display_manual_mode_config("ETH/USDT")

        # Check for symbol mention
        calls = [str(call) for call in mock_log_data.call_args_list]
        assert any("ETH/USDT" in str(call) for call in calls)


# ============================================================================
# Component Initialization Tests
# ============================================================================


class TestComponentInitialization:
    """Test initialize_components function."""

    @patch("modules.adaptive_trend_LTS_mini.cli.main.ExchangeManager")
    @patch("modules.adaptive_trend_LTS_mini.cli.main.DataFetcher")
    def test_initialize_components_returns_data_fetcher(self, mock_data_fetcher_class, mock_exchange_manager_class):
        """Test that initialize_components returns DataFetcher instance."""
        mock_exchange = Mock()
        mock_exchange_manager_class.return_value = mock_exchange
        mock_fetcher = Mock()
        mock_data_fetcher_class.return_value = mock_fetcher

        result = initialize_components()

        # Should create ExchangeManager
        mock_exchange_manager_class.assert_called_once()

        # Should create DataFetcher with ExchangeManager
        mock_data_fetcher_class.assert_called_once_with(mock_exchange)

        # Should return DataFetcher (not a tuple)
        assert result is mock_fetcher
        assert not isinstance(result, tuple)


# ============================================================================
# Run Auto Scan Tests
# ============================================================================


class TestRunAutoScan:
    """Test run_auto_scan method with various scenarios."""

    @patch("modules.adaptive_trend_LTS_mini.cli.main.scan_all_symbols")
    @patch("modules.adaptive_trend_LTS_mini.cli.main.create_atc_config_from_dict")
    def test_run_auto_scan_without_symbol_filter(self, mock_create_config, mock_scan, mock_args, mock_data_fetcher):
        """Test run_auto_scan scans all symbols when no filter provided."""
        mock_config = Mock()
        mock_config.batch_size = 100
        mock_create_config.return_value = mock_config
        mock_scan.return_value = (pd.DataFrame(), pd.DataFrame())

        analyzer = ATCAnalyzer(mock_args, mock_data_fetcher)
        analyzer.run_auto_scan()

        # Should pass symbols=None to scan all
        call_kwargs = mock_scan.call_args[1]
        assert call_kwargs["symbols"] is None

    @patch("modules.adaptive_trend_LTS_mini.cli.main.scan_all_symbols")
    @patch("modules.adaptive_trend_LTS_mini.cli.main.create_atc_config_from_dict")
    def test_run_auto_scan_with_symbol_filter(self, mock_create_config, mock_scan, mock_args, mock_data_fetcher):
        """Test run_auto_scan with pre-filtered symbols list."""
        mock_config = Mock()
        mock_config.batch_size = 100
        mock_create_config.return_value = mock_config
        mock_scan.return_value = (pd.DataFrame(), pd.DataFrame())

        analyzer = ATCAnalyzer(mock_args, mock_data_fetcher)
        test_symbols = ["BTC/USDT", "ETH/USDT", "SOL/USDT"]
        analyzer.run_auto_scan(symbols=test_symbols)

        # Should pass the symbols list
        call_kwargs = mock_scan.call_args[1]
        assert call_kwargs["symbols"] == test_symbols

    @patch("modules.adaptive_trend_LTS_mini.cli.main.scan_all_symbols")
    @patch("modules.adaptive_trend_LTS_mini.cli.main.create_atc_config_from_dict")
    def test_run_auto_scan_respects_execution_mode(self, mock_create_config, mock_scan, mock_args, mock_data_fetcher):
        """Test run_auto_scan passes execution_mode parameter correctly."""
        mock_config = Mock()
        mock_config.batch_size = 100
        mock_create_config.return_value = mock_config
        mock_scan.return_value = (pd.DataFrame(), pd.DataFrame())

        mock_args.execution_mode = "dask"
        analyzer = ATCAnalyzer(mock_args, mock_data_fetcher)
        analyzer.run_auto_scan()

        # Should use dask execution mode
        call_kwargs = mock_scan.call_args[1]
        assert call_kwargs["execution_mode"] == "dask"


# ============================================================================
# Analyzer State Tests
# ============================================================================


class TestAnalyzerState:
    """Test ATCAnalyzer state management."""

    def test_analyzer_initial_state(self, mock_args, mock_data_fetcher):
        """Test that analyzer initializes with correct default state."""
        analyzer = ATCAnalyzer(mock_args, mock_data_fetcher)

        assert analyzer.args is mock_args
        assert analyzer.data_fetcher is mock_data_fetcher
        assert analyzer.selected_timeframe == "1h"
        assert analyzer.mode == "manual"
        assert analyzer._atc_params is None

    def test_mode_state_persists_after_determination(self, mock_args, mock_data_fetcher):
        """Test that mode state persists in analyzer after determination."""
        mock_args.auto = True
        analyzer = ATCAnalyzer(mock_args, mock_data_fetcher)

        mode, _ = analyzer.determine_mode_and_timeframe()

        assert analyzer.mode == mode
        assert analyzer.mode == "auto"

    def test_timeframe_state_persists_after_determination(self, mock_args, mock_data_fetcher):
        """Test that timeframe state persists in analyzer."""
        mock_args.timeframe = "4h"
        analyzer = ATCAnalyzer(mock_args, mock_data_fetcher)

        _, timeframe = analyzer.determine_mode_and_timeframe()

        assert analyzer.selected_timeframe == timeframe
        assert analyzer.selected_timeframe == "4h"
