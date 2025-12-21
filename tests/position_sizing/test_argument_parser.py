"""
Tests for modules/position_sizing/cli/argument_parser.py interactive_config_menu.

Tests cover:
- Main menu display
- Navigation with back option
- Configuration updates
- Review and confirm
"""

import pytest
import sys
from pathlib import Path
from unittest.mock import patch

# Add parent directory to path
ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from modules.position_sizing.cli.argument_parser import (
    interactive_config_menu,
    _display_main_menu_ps,
    _format_current_value_ps,
    _prompt_with_back_ps,
    _configure_symbol_source,
    _configure_account_balance,
    _configure_backtest_settings,
    _configure_position_size,
    _review_and_confirm_ps,
    parse_symbols_string,
    normalize_symbol,
)


def _aggregate_print_output(mock_print):
    """
    Helper function to aggregate printed mock output into a single string.
    
    Args:
        mock_print: Mock object for the print function
        
    Returns:
        str: Combined string of all print call arguments
    """
    result_parts = []
    for call in mock_print.call_args_list:
        if call.args:
            # Convert each argument to string and join with space (preserving print's default separator)
            call_output = ' '.join(str(arg) for arg in call.args)
            result_parts.append(call_output)
    return ' '.join(result_parts)


class TestFormatCurrentValuePS:
    """Test _format_current_value_ps helper function."""
    
    def test_format_none(self):
        """Test formatting None value."""
        result = _format_current_value_ps(None)
        assert result == "not set"
    
    def test_format_bool_true(self):
        """Test formatting True boolean."""
        result = _format_current_value_ps(True)
        assert result == "enabled"
    
    def test_format_bool_false(self):
        """Test formatting False boolean."""
        result = _format_current_value_ps(False)
        assert result == "disabled"
    
    def test_format_string(self):
        """Test formatting string value."""
        result = _format_current_value_ps("1h")
        assert result == "1h"
    
    def test_format_empty_string(self):
        """Test formatting empty string."""
        result = _format_current_value_ps("")
        assert result == "not set"


class TestPromptWithBackPS:
    """Test _prompt_with_back_ps helper function."""
    
    @patch('modules.position_sizing.cli.argument_parser.prompt_user_input_with_backspace')
    def test_prompt_with_back_returns_continue(self, mock_prompt):
        """Test that normal input returns continue action."""
        mock_prompt.return_value = ("y", False)  # (user_input, is_back)
        value, action = _prompt_with_back_ps("Test prompt: ", default="n")
        assert value == "y"
        assert action == 'continue'
    
    @patch('modules.position_sizing.cli.argument_parser.prompt_user_input_with_backspace')
    def test_prompt_with_back_returns_main(self, mock_prompt):
        """Test that backspace (is_back=True) returns main action."""
        mock_prompt.return_value = (None, True)  # (user_input, is_back)
        value, action = _prompt_with_back_ps("Test prompt: ", default="n")
        assert value is None
        assert action == 'main'
    
    @patch('modules.position_sizing.cli.argument_parser.prompt_user_input')
    def test_prompt_with_back_no_back_option(self, mock_prompt):
        """Test prompt without back option."""
        mock_prompt.return_value = "y"
        value, action = _prompt_with_back_ps("Test prompt: ", default="n", allow_back=False)
        assert value == "y"
        assert action == 'continue'


class TestDisplayMainMenuPS:
    """Test _display_main_menu_ps function."""
    
    def test_display_main_menu_with_values(self):
        """Test main menu display with configuration values."""
        config = {
            'source': 'hybrid',
            'fetch_balance': True,
            'timeframe': '1h',
            'auto_timeframe': False,
            'lookback_days': 30,
            'max_position_size': 0.1,
        }
        
        with patch('builtins.print') as mock_print:
            _display_main_menu_ps(config)
            
            # Combine all print call arguments into a single string
            output = _aggregate_print_output(mock_print)
            
            # Assert menu option labels are present
            assert 'Symbol Source' in output
            assert 'Account Balance' in output
            assert 'Backtest Settings' in output
            assert 'Position Size Constraints' in output
            
            # Assert config values are displayed
            assert 'hybrid' in output
            assert 'fetch' in output.lower() and 'binance' in output.lower()
            assert '1h' in output
            assert '30' in output
            assert '0.1' in output
            assert 'disabled' in output  # auto_timeframe=False
    
    def test_display_main_menu_empty_config(self):
        """Test main menu display with empty config."""
        config = {}
        
        with patch('builtins.print') as mock_print:
            _display_main_menu_ps(config)
            
            # Combine all print call arguments into a single string
            output = _aggregate_print_output(mock_print)
            
            # Assert menu option labels are present
            assert 'Symbol Source' in output
            assert 'Account Balance' in output
            assert 'Backtest Settings' in output
            assert 'Position Size Constraints' in output
            
            # Assert default placeholders or values are shown for empty config
            assert 'not set' in output  # Should appear for source and balance
            # Check for default values being displayed (both timeframe and lookback are shown in Backtest Settings)
            assert '1h' in output  # Default timeframe
            assert '15' in output  # Default lookback days
            assert '1.0' in output  # Default max_position_size


class TestNormalizeSymbol:
    """Test normalize_symbol function."""
    
    def test_normalize_symbol_with_slash(self):
        """Test normalizing symbol with slash."""
        result = normalize_symbol("BTC/USDT")
        assert result == "BTC/USDT"
    
    def test_normalize_symbol_without_slash(self):
        """Test normalizing symbol without slash."""
        result = normalize_symbol("BTC")
        assert result == "BTC/USDT"
    
    def test_normalize_symbol_lowercase(self):
        """Test normalizing lowercase symbol."""
        result = normalize_symbol("btc")
        assert result == "BTC/USDT"
    
    def test_normalize_symbol_ends_with_usdt(self):
        """Test normalizing symbol ending with USDT."""
        result = normalize_symbol("BTCUSDT")
        assert result == "BTC/USDT"
    
    def test_normalize_symbol_empty(self):
        """Test normalizing empty symbol."""
        result = normalize_symbol("")
        assert result == ""


class TestParseSymbolsString:
    """Test parse_symbols_string function."""
    
    def test_parse_single_symbol(self):
        """Test parsing single symbol."""
        result = parse_symbols_string("BTC")
        assert len(result) == 1
        assert result[0] == "BTC/USDT"
    
    def test_parse_multiple_symbols(self):
        """Test parsing multiple symbols."""
        result = parse_symbols_string("BTC,ETH")
        assert len(result) == 2
        assert result[0] == "BTC/USDT"
        assert result[1] == "ETH/USDT"
    
    def test_parse_symbols_with_slash(self):
        """Test parsing symbols with slash."""
        result = parse_symbols_string("BTC/USDT,ETH/USDT")
        assert len(result) == 2
        assert result[0] == "BTC/USDT"
        assert result[1] == "ETH/USDT"
    
    def test_parse_empty_string(self):
        """Test parsing empty string."""
        result = parse_symbols_string("")
        assert len(result) == 0


class TestConfigureFunctions:
    """Test individual configure functions."""
    
    @patch('modules.position_sizing.cli.argument_parser._prompt_with_back_ps')
    def test_configure_symbol_source_hybrid(self, mock_prompt):
        """Test symbol source configuration for hybrid."""
        config = {}
        mock_prompt.return_value = ("a", 'continue')
        
        result = _configure_symbol_source(config)
        
        assert result == 'main'
        assert config['source'] == 'hybrid'
    
    @patch('modules.position_sizing.cli.argument_parser._prompt_with_back_ps')
    def test_configure_symbol_source_voting(self, mock_prompt):
        """Test symbol source configuration for voting."""
        config = {}
        mock_prompt.return_value = ("b", 'continue')
        
        result = _configure_symbol_source(config)
        
        assert result == 'main'
        assert config['source'] == 'voting'
    
    @patch('modules.position_sizing.cli.argument_parser._prompt_with_back_ps')
    def test_configure_symbol_source_file(self, mock_prompt):
        """Test symbol source configuration for file."""
        config = {}
        mock_prompt.side_effect = [("c", 'continue'), ("test.csv", 'continue')]
        
        result = _configure_symbol_source(config)
        
        assert result == 'main'
        assert config['symbols_file'] == "test.csv"
    
    @patch('modules.position_sizing.cli.argument_parser._prompt_with_back_ps')
    def test_configure_symbol_source_manual(self, mock_prompt):
        """Test symbol source configuration for manual input."""
        config = {}
        mock_prompt.side_effect = [("d", 'continue'), ("BTC,ETH", 'continue')]
        
        result = _configure_symbol_source(config)
        
        assert result == 'main'
        assert 'symbols' in config
    
    @patch('modules.position_sizing.cli.argument_parser._prompt_with_back_ps')
    def test_configure_account_balance_fetch(self, mock_prompt):
        """Test account balance configuration with fetch option."""
        config = {}
        mock_prompt.return_value = ("a", 'continue')
        
        result = _configure_account_balance(config)
        
        assert result == 'main'
        assert config['fetch_balance'] is True
        assert config['account_balance'] is None
    
    @patch('modules.position_sizing.cli.argument_parser._prompt_with_back_ps')
    def test_configure_account_balance_manual(self, mock_prompt):
        """Test account balance configuration with manual input."""
        config = {}
        mock_prompt.side_effect = [("b", 'continue'), ("10000", 'continue')]
        
        result = _configure_account_balance(config)
        
        assert result == 'main'
        assert config['account_balance'] == 10000.0
        assert config['fetch_balance'] is False
    
    @patch('modules.position_sizing.cli.argument_parser._prompt_with_back_ps')
    def test_configure_backtest_settings(self, mock_prompt):
        """Test backtest settings configuration."""
        config = {}
        mock_prompt.side_effect = [
            ("b", 'continue'),  # Use specified timeframe
            ("1h", 'continue'),  # Timeframe
            ("30", 'continue'),  # Lookback days
        ]
        
        result = _configure_backtest_settings(config)
        
        assert result == 'main'
        assert config['auto_timeframe'] is False
        assert config['timeframe'] == "1h"
        assert config['lookback_days'] == 30
    
    @patch('modules.position_sizing.cli.argument_parser._prompt_with_back_ps')
    def test_configure_position_size(self, mock_prompt):
        """Test position size configuration."""
        config = {}
        mock_prompt.return_value = ("0.2", 'continue')
        
        result = _configure_position_size(config)
        
        assert result == 'main'
        assert config['max_position_size'] == 0.2


class TestInteractiveConfigMenu:
    """Test interactive_config_menu main function."""
    
    @patch('modules.position_sizing.cli.argument_parser._display_main_menu_ps')
    @patch('modules.position_sizing.cli.argument_parser.prompt_user_input')
    @patch('modules.position_sizing.cli.argument_parser._review_and_confirm_ps')
    def test_interactive_config_menu_review_and_confirm(self, mock_review, mock_prompt, mock_display):
        """Test menu flow with review and confirm."""
        mock_prompt.return_value = "5"  # Review and Confirm
        mock_review.return_value = 'done'
        
        config = interactive_config_menu()
        
        assert config is not None
        assert isinstance(config, dict)
    
    @patch('modules.position_sizing.cli.argument_parser._display_main_menu_ps')
    @patch('modules.position_sizing.cli.argument_parser.prompt_user_input')
    @patch('modules.position_sizing.cli.argument_parser._configure_symbol_source')
    def test_interactive_config_menu_configure_symbol_source(self, mock_configure, mock_prompt, mock_display):
        """Test menu flow with symbol source configuration."""
        mock_prompt.side_effect = ["1", "5"]  # Configure symbol source, then review
        mock_configure.return_value = 'main'
        
        with patch('modules.position_sizing.cli.argument_parser._review_and_confirm_ps', return_value='done'):
            config = interactive_config_menu()
        
        assert config is not None
        mock_configure.assert_called_once()
    
    @patch('modules.position_sizing.cli.argument_parser._display_main_menu_ps')
    @patch('modules.position_sizing.cli.argument_parser.prompt_user_input')
    def test_interactive_config_menu_exit(self, mock_prompt, mock_display):
        """Test menu exit option."""
        mock_prompt.return_value = "6"  # Exit
        
        with patch('sys.exit') as mock_exit:
            interactive_config_menu()
        
        mock_exit.assert_called_once_with(0)
    
    @patch('modules.position_sizing.cli.argument_parser._display_main_menu_ps')
    @patch('modules.position_sizing.cli.argument_parser.prompt_user_input')
    def test_interactive_config_menu_invalid_choice(self, mock_prompt, mock_display):
        """Test menu with invalid choice."""
        mock_prompt.side_effect = ["99", "6"]  # Invalid choice, then exit
        
        with patch('sys.exit'):
            with patch('builtins.print') as mock_print:
                try:
                    interactive_config_menu()
                except SystemExit:
                    pass
                
                # Should print error message for invalid choice
                print_calls = [str(call) for call in mock_print.call_args_list]
                assert any('Invalid' in call for call in print_calls)
    
    @patch('modules.position_sizing.cli.argument_parser._display_main_menu_ps')
    @patch('modules.position_sizing.cli.argument_parser.prompt_user_input')
    @patch('modules.position_sizing.cli.argument_parser._configure_account_balance')
    def test_interactive_config_menu_configure_account_balance(self, mock_configure, mock_prompt, mock_display):
        """Test menu flow with account balance configuration."""
        mock_prompt.side_effect = ["2", "5"]  # Configure account balance, then review
        mock_configure.return_value = 'main'
        
        with patch('modules.position_sizing.cli.argument_parser._review_and_confirm_ps', return_value='done'):
            config = interactive_config_menu()
        
        assert config is not None
        mock_configure.assert_called_once()

