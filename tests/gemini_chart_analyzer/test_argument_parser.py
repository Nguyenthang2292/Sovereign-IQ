"""
Tests for argument_parser module.

Tests cover:
- Argument parsing with various options
- Default values
- Validation logic
- MA periods parsing
- Chart figsize parsing
"""

import pytest
import sys
from unittest.mock import patch

from modules.gemini_chart_analyzer.cli.argument_parser import (
    parse_args,
    _format_current_value,
)


class TestFormatCurrentValue:
    """Test _format_current_value helper function."""
    
    def test_format_none(self):
        """Test formatting None value."""
        result = _format_current_value(None)
        assert result == "not set"
    
    def test_format_bool_true(self):
        """Test formatting True boolean."""
        result = _format_current_value(True)
        assert result == "enabled"
    
    def test_format_bool_false(self):
        """Test formatting False boolean."""
        result = _format_current_value(False)
        assert result == "disabled"
    
    def test_format_string(self):
        """Test formatting string value."""
        result = _format_current_value("1h")
        assert result == "1h"
    
    def test_format_list(self):
        """Test formatting list value."""
        result = _format_current_value([20, 50, 200])
        assert result == "20, 50, 200"
    
    def test_format_dict_with_periods(self):
        """Test formatting dict with periods."""
        result = _format_current_value({'periods': [20, 50]})
        assert result == "periods=[20, 50]"
    
    def test_format_dict_with_period(self):
        """Test formatting dict with period."""
        result = _format_current_value({'period': 14})
        assert result == "period=14"
    
    def test_format_dict_other(self):
        """Test formatting other dict."""
        result = _format_current_value({'key': 'value'})
        assert result == "{'key': 'value'}"


class TestParseArgs:
    """Test parse_args function."""
    
    @patch('sys.argv', ['script.py'])
    def test_parse_args_no_arguments_returns_none(self):
        """Test that parse_args returns None when no arguments provided."""
        result = parse_args()
        assert result is None
    
    @patch('sys.argv', ['script.py', '--symbol', 'BTC/USDT'])
    def test_parse_args_with_symbol(self):
        """Test parsing with symbol argument."""
        result = parse_args()
        
        assert result is not None
        assert result.symbol == 'BTC/USDT'
        assert result.timeframe == '1h'  # Default
    
    @patch('sys.argv', ['script.py', '--symbol', 'ETH/USDT', '--timeframe', '4h'])
    def test_parse_args_with_symbol_and_timeframe(self):
        """Test parsing with symbol and timeframe."""
        result = parse_args()
        
        assert result.symbol == 'ETH/USDT'
        assert result.timeframe == '4h'
    
    @patch('sys.argv', ['script.py', '--symbol', 'BTC/USDT', '--ma-periods', '20,50,200'])
    def test_parse_args_ma_periods(self):
        """Test parsing MA periods."""
        result = parse_args()
        
        assert result.ma_periods_list == [20, 50, 200]
    
    @patch('sys.argv', ['script.py', '--symbol', 'BTC/USDT', '--ma-periods', '20, 50, 200'])
    def test_parse_args_ma_periods_with_spaces(self):
        """Test parsing MA periods with spaces."""
        result = parse_args()
        
        assert result.ma_periods_list == [20, 50, 200]
    
    @patch('sys.argv', ['script.py', '--symbol', 'BTC/USDT', '--ma-periods', 'invalid'])
    def test_parse_args_ma_periods_invalid(self):
        """Test parsing invalid MA periods falls back to default."""
        result = parse_args()
        
        # Should fall back to default when invalid
        assert result.ma_periods_list == [20, 50, 200]
    
    @patch('sys.argv', ['script.py', '--symbol', 'BTC/USDT', '--no-ma'])
    def test_parse_args_no_ma(self):
        """Test parsing --no-ma flag."""
        result = parse_args()
        
        assert result.no_ma is True
    
    @patch('sys.argv', ['script.py', '--symbol', 'BTC/USDT', '--rsi-period', '21'])
    def test_parse_args_rsi_period(self):
        """Test parsing RSI period."""
        result = parse_args()
        
        assert result.rsi_period == 21
    
    @patch('sys.argv', ['script.py', '--symbol', 'BTC/USDT', '--no-rsi'])
    def test_parse_args_no_rsi(self):
        """Test parsing --no-rsi flag."""
        result = parse_args()
        
        assert result.no_rsi is True
    
    @patch('sys.argv', ['script.py', '--symbol', 'BTC/USDT', '--no-macd'])
    def test_parse_args_no_macd(self):
        """Test parsing --no-macd flag."""
        result = parse_args()
        
        assert result.no_macd is True
    
    @patch('sys.argv', ['script.py', '--symbol', 'BTC/USDT', '--enable-bb', '--bb-period', '30'])
    def test_parse_args_bollinger_bands(self):
        """Test parsing Bollinger Bands configuration."""
        result = parse_args()
        
        assert result.enable_bb is True
        assert result.bb_period == 30
    
    @patch('sys.argv', ['script.py', '--symbol', 'BTC/USDT', '--prompt-type', 'simple'])
    def test_parse_args_prompt_type(self):
        """Test parsing prompt type."""
        result = parse_args()
        
        assert result.prompt_type == 'simple'
    
    @patch('sys.argv', ['script.py', '--symbol', 'BTC/USDT', '--prompt-type', 'custom', '--custom-prompt', 'Analyze this chart'])
    def test_parse_args_custom_prompt(self):
        """Test parsing custom prompt."""
        result = parse_args()
        
        assert result.prompt_type == 'custom'
        assert result.custom_prompt == 'Analyze this chart'
    
    @patch('sys.argv', ['script.py', '--symbol', 'BTC/USDT', '--prompt-type', 'custom'])
    def test_parse_args_custom_prompt_missing_raises_error(self):
        """Test that missing custom prompt with custom type raises error."""
        with pytest.raises(SystemExit):  # argparse calls sys.exit on error
            parse_args()
    
    @patch('sys.argv', ['script.py', '--symbol', 'BTC/USDT', '--chart-figsize', '20,12'])
    def test_parse_args_chart_figsize(self):
        """Test parsing chart figsize."""
        result = parse_args()
        
        assert result.chart_figsize_tuple == (20, 12)
    
    @patch('sys.argv', ['script.py', '--symbol', 'BTC/USDT', '--chart-figsize', 'invalid'])
    def test_parse_args_chart_figsize_invalid(self):
        """Test parsing invalid chart figsize falls back to default."""
        result = parse_args()
        
        assert result.chart_figsize_tuple == (16, 10)  # Default
    
    @patch('sys.argv', ['script.py', '--symbol', 'BTC/USDT', '--chart-dpi', '200'])
    def test_parse_args_chart_dpi(self):
        """Test parsing chart DPI."""
        result = parse_args()
        
        assert result.chart_dpi == 200
    
    @patch('sys.argv', ['script.py', '--symbol', 'BTC/USDT', '--limit', '1000'])
    def test_parse_args_limit(self):
        """Test parsing limit."""
        result = parse_args()
        
        assert result.limit == 1000
    
    @patch('sys.argv', ['script.py', '--symbol', 'BTC/USDT', '--no-cleanup'])
    def test_parse_args_no_cleanup(self):
        """Test parsing --no-cleanup flag."""
        result = parse_args()
        
        assert result.no_cleanup is True
    
    @patch('sys.argv', ['script.py', '--symbol', 'BTC/USDT', '--timeframe', '1h', '--rsi-period', '14', '--no-ma', '--enable-bb'])
    def test_parse_args_multiple_options(self):
        """Test parsing multiple options together."""
        result = parse_args()
        
        assert result.symbol == 'BTC/USDT'
        assert result.timeframe == '1h'
        assert result.rsi_period == 14
        assert result.no_ma is True
        assert result.enable_bb is True

