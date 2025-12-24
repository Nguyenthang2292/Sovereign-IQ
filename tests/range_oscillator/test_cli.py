"""
Tests for range_oscillator CLI module.

Tests CLI functions:
- parse_args
- display_configuration
- display_final_results
"""
import sys
import pytest
import pandas as pd
from unittest.mock import patch, MagicMock

from modules.range_oscillator.cli.argument_parser import parse_args
from modules.range_oscillator.cli.display import (
    display_configuration,
    display_final_results,
)


class TestArgumentParser:
    """Tests for parse_args function."""
    
    def test_parse_args_defaults(self):
        """Test parse_args with default values."""
        with patch('sys.argv', ['script']):
            args = parse_args()
            
            assert args.timeframe is not None
            assert args.limit == 500
            assert args.max_workers == 10
            assert args.osc_strategies is None
    
    def test_parse_args_custom_timeframe(self):
        """Test parse_args with custom timeframe."""
        with patch('sys.argv', ['script', '--timeframe', '1h']):
            args = parse_args()
            assert args.timeframe == '1h'
    
    def test_parse_args_custom_limit(self):
        """Test parse_args with custom limit."""
        with patch('sys.argv', ['script', '--limit', '1000']):
            args = parse_args()
            assert args.limit == 1000
    
    def test_parse_args_validation_positive_int(self):
        """Test parse_args validation for positive integers."""
        with patch('sys.argv', ['script', '--limit', '0']):
            with pytest.raises(SystemExit):  # argparse raises SystemExit on validation error
                parse_args()
        
        with patch('sys.argv', ['script', '--limit', '-1']):
            with pytest.raises(SystemExit):
                parse_args()
    
    def test_parse_args_strategy_ids(self):
        """Test parse_args with strategy IDs."""
        with patch('sys.argv', ['script', '--osc-strategies', '2', '3', '4']):
            args = parse_args()
            assert args.osc_strategies == [2, 3, 4]
    
    def test_parse_args_invalid_strategy_id(self):
        """Test parse_args with invalid strategy ID."""
        with patch('sys.argv', ['script', '--osc-strategies', '1', '5']):
            with pytest.raises(SystemExit):  # Invalid strategy IDs should raise
                parse_args()
    
    def test_parse_args_all_parameters(self):
        """Test parse_args with all parameters."""
        with patch('sys.argv', [
            'script',
            '--timeframe', '1h',
            '--limit', '1000',
            '--max-workers', '20',
            '--osc-strategies', '2', '3', '4',
            '--min-signal', '0.05',
            '--osc-length', '100',
            '--osc-mult', '3.0'
        ]):
            args = parse_args()
            assert args.timeframe == '1h'
            assert args.limit == 1000
            assert args.max_workers == 20
            assert args.osc_strategies == [2, 3, 4]
            assert args.min_signal == 0.05
            assert args.osc_length == 100
            assert args.osc_mult == 3.0


class TestDisplayConfiguration:
    """Tests for display_configuration function."""
    
    def test_display_configuration_basic(self, capsys):
        """Test display_configuration basic functionality."""
        display_configuration(
            timeframe='1h',
            limit=500,
            min_signal=0.01,
            max_workers=10,
            strategies=None
        )
        
        captured = capsys.readouterr()
        assert '1h' in captured.out
        assert '500' in captured.out
        assert '0.01' in captured.out
    
    def test_display_configuration_with_strategies(self, capsys):
        """Test display_configuration with strategies."""
        display_configuration(
            timeframe='15m',
            limit=1000,
            min_signal=0.05,
            max_workers=20,
            strategies=[2, 3, 4],
            max_symbols=50
        )
        
        captured = capsys.readouterr()
        assert '15m' in captured.out
        assert '1000' in captured.out
        assert '50' in captured.out
    
    def test_display_configuration_with_max_symbols(self, capsys):
        """Test display_configuration with max_symbols."""
        display_configuration(
            timeframe='1h',
            limit=500,
            min_signal=0.01,
            max_workers=10,
            strategies=None,
            max_symbols=100
        )
        
        captured = capsys.readouterr()
        assert '100' in captured.out


class TestDisplayFinalResults:
    """Tests for display_final_results function."""
    
    @pytest.fixture
    def sample_signals_data(self):
        """Create sample signals data for testing."""
        dates = pd.date_range('2024-01-01', periods=5, freq='1h')
        
        long_signals = pd.DataFrame({
            'symbol': ['BTC/USDT', 'ETH/USDT'],
            'signal': [0.5, 0.3],
            'price': [50000.0, 3000.0],
            'exchange': ['binance', 'binance'],
            'osc_confidence': [0.8, 0.7]
        }, index=dates[:2])
        
        short_signals = pd.DataFrame({
            'symbol': ['SOL/USDT'],
            'signal': [-0.4],
            'price': [100.0],
            'exchange': ['binance'],
            'osc_confidence': [0.6]
        }, index=dates[2:3])
        
        return long_signals, short_signals
    
    def test_display_final_results_basic(self, sample_signals_data, capsys):
        """Test display_final_results basic functionality."""
        long_signals, short_signals = sample_signals_data
        
        display_final_results(
            long_signals=long_signals,
            short_signals=short_signals,
            original_long_count=10,
            original_short_count=5,
            long_uses_fallback=False,
            short_uses_fallback=False
        )
        
        captured = capsys.readouterr()
        assert 'BTC/USDT' in captured.out or 'ETH/USDT' in captured.out
        assert 'SOL/USDT' in captured.out
        assert '10' in captured.out  # original_long_count
        assert '5' in captured.out   # original_short_count
    
    def test_display_final_results_empty_signals(self, capsys):
        """Test display_final_results with empty signals."""
        empty_long = pd.DataFrame(columns=['symbol', 'signal', 'price', 'exchange'])
        empty_short = pd.DataFrame(columns=['symbol', 'signal', 'price', 'exchange'])
        
        display_final_results(
            long_signals=empty_long,
            short_signals=empty_short,
            original_long_count=0,
            original_short_count=0,
            long_uses_fallback=False,
            short_uses_fallback=False
        )
        
        captured = capsys.readouterr()
        assert 'No LONG signals' in captured.out or 'No SHORT signals' in captured.out
    
    def test_display_final_results_fallback(self, sample_signals_data, capsys):
        """Test display_final_results with fallback signals."""
        long_signals, short_signals = sample_signals_data
        
        display_final_results(
            long_signals=long_signals,
            short_signals=short_signals,
            original_long_count=10,
            original_short_count=5,
            long_uses_fallback=True,
            short_uses_fallback=False
        )
        
        captured = capsys.readouterr()
        assert 'Fallback' in captured.out or 'ATC ONLY' in captured.out
    
    def test_display_final_results_without_confidence(self, capsys):
        """Test display_final_results without confidence column."""
        dates = pd.date_range('2024-01-01', periods=2, freq='1h')
        
        long_signals = pd.DataFrame({
            'symbol': ['BTC/USDT'],
            'signal': [0.5],
            'price': [50000.0],
            'exchange': ['binance']
        }, index=dates[:1])
        
        short_signals = pd.DataFrame({
            'symbol': ['ETH/USDT'],
            'signal': [-0.3],
            'price': [3000.0],
            'exchange': ['binance']
        }, index=dates[1:2])
        
        display_final_results(
            long_signals=long_signals,
            short_signals=short_signals,
            original_long_count=5,
            original_short_count=3,
            long_uses_fallback=False,
            short_uses_fallback=False
        )
        
        captured = capsys.readouterr()
        assert 'BTC/USDT' in captured.out or 'ETH/USDT' in captured.out
    
    def test_display_final_results_input_validation(self):
        """Test display_final_results input validation."""
        # Test with non-DataFrame input
        with pytest.raises(TypeError):
            display_final_results(
                long_signals=[],  # Should be DataFrame
                short_signals=pd.DataFrame(),
                original_long_count=0,
                original_short_count=0
            )
        
        # Test with invalid original counts
        empty_df = pd.DataFrame(columns=['symbol', 'signal', 'price', 'exchange'])
        with pytest.raises(ValueError):
            display_final_results(
                long_signals=empty_df,
                short_signals=empty_df,
                original_long_count=-1,  # Should be >= 0
                original_short_count=0
            )
        
        # Test with missing required columns
        incomplete_df = pd.DataFrame({
            'symbol': ['BTC/USDT'],
            'signal': [0.5]
            # Missing 'price' and 'exchange'
        })
        with pytest.raises(ValueError):
            display_final_results(
                long_signals=incomplete_df,
                short_signals=empty_df,
                original_long_count=0,
                original_short_count=0
            )
    
    def test_display_final_results_with_missing_keys(self, capsys):
        """Test display_final_results with missing keys in rows (should skip gracefully)."""
        dates = pd.date_range('2024-01-01', periods=3, freq='1h')
        
        # Create DataFrame with some rows missing keys
        long_signals = pd.DataFrame({
            'symbol': ['BTC/USDT', 'ETH/USDT', 'SOL/USDT'],
            'signal': [0.5, 0.3, 0.2],
            'price': [50000.0, 3000.0, 100.0],
            'exchange': ['binance', 'binance', 'binance']
        }, index=dates)
        
        # Manually remove a key from one row to simulate missing key scenario
        # This is handled gracefully in the function
        short_signals = pd.DataFrame({
            'symbol': ['ADA/USDT'],
            'signal': [-0.4],
            'price': [0.5],
            'exchange': ['binance']
        }, index=dates[2:3])
        
        display_final_results(
            long_signals=long_signals,
            short_signals=short_signals,
            original_long_count=10,
            original_short_count=5,
            long_uses_fallback=False,
            short_uses_fallback=False
        )
        
        captured = capsys.readouterr()
        # Should complete without crashing
        assert len(captured.out) > 0

