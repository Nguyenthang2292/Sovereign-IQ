"""
Tests for ChartMultiTimeframeBatchGenerator class.

Tests cover:
- Initialization with default and custom parameters
- Grid calculation
- Multi-timeframe batch chart creation
- Error handling
- File output validation
"""

import os
import pytest
import pandas as pd
import numpy as np
from pathlib import Path
from datetime import datetime, timedelta
from unittest.mock import patch, MagicMock

from modules.gemini_chart_analyzer.core.generators.chart_multi_timeframe_batch_generator import (
    ChartMultiTimeframeBatchGenerator
)
from modules.gemini_chart_analyzer.core.utils.chart_paths import get_charts_dir


@pytest.fixture
def sample_ohlcv_data():
    """Create sample OHLCV DataFrame for testing."""
    dates = pd.date_range(start='2024-01-01', periods=100, freq='1h')
    np.random.seed(42)
    
    base_price = 50000
    prices = []
    for i in range(100):
        change = np.random.randn() * 100
        base_price = max(base_price + change, 1000)
        prices.append(base_price)
    
    df = pd.DataFrame({
        'open': prices,
        'high': [p * 1.01 for p in prices],
        'low': [p * 0.99 for p in prices],
        'close': [p * 1.005 for p in prices],
        'volume': np.random.randint(1000, 10000, 100)
    }, index=dates)
    
    return df


@pytest.fixture
def symbols_data(sample_ohlcv_data):
    """Create sample symbols_data dict for testing."""
    return {
        'BTC/USDT': {
            '15m': sample_ohlcv_data,
            '1h': sample_ohlcv_data,
            '4h': sample_ohlcv_data,
            '1d': sample_ohlcv_data
        },
        'ETH/USDT': {
            '15m': sample_ohlcv_data,
            '1h': sample_ohlcv_data,
            '4h': sample_ohlcv_data,
            '1d': sample_ohlcv_data
        }
    }


@pytest.fixture
def default_generator():
    """Create ChartMultiTimeframeBatchGenerator with default parameters."""
    return ChartMultiTimeframeBatchGenerator()


@pytest.fixture
def custom_generator():
    """Create ChartMultiTimeframeBatchGenerator with custom parameters."""
    return ChartMultiTimeframeBatchGenerator(
        charts_per_batch=10,
        timeframes_per_symbol=3,
        chart_size=(2.0, 1.5),
        dpi=100
    )


@pytest.fixture
def temp_output_dir(tmp_path):
    """Create temporary directory for chart output."""
    output_dir = tmp_path / "charts" / "batch"
    output_dir.mkdir(parents=True, exist_ok=True)
    return output_dir


class TestChartMultiTimeframeBatchGeneratorInit:
    """Test ChartMultiTimeframeBatchGenerator initialization."""
    
    def test_init_default_params(self, default_generator):
        """Test initialization with default parameters."""
        # charts_per_batch is now total_subplots (25 * 4 = 100)
        # symbols_per_batch stores the original value (25)
        assert default_generator.symbols_per_batch == 25
        assert default_generator.charts_per_batch == 100  # total_subplots
        assert default_generator.timeframes_per_symbol == 4
        assert default_generator.chart_size == (3.0, 2.0)
        assert default_generator.dpi == 100
        assert default_generator.symbols_per_row > 0
        assert default_generator.symbols_per_col > 0
        assert default_generator.tf_rows > 0
        assert default_generator.tf_cols > 0
    
    def test_init_custom_params(self, custom_generator):
        """Test initialization with custom parameters."""
        # charts_per_batch is now total_subplots (10 * 3 = 30)
        # symbols_per_batch stores the original value (10)
        assert custom_generator.symbols_per_batch == 10
        assert custom_generator.charts_per_batch == 30  # total_subplots
        assert custom_generator.timeframes_per_symbol == 3
        assert custom_generator.chart_size == (2.0, 1.5)
        assert custom_generator.dpi == 100
    
    def test_grid_calculation(self, default_generator):
        """Test that grid dimensions are calculated correctly."""
        # For 25 symbols with 4 timeframes each
        # symbols_per_row should be around sqrt(25) = 5
        assert default_generator.symbols_per_row * default_generator.symbols_per_col >= \
               default_generator.symbols_per_batch
        
        # For 4 timeframes, should be 2x2 grid
        assert default_generator.tf_rows * default_generator.tf_cols >= \
               default_generator.timeframes_per_symbol
        
        # charts_per_batch should equal total_subplots (symbols * timeframes)
        assert default_generator.charts_per_batch == \
               default_generator.symbols_per_batch * default_generator.timeframes_per_symbol
    
    def test_inherits_from_batch_chart_generator(self, default_generator):
        """Test that ChartMultiTimeframeBatchGenerator inherits from ChartBatchGenerator."""
        from modules.gemini_chart_analyzer.core.generators.chart_batch_generator import ChartBatchGenerator
        assert isinstance(default_generator, ChartBatchGenerator)


class TestCreateMultiTFBatchChart:
    """Test create_multi_tf_batch_chart method."""
    
    def test_create_chart_success(self, default_generator, symbols_data, temp_output_dir):
        """Test successful chart creation."""
        timeframes = ['15m', '1h', '4h', '1d']
        output_path = str(temp_output_dir / "test_batch_chart.png")
        
        result_path, truncated = default_generator.create_multi_tf_batch_chart(
            symbols_data=symbols_data,
            timeframes=timeframes,
            output_path=output_path
        )
        
        assert result_path == output_path
        assert truncated is False
        assert os.path.exists(result_path)
        assert os.path.getsize(result_path) > 0
    
    def test_create_chart_auto_path(self, default_generator, symbols_data, temp_output_dir):
        """Test chart creation with auto-generated path."""
        timeframes = ['15m', '1h', '4h', '1d']
        
        with patch('modules.gemini_chart_analyzer.core.generators.chart_multi_timeframe_batch_generator.get_charts_dir') as mock_dir:
            mock_dir.return_value = temp_output_dir.parent
            
            result_path, truncated = default_generator.create_multi_tf_batch_chart(
                symbols_data=symbols_data,
                timeframes=timeframes,
                batch_id=1
            )
            
            assert result_path is not None
            assert 'batch_chart_multi_tf' in result_path
            assert 'batch1' in result_path
            assert truncated is False
    
    def test_create_chart_truncates_symbols(self, default_generator, sample_ohlcv_data, temp_output_dir):
        """Test that chart creation truncates when too many symbols."""
        # Create more symbols than symbols_per_batch (25)
        symbols_data = {}
        for i in range(30):
            symbol = f'SYMBOL{i}/USDT'
            symbols_data[symbol] = {
                '15m': sample_ohlcv_data,
                '1h': sample_ohlcv_data
            }
        
        timeframes = ['15m', '1h']
        output_path = str(temp_output_dir / "test_batch_chart.png")
        
        result_path, truncated = default_generator.create_multi_tf_batch_chart(
            symbols_data=symbols_data,
            timeframes=timeframes,
            output_path=output_path
        )
        
        assert truncated is True
        assert os.path.exists(result_path)
    
    def test_create_chart_empty_data(self, default_generator):
        """Test that chart creation raises error for empty data."""
        with pytest.raises(ValueError, match="cannot be empty"):
            default_generator.create_multi_tf_batch_chart(
                symbols_data={},
                timeframes=['15m', '1h']
            )
    
    def test_create_chart_missing_timeframe(self, default_generator, sample_ohlcv_data, temp_output_dir):
        """Test chart creation when some timeframes are missing."""
        symbols_data = {
            'BTC/USDT': {
                '15m': sample_ohlcv_data,
                '1h': sample_ohlcv_data
                # Missing '4h' and '1d'
            }
        }
        timeframes = ['15m', '1h', '4h', '1d']
        output_path = str(temp_output_dir / "test_batch_chart.png")
        
        # Should not raise error, just skip missing timeframes
        result_path, truncated = default_generator.create_multi_tf_batch_chart(
            symbols_data=symbols_data,
            timeframes=timeframes,
            output_path=output_path
        )
        
        assert os.path.exists(result_path)
    
    def test_create_chart_handles_plotting_error(self, default_generator, temp_output_dir):
        """Test that chart creation handles plotting errors gracefully."""
        # Create invalid DataFrame
        invalid_df = pd.DataFrame({'invalid': [1, 2, 3]})
        
        symbols_data = {
            'BTC/USDT': {
                '15m': invalid_df,
                '1h': invalid_df
            }
        }
        timeframes = ['15m', '1h']
        output_path = str(temp_output_dir / "test_batch_chart.png")
        
        # Should not raise error, should handle gracefully
        result_path, truncated = default_generator.create_multi_tf_batch_chart(
            symbols_data=symbols_data,
            timeframes=timeframes,
            output_path=output_path
        )
        
        assert os.path.exists(result_path)
    
    def test_create_chart_multiple_symbols(self, default_generator, sample_ohlcv_data, temp_output_dir):
        """Test chart creation with multiple symbols."""
        symbols_data = {
            'BTC/USDT': {
                '15m': sample_ohlcv_data,
                '1h': sample_ohlcv_data
            },
            'ETH/USDT': {
                '15m': sample_ohlcv_data,
                '1h': sample_ohlcv_data
            },
            'BNB/USDT': {
                '15m': sample_ohlcv_data,
                '1h': sample_ohlcv_data
            }
        }
        timeframes = ['15m', '1h']
        output_path = str(temp_output_dir / "test_batch_chart.png")
        
        result_path, truncated = default_generator.create_multi_tf_batch_chart(
            symbols_data=symbols_data,
            timeframes=timeframes,
            output_path=output_path
        )
        
        assert os.path.exists(result_path)
        assert truncated is False
    
    def test_create_chart_custom_timeframes_per_symbol(self, custom_generator, sample_ohlcv_data, temp_output_dir):
        """Test chart creation with custom timeframes_per_symbol."""
        symbols_data = {
            'BTC/USDT': {
                '15m': sample_ohlcv_data,
                '1h': sample_ohlcv_data,
                '4h': sample_ohlcv_data
            }
        }
        timeframes = ['15m', '1h', '4h', '1d']  # More than timeframes_per_symbol (3)
        output_path = str(temp_output_dir / "test_batch_chart.png")
        
        result_path, truncated = custom_generator.create_multi_tf_batch_chart(
            symbols_data=symbols_data,
            timeframes=timeframes,
            output_path=output_path
        )
        
        # Should only use first 3 timeframes (timeframes_per_symbol)
        assert os.path.exists(result_path)


class TestGetChartsDir:
    """Test get_charts_dir function."""
    
    def test_get_charts_dir_returns_path(self):
        """Test that get_charts_dir returns a Path object."""
        charts_dir = get_charts_dir()
        assert isinstance(charts_dir, Path)
    
    def test_get_charts_dir_relative_to_module(self):
        """Test that charts directory is relative to module root."""
        charts_dir = get_charts_dir()
        # Should be in modules/gemini_chart_analyzer/charts
        assert charts_dir.parts[-3:] == ("modules", "gemini_chart_analyzer", "charts")


class TestErrorHandling:
    """Test error handling scenarios."""
    
    def test_save_error_raises_ioerror(self, default_generator, symbols_data, temp_output_dir):
        """Test that save errors raise IOError."""
        timeframes = ['15m', '1h']
        
        # Create invalid output path (directory doesn't exist)
        invalid_path = str(temp_output_dir.parent / "nonexistent" / "chart.png")
        
        with pytest.raises(IOError):
            default_generator.create_multi_tf_batch_chart(
                symbols_data=symbols_data,
                timeframes=timeframes,
                output_path=invalid_path
            )
    
    def test_empty_dataframe_handled(self, default_generator, temp_output_dir):
        """Test that empty DataFrames are handled."""
        symbols_data = {
            'BTC/USDT': {
                '15m': pd.DataFrame(),  # Empty DataFrame
                '1h': pd.DataFrame()
            }
        }
        timeframes = ['15m', '1h']
        output_path = str(temp_output_dir / "test_batch_chart.png")
        
        # Should handle gracefully (may skip or show error message)
        result_path, truncated = default_generator.create_multi_tf_batch_chart(
            symbols_data=symbols_data,
            timeframes=timeframes,
            output_path=output_path
        )
        
        # Should still create file (may be mostly empty)
        assert os.path.exists(result_path)


class TestIntegration:
    """Integration tests."""
    
    def test_full_workflow(self, default_generator, sample_ohlcv_data, temp_output_dir):
        """Test complete workflow from data to chart file."""
        symbols_data = {
            'BTC/USDT': {
                '15m': sample_ohlcv_data,
                '1h': sample_ohlcv_data,
                '4h': sample_ohlcv_data,
                '1d': sample_ohlcv_data
            },
            'ETH/USDT': {
                '15m': sample_ohlcv_data,
                '1h': sample_ohlcv_data,
                '4h': sample_ohlcv_data,
                '1d': sample_ohlcv_data
            }
        }
        timeframes = ['15m', '1h', '4h', '1d']
        output_path = str(temp_output_dir / "integration_test.png")
        
        result_path, truncated = default_generator.create_multi_tf_batch_chart(
            symbols_data=symbols_data,
            timeframes=timeframes,
            output_path=output_path,
            batch_id=1
        )
        
        # Verify file exists and has content
        assert os.path.exists(result_path)
        assert os.path.getsize(result_path) > 1000  # Should be substantial file
        
        # Verify path matches
        assert result_path == output_path
        assert truncated is False
    
    def test_batch_id_in_filename(self, default_generator, symbols_data, temp_output_dir):
        """Test that batch_id appears in auto-generated filename."""
        timeframes = ['15m', '1h']
        
        with patch('modules.gemini_chart_analyzer.core.generators.chart_multi_timeframe_batch_generator.get_charts_dir') as mock_dir:
            mock_dir.return_value = temp_output_dir.parent
            
            result_path, _ = default_generator.create_multi_tf_batch_chart(
                symbols_data=symbols_data,
                timeframes=timeframes,
                batch_id=42
            )
            
            assert 'batch42' in result_path or 'batch_42' in result_path


