"""
Tests for BatchChartGenerator class.

Tests cover:
- Initialization with custom parameters
- Batch chart creation with multiple symbols
- Grid layout validation
- Error handling for invalid inputs
- Edge cases (empty data, insufficient symbols)
"""

import os
import pytest
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from datetime import datetime, timedelta

from modules.gemini_chart_analyzer.core.batch_chart_generator import BatchChartGenerator


@pytest.fixture
def sample_ohlcv_data():
    """Create sample OHLCV DataFrame for testing."""
    dates = pd.date_range(start='2024-01-01', periods=50, freq='1h')
    np.random.seed(42)
    
    base_price = 50000
    prices = []
    for i in range(50):
        change = np.random.randn() * 100
        base_price = max(base_price + change, 1000)
        prices.append(base_price)
    
    df = pd.DataFrame({
        'open': prices,
        'high': [p * (1 + abs(np.random.randn() * 0.01)) for p in prices],
        'low': [p * (1 - abs(np.random.randn() * 0.01)) for p in prices],
        'close': [p * (1 + np.random.randn() * 0.005) for p in prices],
        'volume': np.random.randint(1000, 10000, 50)
    }, index=dates)
    
    df['high'] = df[['open', 'high', 'low', 'close']].max(axis=1)
    df['low'] = df[['open', 'high', 'low', 'close']].min(axis=1)
    
    return df


@pytest.fixture
def multiple_symbols_data(sample_ohlcv_data):
    """Create data for multiple symbols."""
    symbols = [f"SYM{i}/USDT" for i in range(1, 11)]  # 10 symbols
    return [{'symbol': sym, 'df': sample_ohlcv_data.copy()} for sym in symbols]


@pytest.fixture
def batch_generator():
    """Create BatchChartGenerator instance for testing."""
    return BatchChartGenerator(
        charts_per_batch=100,
        grid_rows=10,
        grid_cols=10,
        chart_size=(2.0, 1.5),
        dpi=100
    )


class TestBatchChartGeneratorInit:
    """Test BatchChartGenerator initialization."""
    
    def test_init_default_params(self):
        """Test initialization with default parameters."""
        generator = BatchChartGenerator()
        assert generator.charts_per_batch == 100
        assert generator.grid_rows == 10
        assert generator.grid_cols == 10
        assert generator.chart_size == (2.0, 1.5)
        assert generator.dpi == 100
    
    def test_init_custom_params(self):
        """Test initialization with custom parameters."""
        generator = BatchChartGenerator(
            charts_per_batch=25,
            grid_rows=5,
            grid_cols=5,
            chart_size=(3.0, 2.0),
            dpi=150
        )
        assert generator.charts_per_batch == 25
        assert generator.grid_rows == 5
        assert generator.grid_cols == 5
        assert generator.chart_size == (3.0, 2.0)
        assert generator.dpi == 150
    
    def test_init_invalid_grid_raises_error(self):
        """Test initialization with invalid grid dimensions raises error."""
        with pytest.raises(ValueError, match="grid_rows.*grid_cols.*must equal"):
            BatchChartGenerator(
                charts_per_batch=100,
                grid_rows=10,
                grid_cols=5  # 10 * 5 != 100
            )


class TestBatchChartGeneratorCreateBatchChart:
    """Test batch chart creation functionality."""
    
    def test_create_batch_chart_basic(self, batch_generator, multiple_symbols_data, tmp_path):
        """Test basic batch chart creation."""
        output_path = tmp_path / "batch_chart.png"
        result_path, truncated = batch_generator.create_batch_chart(
            symbols_data=multiple_symbols_data,
            timeframe="1h",
            output_path=str(output_path),
            batch_id=1
        )
        
        assert result_path == str(output_path)
        assert not truncated
        assert os.path.exists(result_path)
        assert os.path.getsize(result_path) > 0
        plt.close('all')
    
    def test_create_batch_chart_auto_path(self, batch_generator, multiple_symbols_data, tmp_path):
        """Test batch chart creation with automatic path generation."""
        # Temporarily change to temp directory
        original_cwd = os.getcwd()
        try:
            os.chdir(str(tmp_path))
            result_path, truncated = batch_generator.create_batch_chart(
                symbols_data=multiple_symbols_data[:5],
                timeframe="1h",
                batch_id=1
            )
            
            assert not truncated
            assert os.path.exists(result_path)
            assert "batch_chart" in result_path
            assert "1h" in result_path
            plt.close('all')
        finally:
            os.chdir(original_cwd)
    
    def test_create_batch_chart_too_many_symbols(self, batch_generator, multiple_symbols_data, tmp_path):
        """Test batch chart creation with too many symbols (should truncate)."""
        # Create 150 symbols (more than charts_per_batch=100)
        many_symbols = [{'symbol': f"SYM{i}/USDT", 'df': multiple_symbols_data[0]['df']} 
                        for i in range(150)]
        
        output_path = tmp_path / "batch_chart_many.png"
        result_path, truncated = batch_generator.create_batch_chart(
            symbols_data=many_symbols,
            timeframe="1h",
            output_path=str(output_path)
        )
        
        assert not truncated
        assert os.path.exists(result_path)
        plt.close('all')
    
    def test_create_batch_chart_empty_data(self, batch_generator, tmp_path):
        """Test batch chart creation with empty symbols data."""
        output_path = tmp_path / "batch_chart_empty.png"
        result_path, truncated = batch_generator.create_batch_chart(
            symbols_data=[],
            timeframe="1h",
            output_path=str(output_path)
        )
        
        assert not truncated
        assert os.path.exists(result_path)
        plt.close('all')
    
    def test_create_batch_chart_with_error_symbol(self, batch_generator, sample_ohlcv_data, tmp_path):
        """Test batch chart creation handles errors gracefully."""
        # Create data with one invalid symbol (empty DataFrame)
        symbols_data = [
            {'symbol': 'VALID/USDT', 'df': sample_ohlcv_data},
            {'symbol': 'INVALID/USDT', 'df': pd.DataFrame()},  # Empty DataFrame
            {'symbol': 'VALID2/USDT', 'df': sample_ohlcv_data}
        ]
        
        output_path = tmp_path / "batch_chart_error.png"
        result_path, truncated = batch_generator.create_batch_chart(
            symbols_data=symbols_data,
            timeframe="1h",
            output_path=str(output_path)
        )
        
        assert not truncated
        assert os.path.exists(result_path)
        plt.close('all')
    
    def test_create_batch_chart_missing_columns(self, batch_generator, tmp_path):
        """Test batch chart creation with missing columns (should show error message)."""
        # Create DataFrame with missing columns
        invalid_df = pd.DataFrame({'open': [1, 2, 3]})
        invalid_df.index = pd.date_range(start='2024-01-01', periods=3, freq='1h')
        
        symbols_data = [
            {'symbol': 'INVALID/USDT', 'df': invalid_df}
        ]
        
        output_path = tmp_path / "batch_chart_missing.png"
        result_path, truncated = batch_generator.create_batch_chart(
            symbols_data=symbols_data,
            timeframe="1h",
            output_path=str(output_path)
        )
        
        assert not truncated
        assert os.path.exists(result_path)
        plt.close('all')


class TestBatchChartGeneratorPlotSimpleChart:
    """Test _plot_simple_chart_on_axes method."""
    
    def test_plot_simple_chart_valid_data(self, batch_generator, sample_ohlcv_data):
        """Test plotting simple chart with valid data."""
        fig, ax = plt.subplots(figsize=(2, 1.5))
        
        batch_generator._plot_simple_chart_on_axes(
            ax=ax,
            df=sample_ohlcv_data,
            symbol="BTC/USDT",
            timeframe="1h"
        )
        
        assert ax is not None
        plt.close(fig)
    
    def test_plot_simple_chart_empty_dataframe(self, batch_generator):
        """Test plotting with empty DataFrame shows error message."""
        empty_df = pd.DataFrame()
        fig, ax = plt.subplots(figsize=(2, 1.5))
        
        batch_generator._plot_simple_chart_on_axes(
            ax=ax,
            df=empty_df,
            symbol="BTC/USDT",
            timeframe="1h"
        )
        
        # Should not raise error, but show error message on chart
        plt.close(fig)
    
    def test_plot_simple_chart_missing_columns(self, batch_generator):
        """Test plotting with missing columns shows error message."""
        invalid_df = pd.DataFrame({'open': [1, 2, 3]})
        invalid_df.index = pd.date_range(start='2024-01-01', periods=3, freq='1h')
        
        fig, ax = plt.subplots(figsize=(2, 1.5))
        
        batch_generator._plot_simple_chart_on_axes(
            ax=ax,
            df=invalid_df,
            symbol="BTC/USDT",
            timeframe="1h"
        )
        
        # Should not raise error, but show error message on chart
        plt.close(fig)
    
    def test_plot_simple_chart_single_candle(self, batch_generator):
        """Test plotting with single candle."""
        single_candle = pd.DataFrame({
            'open': [50000],
            'high': [51000],
            'low': [49000],
            'close': [50500]
        }, index=pd.date_range(start='2024-01-01', periods=1, freq='1h'))
        
        fig, ax = plt.subplots(figsize=(2, 1.5))
        
        batch_generator._plot_simple_chart_on_axes(
            ax=ax,
            df=single_candle,
            symbol="BTC/USDT",
            timeframe="1h"
        )
        
        plt.close(fig)

