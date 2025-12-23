"""
Tests for ChartGenerator class.

Tests cover:
- Initialization with custom parameters
- Chart creation with valid data
- Indicator calculations (MA, RSI, MACD, Bollinger Bands)
- Error handling for invalid inputs
- File output validation
"""

import os
import pytest
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from datetime import datetime, timedelta

from modules.gemini_chart_analyzer.core.chart_generator import ChartGenerator


@pytest.fixture
def sample_ohlcv_data():
    """Create sample OHLCV DataFrame for testing."""
    dates = pd.date_range(start='2024-01-01', periods=100, freq='1h')
    np.random.seed(42)
    
    # Generate realistic OHLCV data
    base_price = 50000
    prices = []
    for i in range(100):
        change = np.random.randn() * 100
        base_price = max(base_price + change, 1000)  # Ensure positive prices
        prices.append(base_price)
    
    df = pd.DataFrame({
        'open': prices,
        'high': [p * (1 + abs(np.random.randn() * 0.01)) for p in prices],
        'low': [p * (1 - abs(np.random.randn() * 0.01)) for p in prices],
        'close': [p * (1 + np.random.randn() * 0.005) for p in prices],
        'volume': np.random.randint(1000, 10000, 100)
    }, index=dates)
    
    # Ensure high >= close >= low and high >= open >= low
    df['high'] = df[['open', 'high', 'low', 'close']].max(axis=1)
    df['low'] = df[['open', 'high', 'low', 'close']].min(axis=1)
    
    return df


@pytest.fixture
def chart_generator():
    """Create ChartGenerator instance for testing."""
    return ChartGenerator()


@pytest.fixture
def temp_output_dir(tmp_path):
    """Create temporary directory for chart output."""
    output_dir = tmp_path / "charts"
    output_dir.mkdir()
    return output_dir


class TestChartGeneratorInit:
    """Test ChartGenerator initialization."""
    
    def test_init_default_params(self):
        """Test initialization with default parameters."""
        generator = ChartGenerator()
        assert generator.figsize == (16, 10)
        assert generator.style == 'dark_background'
        assert generator.dpi == 150
    
    def test_init_custom_params(self):
        """Test initialization with custom parameters."""
        generator = ChartGenerator(
            figsize=(20, 12),
            style='default',
            dpi=200
        )
        assert generator.figsize == (20, 12)
        assert generator.style == 'default'
        assert generator.dpi == 200


class TestChartGeneratorCreateChart:
    """Test chart creation functionality."""
    
    def test_create_chart_basic(self, chart_generator, sample_ohlcv_data, temp_output_dir):
        """Test basic chart creation without indicators."""
        output_path = temp_output_dir / "test_chart.png"
        result_path = chart_generator.create_chart(
            df=sample_ohlcv_data,
            symbol="BTC/USDT",
            timeframe="1h",
            output_path=str(output_path)
        )
        
        assert result_path == str(output_path)
        assert os.path.exists(result_path)
        assert os.path.getsize(result_path) > 0
        plt.close('all')
    
    def test_create_chart_with_ma(self, chart_generator, sample_ohlcv_data, temp_output_dir):
        """Test chart creation with Moving Averages."""
        output_path = temp_output_dir / "test_chart_ma.png"
        indicators = {
            'MA': {'periods': [20, 50]}
        }
        result_path = chart_generator.create_chart(
            df=sample_ohlcv_data,
            symbol="BTC/USDT",
            timeframe="1h",
            indicators=indicators,
            output_path=str(output_path)
        )
        
        assert os.path.exists(result_path)
        plt.close('all')
    
    def test_create_chart_with_rsi(self, chart_generator, sample_ohlcv_data, temp_output_dir):
        """Test chart creation with RSI indicator."""
        output_path = temp_output_dir / "test_chart_rsi.png"
        indicators = {
            'RSI': {'period': 14}
        }
        result_path = chart_generator.create_chart(
            df=sample_ohlcv_data,
            symbol="ETH/USDT",
            timeframe="4h",
            indicators=indicators,
            output_path=str(output_path)
        )
        
        assert os.path.exists(result_path)
        plt.close('all')
    
    def test_create_chart_with_macd(self, chart_generator, sample_ohlcv_data, temp_output_dir):
        """Test chart creation with MACD indicator."""
        output_path = temp_output_dir / "test_chart_macd.png"
        indicators = {
            'MACD': {'fast': 12, 'slow': 26, 'signal': 9}
        }
        result_path = chart_generator.create_chart(
            df=sample_ohlcv_data,
            symbol="BTC/USDT",
            timeframe="1h",
            indicators=indicators,
            output_path=str(output_path)
        )
        
        assert os.path.exists(result_path)
    
    def test_create_chart_with_bollinger_bands(self, chart_generator, sample_ohlcv_data, temp_output_dir):
        """Test chart creation with Bollinger Bands."""
        output_path = temp_output_dir / "test_chart_bb.png"
        indicators = {
            'BB': {'period': 20, 'std': 2}
        }
        result_path = chart_generator.create_chart(
            df=sample_ohlcv_data,
            symbol="BTC/USDT",
            timeframe="1h",
            indicators=indicators,
            output_path=str(output_path)
        )
        
        assert os.path.exists(result_path)
        plt.close('all')
    
    def test_create_chart_all_indicators(self, chart_generator, sample_ohlcv_data, temp_output_dir):
        """Test chart creation with all indicators."""
        output_path = temp_output_dir / "test_chart_all.png"
        indicators = {
            'MA': {'periods': [20, 50, 200]},
            'RSI': {'period': 14, 'overbought': 70, 'oversold': 30},
            'MACD': {'fast': 12, 'slow': 26, 'signal': 9},
            'BB': {'period': 20, 'std': 2}
        }
        result_path = chart_generator.create_chart(
            df=sample_ohlcv_data,
            symbol="BTC/USDT",
            timeframe="1h",
            indicators=indicators,
            output_path=str(output_path)
        )
        
        assert os.path.exists(result_path)
        plt.close('all')
    
    def test_create_chart_auto_output_path(self, chart_generator, sample_ohlcv_data, tmp_path):
        """Test chart creation with automatic output path generation."""
        # Set charts directory
        charts_dir = tmp_path / "charts"
        charts_dir.mkdir()
        
        # Change to temp directory temporarily
        original_cwd = os.getcwd()
        try:
            os.chdir(str(tmp_path))
            result_path = chart_generator.create_chart(
                df=sample_ohlcv_data,
                symbol="BTC/USDT",
                timeframe="1h"
            )
            
            assert os.path.exists(result_path)
            assert "charts" in result_path
            assert "BTC_USDT" in result_path
            assert "1h" in result_path
            plt.close('all')
        finally:
            os.chdir(original_cwd)
    
    def test_create_chart_no_volume(self, chart_generator, sample_ohlcv_data, temp_output_dir):
        """Test chart creation without volume chart."""
        output_path = temp_output_dir / "test_chart_no_volume.png"
        result_path = chart_generator.create_chart(
            df=sample_ohlcv_data,
            symbol="BTC/USDT",
            timeframe="1h",
            show_volume=False,
            output_path=str(output_path)
        )
        
        assert os.path.exists(result_path)
        plt.close('all')
    
    def test_create_chart_empty_dataframe(self, chart_generator, temp_output_dir):
        """Test chart creation with empty DataFrame raises error."""
        empty_df = pd.DataFrame()
        output_path = temp_output_dir / "test_chart_empty.png"
        
        with pytest.raises(ValueError, match="DataFrame rỗng"):
            chart_generator.create_chart(
                df=empty_df,
                symbol="BTC/USDT",
                timeframe="1h",
                output_path=str(output_path)
            )
    
    def test_create_chart_missing_columns(self, chart_generator, temp_output_dir):
        """Test chart creation with missing required columns."""
        df = pd.DataFrame({'open': [1, 2, 3]})  # Missing high, low, close
        output_path = temp_output_dir / "test_chart_missing.png"
        
        with pytest.raises(ValueError, match="Thiếu các cột"):
            chart_generator.create_chart(
                df=df,
                symbol="BTC/USDT",
                timeframe="1h",
                output_path=str(output_path)
            )
    
    def test_create_chart_no_datetime_index(self, chart_generator, temp_output_dir):
        """Test chart creation with DataFrame without DatetimeIndex."""
        df = pd.DataFrame({
            'open': [1, 2, 3],
            'high': [1.1, 2.1, 3.1],
            'low': [0.9, 1.9, 2.9],
            'close': [1.05, 2.05, 3.05]
        })
        output_path = temp_output_dir / "test_chart_no_datetime.png"
        
        with pytest.raises(ValueError, match="DatetimeIndex"):
            chart_generator.create_chart(
                df=df,
                symbol="BTC/USDT",
                timeframe="1h",
                output_path=str(output_path)
            )
    
    def test_create_chart_with_timestamp_column(self, chart_generator, temp_output_dir):
        """Test chart creation with timestamp column instead of DatetimeIndex."""
        dates = pd.date_range(start='2024-01-01', periods=50, freq='1h')
        n = len(dates)
        
        # Generate open and close first
        base_price = 50000
        open_prices = np.random.rand(n) * 100 + base_price
        close_prices = open_prices + np.random.randn(n) * 50  # Small delta from open
        
        # Generate small deltas for high and low
        delta_high = np.abs(np.random.randn(n) * 20)  # Positive delta for high
        delta_low = np.abs(np.random.randn(n) * 20)   # Positive delta for low
        
        # Set high = max(open, close) + abs(delta_high)
        high_prices = np.maximum(open_prices, close_prices) + delta_high
        
        # Set low = min(open, close) - abs(delta_low)
        low_prices = np.minimum(open_prices, close_prices) - delta_low
        
        # Ensure volume is positive integer
        volume = np.random.randint(1000, 10000, n)
        
        # Ensure all arrays have the same length and types are preserved
        assert len(open_prices) == len(close_prices) == len(high_prices) == len(low_prices) == len(volume) == n
        
        df = pd.DataFrame({
            'timestamp': dates,
            'open': open_prices,
            'high': high_prices,
            'low': low_prices,
            'close': close_prices,
            'volume': volume
        })
        
        output_path = temp_output_dir / "test_chart_timestamp.png"
        result_path = chart_generator.create_chart(
            df=df,
            symbol="BTC/USDT",
            timeframe="1h",
            output_path=str(output_path)
        )
        
        assert os.path.exists(result_path)


class TestChartGeneratorIndicators:
    """Test indicator calculations."""
    
    def test_add_indicators_ma(self, chart_generator, sample_ohlcv_data):
        """Test Moving Average calculation."""
        indicators = {'MA': {'periods': [20, 50]}}
        df_with_indicators = chart_generator._add_indicators(sample_ohlcv_data.copy(), indicators)
        
        assert 'MA_20' in df_with_indicators.columns
        assert 'MA_50' in df_with_indicators.columns
        assert not df_with_indicators['MA_20'].iloc[:19].notna().any()  # First 19 should be NaN
        assert df_with_indicators['MA_20'].iloc[19:].notna().all()  # All values after warm-up should be computed
        assert not df_with_indicators['MA_50'].iloc[:49].notna().any()  # First 49 should be NaN
        assert df_with_indicators['MA_50'].iloc[49:].notna().all()  # All values after warm-up should be computed
    
    def test_add_indicators_rsi(self, chart_generator, sample_ohlcv_data):
        """Test RSI calculation."""
        indicators = {'RSI': {'period': 14}}
        df_with_indicators = chart_generator._add_indicators(sample_ohlcv_data.copy(), indicators)
        
        assert 'RSI_14' in df_with_indicators.columns
        # RSI should be between 0 and 100
        rsi_values = df_with_indicators['RSI_14'].dropna()
        assert (rsi_values >= 0).all()
        assert (rsi_values <= 100).all()
    
    def test_add_indicators_macd(self, chart_generator, sample_ohlcv_data):
        """Test MACD calculation."""
        indicators = {'MACD': {'fast': 12, 'slow': 26, 'signal': 9}}
        df_with_indicators = chart_generator._add_indicators(sample_ohlcv_data.copy(), indicators)
        
        assert 'MACD' in df_with_indicators.columns
        assert 'MACD_signal' in df_with_indicators.columns
        assert 'MACD_hist' in df_with_indicators.columns
    
    def test_add_indicators_bollinger_bands(self, chart_generator, sample_ohlcv_data):
        """Test Bollinger Bands calculation."""
        indicators = {'BB': {'period': 20, 'std': 2}}
        df_with_indicators = chart_generator._add_indicators(sample_ohlcv_data.copy(), indicators)
        
        assert 'BB_upper_20' in df_with_indicators.columns
        assert 'BB_lower_20' in df_with_indicators.columns
        assert 'BB_middle_20' in df_with_indicators.columns
        
        # Upper band should be >= middle >= lower band
        valid_rows = df_with_indicators[['BB_upper_20', 'BB_middle_20', 'BB_lower_20']].dropna()
        assert (valid_rows['BB_upper_20'] >= valid_rows['BB_middle_20']).all()
        assert (valid_rows['BB_middle_20'] >= valid_rows['BB_lower_20']).all()
    
    def test_add_indicators_empty_dict(self, chart_generator, sample_ohlcv_data):
        """Test adding indicators with empty indicators dict."""
        df_with_indicators = chart_generator._add_indicators(sample_ohlcv_data.copy(), {})
        
        # Should return original dataframe with no new columns
        assert len(df_with_indicators.columns) == len(sample_ohlcv_data.columns)
    
    def test_add_indicators_none(self, chart_generator, sample_ohlcv_data):
        """Test adding indicators with None indicators."""
        # None should be treated as empty dict
        df_with_indicators = chart_generator._add_indicators(sample_ohlcv_data.copy(), None)
        
        # Should return original dataframe
        assert len(df_with_indicators.columns) == len(sample_ohlcv_data.columns)

class TestChartGeneratorSubplots:
    """Test subplot creation."""
    
    def test_create_subplots_with_volume(self, chart_generator):
        """Test subplot creation with volume."""
        indicators = {'RSI': {'period': 14}, 'MACD': {'fast': 12, 'slow': 26, 'signal': 9}}
        fig, axes, total_rows = chart_generator._create_subplots(indicators=indicators, show_volume=True)
        
        assert len(axes) == 4  # Price + Volume + RSI + MACD
        assert total_rows == 4
        assert fig is not None
        plt.close(fig)
    
    def test_create_subplots_without_volume(self, chart_generator):
        """Test subplot creation without volume."""
        indicators = {'RSI': {'period': 14}, 'MACD': {'fast': 12, 'slow': 26, 'signal': 9}}
        fig, axes, total_rows = chart_generator._create_subplots(indicators=indicators, show_volume=False)
        
        assert len(axes) == 3  # Price + RSI + MACD
        assert total_rows == 3
        assert fig is not None
        plt.close(fig)
    
    def test_create_subplots_no_indicators_none_with_volume(self, chart_generator):
        """Test subplot creation with None indicators and volume."""
        fig, axes, total_rows = chart_generator._create_subplots(indicators=None, show_volume=True)
        
        assert len(axes) == 2  # Price + Volume
        assert total_rows == 2
        assert fig is not None
        plt.close(fig)
    
    def test_create_subplots_no_indicators_none_without_volume(self, chart_generator):
        """Test subplot creation with None indicators without volume."""
        fig, axes, total_rows = chart_generator._create_subplots(indicators=None, show_volume=False)
        
        assert len(axes) == 1  # Price only
        assert total_rows == 1
        assert fig is not None
        plt.close(fig)
    
    def test_create_subplots_no_indicators_empty_dict_with_volume(self, chart_generator):
        """Test subplot creation with empty indicators dict and volume."""
        fig, axes, total_rows = chart_generator._create_subplots(indicators={}, show_volume=True)
        
        assert len(axes) == 2  # Price + Volume
        assert total_rows == 2
        assert fig is not None
        plt.close(fig)
    
    def test_create_subplots_no_indicators_empty_dict_without_volume(self, chart_generator):
        """Test subplot creation with empty indicators dict without volume."""
        fig, axes, total_rows = chart_generator._create_subplots(indicators={}, show_volume=False)
        
        assert len(axes) == 1  # Price only
        assert total_rows == 1
        assert fig is not None
        plt.close(fig)
    
    def test_create_subplots_rsi_only_with_volume(self, chart_generator):
        """Test subplot creation with RSI only and volume."""
        indicators = {'RSI': {'period': 14}}
        fig, axes, total_rows = chart_generator._create_subplots(indicators=indicators, show_volume=True)
        
        assert len(axes) == 3  # Price + Volume + RSI
        assert total_rows == 3
        assert fig is not None
        plt.close(fig)
    
    def test_create_subplots_rsi_only_without_volume(self, chart_generator):
        """Test subplot creation with RSI only without volume."""
        indicators = {'RSI': {'period': 14}}
        fig, axes, total_rows = chart_generator._create_subplots(indicators=indicators, show_volume=False)
        
        assert len(axes) == 2  # Price + RSI
        assert total_rows == 2
        assert fig is not None
        plt.close(fig)
    
    def test_create_subplots_ma_only_with_volume(self, chart_generator):
        """Test subplot creation with MA only and volume (MA doesn't add subplot)."""
        indicators = {'MA': {'periods': [20, 50]}}
        fig, axes, total_rows = chart_generator._create_subplots(indicators=indicators, show_volume=True)
        
        assert len(axes) == 2  # Price + Volume (MA plotted on price chart)
        assert total_rows == 2
        assert fig is not None
        plt.close(fig)
    
    def test_create_subplots_ma_only_without_volume(self, chart_generator):
        """Test subplot creation with MA only without volume (MA doesn't add subplot)."""
        indicators = {'MA': {'periods': [20, 50]}}
        fig, axes, total_rows = chart_generator._create_subplots(indicators=indicators, show_volume=False)
        
        assert len(axes) == 1  # Price only (MA plotted on price chart)
        assert total_rows == 1
        assert fig is not None
        plt.close(fig)
    
    def test_create_subplots_bb_only_with_volume(self, chart_generator):
        """Test subplot creation with BB only and volume (BB doesn't add subplot)."""
        indicators = {'BB': {'period': 20, 'std': 2}}
        fig, axes, total_rows = chart_generator._create_subplots(indicators=indicators, show_volume=True)
        
        assert len(axes) == 2  # Price + Volume (BB plotted on price chart)
        assert total_rows == 2
        assert fig is not None
        plt.close(fig)
    
    def test_create_subplots_bb_only_without_volume(self, chart_generator):
        """Test subplot creation with BB only without volume (BB doesn't add subplot)."""
        indicators = {'BB': {'period': 20, 'std': 2}}
        fig, axes, total_rows = chart_generator._create_subplots(indicators=indicators, show_volume=False)
        
        assert len(axes) == 1  # Price only (BB plotted on price chart)
        assert total_rows == 1
        assert fig is not None
        plt.close(fig)
    
    def test_create_subplots_non_plotting_indicators_only_with_volume(self, chart_generator):
        """Test subplot creation with only non-plotting indicators (MA+BB) and volume."""
        indicators = {'MA': {'periods': [20]}, 'BB': {'period': 20, 'std': 2}}
        fig, axes, total_rows = chart_generator._create_subplots(indicators=indicators, show_volume=True)
        
        assert len(axes) == 2  # Price + Volume (MA and BB plotted on price chart)
        assert total_rows == 2
        assert fig is not None
        plt.close(fig)
    
    def test_create_subplots_non_plotting_indicators_only_without_volume(self, chart_generator):
        """Test subplot creation with only non-plotting indicators (MA+BB) without volume."""
        indicators = {'MA': {'periods': [20]}, 'BB': {'period': 20, 'std': 2}}
        fig, axes, total_rows = chart_generator._create_subplots(indicators=indicators, show_volume=False)
        
        assert len(axes) == 1  # Price only (MA and BB plotted on price chart)
        assert total_rows == 1
        assert fig is not None
        plt.close(fig)

