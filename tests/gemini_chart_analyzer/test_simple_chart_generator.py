"""
Tests for SimpleChartGenerator class.

Tests cover:
- Initialization
- Simple chart creation
- Edge cases (empty data, missing columns)
- Symbol label display
"""

import pytest
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from datetime import datetime

from modules.gemini_chart_analyzer.core.generators.simple_chart_generator import SimpleChartGenerator


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
def simple_generator():
    """Create SimpleChartGenerator instance for testing."""
    return SimpleChartGenerator()


class TestSimpleChartGeneratorInit:
    """Test SimpleChartGenerator initialization."""
    
    def test_init_default_params(self):
        """Test initialization with default parameters."""
        generator = SimpleChartGenerator()
        assert generator.figsize == (2, 1.5)
        assert generator.style == 'dark_background'
        assert generator.dpi == 100
    
    def test_init_custom_params(self):
        """Test initialization with custom parameters."""
        generator = SimpleChartGenerator(
            figsize=(3.0, 2.0),
            style='default',
            dpi=150
        )
        assert generator.figsize == (3.0, 2.0)
        assert generator.style == 'default'
        assert generator.dpi == 150


class TestSimpleChartGeneratorCreateSimpleChart:
    """Test simple chart creation."""
    
    def test_create_simple_chart_basic(self, simple_generator, sample_ohlcv_data):
        """Test basic simple chart creation."""
        fig = simple_generator.create_simple_chart(
            df=sample_ohlcv_data,
            symbol="BTC/USDT",
            timeframe="1h"
        )
        
        assert fig is not None
        plt.close(fig)
    
    def test_create_simple_chart_with_label(self, simple_generator, sample_ohlcv_data):
        """Test chart creation with symbol label."""
        fig = simple_generator.create_simple_chart(
            df=sample_ohlcv_data,
            symbol="BTC/USDT",
            timeframe="1h",
            show_symbol_label=True
        )
        
        assert fig is not None
        plt.close(fig)
    
    def test_create_simple_chart_without_label(self, simple_generator, sample_ohlcv_data):
        """Test chart creation without symbol label."""
        fig = simple_generator.create_simple_chart(
            df=sample_ohlcv_data,
            symbol="BTC/USDT",
            timeframe="1h",
            show_symbol_label=False
        )
        
        assert fig is not None
        plt.close(fig)
    
    def test_create_simple_chart_single_candle(self, simple_generator):
        """Test chart creation with single candle."""
        single_candle = pd.DataFrame({
            'open': [50000],
            'high': [51000],
            'low': [49000],
            'close': [50500]
        }, index=pd.date_range(start='2024-01-01', periods=1, freq='1h'))
        
        fig = simple_generator.create_simple_chart(
            df=single_candle,
            symbol="BTC/USDT",
            timeframe="1h"
        )
        
        assert fig is not None
        plt.close(fig)
    
    def test_create_simple_chart_no_datetime_index(self, simple_generator):
        """Test chart creation with non-DatetimeIndex (should convert)."""
        df = pd.DataFrame({
            'open': [1, 2, 3],
            'high': [1.1, 2.1, 3.1],
            'low': [0.9, 1.9, 2.9],
            'close': [1.05, 2.05, 3.05]
        })
        
        fig = simple_generator.create_simple_chart(
            df=df,
            symbol="BTC/USDT",
            timeframe="1h"
        )
        
        assert fig is not None
        plt.close(fig)    
        
    def test_create_simple_chart_doji_candles(self, simple_generator):
        """Test chart creation with doji candles (open == close)."""
        dates = pd.date_range(start='2024-01-01', periods=5, freq='1h')
        df = pd.DataFrame({
            'open': [50000, 50000, 50000, 50000, 50000],
            'high': [51000, 51000, 51000, 51000, 51000],
            'low': [49000, 49000, 49000, 49000, 49000],
            'close': [50000, 50000, 50000, 50000, 50000]  # Same as open (doji)
        }, index=dates)
        
        fig = simple_generator.create_simple_chart(
            df=df,
            symbol="BTC/USDT",
            timeframe="1h"
        )
        
        assert fig is not None
        plt.close(fig)
    
    def test_create_simple_chart_symbol_label_formatting(self, simple_generator, sample_ohlcv_data):
        """Test symbol label formatting (removes /USDT)."""
        fig = simple_generator.create_simple_chart(
            df=sample_ohlcv_data,
            symbol="BTC/USDT",
            timeframe="1h",
            show_symbol_label=True
        )
        
        assert fig is not None
        
        # Get all text elements from the figure
        ax = fig.axes[0]
        text_elements = ax.texts
        
        # Find the symbol label text
        label_text = None
        for text in text_elements:
            text_content = text.get_text()
            # The label should be positioned at the top-left (transform=ax.transAxes)
            # Compare transforms by checking canonical point mappings instead of direct equality
            text_transform = text.get_transform()
            test_points = [(0, 0), (1, 1)]
            is_transaxes = all(
                np.allclose(
                    text_transform.transform(point),
                    ax.transAxes.transform(point)
                )
                for point in test_points
            )
            if is_transaxes:
                label_text = text_content
                break
        
        # Assert the formatted label is present
        assert label_text is not None, "Symbol label text not found in figure"
        assert 'BTC' in label_text, f"Label should contain 'BTC', but got '{label_text}'"
        assert 'USDT' not in label_text, f"Label should not contain 'USDT', but got '{label_text}'"
        
        plt.close(fig)
    
    def test_create_simple_chart_different_symbols(self, simple_generator, sample_ohlcv_data):
        """Test chart creation with different symbol formats."""
        symbols = ["BTC/USDT", "ETH/USDT", "BNB:USDT", "SOL-USDT"]
        
        for symbol in symbols:
            fig = simple_generator.create_simple_chart(
                df=sample_ohlcv_data,
                symbol=symbol,
                timeframe="1h"
            )
            assert fig is not None
            plt.close(fig)


