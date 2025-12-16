"""
Tests for range_oscillator utils module.

Tests utility functions:
- get_oscillator_data
"""
import numpy as np
import pandas as pd
import pytest

from modules.range_oscillator.utils.oscillator_data import get_oscillator_data


@pytest.fixture
def sample_ohlc_data():
    """Create sample OHLC data for testing."""
    np.random.seed(42)
    n = 200
    dates = pd.date_range('2024-01-01', periods=n, freq='1h')
    
    close = 50000 + np.cumsum(np.random.randn(n) * 100)
    high = close + np.abs(np.random.randn(n) * 50)
    low = close - np.abs(np.random.randn(n) * 50)
    
    return (
        pd.Series(high, index=dates, name='high'),
        pd.Series(low, index=dates, name='low'),
        pd.Series(close, index=dates, name='close')
    )


class TestGetOscillatorData:
    """Tests for get_oscillator_data function."""
    
    def test_get_oscillator_data_with_ohlc(self, sample_ohlc_data):
        """Test get_oscillator_data with OHLC data."""
        high, low, close = sample_ohlc_data
        
        oscillator, ma, range_atr = get_oscillator_data(
            high=high,
            low=low,
            close=close,
            length=50,
            mult=2.0
        )
        
        assert isinstance(oscillator, pd.Series)
        assert isinstance(ma, pd.Series)
        assert isinstance(range_atr, pd.Series)
        assert len(oscillator) == len(close)
        assert len(ma) == len(close)
        assert len(range_atr) == len(close)
        assert oscillator.index.equals(close.index)
        assert ma.index.equals(close.index)
        assert range_atr.index.equals(close.index)
    
    def test_get_oscillator_data_with_precalculated(self, sample_ohlc_data):
        """Test get_oscillator_data with pre-calculated values."""
        high, low, close = sample_ohlc_data
        
        # Create pre-calculated values
        oscillator = pd.Series(np.sin(np.linspace(0, 4*np.pi, len(close))) * 50, index=close.index)
        ma = close.rolling(50).mean()
        range_atr = pd.Series(np.ones(len(close)) * 1000, index=close.index)
        
        result_oscillator, result_ma, result_range_atr = get_oscillator_data(
            oscillator=oscillator,
            ma=ma,
            range_atr=range_atr
        )
        
        # Should return pre-calculated values without modification
        assert result_oscillator is oscillator
        assert result_ma is ma
        assert result_range_atr is range_atr
    
    def test_get_oscillator_data_precalculated_takes_precedence(self, sample_ohlc_data):
        """Test that pre-calculated values take precedence over OHLC."""
        high, low, close = sample_ohlc_data
        
        # Create pre-calculated values
        oscillator = pd.Series([10.0] * len(close), index=close.index)
        ma = pd.Series([100.0] * len(close), index=close.index)
        range_atr = pd.Series([1000.0] * len(close), index=close.index)
        
        result_oscillator, result_ma, result_range_atr = get_oscillator_data(
            high=high,  # Provided but should be ignored
            low=low,    # Provided but should be ignored
            close=close,  # Provided but should be ignored
            oscillator=oscillator,  # Should be used
            ma=ma,      # Should be used
            range_atr=range_atr  # Should be used
        )
        
        # Should use pre-calculated values
        assert result_oscillator is oscillator
        assert result_ma is ma
        assert result_range_atr is range_atr
    
    def test_get_oscillator_data_different_parameters(self, sample_ohlc_data):
        """Test get_oscillator_data with different parameters."""
        high, low, close = sample_ohlc_data
        
        for length in [20, 50, 100]:
            for mult in [1.5, 2.0, 3.0]:
                oscillator, ma, range_atr = get_oscillator_data(
                    high=high,
                    low=low,
                    close=close,
                    length=length,
                    mult=mult
                )
                
                assert isinstance(oscillator, pd.Series)
                assert isinstance(ma, pd.Series)
                assert isinstance(range_atr, pd.Series)
                assert len(oscillator) == len(close)
    
    def test_get_oscillator_data_missing_required(self):
        """Test get_oscillator_data with missing required parameters."""
        dates = pd.date_range('2024-01-01', periods=10, freq='1h')
        close = pd.Series([100.0] * 10, index=dates)
        
        # Missing both pre-calculated and OHLC
        with pytest.raises(ValueError, match="Either provide"):
            get_oscillator_data(close=close)  # Missing high and low
        
        # Missing high
        with pytest.raises(ValueError, match="Either provide"):
            get_oscillator_data(low=close, close=close)
        
        # Missing low
        with pytest.raises(ValueError, match="Either provide"):
            get_oscillator_data(high=close, close=close)
        
        # Missing close
        with pytest.raises(ValueError, match="Either provide"):
            get_oscillator_data(high=close, low=close)
    
    def test_get_oscillator_data_partial_precalculated(self):
        """Test get_oscillator_data with partial pre-calculated values."""
        dates = pd.date_range('2024-01-01', periods=10, freq='1h')
        close = pd.Series([100.0] * 10, index=dates)
        high = close + 1
        low = close - 1
        
        oscillator = pd.Series([10.0] * 10, index=dates)
        
        # Only oscillator provided, missing ma and range_atr
        # Should fall back to calculation
        result_oscillator, result_ma, result_range_atr = get_oscillator_data(
            high=high,
            low=low,
            close=close,
            oscillator=oscillator  # Provided but ma and range_atr missing
        )
        
        # Should calculate ma and range_atr
        assert isinstance(result_ma, pd.Series)
        assert isinstance(result_range_atr, pd.Series)
        # Oscillator might be recalculated or used, depending on implementation
        assert isinstance(result_oscillator, pd.Series)
    
    def test_get_oscillator_data_empty_series(self):
        """Test get_oscillator_data with empty series."""
        empty_series = pd.Series([], dtype=float)
        
        # Should raise error or handle gracefully
        with pytest.raises((ValueError, IndexError)):
            get_oscillator_data(
                high=empty_series,
                low=empty_series,
                close=empty_series
            )
    
    def test_get_oscillator_data_index_mismatch(self):
        """Test get_oscillator_data with index mismatch in pre-calculated values."""
        dates1 = pd.date_range('2024-01-01', periods=10, freq='1h')
        dates2 = pd.date_range('2024-01-02', periods=10, freq='1h')
        
        oscillator = pd.Series([10.0] * 10, index=dates1)
        ma = pd.Series([100.0] * 10, index=dates2)  # Different index
        range_atr = pd.Series([1000.0] * 10, index=dates1)
        
        # Should raise ValueError due to index mismatch (validation added)
        with pytest.raises(ValueError, match="must have the same index"):
            get_oscillator_data(
                oscillator=oscillator,
                ma=ma,
                range_atr=range_atr
            )

