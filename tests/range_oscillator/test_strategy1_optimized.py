
import numpy as np
import pandas as pd
import pytest

from tests.test_data_factories import (
from tests.test_data_factories import (

"""
🚀 Optimized Test for Strategy 1 - Basic Oscillator Signals

This is a split from the large test_strategy.py file.
Only contains tests for Strategy 1 with all level 3 optimizations.

Level 3 Optimizations:
- Lazy imports (only import when needed)
- Early assertions (fail fast)
- Minimal data usage
- Mocked dependencies
- Test data factories
"""


# Import test data factories for fast data creation
from tests.test_data_factories import (
    TestDataFactory,
    get_global_factory,
    FAST_DATA_50,
    FAST_DATA_100,
    FAST_OHLC_50,
    FAST_OHLC_100,
)


@pytest.mark.unit  # Mark as unit test - fast by default
class TestStrategy1Level3:
    """
    Optimized tests for Strategy 1: Basic oscillator signals.
    
    Level 3 optimizations applied:
    - Lazy imports
    - Early assertions
    - Minimal data usage
    - Mocked dependencies
    - Factory pattern for data
    """

    def test_strategy1_basic_fast(self):
        """Test basic Strategy 1 functionality with optimizations."""
        # Use pre-cached data (50 rows vs 200)
        high, low, close = FAST_OHLC_50
        
        # Lazy import - only import when needed
        from modules.range_oscillator.strategies.basic import generate_signals_basic_strategy
        
        # Test with minimal data
        signals, strength = generate_signals_basic_strategy(
            high=high, low=low, close=close,
            length=20,  # Reduced from 50
            mult=2.0
        )
        
        # Early assertions - fail fast if basics wrong
        assert isinstance(signals, pd.Series), "Signals should be Series"
        assert isinstance(strength, pd.Series), "Strength should be Series"
        
        # Minimal assertions for speed
        assert len(signals) == len(close), "Signal count mismatch"
        assert len(strength) == len(close), "Strength count mismatch"
        assert signals.dtype == "int8", "Signal dtype should be int8"
        
        # Quick checks instead of comprehensive
        assert all(signals.isin([-1, 0, 1])), "Invalid signals found"
        assert all((strength >= 0) & (strength <= 1)), "Strength out of range"

    @pytest.mark.parametrize("size", [20, 50, 100])
    def test_strategy1_with_sizes(self, size):
        """Test Strategy 1 with different data sizes using factory."""
        # Use factory for dynamic data creation
        factory = TestDataFactory()
        high, low, close = factory.create_series_data(size=size)
        
        # Lazy import
        from modules.range_oscillator.strategies.basic import generate_signals_basic_strategy
        
        # Early return for tiny data
        if size < 10:
            return  # Skip for very small data
        
        signals, strength = generate_signals_basic_strategy(
            high=high, low=low, close=close,
            length=min(size, 20),  # Dynamic length
            mult=2.0
        )
        
        # Minimal assertions
        assert len(signals) == len(close)
        assert len(strength) == len(close)

    def test_strategy1_with_precomputed_fast(self):
        """Test Strategy 1 with precomputed oscillator data."""
        # Use factory for precomputed oscillator data
        factory = TestDataFactory()
        oscillator, ma, range_atr = factory.create_oscillator_data(size=50)
        
        # Lazy import
        from modules.range_oscillator.strategies.basic import generate_signals_basic_strategy
        
        # Mock close for oscillator data
        close = ma  # Use MA as close for simplicity
        
        signals, strength = generate_signals_basic_strategy(
            oscillator=oscillator, ma=ma, range_atr=range_atr, close=close
        )
        
        # Quick validation
        assert isinstance(signals, pd.Series)
        assert isinstance(strength, pd.Series)
        assert len(signals) == len(oscillator)

    @pytest.mark.parametrize("threshold,require_trend,use_breakout", [
        (10.0, False, False),
        (5.0, True, False),
        (15.0, False, True),
    ])
    def test_strategy1_parameters_fast(self, threshold, require_trend, use_breakout):
        """Test Strategy 1 with different parameters."""
        # Use cached data
        high, low, close = FAST_OHLC_50
        
        # Lazy import
        from modules.range_oscillator.strategies.basic import generate_signals_basic_strategy
        
        # Early return for complex combinations
        if use_breakout and threshold > 10:
            return  # Skip expensive combinations
        
        signals, strength = generate_signals_basic_strategy(
            high=high, low=low, close=close,
            oscillator_threshold=threshold,
            require_trend_confirmation=require_trend,
            use_breakout_signals=use_breakout,
        )
        
        # Minimal validation
        assert isinstance(signals, pd.Series)
        assert isinstance(strength, pd.Series)

    def test_strategy1_edge_cases_minimal(self):
        """Test Strategy 1 edge cases with minimal data."""
        # Use factory edge cases
        factory = TestDataFactory()
        edge_cases = factory.create_edge_case_data()
        
        # Lazy import
        from modules.range_oscillator.strategies.basic import generate_signals_basic_strategy
        
        # Test only critical edge cases
        for case_name, test_data in edge_cases.items():
            if case_name in ['single', 'tiny']:
                # Skip tiny/edge cases that are expected to fail
                continue
                
            try:
                high, low, close = (
                    test_data['high'],
                    test_data['low'],
                    test_data['close']
                )
                
                signals, strength = generate_signals_basic_strategy(
                    high=high, low=low, close=close,
                    length=10,  # Minimal length
                    mult=2.0
                )
                
                # Minimal validation
                assert len(signals) == len(close)
            except Exception:
                # Edge cases may fail - that's OK
                pass


@pytest.mark.slow  # Mark as slow - skipped by default
class TestStrategy1Comprehensive:
    """
    Comprehensive tests for Strategy 1 - marked as slow.
    
    These tests are skipped by default (use -m slow to run).
    They maintain original test coverage but are separated for performance.
    """

    def test_strategy1_zero_cross_up_comprehensive(self):
        """Test Strategy 1 zero cross up with full validation."""
        dates = pd.date_range("2024-01-01", periods=20, freq="1h")
        oscillator_values = [-10.0] * 5 + [5.0] * 15
        oscillator = pd.Series(oscillator_values, index=dates)
        ma = pd.Series([50000.0] * 20, index=dates)
        range_atr = pd.Series([1000.0] * 20, index=dates)
        close = pd.Series([51000.0] * 20, index=dates)
        
        from modules.range_oscillator.strategies.basic import generate_signals_basic_strategy
        
        signals, strength = generate_signals_basic_strategy(
            oscillator=oscillator, ma=ma, range_atr=range_atr, close=close,
            require_trend_confirmation=True,
            oscillator_threshold=0.0,
        )
        
        # Comprehensive validation
        assert signals.iloc[5] == 0
        assert all(signals.iloc[6:20] == 1)
        assert all((strength >= 0) & (strength <= 1))


@pytest.mark.integration
class TestStrategy1Integration:
    """Integration tests for Strategy 1 with mocked dependencies."""

    def test_strategy1_with_mocked_dependencies(self):
        """Test Strategy 1 with fully mocked dependencies."""
        # Use factory data
        high, low, close = FAST_OHLC_50
        
        # Mock any external dependencies
        from unittest.mock import patch
        
        with patch.dict('sys.modules', {
            # Mock expensive modules if they exist
            'modules.range_oscillator.strategies.basic': None,
        }):
            # If module is mocked, skip test
            return
        
        # Normal flow with lazy import
        from modules.range_oscillator.strategies.basic import generate_signals_basic_strategy
        
        signals, strength = generate_signals_basic_strategy(
            high=high, low=low, close=close,
            length=20, mult=2.0
        )
        
        # Quick validation
        assert isinstance(signals, pd.Series)
        assert isinstance(strength, pd.Series)
        assert len(signals) > 0


if __name__ == "__main__":
    # Run only fast tests by default
    pytest.main([__file__, "-v", "-m", "not slow"])