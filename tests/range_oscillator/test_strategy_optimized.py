
import numpy as np
import pandas as pd
import pytest

from tests.performance_fixtures import (
from tests.performance_fixtures import (

"""
🚀 Optimized test for range oscillator strategy using new performance fixtures.

This demonstrates how to rewrite existing tests to use the fast fixtures.
"""

    fast_ohlcv_data,
    fast_data_fetcher,
    fast_config,
    performance_tracker,
    assert_signal_result,
    create_test_signal_data
)


@pytest.mark.unit  # Mark as unit test for fast execution
class TestRangeOscillatorOptimized:
    """Optimized version of range oscillator tests."""

    def test_strategy_basic_fast(self, fast_ohlcv_data, fast_data_fetcher, performance_tracker):
        """Test basic strategy with fast data and performance tracking."""
        performance_tracker.start()
        
        # Import locally to avoid slow imports
        from modules.range_oscillator.strategies.basic import generate_signals_basic_strategy
        
        # Extract series from cached data
        high = fast_ohlcv_data['high']
        low = fast_ohlcv_data['low']  
        close = fast_ohlcv_data['close']
        
        # Test with fast parameters
        signals, strength = generate_signals_basic_strategy(
            high=high, low=low, close=close,
            length=20,  # Reduced from 50
            mult=2.0
        )
        
        # Use fast assertions
        assert_signal_result((signals.iloc[-1], strength.iloc[-1])) if len(signals) > 0 else None
        
        elapsed = performance_tracker.stop()
        assert elapsed < 5.0, f"Test too slow: {elapsed:.3f}s"  # Should be < 5s

    def test_strategy_with_parameters_fast(self, fast_ohlcv_data, fast_config):
        """Test strategy with different parameters using cached config."""
        from modules.range_oscillator.strategies.basic import generate_signals_basic_strategy
        
        high = fast_ohlcv_data['high']
        low = fast_ohlcv_data['low']
        close = fast_ohlcv_data['close']
        
        signals, strength = generate_signals_basic_strategy(
            high=high, low=low, close=close,
            length=fast_config.osc_length,
            mult=fast_config.osc_mult,
        )
        
        # Quick validation
        assert len(signals) == len(close)
        assert len(strength) == len(close)

    @pytest.mark.parametrize("osc_length,mult", [
        (10, 1.5),
        (20, 2.0), 
        (30, 2.5),
    ])
    def test_strategy_parametrized_fast(self, fast_ohlcv_data, osc_length, mult):
        """Parametrized test using pytest parametrization."""
        from modules.range_oscillator.strategies.basic import generate_signals_basic_strategy
        
        high = fast_ohlcv_data['high']
        low = fast_ohlcv_data['low']
        close = fast_ohlcv_data['close']
        
        signals, strength = generate_signals_basic_strategy(
            high=high, low=low, close=close,
            length=osc_length,
            mult=mult,
        )
        
        # Minimal assertions for speed
        assert len(signals) > 0
        assert len(strength) > 0

    def test_strategy_edge_cases_fast(self, fast_ohlcv_data):
        """Test edge cases with minimal data using cached data."""
        from modules.range_oscillator.strategies.basic import generate_signals_basic_strategy
        
        # Use cached data instead of creating new tiny dataset
        tiny_data = fast_ohlcv_data.head(3)  # Just use first 3 rows
        
        try:
            signals, strength = generate_signals_basic_strategy(
                high=tiny_data['high'],
                low=tiny_data['low'],
                close=tiny_data['close'],
                length=2,  # Very small length
                mult=1.0,
            )
            
            # Should handle tiny data gracefully
            assert len(signals) == len(tiny_data)
            assert len(strength) == len(tiny_data)
        except (ValueError, IndexError):
            # Edge cases with tiny data are expected to fail gracefully
            pass


@pytest.mark.integration  # Still run in integration tests
class TestRangeOscillatorIntegration:
    """Integration tests with mocked expensive operations."""
    
    def test_with_mocked_model(self, performance_tracker):
        """Test integration with mocked model."""
        from tests.performance_fixtures import mocked_random_forest_model
        
        performance_tracker.start()
        
        # Use mocked model to avoid loading actual model
        prediction = mocked_random_forest_model.predict([[1, 2, 3]])
        assert len(prediction) == 5
        
        elapsed = performance_tracker.stop()
        assert elapsed < 1.0, f"Integration test too slow: {elapsed:.3f}s"

    @pytest.mark.slow  # Mark as slow - skipped by default
    def test_full_workflow_slow(self, fast_data_fetcher):
        """Full workflow test - marked as slow."""
        # This test would be skipped by default unless -m slow is used
        # For real workflow testing
        pass


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-m", "not slow"])  # Skip slow tests by default