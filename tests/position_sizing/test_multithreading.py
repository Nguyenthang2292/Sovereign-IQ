from types import SimpleNamespace
from unittest.mock import patch

import numpy as np
import pandas as pd
import pytest

from modules.position_sizing.core.hybrid_signal_calculator import HybridSignalCalculator

"""
Tests for multithreading in Hybrid Signal Calculator.
"""


@pytest.fixture
def mock_data_fetcher():
    """Create a mock data fetcher for testing."""

    def fake_fetch(symbol, **kwargs):
        dates = pd.date_range("2023-01-01", periods=200, freq="h")
        prices = 100 + np.cumsum(np.random.randn(200) * 0.5)
        df = pd.DataFrame(
            {
                "open": prices,
                "high": prices * 1.01,
                "low": prices * 0.99,
                "close": prices,
            },
            index=dates,
        )
        return df, "binance"

    return SimpleNamespace(
        fetch_ohlcv_with_fallback_exchange=fake_fetch,
    )


@pytest.fixture
def sample_dataframe():
    """Create a sample DataFrame for testing."""
    dates = pd.date_range("2023-01-01", periods=100, freq="h")
    prices = 100 + np.cumsum(np.random.randn(100) * 0.5)
    return pd.DataFrame(
        {
            "open": prices,
            "high": prices * 1.01,
            "low": prices * 0.99,
            "close": prices,
        },
        index=dates,
    )


def test_multithreading_enabled(mock_data_fetcher, sample_dataframe):
    """Test that multithreading can be enabled for indicator calculations."""
    with patch("config.position_sizing.ENABLE_MULTITHREADING", True):
        calculator = HybridSignalCalculator(mock_data_fetcher)

        # Mock indicator functions
        with (
            patch.object(calculator, "_calc_range_oscillator") as mock_osc,
            patch.object(calculator, "_calc_spc") as mock_spc,
            patch.object(calculator, "_calc_hmm") as mock_hmm,
        ):
            mock_osc.return_value = {"indicator": "range_oscillator", "signal": 1, "confidence": 0.8}
            mock_spc.return_value = {"indicator": "spc_cluster", "signal": 1, "confidence": 0.7}
            mock_hmm.return_value = {"indicator": "hmm", "signal": 1, "confidence": 0.6}

            signal, confidence = calculator.calculate_hybrid_signal(
                df=sample_dataframe,
                symbol="BTC/USDT",
                timeframe="1h",
                period_index=50,
                signal_type="LONG",
            )

            assert signal in [-1, 0, 1]
            assert 0.0 <= confidence <= 1.0


def test_multithreading_disabled(mock_data_fetcher, sample_dataframe):
    """Test that sequential processing works when multithreading is disabled."""
    with patch("config.position_sizing.ENABLE_MULTITHREADING", False):
        calculator = HybridSignalCalculator(mock_data_fetcher)

        # Mock indicator functions
        with patch.object(calculator, "_calc_range_oscillator") as mock_osc:
            mock_osc.return_value = {"indicator": "range_oscillator", "signal": 1, "confidence": 0.8}

            signal, confidence = calculator.calculate_hybrid_signal(
                df=sample_dataframe,
                symbol="BTC/USDT",
                timeframe="1h",
                period_index=50,
                signal_type="LONG",
            )

            assert signal in [-1, 0, 1]
            assert 0.0 <= confidence <= 1.0


def test_indicator_caching(mock_data_fetcher):
    """Test that indicator results are cached."""
    calculator = HybridSignalCalculator(mock_data_fetcher)

    # Add a cached indicator result
    cache_key = ("BTC/USDT", 50, "range_oscillator")
    cached_result = {"indicator": "range_oscillator", "signal": 1, "confidence": 0.8}
    calculator._indicator_cache[cache_key] = cached_result

    # Call the indicator calculation method
    result = calculator._calc_range_oscillator(
        symbol="BTC/USDT",
        timeframe="1h",
        limit=100,
        osc_length=50,
        osc_mult=2.0,
        osc_strategies=[2, 3],
        period_index=50,
    )

    # Should return cached result without calling the actual function
    assert result == cached_result


def test_indicator_cache_eviction(mock_data_fetcher):
    """Test that indicator cache evicts old entries when full."""
    calculator = HybridSignalCalculator(mock_data_fetcher)
    calculator._indicator_cache_max_size = 5  # Small cache for testing

    # Fill cache beyond max size using the proper method that handles eviction
    for i in range(10):
        cache_key = ("BTC/USDT", i, "range_oscillator")
        calculator._cache_indicator_result(cache_key, {"signal": 1})

    # Cache should not exceed max size
    assert len(calculator._indicator_cache) <= calculator._indicator_cache_max_size


def test_calc_indicator_methods(mock_data_fetcher):
    """Test individual indicator calculation methods."""
    calculator = HybridSignalCalculator(mock_data_fetcher)

    # Test that methods exist and can be called
    # Patch indicator_calculators module instead of hybrid_signal_calculator
    with patch("modules.position_sizing.core.indicator_calculators.get_range_oscillator_signal") as mock_func:
        mock_func.return_value = (1, 0.8)

        result = calculator._calc_range_oscillator(
            symbol="BTC/USDT",
            timeframe="1h",
            limit=100,
            osc_length=50,
            osc_mult=2.0,
            osc_strategies=[2, 3],
            period_index=50,
        )

        assert result is not None
        assert "indicator" in result
        assert "signal" in result
        assert "confidence" in result


def test_parallel_indicator_calculation_timeout(mock_data_fetcher, sample_dataframe):
    """Test that parallel indicator calculation handles timeouts."""
    with patch("config.position_sizing.ENABLE_MULTITHREADING", True):
        calculator = HybridSignalCalculator(mock_data_fetcher)

        # Mock a slow indicator that times out
        def slow_indicator(*args, **kwargs):
            import time

            time.sleep(2)  # Simulate slow operation
            return {"indicator": "slow", "signal": 1, "confidence": 0.5}

        with patch.object(calculator, "_calc_range_oscillator", side_effect=slow_indicator):
            # Should handle timeout gracefully
            signal, confidence = calculator.calculate_hybrid_signal(
                df=sample_dataframe,
                symbol="BTC/USDT",
                timeframe="1h",
                period_index=50,
                signal_type="LONG",
            )

            # Should still return a valid result (even if some indicators timed out)
            assert signal in [-1, 0, 1]
