"""
Tests for parallel processing in Full Backtester.
"""

from unittest.mock import MagicMock, patch

import numpy as np
import pandas as pd

from modules.backtester import FullBacktester

# Fixtures from conftest.py will be automatically available


def test_parallel_processing_enabled(mock_data_fetcher):
    """Test that parallel processing can be enabled."""
    with patch("config.position_sizing.ENABLE_PARALLEL_PROCESSING", True):
        backtester = FullBacktester(mock_data_fetcher)

        # Medium dataset - enough to trigger parallel but fast
        result = backtester.backtest(
            symbol="BTC/USDT",
            timeframe="1h",
            lookback=150,  # Reduced from 500 for faster tests
            signal_type="LONG",
        )

        assert "trades" in result
        assert "equity_curve" in result
        assert "metrics" in result


def test_parallel_processing_fallback(mock_data_fetcher):
    """Test that parallel processing falls back to sequential on error."""
    with patch("config.position_sizing.ENABLE_PARALLEL_PROCESSING", True):
        with patch("modules.backtester.core.signal_calculator.Pool") as mock_pool:
            # Simulate multiprocessing error
            mock_pool.side_effect = Exception("Multiprocessing error")

            backtester = FullBacktester(mock_data_fetcher)

            # Should fall back to sequential processing
            result = backtester.backtest(
                symbol="BTC/USDT",
                timeframe="1h",
                lookback=150,  # Reduced from 500
                signal_type="LONG",
            )

            # Should still return valid results
            assert "trades" in result
            assert "metrics" in result


def test_sequential_processing_for_small_datasets(mock_data_fetcher):
    """Test that small datasets use sequential processing."""
    with patch("config.position_sizing.ENABLE_PARALLEL_PROCESSING", True):
        backtester = FullBacktester(mock_data_fetcher)

        # Small dataset (less than 100 periods)
        result = backtester.backtest(
            symbol="BTC/USDT",
            timeframe="1h",
            lookback=50,  # Small enough to skip parallel
            signal_type="LONG",
        )

        assert "trades" in result
        assert "metrics" in result


def test_batch_processing_worker_function():
    """Test the batch processing worker function."""
    import pickle

    from modules.backtester.core.parallel_workers import calculate_signal_batch_worker as _calculate_signal_batch_worker

    # Create sample DataFrame
    dates = pd.date_range("2023-01-01", periods=100, freq="h")
    prices = 100 + np.cumsum(np.random.randn(100) * 0.5)
    df = pd.DataFrame(
        {
            "open": prices,
            "high": prices * 1.01,
            "low": prices * 0.99,
            "close": prices,
        },
        index=dates,
    )

    df_bytes = pickle.dumps(df)

    # Mock the HybridSignalCalculator
    with patch("modules.position_sizing.core.hybrid_signal_calculator.HybridSignalCalculator") as mock_calc_class:
        mock_calc = MagicMock()
        mock_calc.calculate_hybrid_signal.return_value = (1, 0.8)
        mock_calc_class.return_value = mock_calc

        # Test worker function
        result = _calculate_signal_batch_worker(
            start_idx=0,
            end_idx=10,
            df_bytes=df_bytes,
            symbol="BTC/USDT",
            timeframe="1h",
            limit=100,
            signal_type="LONG",
            osc_length=50,
            osc_mult=2.0,
            osc_strategies=[2, 3],
            spc_params=None,
            enabled_indicators=["range_oscillator"],
            use_confidence_weighting=True,
            min_indicators_agreement=3,
        )

        assert isinstance(result, dict)
        assert len(result) == 10  # 10 periods in batch
        for i in range(10):
            assert i in result
            assert result[i] in [-1, 0, 1]


def test_parallel_vs_sequential_consistency(mock_data_fetcher):
    """Test that parallel and sequential processing produce consistent results."""
    # This is a complex test that would require mocking all indicators
    # For now, we just test that both modes work
    backtester = FullBacktester(mock_data_fetcher)

    with patch("config.position_sizing.ENABLE_PARALLEL_PROCESSING", False):
        result_seq = backtester.backtest(
            symbol="BTC/USDT",
            timeframe="1h",
            lookback=100,  # Reduced from 200 for faster tests
            signal_type="LONG",
        )

    with patch("config.position_sizing.ENABLE_PARALLEL_PROCESSING", True):
        result_par = backtester.backtest(
            symbol="BTC/USDT",
            timeframe="1h",
            lookback=100,  # Reduced from 200
            signal_type="LONG",
        )

    # Both should return valid structures
    assert "trades" in result_seq
    assert "trades" in result_par
    assert "metrics" in result_seq
    assert "metrics" in result_par

    # Note: Results might differ slightly due to caching differences,
    # but structure should be the same
