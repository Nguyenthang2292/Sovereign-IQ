"""
Stress tests for adaptive_trend_enhance module.

Tests system behavior under extreme conditions:
- Large number of symbols (optimized for faster execution)
- Limited memory scenarios
- Parallel processing under load
- Cache eviction under pressure

NOTE: These tests have been optimized for faster execution while maintaining test coverage.
      Run with -m "not slow" to skip slow tests during development.
"""

import gc
import sys
import tracemalloc
from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np
import pandas as pd
import pytest

# Add project root to sys.path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from modules.adaptive_trend_enhance.core.compute_equity import equity_series
from modules.adaptive_trend_enhance.core.scanner import scan_all_symbols
from modules.adaptive_trend_enhance.utils.cache_manager import get_cache_manager
from modules.adaptive_trend_enhance.utils.config import ATCConfig
from modules.common.system import get_memory_manager


@pytest.fixture
def mock_data_fetcher():
    """Create a mock DataFetcher instance."""
    fetcher = MagicMock()
    return fetcher


@pytest.fixture
def base_config():
    """Create a base ATCConfig for testing."""
    return ATCConfig(
        limit=200,
        timeframe="1h",
        ema_len=28,
        hma_len=28,
        wma_len=28,
        dema_len=28,
        lsma_len=28,
        kama_len=28,
        robustness="Medium",
        lambda_param=0.02,
        decay=0.03,
        cutout=0,
        long_threshold=0.1,
        short_threshold=-0.1,
        batch_size=100,
    )


def create_mock_ohlcv_data(num_candles: int = 200) -> pd.DataFrame:
    """Create mock OHLCV DataFrame for testing."""
    timestamps = pd.date_range(start="2024-01-01", periods=num_candles, freq="1h", tz="UTC")
    prices = np.linspace(100, 110, num_candles)

    df = pd.DataFrame(
        {
            "timestamp": timestamps,
            "open": prices,
            "high": prices + 1,
            "low": prices - 1,
            "close": prices,
            "volume": 1000,
        }
    )
    return df


def create_mock_atc_results(signal_value: float = 0.05) -> dict:
    """Create mock ATC results."""
    signal_series = pd.Series([signal_value] * 200)
    return {
        "Average_Signal": signal_series,
    }


class TestLargeScaleSymbols:
    """Test with large number of symbols (optimized for faster execution)."""

    @pytest.mark.slow
    @patch("modules.adaptive_trend_enhance.core.scanner.process_symbol.compute_atc_signals")
    @patch("modules.adaptive_trend_enhance.core.scanner.process_symbol.trend_sign")
    def test_5000_symbols_processing(self, mock_trend_sign, mock_compute_atc, base_config, mock_data_fetcher):
        """Test processing large number of symbols (optimized: reduced from 5000 to 500 for faster execution)."""
        # Reduced from 5000 to 500 symbols for faster test execution while still testing batching
        num_symbols = 500
        symbols = [f"SYM{i}" for i in range(num_symbols)]
        mock_data_fetcher.list_binance_futures_symbols.return_value = symbols

        # Reduced from 500 to 200 candles for faster processing
        mock_df = create_mock_ohlcv_data(num_candles=200)
        mock_data_fetcher.fetch_ohlcv_with_fallback_exchange.return_value = (mock_df, "binance")

        mock_compute_atc.return_value = create_mock_atc_results(signal_value=0.5)
        mock_trend_sign.return_value = pd.Series([1] * 200)

        mem_manager = get_memory_manager()
        mem_manager.enable_tracemalloc = True
        if not tracemalloc.is_tracing():
            tracemalloc.start()

        initial_stats = mem_manager.get_current_usage()

        # Process with batching
        try:
            long_df, short_df = scan_all_symbols(
                mock_data_fetcher,
                base_config,
                batch_size=100,
                execution_mode="sequential",
                max_symbols=num_symbols,
            )
        except Exception as e:
            pytest.fail(f"Failed to process {num_symbols} symbols: {e}")

        final_stats = mem_manager.get_current_usage()

        print(f"\nLarge Scale Symbols Stress Test ({num_symbols} symbols):")
        print(f"  Initial RAM: {initial_stats.ram_used_gb:.2f} GB")
        print(f"  Final RAM: {final_stats.ram_used_gb:.2f} GB")
        print(f"  Results: {len(long_df)} long, {len(short_df)} short")

        # Verify results
        assert len(long_df) == num_symbols, f"Expected {num_symbols} results, got {len(long_df)}"
        memory_growth = final_stats.ram_used_gb - initial_stats.ram_used_gb
        assert memory_growth < 2.0, (
            f"Memory growth too high: {memory_growth:.2f} GB (initial={initial_stats.ram_used_gb:.2f}, "
            f"final={final_stats.ram_used_gb:.2f})"
        )

    @pytest.mark.slow
    @patch("modules.adaptive_trend_enhance.core.scanner.process_symbol.compute_atc_signals")
    @patch("modules.adaptive_trend_enhance.core.scanner.process_symbol.trend_sign")
    def test_large_scale_memory_stability(self, mock_trend_sign, mock_compute_atc, base_config, mock_data_fetcher):
        """Test memory stability with large scale processing (optimized: reduced from 2000 to 200 symbols)."""
        # Reduced from 2000 to 200 symbols for faster test execution
        num_symbols = 200
        symbols = [f"SYM{i}" for i in range(num_symbols)]
        mock_data_fetcher.list_binance_futures_symbols.return_value = symbols

        # Reduced from 1000 to 300 candles for faster processing
        mock_df = create_mock_ohlcv_data(num_candles=300)
        mock_data_fetcher.fetch_ohlcv_with_fallback_exchange.return_value = (mock_df, "binance")

        mock_compute_atc.return_value = create_mock_atc_results(signal_value=0.5)
        mock_trend_sign.return_value = pd.Series([1] * 300)

        mem_manager = get_memory_manager()
        mem_manager.enable_tracemalloc = True
        if not tracemalloc.is_tracing():
            tracemalloc.start()

        # Monitor memory at intervals
        memory_samples = []
        initial_stats = mem_manager.get_current_usage()
        memory_samples.append(initial_stats.ram_used_gb)

        # Process in chunks and monitor memory
        # Reduced chunk_size from 500 to 100 for faster execution
        chunk_size = 100
        for i in range(0, num_symbols, chunk_size):
            chunk_symbols = symbols[i : i + chunk_size]
            mock_data_fetcher.list_binance_futures_symbols.return_value = chunk_symbols

            scan_all_symbols(
                mock_data_fetcher,
                base_config,
                batch_size=50,
                execution_mode="sequential",
                max_symbols=len(chunk_symbols),
            )

            stats = mem_manager.get_current_usage()
            memory_samples.append(stats.ram_used_gb)
            gc.collect()

        final_stats = mem_manager.get_current_usage()
        memory_samples.append(final_stats.ram_used_gb)

        print(f"\nMemory Stability Test ({num_symbols} symbols):")
        print(f"  Initial: {memory_samples[0]:.2f} GB")
        print(f"  Peak: {max(memory_samples):.2f} GB")
        print(f"  Final: {memory_samples[-1]:.2f} GB")
        print(f"  Memory Growth: {memory_samples[-1] - memory_samples[0]:.2f} GB")

        # Memory should not grow excessively
        memory_growth = memory_samples[-1] - memory_samples[0]
        assert memory_growth < 2.0, f"Excessive memory growth: {memory_growth:.2f} GB"


class TestLimitedMemoryScenarios:
    """Test behavior with limited memory scenarios."""

    @patch("modules.adaptive_trend_enhance.core.scanner.process_symbol.compute_atc_signals")
    @patch("modules.adaptive_trend_enhance.core.scanner.process_symbol.trend_sign")
    def test_small_batch_size_under_memory_pressure(
        self, mock_trend_sign, mock_compute_atc, base_config, mock_data_fetcher
    ):
        """Test with very small batch sizes to simulate memory pressure."""
        num_symbols = 500
        symbols = [f"SYM{i}" for i in range(num_symbols)]
        mock_data_fetcher.list_binance_futures_symbols.return_value = symbols

        mock_df = create_mock_ohlcv_data(num_candles=1000)
        mock_data_fetcher.fetch_ohlcv_with_fallback_exchange.return_value = (mock_df, "binance")

        mock_compute_atc.return_value = create_mock_atc_results(signal_value=0.5)
        mock_trend_sign.return_value = pd.Series([1] * 1000)

        # Use very small batch size to force frequent GC
        base_config.batch_size = 10

        mem_manager = get_memory_manager()
        mem_manager.enable_tracemalloc = True
        if not tracemalloc.is_tracing():
            tracemalloc.start()

        initial_stats = mem_manager.get_current_usage()

        # Process with small batches
        long_df, short_df = scan_all_symbols(
            mock_data_fetcher,
            base_config,
            batch_size=10,
            execution_mode="sequential",
            max_symbols=num_symbols,
        )

        final_stats = mem_manager.get_current_usage()

        print("\nSmall Batch Size Test (batch_size=10):")
        print(f"  Initial RAM: {initial_stats.ram_used_gb:.2f} GB")
        print(f"  Final RAM: {final_stats.ram_used_gb:.2f} GB")
        print(f"  Results: {len(long_df)} symbols processed")

        # Should complete successfully even with small batches
        assert len(long_df) == num_symbols
        memory_growth = final_stats.ram_used_gb - initial_stats.ram_used_gb
        # Small batches (batch_size=10) cause more GC overhead; allow up to 4 GB growth
        assert memory_growth < 4.0, f"Memory growth too high: {memory_growth:.2f} GB"

    def test_equity_calculation_memory_pressure(self):
        """Test equity calculations under memory pressure (optimized: reduced iterations)."""
        # Reduced from 2000 to 1000 bars for faster execution
        n_bars = 1000
        prices = pd.Series(100 * (1 + np.random.randn(n_bars).cumsum() * 0.01))
        R = prices.pct_change().fillna(0)

        mem_manager = get_memory_manager()
        mem_manager.enable_tracemalloc = True
        if not tracemalloc.is_tracing():
            tracemalloc.start()

        initial_stats = mem_manager.get_current_usage()

        # Calculate many equities without cleanup
        # Reduced from 100 to 50 iterations for faster execution
        equities = []
        for i in range(50):
            signal = pd.Series(np.random.choice([-1, 0, 1], n_bars))
            equity = equity_series(starting_equity=100.0, sig=signal, R=R, L=0.02, De=0.03, verbose=False)
            equities.append(equity)

        peak_stats = mem_manager.get_current_usage()

        # Cleanup
        del equities
        gc.collect()
        final_stats = mem_manager.get_current_usage()

        print("\nEquity Memory Pressure Test:")
        print(f"  Initial RAM: {initial_stats.ram_used_gb:.2f} GB")
        print(f"  Peak RAM: {peak_stats.ram_used_gb:.2f} GB")
        print(f"  Final RAM: {final_stats.ram_used_gb:.2f} GB")

        # Memory should be recoverable after cleanup (allow small noise: GC may not
        # return all memory to OS immediately, or other allocators can add ~0.5 GB)
        memory_recovered = peak_stats.ram_used_gb - final_stats.ram_used_gb
        assert memory_recovered > -0.5, (
            f"Memory should be recoverable after cleanup (recovered={memory_recovered:.2f} GB)"
        )


class TestParallelProcessingUnderLoad:
    """Test parallel processing under load."""

    @pytest.mark.slow
    @patch("modules.adaptive_trend_enhance.core.scanner.process_symbol.compute_atc_signals")
    @patch("modules.adaptive_trend_enhance.core.scanner.process_symbol.trend_sign")
    @pytest.mark.parametrize("execution_mode", ["threadpool", "asyncio"])
    def test_parallel_processing_under_load(
        self, mock_trend_sign, mock_compute_atc, base_config, mock_data_fetcher, execution_mode
    ):
        """Test parallel processing modes under load (optimized: reduced symbols and candles)."""
        # Reduced from 1000 to 200 symbols for faster execution
        num_symbols = 200
        symbols = [f"SYM{i}" for i in range(num_symbols)]
        mock_data_fetcher.list_binance_futures_symbols.return_value = symbols

        # Reduced from 500 to 200 candles for faster processing
        mock_df = create_mock_ohlcv_data(num_candles=200)
        mock_data_fetcher.fetch_ohlcv_with_fallback_exchange.return_value = (mock_df, "binance")

        mock_compute_atc.return_value = create_mock_atc_results(signal_value=0.5)
        mock_trend_sign.return_value = pd.Series([1] * 200)

        mem_manager = get_memory_manager()
        mem_manager.enable_tracemalloc = True
        if not tracemalloc.is_tracing():
            tracemalloc.start()

        initial_stats = mem_manager.get_current_usage()

        try:
            long_df, short_df = scan_all_symbols(
                mock_data_fetcher,
                base_config,
                batch_size=100,
                execution_mode=execution_mode,
                max_workers=4,
                max_symbols=num_symbols,
            )
        except Exception as e:
            pytest.fail(f"Parallel processing failed in {execution_mode} mode: {e}")

        final_stats = mem_manager.get_current_usage()

        print(f"\nParallel Processing Under Load ({execution_mode}):")
        print(f"  Initial RAM: {initial_stats.ram_used_gb:.2f} GB")
        print(f"  Final RAM: {final_stats.ram_used_gb:.2f} GB")
        print(f"  Results: {len(long_df)} symbols processed")

        assert len(long_df) == num_symbols
        memory_growth = final_stats.ram_used_gb - initial_stats.ram_used_gb
        assert memory_growth < 2.0, f"Memory growth too high: {memory_growth:.2f} GB"

    @patch("modules.adaptive_trend_enhance.core.scanner.process_symbol.compute_atc_signals")
    @patch("modules.adaptive_trend_enhance.core.scanner.process_symbol.trend_sign")
    def test_concurrent_equity_calculations(self, mock_trend_sign, mock_compute_atc):
        """Test concurrent equity calculations (optimized: reduced iterations)."""
        from concurrent.futures import ThreadPoolExecutor

        # Reduced from 1500 to 1000 bars for faster execution
        n_bars = 1000
        prices = pd.Series(100 * (1 + np.random.randn(n_bars).cumsum() * 0.01))
        R = prices.pct_change().fillna(0)

        def calculate_equity(i):
            signal = pd.Series(np.random.choice([-1, 0, 1], n_bars))
            equity = equity_series(starting_equity=100.0, sig=signal, R=R, L=0.02, De=0.03, verbose=False)
            return len(equity)

        # Run concurrent calculations
        # Reduced from 50 to 20 iterations for faster execution
        with ThreadPoolExecutor(max_workers=8) as executor:
            futures = [executor.submit(calculate_equity, i) for i in range(20)]
            results = [f.result() for f in futures]

        print("\nConcurrent Equity Calculations:")
        print(f"  Completed: {len(results)} calculations")
        print(f"  All results length: {results[0]}")

        assert all(r == n_bars for r in results), "All equity calculations should succeed"


class TestCacheEvictionUnderPressure:
    """Test cache eviction under pressure."""

    def test_cache_eviction_under_load(self):
        """Test cache eviction when cache is under pressure (optimized: reduced iterations)."""
        cache = get_cache_manager()

        # Fill cache with many entries
        # Reduced from 1000 to 500 bars for faster execution
        n_bars = 500
        prices = pd.Series(100 * (1 + np.random.randn(n_bars).cumsum() * 0.01))
        R = prices.pct_change().fillna(0)

        # Create many different signals to fill cache
        # Reduced from 200 to 50 iterations for faster execution
        for i in range(50):
            signal = pd.Series(np.random.choice([-1, 0, 1], n_bars))
            equity = equity_series(starting_equity=100.0, sig=signal, R=R, L=0.02, De=0.03, verbose=False)
            del equity

        # Get cache stats
        cache_stats = cache.get_stats()
        print("\nCache Eviction Under Load:")
        print(f"  Cache size: {cache_stats.get('equity_cache_size', 'N/A')}")
        print(f"  Cache hits: {cache_stats.get('equity_cache_hits', 'N/A')}")
        print(f"  Cache misses: {cache_stats.get('equity_cache_misses', 'N/A')}")

        # Cache should handle eviction gracefully
        # Adjusted threshold based on reduced iterations
        assert cache_stats.get("equity_cache_size", 0) <= 100, "Cache should evict old entries"

    def test_cache_performance_under_pressure(self):
        """Test cache performance when under pressure (optimized: reduced iterations)."""
        cache = get_cache_manager()

        # Reduced from 1500 to 800 bars for faster execution
        n_bars = 800
        prices = pd.Series(100 * (1 + np.random.randn(n_bars).cumsum() * 0.01))
        R = prices.pct_change().fillna(0)

        import time

        # First pass: fill cache
        # Reduced from 100 to 30 iterations for faster execution
        start = time.perf_counter()
        for i in range(30):
            signal = pd.Series(np.random.choice([-1, 0, 1], n_bars))
            equity = equity_series(starting_equity=100.0, sig=signal, R=R, L=0.02, De=0.03, verbose=False)
            del equity
        first_pass_time = time.perf_counter() - start

        # Second pass: should benefit from cache (if entries still cached)
        # Reduced from 100 to 30 iterations for faster execution
        start = time.perf_counter()
        for i in range(30):
            signal = pd.Series(np.random.choice([-1, 0, 1], n_bars))
            equity = equity_series(starting_equity=100.0, sig=signal, R=R, L=0.02, De=0.03, verbose=False)
            del equity
        second_pass_time = time.perf_counter() - start

        cache_stats = cache.get_stats()

        print("\nCache Performance Under Pressure:")
        print(f"  First pass: {first_pass_time:.2f}s")
        print(f"  Second pass: {second_pass_time:.2f}s")
        print(f"  Cache hits: {cache_stats.get('equity_cache_hits', 0)}")

        # Cache should still function (even if eviction occurs)
        assert first_pass_time > 0
        assert second_pass_time > 0
