"""
Benchmark and memory verification for scanner batching.
"""

import gc
import os
import sys
import time
import tracemalloc
from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np
import pandas as pd
import psutil

# Add project root to sys.path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from modules.adaptive_trend_enhance.core.scanner import scan_all_symbols
from modules.adaptive_trend_enhance.utils.config import ATCConfig


def get_process_memory():
    process = psutil.Process(os.getpid())
    return process.memory_info().rss / (1024 * 1024)  # MB


def create_mock_ohlcv_data(num_candles: int = 500) -> pd.DataFrame:
    """Create mock OHLCV DataFrame."""
    prices = np.random.uniform(100, 200, num_candles)
    return pd.DataFrame(
        {
            "timestamp": pd.date_range("2024-01-01", periods=num_candles, freq="1h"),
            "open": prices,
            "high": prices + 1,
            "low": prices - 1,
            "close": prices,
            "volume": 1000,
        }
    )


def create_mock_atc_results() -> dict:
    return {"Average_Signal": pd.Series([0.5] * 500)}


@patch("modules.adaptive_trend_enhance.core.scanner.compute_atc_signals")
@patch("modules.adaptive_trend_enhance.core.scanner.trend_sign")
def run_benchmark(mock_trend_sign, mock_compute_atc, num_symbols=200, batch_size=50):
    print(f"\n--- Benchmarking {num_symbols} symbols with batch_size={batch_size} ---")

    # Mock setup
    symbols = [f"SYM{i}" for i in range(num_symbols)]
    mock_fetcher = MagicMock()
    mock_fetcher.list_binance_futures_symbols.return_value = symbols

    # Each call returns a new DF to simulate real memory allocation
    def fetch_side_effect(*args, **kwargs):
        # Create a large DF to really stress memory (50,000 bars)
        return (create_mock_ohlcv_data(num_candles=50000), "binance")

    mock_fetcher.fetch_ohlcv_with_fallback_exchange.side_effect = fetch_side_effect

    mock_compute_atc.return_value = create_mock_atc_results()
    mock_trend_sign.return_value = pd.Series([1] * 5000)

    config = ATCConfig(limit=200)

    gc.collect()
    tracemalloc.start()
    start_time = time.time()

    # We use sequential to avoid noise from threads for memory measurement
    scan_all_symbols(mock_fetcher, config, batch_size=batch_size, execution_mode="sequential")

    duration = time.time() - start_time
    current, peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()

    print(f"Duration: {duration:.2f}s")
    print(f"Peak Memory: {peak / (1024 * 1024):.2f} MB")
    print(f"Duration: {duration:.2f}s", flush=True)
    print(f"Peak Memory: {peak / (1024 * 1024):.2f} MB", flush=True)
    print(f"Current Memory Leak: {current / (1024 * 1024):.2f} MB", flush=True)

    return duration, peak


if __name__ == "__main__":
    print("Starting benchmarks...", flush=True)
    # Test different batch sizes
    # Large batch (effectively no batching)
    run_benchmark(num_symbols=20, batch_size=50)
    print("Finished Benchmark 1", flush=True)

    # Small batch
    run_benchmark(num_symbols=20, batch_size=5)
    print("Finished Benchmark 2", flush=True)

    # Very small batch
    run_benchmark(num_symbols=20, batch_size=1)
    print("Finished Benchmark 3", flush=True)
    print("All benchmarks completed.", flush=True)
