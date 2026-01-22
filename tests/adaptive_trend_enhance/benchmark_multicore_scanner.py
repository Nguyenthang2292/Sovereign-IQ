import time
from unittest.mock import MagicMock

import numpy as np
import pandas as pd

from modules.adaptive_trend_enhance.core.scanner.scan_all_symbols import scan_all_symbols
from modules.adaptive_trend_enhance.utils.config import ATCConfig
from modules.common.core.data_fetcher import DataFetcher


def benchmark_scanner():
    """Benchmark different scanner execution modes."""
    # Setup mock data fetcher
    data_fetcher = MagicMock(spec=DataFetcher)

    # Mock return value for fetch_ohlcv_with_fallback_exchange
    n = 1000
    df = pd.DataFrame(
        {
            "close": 100 * (1 + np.random.randn(n).cumsum() * 0.01),
            "open": 100 * (1 + np.random.randn(n).cumsum() * 0.01),
            "high": 100 * (1 + np.random.randn(n).cumsum() * 0.01),
            "low": 100 * (1 + np.random.randn(n).cumsum() * 0.01),
            "volume": np.random.rand(n) * 1000,
        },
        index=pd.date_range("2023-01-01", periods=n, freq="15min"),
    )

    data_fetcher.fetch_ohlcv_with_fallback_exchange.return_value = (df, "binance")

    # Mock list_symbols
    symbols = [f"SYMBOL_{i}" for i in range(20)]

    # Configuration
    atc_config = ATCConfig()
    min_signal = 0.01

    # Modes to test
    modes = ["sequential", "threadpool", "processpool"]

    print(f"\nBenchmarking {len(symbols)} symbols...")
    results_perf = {}

    for mode in modes:
        print(f"Testing mode: {mode}...")
        start_time = time.time()

        # We need to mock the scan functions or ensure data_fetcher works in workers
        # Since we are using MagicMock, ProcessPoolExecutor might fail to pickle it
        # So for processpool benchmark, we might need a more realistic mock.

        try:
            results, _ = scan_all_symbols(
                data_fetcher=data_fetcher,
                atc_config=atc_config,
                max_symbols=len(symbols),
                execution_mode=mode,
                max_workers=4,
                batch_size=10,
            )
            duration = time.time() - start_time
            results_perf[mode] = duration
            print(f"Mode {mode} took {duration:.2f}s")
        except Exception as e:
            print(f"Mode {mode} failed: {e}")

    # Print comparison
    if "sequential" in results_perf and "processpool" in results_perf:
        speedup = results_perf["sequential"] / results_perf["processpool"]
        print(f"\nProcessPool Speedup vs Sequential: {speedup:.2f}x")

    if "threadpool" in results_perf and "processpool" in results_perf:
        speedup = results_perf["threadpool"] / results_perf["processpool"]
        print(f"ProcessPool Speedup vs ThreadPool: {speedup:.2f}x")


if __name__ == "__main__":
    # Note: scan_all_symbols calls list_symbols if symbols not provided,
    # but we can't easily mock that if it uses data_fetcher.list_binance_futures_symbols internally.
    # So we'll need to wrap it.

    # This benchmark might need more care with the mock for ProcessPool.
    # For now, let's just run it and see.
    benchmark_scanner()
