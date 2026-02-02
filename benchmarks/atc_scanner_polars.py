import os
import sys
import time
import tracemalloc

# Add project root to path
sys.path.append(os.getcwd())

from unittest.mock import MagicMock, patch

import pandas as pd

from modules.auto_trade.core.atc_scanner import ATCScanner
from modules.common.core.data_fetcher import DataFetcher

# Mock dependencies
mock_data_fetcher = MagicMock(spec=DataFetcher)


def mock_scan_all_symbols_side_effect(data_fetcher, atc_config, symbols, **kwargs):
    # Simulate return data (Pandas, as expected by scan_all_symbols)
    longs = pd.DataFrame({"symbol": symbols, "signal": [0.8] * len(symbols)})
    shorts = pd.DataFrame()
    return longs, shorts


def run_benchmark(num_symbols):
    symbols = [f"SYM_{i}" for i in range(num_symbols)]

    # Initialize scanner (caches are empty)
    scanner = ATCScanner(mock_data_fetcher)

    # Patch scan_all_symbols to avoid network calls and purely measure overhead/processing
    with patch("modules.auto_trade.core.atc_scanner.scan_all_symbols", side_effect=mock_scan_all_symbols_side_effect):
        tracemalloc.start()
        start_time = time.perf_counter()

        results = scanner.scan_symbols(symbols)

        end_time = time.perf_counter()
        current, peak = tracemalloc.get_traced_memory()
        tracemalloc.stop()

        duration = end_time - start_time
        memory_mb = peak / 1024 / 1024

        print(
            f"Symbols: {num_symbols:3d} | Time: {duration:.4f}s | Peak Memory: {memory_mb:.2f}MB | Results: {len(results)}"
        )


if __name__ == "__main__":
    print("Benchmarking ATCScanner (Polars + Rust)...")
    print("-" * 65)
    for n in [10, 50, 100, 500]:
        run_benchmark(n)
