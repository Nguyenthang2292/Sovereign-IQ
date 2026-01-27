"""Simple benchmark comparing batch vs individual incremental ATC updates."""

import time
import pandas as pd
import numpy as np

from modules.adaptive_trend_LTS.core.compute_atc_signals.batch_incremental_atc import BatchIncrementalATC
from modules.adaptive_trend_LTS.core.compute_atc_signals.incremental_atc import IncrementalATC


def generate_prices(base=1000, bars=100):
    """Generate simple price series."""
    np.random.seed(42)
    returns = np.random.normal(0, 0.01, bars)
    return pd.Series(base * (1 + returns).cumprod())


def benchmark():
    config = {
        "ema_len": 28,
        "hull_len": 28,
        "wma_len": 28,
        "dema_len": 28,
        "lsma_len": 28,
        "kama_len": 28,
        "De": 0.03,
        "La": 0.02,
        "long_threshold": 0.1,
        "short_threshold": -0.1,
    }

    num_symbols = 10
    num_updates = 50
    symbols = [f"SYMBOL_{i}" for i in range(num_symbols)]

    print("=" * 60)
    print("Batch vs Individual Incremental ATC Benchmark")
    print("=" * 60)
    print(f"Symbols: {num_symbols}, Updates: {num_updates}")
    print()

    # Generate prices for all symbols
    prices_dict = {}
    for symbol in symbols:
        prices_dict[symbol] = generate_prices(base=1000 + np.random.rand() * 500)

    # Benchmark 1: Individual updates
    print("Benchmarking individual incremental updates...")
    instances = {}
    for symbol in symbols:
        atc = IncrementalATC(config)
        atc.initialize(prices_dict[symbol])
        instances[symbol] = atc

    start = time.time()
    for i in range(num_updates):
        for symbol in symbols:
            new_price = prices_dict[symbol].iloc[-1] * (1 + 0.01 * np.random.randn())
            instances[symbol].update(new_price)
    individual_time = time.time() - start

    print(f"  Individual Time: {individual_time:.4f}s")
    print(f"  Per Update: {individual_time / (num_symbols * num_updates) * 1000:.4f}ms")

    # Benchmark 2: Batch updates
    print("\nBenchmarking batch incremental updates...")
    batch = BatchIncrementalATC(config)
    for symbol in symbols:
        batch.add_symbol(symbol, prices_dict[symbol])

    start = time.time()
    for i in range(num_updates):
        update_dict = {symbol: prices_dict[symbol].iloc[-1] * (1 + 0.01 * np.random.randn()) for symbol in symbols}
        batch.update_all(update_dict)
    batch_time = time.time() - start

    print(f"  Batch Time: {batch_time:.4f}s")
    print(f"  Per Update: {batch_time / (num_symbols * num_updates) * 1000:.4f}ms")

    # Calculate speedup
    speedup = individual_time / batch_time
    time_saved = ((individual_time - batch_time) / individual_time) * 100

    print()
    print("=" * 60)
    print("Results:")
    print(f"  Speedup: {speedup:.2f}x")
    print(f"  Time Saved: {time_saved:.1f}%")

    if speedup >= 2.0:
        print("  ✅ Batch mode shows 2-5x speedup (target met)")
    elif speedup >= 1.5:
        print("  ⚠️  Batch mode shows moderate speedup")
    else:
        print("  ❌ Batch mode needs optimization")


if __name__ == "__main__":
    benchmark()
