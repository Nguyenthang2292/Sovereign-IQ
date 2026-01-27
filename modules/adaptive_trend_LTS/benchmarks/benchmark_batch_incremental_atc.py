"""Benchmark script comparing batch incremental vs individual incremental updates.

This script measures the performance difference between:
1. Individual incremental updates (updating each symbol one by one)
2. Batch incremental updates (updating all symbols at once)
"""

import time
import pandas as pd
import numpy as np
from typing import Dict, List

from modules.adaptive_trend_LTS.core.compute_atc_signals.incremental_atc import IncrementalATC
from modules.adaptive_trend_LTS.core.compute_atc_signals.batch_incremental_atc import BatchIncrementalATC


def generate_price_series(base_price: float, bars: int, volatility: float = 0.01) -> pd.Series:
    """Generate a synthetic price series for testing."""
    np.random.seed(42)
    returns = np.random.normal(0, volatility, bars)
    prices = base_price * (1 + returns).cumprod()
    return pd.Series(prices)


def benchmark_individual_updates(config: Dict, symbols: List[str], price_updates: Dict[str, List[float]]) -> float:
    """Benchmark individual incremental updates.

    Args:
        config: ATC configuration
        symbols: List of symbol identifiers
        price_updates: Dictionary mapping symbol to list of price updates

    Returns:
        Total time for all updates
    """
    # Generate initialization data
    init_data = {}
    for symbol in symbols:
        init_data[symbol] = generate_price_series(base_price=1000, bars=100)

    # Create individual instances
    instances = {}
    for symbol, prices in init_data.items():
        atc = IncrementalATC(config)
        atc.initialize(prices)
        instances[symbol] = atc

    # Benchmark individual updates
    start_time = time.time()

    num_updates = len(list(price_updates.values())[0])
    for i in range(num_updates):
        for symbol in symbols:
            new_price = price_updates[symbol][i]
            instances[symbol].update(new_price)

    elapsed_time = time.time() - start_time
    return elapsed_time


def benchmark_batch_updates(config: Dict, symbols: List[str], price_updates: Dict[str, List[float]]) -> float:
    """Benchmark batch incremental updates.

    Args:
        config: ATC configuration
        symbols: List of symbol identifiers
        price_updates: Dictionary mapping symbol to list of price updates

    Returns:
        Total time for all updates
    """
    # Generate initialization data
    init_data = {}
    for symbol in symbols:
        init_data[symbol] = generate_price_series(base_price=1000, bars=100)

    # Create batch instance
    batch_atc = BatchIncrementalATC(config)
    for symbol, prices in init_data.items():
        batch_atc.add_symbol(symbol, prices)

    # Benchmark batch updates
    start_time = time.time()

    num_updates = len(list(price_updates.values())[0])
    for i in range(num_updates):
        batch_update = {symbol: price_updates[symbol][i] for symbol in symbols}
        batch_atc.update_all(batch_update)

    elapsed_time = time.time() - start_time
    return elapsed_time


def run_benchmark():
    """Run comprehensive benchmark comparing batch vs individual updates."""
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

    # Test with different numbers of symbols
    symbol_counts = [10, 50, 100]
    num_updates = 20

    results = []

    print("=" * 80)
    print("Batch Incremental ATC vs Individual Incremental ATC Benchmark")
    print("=" * 80)
    print()

    for num_symbols in symbol_counts:
        symbols = [f"SYMBOL_{i}" for i in range(num_symbols)]

        # Generate price updates for all symbols
        price_updates = {}
        for symbol in symbols:
            np.random.seed(42 + hash(symbol) % 1000)
            base_price = 1000 + np.random.rand() * 1000
            prices = generate_price_series(base_price=base_price, bars=num_updates + 100)
            price_updates[symbol] = prices.iloc[-num_updates:].tolist()

        print(f"Benchmarking with {num_symbols} symbols, {num_updates} updates...")
        print("-" * 80)

        # Benchmark individual updates
        individual_time = benchmark_individual_updates(config, symbols, price_updates)
        individual_avg = individual_time / (num_symbols * num_updates)

        print(f"  Individual Updates:")
        print(f"    Total Time:    {individual_time:.4f}s")
        print(f"    Per Update:     {individual_avg * 1000:.4f}ms")

        # Benchmark batch updates
        batch_time = benchmark_batch_updates(config, symbols, price_updates)
        batch_avg = batch_time / (num_symbols * num_updates)

        print(f"  Batch Updates:")
        print(f"    Total Time:    {batch_time:.4f}s")
        print(f"    Per Update:     {batch_avg * 1000:.4f}ms")

        # Calculate speedup
        speedup = individual_time / batch_time if batch_time > 0 else 0
        time_saved = ((individual_time - batch_time) / individual_time) * 100

        print(f"  Results:")
        print(f"    Speedup:       {speedup:.2f}x")
        print(f"    Time Saved:     {time_saved:.1f}%")
        print()

        results.append(
            {
                "num_symbols": num_symbols,
                "individual_time": individual_time,
                "batch_time": batch_time,
                "speedup": speedup,
                "time_saved_percent": time_saved,
            }
        )

    print("=" * 80)
    print("Summary")
    print("=" * 80)
    print()
    print(f"{'Symbols':>10} | {'Individual':>12} | {'Batch':>12} | {'Speedup':>8} | {'Saved':>8}")
    print("-" * 80)
    for r in results:
        print(
            f"{r['num_symbols']:>10} | {r['individual_time']:>12.4f}s | {r['batch_time']:>12.4f}s | {r['speedup']:>8.2f}x | {r['time_saved_percent']:>7.1f}%"
        )

    print()
    print("Conclusion:")
    print("-" * 80)
    avg_speedup = np.mean([r["speedup"] for r in results])
    print(f"Average speedup across all symbol counts: {avg_speedup:.2f}x")

    if avg_speedup >= 2.0:
        print("✅ Batch mode shows 2-5x speedup as expected (target met)")
    elif avg_speedup >= 1.5:
        print("⚠️  Batch mode shows moderate speedup (below target but beneficial)")
    else:
        print("❌ Batch mode does not show significant speedup (needs optimization)")


if __name__ == "__main__":
    run_benchmark()
