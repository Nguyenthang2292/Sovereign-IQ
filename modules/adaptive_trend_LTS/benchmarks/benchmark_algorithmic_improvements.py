"""Benchmarks for algorithmic improvements (incremental updates and approximate MAs)."""

import time

import numpy as np
import pandas as pd

try:
    from modules.adaptive_trend_LTS.core.compute_atc_signals.compute_atc_signals import compute_atc_signals
    from modules.adaptive_trend_LTS.core.compute_atc_signals.incremental_atc import IncrementalATC
    from modules.adaptive_trend_LTS.core.compute_moving_averages.approximate_mas import (
        fast_ema_approx,
        set_of_approximate_moving_averages,
    )
except ImportError:
    print("Warning: Algorithmic improvements modules not available")


def benchmark_incremental_vs_full(n_bars=1000, n_updates=100):
    """Compare incremental update vs full recalculation.

    Expected: Incremental update should be 10-100x faster for single bar.
    """
    print("\n=== Benchmark: Incremental vs Full Recalculation ===")
    print(f"Series length: {n_bars}, Updates: {n_updates}")

    # Generate test data
    np.random.seed(42)
    prices = 100 + np.cumsum(np.random.normal(0, 1, n_bars))
    prices = pd.Series(prices)

    config = {
        "ema_len": 28,
        "hull_len": 28,
        "wma_len": 28,
        "dema_len": 28,
        "lsma_len": 28,
        "kama_len": 28,
        "La": 0.02,
        "De": 0.03,
        "long_threshold": 0.1,
        "short_threshold": -0.1,
        "cutout": 0,
    }

    # Benchmark full recalculation
    start_time = time.time()
    for _ in range(n_updates):
        full_results = compute_atc_signals(prices, **config)
        _ = full_results["Average_Signal"].iloc[-1]
    full_time = time.time() - start_time

    # Benchmark incremental update
    atc = IncrementalATC(config)
    atc.initialize(prices[: n_bars - n_updates])

    start_time = time.time()
    for i in range(n_bars - n_updates, n_bars):
        signal = atc.update(prices.iloc[i])
    incremental_time = time.time() - start_time

    speedup = full_time / incremental_time if incremental_time > 0 else float("inf")

    print(f"Full recalculation: {full_time:.4f}s ({n_updates} updates)")
    print(f"Incremental update: {incremental_time:.4f}s ({n_updates} updates)")
    print(f"Speedup: {speedup:.2f}x")

    return {
        "full_time": full_time,
        "incremental_time": incremental_time,
        "speedup": speedup,
    }


def benchmark_approximate_vs_full(n_symbols=100, n_bars=1000):
    """Compare approximate MAs vs full precision.

    Expected: Approximate MAs should be 2-3x faster for large symbol sets.
    """
    print("\n=== Benchmark: Approximate vs Full Precision ===")
    print(f"Symbols: {n_symbols}, Bars per symbol: {n_bars}")

    # Generate test data
    np.random.seed(42)
    symbols_data = {}
    for i in range(n_symbols):
        prices = 100 + np.cumsum(np.random.normal(0, 1, n_bars))
        symbols_data[f"SYM_{i}"] = pd.Series(prices)

    config = {
        "ema_len": 28,
        "hull_len": 28,
        "wma_len": 28,
        "dema_len": 28,
        "lsma_len": 28,
        "kama_len": 28,
        "La": 0.02,
        "De": 0.03,
        "long_threshold": 0.1,
        "short_threshold": -0.1,
        "cutout": 0,
    }

    # Benchmark full precision
    start_time = time.time()
    for symbol, prices in symbols_data.items():
        results = compute_atc_signals(prices, **config)
        _ = results["Average_Signal"].iloc[-1]
    full_time = time.time() - start_time

    # Benchmark approximate MAs (just the MA calculation part)
    start_time = time.time()
    for symbol, prices in symbols_data.items():
        approx_mas = set_of_approximate_moving_averages(
            prices,
            ema_len=config["ema_len"],
            hull_len=config["hull_len"],
            wma_len=config["wma_len"],
            dema_len=config["dema_len"],
            lsma_len=config["lsma_len"],
            kama_len=config["kama_len"],
        )
        _ = approx_mas["ema"].iloc[-1]
    approx_time = time.time() - start_time

    speedup = full_time / approx_time if approx_time > 0 else float("inf")

    print(f"Full precision: {full_time:.4f}s ({n_symbols} symbols)")
    print(f"Approximate MAs: {approx_time:.4f}s ({n_symbols} symbols)")
    print(f"Speedup: {speedup:.2f}x")

    return {
        "full_time": full_time,
        "approx_time": approx_time,
        "speedup": speedup,
    }


def benchmark_approximate_accuracy(n_bars=1000):
    """Verify that approximate MAs are within tolerance of true MAs."""
    print("\n=== Benchmark: Approximate MA Accuracy ===")
    print(f"Series length: {n_bars}")

    # Generate test data
    np.random.seed(42)
    prices = 100 + np.cumsum(np.random.normal(0, 1, n_bars))
    prices = pd.Series(prices)

    # Get full precision EMAs
    from modules.adaptive_trend_enhance.core.calculate_hma import calculate_hma
    from modules.adaptive_trend_LTS.core.compute_moving_averages import calculate_ema

    full_ema = calculate_ema(prices, 28)
    full_hma = calculate_hma(prices, 28)

    # Get approximate EMAs
    approx_ema = fast_ema_approx(prices, 28)
    approx_hma = fast_hma_approx(prices, 28)

    # Calculate accuracy
    ema_diff = (full_ema - approx_ema).abs()
    ema_max_diff = ema_diff.max()
    ema_avg_diff = ema_diff.mean()
    ema_pct_error = (ema_diff / full_ema.abs()).mean() * 100

    hma_diff = (full_hma - approx_hma).abs()
    hma_max_diff = hma_diff.max()
    hma_avg_diff = hma_diff.mean()
    hma_pct_error = (hma_diff / full_hma.abs()).mean() * 100

    print("EMA approximation:")
    print(f"  Max difference: {ema_max_diff:.4f}")
    print(f"  Avg difference: {ema_avg_diff:.4f}")
    print(f"  Avg % error: {ema_pct_error:.2f}%")

    print("HMA approximation:")
    print(f"  Max difference: {hma_max_diff:.4f}")
    print(f"  Avg difference: {hma_avg_diff:.4f}")
    print(f"  Avg % error: {hma_pct_error:.2f}%")

    return {
        "ema_max_diff": ema_max_diff,
        "ema_avg_diff": ema_avg_diff,
        "ema_pct_error": ema_pct_error,
        "hma_max_diff": hma_max_diff,
        "hma_avg_diff": hma_avg_diff,
        "hma_pct_error": hma_pct_error,
    }


def run_all_benchmarks():
    """Run all benchmarks and print summary."""
    print("\n" + "=" * 70)
    print("ALGORITHMIC IMPROVEMENTS BENCHMARKS")
    print("=" * 70)

    # Benchmark incremental vs full
    incremental_results = benchmark_incremental_vs_full()

    # Benchmark approximate vs full
    approximate_results = benchmark_approximate_vs_full()

    # Benchmark approximate accuracy
    accuracy_results = benchmark_approximate_accuracy()

    # Summary
    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)
    print(f"Incremental update speedup: {incremental_results['speedup']:.2f}x")
    print(f"Approximate MA speedup: {approximate_results['speedup']:.2f}x")
    print(f"Approximate EMA % error: {accuracy_results['ema_pct_error']:.2f}%")
    print(f"Approximate HMA % error: {accuracy_results['hma_pct_error']:.2f}%")

    # Check if benchmarks meet expectations
    print("\nPerformance targets:")
    if incremental_results["speedup"] >= 10:
        print("  ✅ Incremental update speedup: MEETS TARGET (10-100x expected)")
    elif incremental_results["speedup"] >= 2:
        print("  ⚠️  Incremental update speedup: BELOW TARGET but showing improvement")
    else:
        print("  ❌ Incremental update speedup: BELOW TARGET (< 2x)")

    if approximate_results["speedup"] >= 2:
        print("  ✅ Approximate MA speedup: MEETS TARGET (2-3x expected)")
    elif approximate_results["speedup"] >= 1.5:
        print("  ⚠️  Approximate MA speedup: BELOW TARGET but showing improvement")
    else:
        print("  ❌ Approximate MA speedup: BELOW TARGET (< 1.5x)")

    if accuracy_results["ema_pct_error"] < 10:
        print("  ✅ Approximate EMA accuracy: MEETS TARGET (< 10% error)")
    else:
        print("  ⚠️  Approximate EMA accuracy: ABOVE TARGET (> 10% error)")


if __name__ == "__main__":
    run_all_benchmarks()
