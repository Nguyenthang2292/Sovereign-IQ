"""
ATC Scanner Conversion Overhead Analysis

This benchmark measures the overhead of Pandas->Polars conversion in the ATCScanner
as a prerequisite for Phase 2 (Rust integration). It does NOT compare fully migrated
Polars implementation vs Pandas, but rather measures the conversion cost.

Usage:
    python tests/performance/atc_scanner_conversion_overhead.py
    python tests/performance/atc_scanner_conversion_overhead.py --test-cases 10 50 100 --iterations 5
    python tests/performance/atc_scanner_conversion_overhead.py --csv results.csv --plot
"""

import argparse
import csv
import pathlib
import random
import statistics
import sys
import time
import tracemalloc
from typing import Any, Dict, List, Tuple

# Add project root to path
project_root = pathlib.Path(__file__).parent.parent.parent
sys.path.append(str(project_root))

from unittest.mock import MagicMock, patch

import pandas as pd
import polars as pl

try:
    import matplotlib.pyplot as plt
    MATPLOTLIB_AVAILABLE = True
except ImportError:
    MATPLOTLIB_AVAILABLE = False

from modules.auto_trade.core.atc_scanner import ATCScanner
from modules.common.core.data_fetcher import DataFetcher

# Mock dependencies
mock_data_fetcher = MagicMock(spec=DataFetcher)


def generate_realistic_mock_data(symbols: List[str]) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Generate realistic mock data with varied signal strengths.

    Args:
        symbols: List of symbol names

    Returns:
        Tuple of (longs_df, shorts_df) with realistic signal distributions
    """
    # 60% of symbols get signals, 40% don't
    num_signals = int(len(symbols) * 0.6)
    if num_signals == 0 and len(symbols) > 0:
        num_signals = 1

    # Split between longs and shorts (70% longs, 30% shorts)
    num_longs = int(num_signals * 0.7)
    num_shorts = num_signals - num_longs

    # Generate varied signal strengths (realistic distribution)
    long_symbols = random.sample(symbols, min(num_longs, len(symbols)))
    short_symbols = random.sample([s for s in symbols if s not in long_symbols],
                                  min(num_shorts, len(symbols) - len(long_symbols)))

    # Signal strengths: mostly strong signals (0.6-0.9), some weak (0.3-0.6)
    long_signals = [random.uniform(0.3, 0.9) for _ in long_symbols]
    short_signals = [random.uniform(-0.9, -0.3) for _ in short_symbols]

    longs_df = pd.DataFrame({"symbol": long_symbols, "signal": long_signals}) if long_symbols else pd.DataFrame()
    shorts_df = pd.DataFrame({"symbol": short_symbols, "signal": short_signals}) if short_symbols else pd.DataFrame()

    return longs_df, shorts_df


def mock_scan_all_symbols_side_effect(data_fetcher: Any, atc_config: Any,
                                      symbols: List[str], **kwargs: Any) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Mock scan_all_symbols to return realistic pre-generated DataFrames.

    Args:
        data_fetcher: Mock data fetcher
        atc_config: ATC configuration
        symbols: List of symbols to scan
        **kwargs: Additional arguments

    Returns:
        Tuple of (longs_df, shorts_df)
    """
    return generate_realistic_mock_data(symbols)


class BenchmarkResult:
    """Container for benchmark results with statistical measures."""

    def __init__(self, version: str, symbols: int, iterations: int):
        self.version = version
        self.symbols = symbols
        self.iterations = iterations
        self.times: List[float] = []
        self.memories: List[float] = []
        self.result_counts: List[int] = []

    def add_run(self, time_sec: float, memory_mb: float, result_count: int) -> None:
        """Add a single benchmark run result."""
        self.times.append(time_sec)
        self.memories.append(memory_mb)
        self.result_counts.append(result_count)

    def get_stats(self) -> Dict[str, Any]:
        """Calculate statistical measures for all runs."""
        return {
            "version": self.version,
            "symbols": self.symbols,
            "iterations": self.iterations,
            "time_mean": statistics.mean(self.times),
            "time_median": statistics.median(self.times),
            "time_stdev": statistics.stdev(self.times) if len(self.times) > 1 else 0.0,
            "time_min": min(self.times),
            "time_max": max(self.times),
            "memory_mean": statistics.mean(self.memories),
            "memory_median": statistics.median(self.memories),
            "memory_stdev": statistics.stdev(self.memories) if len(self.memories) > 1 else 0.0,
            "memory_peak": max(self.memories),
            "results": int(statistics.mean(self.result_counts)),
        }


def run_benchmark_pandas(num_symbols: int, warmup: bool = False) -> Dict[str, Any]:
    """
    Benchmark Pandas-based ATCScanner (baseline).

    Args:
        num_symbols: Number of symbols to scan
        warmup: If True, this is a warmup run (not counted)

    Returns:
        Dictionary with benchmark results

    Raises:
        Exception: Re-raises any exception after cleanup
    """
    symbols = [f"SYM_{i}" for i in range(num_symbols)]
    scanner = ATCScanner(mock_data_fetcher)

    with patch(
        "modules.auto_trade.core.atc_scanner.scan_all_symbols",
        side_effect=mock_scan_all_symbols_side_effect,
    ):
        tracemalloc.start()
        try:
            start_time = time.perf_counter()
            results = scanner.scan_symbols(symbols)
            end_time = time.perf_counter()
        finally:
            current, peak = tracemalloc.get_traced_memory()
            tracemalloc.stop()

        duration = end_time - start_time
        memory_mb = peak / 1024 / 1024

        return {
            "version": "Pandas (Baseline)",
            "symbols": num_symbols,
            "time": duration,
            "memory": memory_mb,
            "results": len(results),
        }


def run_benchmark_polars_conversion(num_symbols: int, warmup: bool = False) -> Dict[str, Any]:
    """
    Benchmark with Pandas->Polars conversion overhead.

    This simulates the conversion overhead of migrating to Polars without
    the benefits of a fully Polars-native implementation.

    Args:
        num_symbols: Number of symbols to scan
        warmup: If True, this is a warmup run (not counted)

    Returns:
        Dictionary with benchmark results

    Raises:
        Exception: Re-raises any exception after cleanup
    """
    symbols = [f"SYM_{i}" for i in range(num_symbols)]
    scanner = ATCScanner(mock_data_fetcher)

    def side_effect_polars_conversion_flow(symbols_batch: List[str], timeframe: str) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """Simulate Polars conversion overhead in scan flow."""
        # 1. Mock Scan returns Pandas DataFrames
        long_pd, short_pd = generate_realistic_mock_data(symbols_batch)

        # 2. Convert to Polars (measure this overhead)
        if not long_pd.empty:
            long_pl = pl.from_pandas(long_pd)
            # Simulate Polars filtering (though result is discarded)
            _ = long_pl.filter(pl.col("signal") > 0.3)

        if not short_pd.empty:
            short_pl = pl.from_pandas(short_pd)
            _ = short_pl.filter(pl.col("signal") < -0.3)

        # Return Pandas to maintain compatibility with current scanner
        return long_pd, short_pd

    with patch.object(scanner, "_run_single_scan", side_effect=side_effect_polars_conversion_flow):
        with patch(
            "modules.auto_trade.core.atc_scanner.scan_all_symbols",
            side_effect=mock_scan_all_symbols_side_effect
        ):
            tracemalloc.start()
            try:
                start_time = time.perf_counter()
                results = scanner.scan_symbols(symbols)
                end_time = time.perf_counter()
            finally:
                current, peak = tracemalloc.get_traced_memory()
                tracemalloc.stop()

            duration = end_time - start_time
            memory_mb = peak / 1024 / 1024

            return {
                "version": "Polars Conversion (Overhead)",
                "symbols": num_symbols,
                "time": duration,
                "memory": memory_mb,
                "results": len(results),
            }


def run_benchmark_with_stats(
    benchmark_fn: Any,
    num_symbols: int,
    iterations: int,
    version_name: str
) -> BenchmarkResult:
    """
    Run benchmark multiple times and collect statistics.

    Args:
        benchmark_fn: Benchmark function to run
        num_symbols: Number of symbols to test
        iterations: Number of iterations to run
        version_name: Name of the benchmark version

    Returns:
        BenchmarkResult with statistical measures
    """
    result = BenchmarkResult(version_name, num_symbols, iterations)

    # Warmup run (not counted)
    print(f"  Warming up {version_name}...", end=" ", flush=True)
    benchmark_fn(num_symbols, warmup=True)
    print("done")

    # Actual benchmark runs
    for i in range(iterations):
        print(f"  Run {i+1}/{iterations}...", end=" ", flush=True)
        run_result = benchmark_fn(num_symbols)
        result.add_run(run_result["time"], run_result["memory"], run_result["results"])
        print(f"{run_result['time']:.4f}s")

    return result


def save_results_to_csv(results: List[Dict[str, Any]], filepath: str) -> None:
    """
    Save benchmark results to CSV file.

    Args:
        results: List of benchmark result dictionaries
        filepath: Path to output CSV file
    """
    if not results:
        return

    with open(filepath, "w", newline="") as f:
        fieldnames = list(results[0].keys())
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(results)

    print(f"\nResults saved to: {filepath}")


def plot_results(results: List[Dict[str, Any]], output_path: str = "benchmark_plot.png") -> None:
    """
    Generate visualization plots for benchmark results.

    Args:
        results: List of benchmark result dictionaries
        output_path: Path to save the plot image
    """
    if not MATPLOTLIB_AVAILABLE:
        print("\nWarning: matplotlib not available. Skipping plot generation.")
        return

    # Separate Pandas and Polars results
    pandas_results = [r for r in results if "Baseline" in r["version"]]
    polars_results = [r for r in results if "Overhead" in r["version"]]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

    # Time comparison plot
    symbols_pandas = [r["symbols"] for r in pandas_results]
    time_pandas = [r["time_mean"] for r in pandas_results]
    time_pandas_err = [r["time_stdev"] for r in pandas_results]

    symbols_polars = [r["symbols"] for r in polars_results]
    time_polars = [r["time_mean"] for r in polars_results]
    time_polars_err = [r["time_stdev"] for r in polars_results]

    ax1.errorbar(symbols_pandas, time_pandas, yerr=time_pandas_err,
                 marker='o', label='Pandas (Baseline)', capsize=5)
    ax1.errorbar(symbols_polars, time_polars, yerr=time_polars_err,
                 marker='s', label='Polars Conversion', capsize=5)
    ax1.set_xlabel('Number of Symbols')
    ax1.set_ylabel('Time (seconds)')
    ax1.set_title('Execution Time Comparison')
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    # Memory comparison plot
    memory_pandas = [r["memory_mean"] for r in pandas_results]
    memory_pandas_err = [r["memory_stdev"] for r in pandas_results]

    memory_polars = [r["memory_mean"] for r in polars_results]
    memory_polars_err = [r["memory_stdev"] for r in polars_results]

    ax2.errorbar(symbols_pandas, memory_pandas, yerr=memory_pandas_err,
                 marker='o', label='Pandas (Baseline)', capsize=5)
    ax2.errorbar(symbols_polars, memory_polars, yerr=memory_polars_err,
                 marker='s', label='Polars Conversion', capsize=5)
    ax2.set_xlabel('Number of Symbols')
    ax2.set_ylabel('Memory (MB)')
    ax2.set_title('Peak Memory Usage Comparison')
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    print(f"Plot saved to: {output_path}")


def print_results_table(all_results: List[Dict[str, Any]]) -> None:
    """
    Print formatted results table with statistical measures.

    Args:
        all_results: List of all benchmark results with statistics
    """
    print("\n" + "=" * 120)
    print("Benchmark Results: Pandas Baseline vs Polars Conversion Overhead")
    print("=" * 120)
    print(f"{'Version':<25} | {'Symbols':<8} | {'Time (s)':<20} | {'Memory (MB)':<20} | {'% Diff'}")
    print("-" * 120)

    # Process results in pairs (Pandas, Polars)
    for i in range(0, len(all_results), 2):
        pd_res = all_results[i]
        pl_res = all_results[i + 1] if i + 1 < len(all_results) else None

        # Print Pandas baseline
        print(
            f"{pd_res['version']:<25} | "
            f"{pd_res['symbols']:<8} | "
            f"{pd_res['time_mean']:.4f} ± {pd_res['time_stdev']:.4f} | "
            f"{pd_res['memory_mean']:.2f} ± {pd_res['memory_stdev']:.2f} | "
            f"-"
        )

        if pl_res:
            # Calculate percentage differences
            diff_time = ((pl_res["time_mean"] - pd_res["time_mean"]) / pd_res["time_mean"]) * 100 if pd_res["time_mean"] > 0 else 0
            diff_mem = ((pl_res["memory_mean"] - pd_res["memory_mean"]) / pd_res["memory_mean"]) * 100 if pd_res["memory_mean"] > 0 else 0

            # Print Polars conversion
            print(
                f"{pl_res['version']:<25} | "
                f"{pl_res['symbols']:<8} | "
                f"{pl_res['time_mean']:.4f} ± {pl_res['time_stdev']:.4f} | "
                f"{pl_res['memory_mean']:.2f} ± {pl_res['memory_stdev']:.2f} | "
                f"T: {diff_time:+.1f}%, M: {diff_mem:+.1f}%"
            )

        print("-" * 120)

    print("=" * 120)
    print("\nNotes:")
    print("  - Time/Memory shown as: mean ± standard deviation")
    print("  - % Diff shows percentage change from Pandas baseline (positive = slower/more memory)")
    print("  - This measures CONVERSION OVERHEAD, not fully optimized Polars implementation")


def parse_arguments() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Benchmark ATCScanner Pandas->Polars conversion overhead",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python tests/performance/atc_scanner_conversion_overhead.py
  python tests/performance/atc_scanner_conversion_overhead.py --test-cases 10 50 100 --iterations 5
  python tests/performance/atc_scanner_conversion_overhead.py --csv results.csv --plot
        """
    )

    parser.add_argument(
        "--test-cases",
        nargs="+",
        type=int,
        default=[10, 50, 100, 500],
        help="Number of symbols to test (default: 10 50 100 500)"
    )

    parser.add_argument(
        "--iterations",
        type=int,
        default=5,
        help="Number of iterations per test case (default: 5)"
    )

    parser.add_argument(
        "--csv",
        type=str,
        help="Save results to CSV file (e.g., results.csv)"
    )

    parser.add_argument(
        "--plot",
        action="store_true",
        help="Generate visualization plots (requires matplotlib)"
    )

    parser.add_argument(
        "--plot-output",
        type=str,
        default="benchmark_plot.png",
        help="Output path for plot image (default: benchmark_plot.png)"
    )

    return parser.parse_args()


def main() -> None:
    """Main benchmark execution."""
    args = parse_arguments()

    print("=" * 80)
    print("ATC Scanner Conversion Overhead Analysis")
    print("=" * 80)
    print(f"Test cases: {args.test_cases}")
    print(f"Iterations per test: {args.iterations}")
    print("\nThis benchmark measures Pandas->Polars CONVERSION OVERHEAD only.")
    print("It does NOT test a fully optimized Polars implementation.")
    print("=" * 80)

    all_results: List[Dict[str, Any]] = []

    for num_symbols in args.test_cases:
        print(f"\n{'='*80}")
        print(f"Testing with {num_symbols} symbols ({args.iterations} iterations)")
        print(f"{'='*80}")

        # Run Pandas baseline benchmark
        print("\nPandas Baseline:")
        pandas_result = run_benchmark_with_stats(
            run_benchmark_pandas,
            num_symbols,
            args.iterations,
            "Pandas (Baseline)"
        )
        all_results.append(pandas_result.get_stats())

        # Run Polars conversion benchmark
        print("\nPolars Conversion:")
        polars_result = run_benchmark_with_stats(
            run_benchmark_polars_conversion,
            num_symbols,
            args.iterations,
            "Polars Conversion (Overhead)"
        )
        all_results.append(polars_result.get_stats())

    # Print results table
    print_results_table(all_results)

    # Save to CSV if requested
    if args.csv:
        save_results_to_csv(all_results, args.csv)

    # Generate plots if requested
    if args.plot:
        plot_results(all_results, args.plot_output)


if __name__ == "__main__":
    main()
