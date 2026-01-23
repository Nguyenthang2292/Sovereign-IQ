"""
Benchmark Comparison: adaptive_trend vs adaptive_trend_enhance

This script loads 500 symbols via ExchangeManager/DataFetcher and compares:
1. Signal output consistency (100% match expected)
2. Execution time (speedup measurement)
3. Memory usage
4. Performance metrics

Usage:
    python benchmark_comparison.py --symbols 1000 --bars 1000
    Or from project root: python -m modules.adaptive_trend_enhance.docs.benchmarks.benchmark_comparison
"""

import argparse
import gc
import sys
import time
from pathlib import Path
from typing import Dict, Tuple

# Add project root to sys.path for imports
# This ensures the script can be run from any directory

try:
    project_root = Path(__file__).parent.parent.parent
    project_root_str = str(project_root.resolve())
    if project_root_str not in sys.path:
        sys.path.insert(0, project_root_str)
except NameError:
    # If __file__ is not defined (e.g., in interactive mode), assume we're at project root
    pass

import numpy as np
import pandas as pd
from tabulate import tabulate

# Import both modules
from modules.adaptive_trend.core import compute_atc_signals as compute_atc_original
from modules.adaptive_trend_enhance.core import compute_atc_signals as compute_atc_enhanced
from modules.common.core import DataFetcher, ExchangeManager
from modules.common.utils import log_error, log_info, log_success, log_warn


def fetch_symbols_data(num_symbols: int = 1000, bars: int = 1000, timeframe: str = "1h") -> Dict[str, pd.Series]:
    """Fetch price data for multiple symbols.

    Args:
        num_symbols: Number of symbols to fetch (default: 1000)
        bars: Number of bars per symbol (default: 1000)
        timeframe: Timeframe for OHLCV data (default: "1h")

    Returns:
        Dictionary mapping symbol -> close price Series
    """
    log_info(f"Fetching {num_symbols} symbols with {bars} bars each...")

    exchange_mgr = ExchangeManager()
    data_fetcher = DataFetcher(exchange_mgr)

    # Get list of symbols
    log_info("Discovering symbols from Binance Futures...")
    symbols = data_fetcher.list_binance_futures_symbols(max_candidates=num_symbols)

    if len(symbols) < num_symbols:
        log_warn(f"Only {len(symbols)} symbols available, requested {num_symbols}")

    # Fetch data for each symbol
    prices_data = {}
    successful = 0
    failed = 0

    for idx, symbol in enumerate(symbols[:num_symbols], 1):
        try:
            df, exchange = data_fetcher.fetch_ohlcv_with_fallback_exchange(
                symbol=symbol, limit=bars, timeframe=timeframe
            )

            if df is not None and len(df) >= bars:
                close_series = data_fetcher.dataframe_to_close_series(df)
                prices_data[symbol] = close_series
                successful += 1

                if idx % 50 == 0:
                    log_info(f"Progress: {idx}/{num_symbols} symbols fetched")
            else:
                failed += 1
                log_warn(f"Insufficient data for {symbol}")

        except Exception as e:
            failed += 1
            log_error(f"Error fetching {symbol}: {e}")

    log_success(f"Fetched {successful} symbols successfully, {failed} failed")
    return prices_data


def run_original_module(prices_data: Dict[str, pd.Series], config: dict) -> Tuple[Dict[str, Dict], float, float]:
    """Run original adaptive_trend module.

    Args:
        prices_data: Dictionary of symbol -> price Series
        config: ATC configuration parameters

    Returns:
        Tuple of (results_dict, execution_time_seconds, peak_memory_mb)
    """
    log_info("Running original adaptive_trend module...")

    results = {}
    start_time = time.time()

    # Track memory (approximate)
    import psutil

    process = psutil.Process()
    mem_before = process.memory_info().rss / 1024 / 1024  # MB

    for idx, (symbol, prices) in enumerate(prices_data.items(), 1):
        try:
            result = compute_atc_original(prices=prices, **config)
            results[symbol] = result

            if idx % 100 == 0:
                log_info(f"Original: Processed {idx}/{len(prices_data)} symbols")

        except Exception as e:
            import traceback

            log_error(f"Original failed for {symbol}: {e}")
            traceback.print_exc()
            results[symbol] = None

    end_time = time.time()
    mem_after = process.memory_info().rss / 1024 / 1024  # MB

    execution_time = end_time - start_time
    peak_memory = mem_after - mem_before

    log_success(f"Original module completed in {execution_time:.2f}s")
    return results, execution_time, peak_memory


def run_enhanced_module(prices_data: Dict[str, pd.Series], config: dict) -> Tuple[Dict[str, Dict], float, float]:
    """Run enhanced adaptive_trend_enhance module.

    Args:
        prices_data: Dictionary of symbol -> price Series
        config: ATC configuration parameters

    Returns:
        Tuple of (results_dict, execution_time_seconds, peak_memory_mb)
    """
    log_info("Running enhanced adaptive_trend_enhance module...")

    results = {}
    start_time = time.time()

    # Track memory (approximate)
    import psutil

    process = psutil.Process()
    mem_before = process.memory_info().rss / 1024 / 1024  # MB

    for idx, (symbol, prices) in enumerate(prices_data.items(), 1):
        try:
            result = compute_atc_enhanced(prices=prices, **config)
            results[symbol] = result

            if idx % 100 == 0:
                log_info(f"Enhanced: Processed {idx}/{len(prices_data)} symbols")

        except Exception as e:
            import traceback

            log_error(f"Enhanced failed for {symbol}: {e}")
            traceback.print_exc()
            results[symbol] = None

    end_time = time.time()
    mem_after = process.memory_info().rss / 1024 / 1024  # MB

    execution_time = end_time - start_time
    peak_memory = mem_after - mem_before

    log_success(f"Enhanced module completed in {execution_time:.2f}s")
    return results, execution_time, peak_memory


def compare_signals(original_results: Dict[str, Dict], enhanced_results: Dict[str, Dict]) -> Dict[str, float]:
    """Compare signal outputs between original and enhanced modules.

    Args:
        original_results: Results from original module
        enhanced_results: Results from enhanced module

    Returns:
        Dictionary of comparison metrics
    """
    log_info("Comparing signal outputs...")

    total_symbols = len(original_results)
    matching_symbols = 0
    signal_diffs = []
    mismatched_symbols = []

    for symbol in original_results.keys():
        if symbol not in enhanced_results:
            log_warn(f"Symbol {symbol} missing in enhanced results")
            continue

        orig = original_results[symbol]
        enh = enhanced_results[symbol]

        if orig is None or enh is None:
            log_warn(f"Symbol {symbol} has None result (orig={orig is None}, enh={enh is None})")
            continue

        # Compare Average_Signal
        orig_s = orig.get("Average_Signal")
        enh_s = enh.get("Average_Signal")

        if orig_s is None or enh_s is None:
            log_warn(f"Symbol {symbol} missing Average_Signal (orig={orig_s is None}, enh={enh_s is None})")
            continue

        # Check index alignment
        if not orig_s.index.equals(enh_s.index):
            log_warn(f"Index mismatch for {symbol}: Orig type={type(orig_s.index)}, Enh type={type(enh_s.index)}")
            log_warn(f"Orig Sample Index: {orig_s.index[:2].tolist()}")
            log_warn(f"Enh Sample Index: {enh_s.index[:2].tolist()}")
            orig_s, enh_s = orig_s.align(enh_s, join="inner")
            if orig_s.empty:
                log_warn(f"Alignment resulted in empty series for {symbol}")
                continue

        # Calculate difference
        diff = np.abs(orig_s - enh_s).max()
        signal_diffs.append(diff)

        # Consider matching if difference < 1e-6 (numerical precision)
        if diff < 1e-6:
            matching_symbols += 1
        else:
            mismatched_symbols.append((symbol, diff))
            log_warn(f"Mismatch for {symbol}: max_diff={diff:.10e}")
            
            # Find where the difference occurs
            diff_series = np.abs(orig_s - enh_s)
            max_diff_idx = diff_series.idxmax()
            max_diff_pos = diff_series.argmax()
            
            log_warn(f"  Max difference at index {max_diff_idx} (position {max_diff_pos})")
            log_warn(f"  Original value: {orig_s.iloc[max_diff_pos]:.10e}")
            log_warn(f"  Enhanced value: {enh_s.iloc[max_diff_pos]:.10e}")
            log_warn(f"  Difference: {diff:.10e}")
            
            # Show context around the difference
            start_idx = max(0, max_diff_pos - 2)
            end_idx = min(len(orig_s), max_diff_pos + 3)
            log_warn(f"  Context (positions {start_idx}-{end_idx-1}):")
            log_warn(f"    Original: {orig_s.iloc[start_idx:end_idx].tolist()}")
            log_warn(f"    Enhanced: {enh_s.iloc[start_idx:end_idx].tolist()}")
            log_warn(f"    Diff:     {diff_series.iloc[start_idx:end_idx].tolist()}")

            # Also check raw EMA and other MAs
            for ma_type in ["EMA", "HMA", "WMA", "DEMA", "LSMA", "KAMA"]:
                orig_ma = orig.get(ma_type)
                enh_ma = enh.get(ma_type)
                if orig_ma is not None and enh_ma is not None:
                    ma_diff = np.abs(orig_ma - enh_ma).max()
                    if ma_diff > 1e-6:
                        log_warn(f"  {ma_type} also differs: max_diff={ma_diff:.10e}")

    match_rate = (matching_symbols / total_symbols) * 100 if total_symbols > 0 else 0
    max_diff = max(signal_diffs) if signal_diffs else 0
    avg_diff = np.mean(signal_diffs) if signal_diffs else 0
    median_diff = np.median(signal_diffs) if signal_diffs else 0

    log_success(f"Signal match rate: {match_rate:.2f}%")
    log_info(f"Max signal difference: {max_diff:.2e}")
    log_info(f"Avg signal difference: {avg_diff:.2e}")
    log_info(f"Median signal difference: {median_diff:.2e}")
    
    if mismatched_symbols:
        log_warn(f"\nFound {len(mismatched_symbols)} mismatched symbols:")
        for symbol, diff in sorted(mismatched_symbols, key=lambda x: x[1], reverse=True):
            log_warn(f"  {symbol}: max_diff={diff:.10e}")

    return {
        "match_rate_percent": match_rate,
        "max_difference": max_diff,
        "avg_difference": avg_diff,
        "median_difference": median_diff,
        "matching_symbols": matching_symbols,
        "total_symbols": total_symbols,
        "mismatched_symbols": [s[0] for s in mismatched_symbols],
    }


def generate_comparison_table(
    original_time: float,
    enhanced_time: float,
    original_memory: float,
    enhanced_memory: float,
    signal_comparison: Dict,
) -> str:
    """Generate formatted comparison table.

    Args:
        original_time: Execution time for original module (seconds)
        enhanced_time: Execution time for enhanced module (seconds)
        original_memory: Peak memory for original module (MB)
        enhanced_memory: Peak memory for enhanced module (MB)
        signal_comparison: Signal comparison metrics

    Returns:
        Formatted table string
    """
    speedup = original_time / enhanced_time if enhanced_time > 0 else 0
    memory_reduction = ((original_memory - enhanced_memory) / original_memory) * 100 if original_memory > 0 else 0

    # Performance table
    perf_data = [
        ["Metric", "Original", "Enhanced", "Improvement"],
        ["─" * 20, "─" * 15, "─" * 15, "─" * 15],
        [
            "Execution Time",
            f"{original_time:.2f}s",
            f"{enhanced_time:.2f}s",
            f"{speedup:.2f}x faster",
        ],
        [
            "Peak Memory",
            f"{original_memory:.1f} MB",
            f"{enhanced_memory:.1f} MB",
            f"{memory_reduction:.1f}% reduction",
        ],
    ]

    # Signal comparison table
    signal_data = [
        ["Signal Comparison", "Value"],
        ["─" * 30, "─" * 20],
        ["Match Rate", f"{signal_comparison['match_rate_percent']:.2f}%"],
        ["Matching Symbols", f"{signal_comparison['matching_symbols']}/{signal_comparison['total_symbols']}"],
        ["Max Difference", f"{signal_comparison['max_difference']:.2e}"],
        ["Avg Difference", f"{signal_comparison['avg_difference']:.2e}"],
        ["Median Difference", f"{signal_comparison.get('median_difference', 0):.2e}"],
    ]

    perf_table = tabulate(perf_data, headers="firstrow", tablefmt="grid")
    signal_table = tabulate(signal_data, headers="firstrow", tablefmt="grid")

    return f"\n{perf_table}\n\n{signal_table}\n"


def main():
    """Main benchmark execution."""
    parser = argparse.ArgumentParser(description="Benchmark adaptive_trend vs adaptive_trend_enhance")
    parser.add_argument("--symbols", type=int, default=100, help="Number of symbols to test (default: 1000)")
    parser.add_argument("--bars", type=int, default=1000, help="Number of bars per symbol (default: 1000)")
    parser.add_argument("--timeframe", type=str, default="1h", help="Timeframe (default: 1h)")

    args = parser.parse_args()

    log_info("=" * 60)
    log_info("Benchmark: adaptive_trend vs adaptive_trend_enhance")
    log_info("=" * 60)

    # Common configuration (matching defaults)
    common_config = {
        "ema_len": 28,
        "hull_len": 28,
        "wma_len": 28,
        "dema_len": 28,
        "lsma_len": 28,
        "kama_len": 28,
        "ema_w": 1.0,
        "hma_w": 1.0,
        "wma_w": 1.0,
        "dema_w": 1.0,
        "lsma_w": 1.0,
        "kama_w": 1.0,
        "robustness": "Medium",
        "La": 0.02,
        "De": 0.03,
        "cutout": 0,
        "long_threshold": 0.1,
        "short_threshold": -0.1,
        "strategy_mode": False,
    }

    enhanced_config = common_config.copy()
    enhanced_config.update(
        {
            "parallel_l1": False,
            "parallel_l2": False,
            "precision": "float64",
            "prefer_gpu": False,
        }
    )

    # Step 1: Fetch data
    prices_data = fetch_symbols_data(num_symbols=args.symbols, bars=args.bars, timeframe=args.timeframe)

    if len(prices_data) == 0:
        log_error("No data fetched, exiting")
        return

    # Step 2: Run original module
    gc.collect()  # Clean memory before benchmark
    original_results, original_time, original_memory = run_original_module(prices_data, common_config)

    # Step 3: Run enhanced module
    gc.collect()  # Clean memory before benchmark
    enhanced_results, enhanced_time, enhanced_memory = run_enhanced_module(prices_data, enhanced_config)

    # Step 4: Compare signals
    signal_comparison = compare_signals(original_results, enhanced_results)

    # Step 5: Generate comparison table
    table = generate_comparison_table(original_time, enhanced_time, original_memory, enhanced_memory, signal_comparison)

    print("\n" + "=" * 60)
    print("BENCHMARK RESULTS")
    print("=" * 60)
    print(table)

    # Save results to file (save in same directory as script)
    import os
    script_dir = os.path.dirname(os.path.abspath(__file__))
    output_file = os.path.join(script_dir, "benchmark_results.txt")
    with open(output_file, "w", encoding="utf-8") as f:
        f.write("Benchmark Results\n")
        f.write("=" * 60 + "\n")
        f.write(f"Symbols: {len(prices_data)}\n")
        f.write(f"Bars per symbol: {args.bars}\n")
        f.write(f"Timeframe: {args.timeframe}\n")
        f.write("\n")
        f.write(table)

    log_success(f"Results saved to {output_file}")


if __name__ == "__main__":
    main()
