"""
Benchmark Comparison: adaptive_trend vs adaptive_trend_enhance vs adaptive_trend_LTS (Rust) vs CUDA

This script loads symbols via ExchangeManager/DataFetcher and compares:
1. Signal output consistency (100% match expected)
2. Execution time (speedup measurement)
3. Memory usage
4. Performance metrics

Compares four versions:
- Original: modules.adaptive_trend.core.compute_atc_signals
- Enhanced: modules.adaptive_trend_enhance.core.compute_atc_signals
- Rust (v2): modules.adaptive_trend_LTS.core.compute_atc_signals (uses Rust backend)
- CUDA: modules.adaptive_trend_LTS.core.compute_atc_signals (uses CUDA kernels)

Usage:
    python benchmark_comparison.py --symbols 1000 --bars 1000
    Or from project root: python modules/adaptive_trend_LTS/benchmarks/benchmark_comparison.py
"""

import argparse
import gc
import subprocess
import sys
import time
from pathlib import Path
from typing import Dict, Tuple

# Add project root to sys.path for imports
# This ensures the script can be run from any directory
# From modules/adaptive_trend_LTS/benchmarks/ -> adaptive_trend_LTS/ -> modules/ -> project root (4 levels)
try:
    project_root = Path(__file__).parent.parent.parent.parent
    project_root_str = str(project_root.resolve())
    if project_root_str not in sys.path:
        sys.path.insert(0, project_root_str)
except NameError:
    # If __file__ is not defined (e.g., in interactive mode), assume we're at project root
    pass

import numpy as np
import pandas as pd
from tabulate import tabulate

# Import all three modules
from modules.adaptive_trend.core import compute_atc_signals as compute_atc_original
from modules.adaptive_trend_enhance.core import compute_atc_signals as compute_atc_enhanced
from modules.adaptive_trend_LTS.core.compute_atc_signals import compute_atc_signals as compute_atc_rust
from modules.adaptive_trend_LTS.core.compute_atc_signals.batch_processor import process_symbols_batch_cuda
from modules.common.core import DataFetcher, ExchangeManager
from modules.common.utils import log_error, log_info, log_success, log_warn


def ensure_rust_extensions_built():
    """Ensure Rust extensions are built and up-to-date."""
    log_info("Ensuring Rust extensions are built...")
    try:
        # Locate rust_extensions directory
        # From benchmarks/ -> ../rust_extensions
        script_dir = Path(__file__).parent
        rust_ext_dir = script_dir.parent / "rust_extensions"

        if not rust_ext_dir.exists():
            log_error(f"Rust extensions directory not found at {rust_ext_dir}")
            return

        log_info(f"Building Rust extensions in {rust_ext_dir}...")

        # Run maturin develop --release
        # Use sys.executable to ensure we use the same python environment
        cmd = [sys.executable, "-m", "maturin", "develop", "--release"]

        subprocess.run(cmd, cwd=str(rust_ext_dir), capture_output=True, text=True, check=True, encoding="utf-8")

        log_success("Rust extensions built successfully")

    except subprocess.CalledProcessError as e:
        log_error(f"Failed to build Rust extensions: {e}")
        log_error(f"Stdout: {e.stdout}")
        log_error(f"Stderr: {e.stderr}")
        # We don't exit here, allowing the script to try running anyway (maybe old binary works)
    except Exception as e:
        log_error(f"Error building Rust extensions: {e}")


def ensure_cuda_extensions_built():
    """Ensure CUDA extensions are built and up-to-date."""
    log_info("Ensuring CUDA extensions are built...")
    try:
        script_dir = Path(__file__).parent
        rust_ext_dir = script_dir.parent / "rust_extensions"
        build_script = rust_ext_dir / "build_cuda.ps1"

        if not build_script.exists():
            log_warn(f"CUDA build script not found at {build_script}, skipping CUDA build")
            return

        log_info(f"Building CUDA extensions using {build_script}...")

        # Run PowerShell script
        cmd = ["powershell.exe", "-ExecutionPolicy", "Bypass", "-File", str(build_script)]
        result = subprocess.run(cmd, cwd=str(rust_ext_dir), capture_output=True, text=True, encoding="utf-8")

        if result.returncode == 0:
            log_success("CUDA extensions built successfully")
        else:
            log_warn(f"CUDA build completed with warnings or errors")
            log_warn(f"Stdout: {result.stdout}")
            log_warn(f"Stderr: {result.stderr}")

    except Exception as e:
        log_warn(f"Error building CUDA extensions: {e}")
        log_warn("CUDA benchmarks may not be available")


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


def run_rust_module(prices_data: Dict[str, pd.Series], config: dict) -> Tuple[Dict[str, Dict], float, float]:
    """Run Rust-accelerated adaptive_trend_LTS module.

    Args:
        prices_data: Dictionary of symbol -> price Series
        config: ATC configuration parameters

    Returns:
        Tuple of (results_dict, execution_time_seconds, peak_memory_mb)
    """
    log_info("Running Rust-accelerated adaptive_trend_LTS module...")

    results = {}
    start_time = time.time()

    # Track memory (approximate)
    import psutil

    process = psutil.Process()
    mem_before = process.memory_info().rss / 1024 / 1024  # MB

    for idx, (symbol, prices) in enumerate(prices_data.items(), 1):
        try:
            result = compute_atc_rust(prices=prices, **config)
            results[symbol] = result

            if idx % 100 == 0:
                log_info(f"Rust: Processed {idx}/{len(prices_data)} symbols")

        except Exception as e:
            import traceback

            log_error(f"Rust failed for {symbol}: {e}")
            traceback.print_exc()
            results[symbol] = None

    end_time = time.time()
    mem_after = process.memory_info().rss / 1024 / 1024  # MB

    execution_time = end_time - start_time
    peak_memory = mem_after - mem_before

    log_success(f"Rust module completed in {execution_time:.2f}s")
    return results, execution_time, peak_memory


def run_cuda_module(prices_data: Dict[str, pd.Series], config: dict) -> Tuple[Dict[str, Dict], float, float]:
    """Run CUDA-accelerated adaptive_trend_LTS module using concurrent batch processing.

    Args:
        prices_data: Dictionary of symbol -> price Series
        config: ATC configuration parameters

    Returns:
        Tuple of (results_dict, execution_time_seconds, peak_memory_mb)
    """
    log_info("Running CUDA-accelerated adaptive_trend_LTS module (Concurrent Batch)...")

    start_time = time.time()

    # Track memory (approximate)
    import psutil

    process = psutil.Process()
    mem_before = process.memory_info().rss / 1024 / 1024  # MB

    try:
        # Use concurrent batch processor
        # This uses ThreadPoolExecutor to overlap Rust-CUDA tasks
        results = process_symbols_batch_cuda(prices_data, config, num_threads=4)

        log_info(f"CUDA: Processed {len(results)}/{len(prices_data)} symbols")

    except Exception as e:
        import traceback

        log_error(f"CUDA batch processing failed: {e}")
        traceback.print_exc()
        results = {}

    end_time = time.time()
    mem_after = process.memory_info().rss / 1024 / 1024  # MB

    execution_time = end_time - start_time
    peak_memory = mem_after - mem_before

    log_success(f"CUDA module completed in {execution_time:.2f}s")
    return results, execution_time, peak_memory


def compare_signals(
    original_results: Dict[str, Dict],
    enhanced_results: Dict[str, Dict],
    rust_results: Dict[str, Dict],
    cuda_results: Dict[str, Dict],
) -> Dict[str, any]:
    """Compare signal outputs between original, enhanced, Rust, and CUDA modules.

    Args:
        original_results: Results from original module
        enhanced_results: Results from enhanced module
        rust_results: Results from Rust module
        cuda_results: Results from CUDA module

    Returns:
        Dictionary of comparison metrics
    """
    log_info("Comparing signal outputs between all four versions...")

    # DEBUG: Print result dict sizes and sample keys
    log_info(
        f"Result dict sizes: orig={len(original_results)}, enh={len(enhanced_results)}, rust={len(rust_results)}, cuda={len(cuda_results)}"
    )
    if original_results:
        sample_key = list(original_results.keys())[0]
        log_info(f"Sample key: {sample_key}")
        log_info(f"Sample orig result type: {type(original_results[sample_key])}")
        if original_results[sample_key]:
            log_info(
                f"Sample orig result keys: {list(original_results[sample_key].keys()) if isinstance(original_results[sample_key], dict) else 'not a dict'}"
            )

    total_symbols = len(original_results)
    processed_symbols = 0

    # Compare Original vs Enhanced
    orig_enh_diffs = []
    orig_enh_matching = 0
    orig_enh_mismatched = []

    # Compare Original vs Rust
    orig_rust_diffs = []
    orig_rust_matching = 0
    orig_rust_mismatched = []

    # Compare Original vs CUDA
    orig_cuda_diffs = []
    orig_cuda_matching = 0
    orig_cuda_mismatched = []

    # Compare Enhanced vs Rust
    enh_rust_diffs = []
    enh_rust_matching = 0
    enh_rust_mismatched = []

    # Compare Rust vs CUDA
    rust_cuda_diffs = []
    rust_cuda_matching = 0
    rust_cuda_mismatched = []

    for symbol in original_results.keys():
        if symbol not in enhanced_results or symbol not in rust_results or symbol not in cuda_results:
            log_warn(f"Symbol {symbol} missing in results")
            continue

        orig = original_results[symbol]
        enh = enhanced_results[symbol]
        rust = rust_results[symbol]
        cuda = cuda_results[symbol]

        if orig is None or enh is None or rust is None or cuda is None:
            log_warn(f"Symbol {symbol} has None result")
            continue

        # Compare Average_Signal
        orig_s = orig.get("Average_Signal")
        enh_s = enh.get("Average_Signal")
        rust_s = rust.get("Average_Signal")
        cuda_s = cuda.get("Average_Signal")

        if orig_s is None or enh_s is None or rust_s is None or cuda_s is None:
            log_warn(f"Symbol {symbol} missing Average_Signal")
            continue

        # Find common index across all four series
        # The previous sequential alignment logic was buggy and caused index desynchronization
        common_index = orig_s.index.intersection(enh_s.index).intersection(rust_s.index).intersection(cuda_s.index)

        if len(common_index) == 0:
            log_warn(f"No common index after alignment for {symbol}")
            continue

        # Reindex all series to common index - this ensures all four have identical indices
        orig_s = orig_s.loc[common_index]
        enh_s = enh_s.loc[common_index]
        rust_s = rust_s.loc[common_index]
        cuda_s = cuda_s.loc[common_index]

        # Original vs Enhanced
        diff_oe = np.abs(orig_s - enh_s).max()
        orig_enh_diffs.append(diff_oe)
        if diff_oe < 1e-6:
            orig_enh_matching += 1
        else:
            orig_enh_mismatched.append((symbol, diff_oe))

        # Original vs Rust
        diff_or = np.abs(orig_s - rust_s).max()
        orig_rust_diffs.append(diff_or)
        if diff_or < 1e-6:
            orig_rust_matching += 1
        else:
            orig_rust_mismatched.append((symbol, diff_or))

        # Original vs CUDA
        diff_oc = np.abs(orig_s - cuda_s).max()
        orig_cuda_diffs.append(diff_oc)
        if diff_oc < 1e-6:
            orig_cuda_matching += 1
        else:
            orig_cuda_mismatched.append((symbol, diff_oc))

        # Enhanced vs Rust
        diff_er = np.abs(enh_s - rust_s).max()
        enh_rust_diffs.append(diff_er)
        if diff_er < 1e-6:
            enh_rust_matching += 1
        else:
            enh_rust_mismatched.append((symbol, diff_er))

        # Rust vs CUDA
        diff_rc = np.abs(rust_s - cuda_s).max()
        rust_cuda_diffs.append(diff_rc)
        if diff_rc < 1e-6:
            rust_cuda_matching += 1
        else:
            rust_cuda_mismatched.append((symbol, diff_rc))

        # Increment processed counter
        processed_symbols += 1

    if orig_rust_mismatched:
        log_info(f"Mismatched symbols (Orig vs Rust): {[s[0] for s in orig_rust_mismatched[:10]]}...")
    if orig_cuda_mismatched:
        log_info(f"Mismatched symbols (Orig vs CUDA): {[s[0] for s in orig_cuda_mismatched[:10]]}...")

    # Calculate metrics
    log_info(f"Processed {processed_symbols}/{total_symbols} symbols successfully")
    orig_enh_match_rate = (orig_enh_matching / total_symbols) * 100 if total_symbols > 0 else 0
    orig_enh_match_rate = (orig_enh_matching / total_symbols) * 100 if total_symbols > 0 else 0
    orig_rust_match_rate = (orig_rust_matching / total_symbols) * 100 if total_symbols > 0 else 0
    orig_cuda_match_rate = (orig_cuda_matching / total_symbols) * 100 if total_symbols > 0 else 0
    enh_rust_match_rate = (enh_rust_matching / total_symbols) * 100 if total_symbols > 0 else 0
    rust_cuda_match_rate = (rust_cuda_matching / total_symbols) * 100 if total_symbols > 0 else 0

    log_success(f"Original vs Enhanced match rate: {orig_enh_match_rate:.2f}%")
    log_success(f"Original vs Rust match rate: {orig_rust_match_rate:.2f}%")
    log_success(f"Original vs CUDA match rate: {orig_cuda_match_rate:.2f}%")
    log_success(f"Enhanced vs Rust match rate: {enh_rust_match_rate:.2f}%")
    log_success(f"Rust vs CUDA match rate: {rust_cuda_match_rate:.2f}%")

    return {
        "orig_enh": {
            "match_rate_percent": orig_enh_match_rate,
            "max_difference": max(orig_enh_diffs) if orig_enh_diffs else 0,
            "avg_difference": np.mean(orig_enh_diffs) if orig_enh_diffs else 0,
            "median_difference": np.median(orig_enh_diffs) if orig_enh_diffs else 0,
            "matching_symbols": orig_enh_matching,
            "mismatched_symbols": [s[0] for s in orig_enh_mismatched],
        },
        "orig_rust": {
            "match_rate_percent": orig_rust_match_rate,
            "max_difference": max(orig_rust_diffs) if orig_rust_diffs else 0,
            "avg_difference": np.mean(orig_rust_diffs) if orig_rust_diffs else 0,
            "median_difference": np.median(orig_rust_diffs) if orig_rust_diffs else 0,
            "matching_symbols": orig_rust_matching,
            "mismatched_symbols": [s[0] for s in orig_rust_mismatched],
        },
        "orig_cuda": {
            "match_rate_percent": orig_cuda_match_rate,
            "max_difference": max(orig_cuda_diffs) if orig_cuda_diffs else 0,
            "avg_difference": np.mean(orig_cuda_diffs) if orig_cuda_diffs else 0,
            "median_difference": np.median(orig_cuda_diffs) if orig_cuda_diffs else 0,
            "matching_symbols": orig_cuda_matching,
            "mismatched_symbols": [s[0] for s in orig_cuda_mismatched],
        },
        "enh_rust": {
            "match_rate_percent": enh_rust_match_rate,
            "max_difference": max(enh_rust_diffs) if enh_rust_diffs else 0,
            "avg_difference": np.mean(enh_rust_diffs) if enh_rust_diffs else 0,
            "median_difference": np.median(enh_rust_diffs) if enh_rust_diffs else 0,
            "matching_symbols": enh_rust_matching,
            "mismatched_symbols": [s[0] for s in enh_rust_mismatched],
        },
        "rust_cuda": {
            "match_rate_percent": rust_cuda_match_rate,
            "max_difference": max(rust_cuda_diffs) if rust_cuda_diffs else 0,
            "avg_difference": np.mean(rust_cuda_diffs) if rust_cuda_diffs else 0,
            "median_difference": np.median(rust_cuda_diffs) if rust_cuda_diffs else 0,
            "matching_symbols": rust_cuda_matching,
            "mismatched_symbols": [s[0] for s in rust_cuda_mismatched],
        },
        "total_symbols": total_symbols,
    }


def generate_comparison_table(
    original_time: float,
    enhanced_time: float,
    rust_time: float,
    cuda_time: float,
    original_memory: float,
    enhanced_memory: float,
    rust_memory: float,
    cuda_memory: float,
    signal_comparison: Dict,
) -> str:
    """Generate formatted comparison table for four versions.

    Args:
        original_time: Execution time for original module (seconds)
        enhanced_time: Execution time for enhanced module (seconds)
        rust_time: Execution time for Rust module (seconds)
        cuda_time: Execution time for CUDA module (seconds)
        original_memory: Peak memory for original module (MB)
        enhanced_memory: Peak memory for enhanced module (MB)
        rust_memory: Peak memory for Rust module (MB)
        cuda_memory: Peak memory for CUDA module (MB)
        signal_comparison: Signal comparison metrics

    Returns:
        Formatted table string
    """
    speedup_enh = original_time / enhanced_time if enhanced_time > 0 else 0
    speedup_rust = original_time / rust_time if rust_time > 0 else 0
    speedup_cuda = original_time / cuda_time if cuda_time > 0 else 0
    speedup_cuda_vs_rust = rust_time / cuda_time if cuda_time > 0 else 0

    memory_reduction_enh = ((original_memory - enhanced_memory) / original_memory) * 100 if original_memory > 0 else 0
    memory_reduction_rust = ((original_memory - rust_memory) / original_memory) * 100 if original_memory > 0 else 0
    memory_reduction_cuda = ((original_memory - cuda_memory) / original_memory) * 100 if original_memory > 0 else 0

    # Performance table
    perf_data = [
        [
            "Metric",
            "Original",
            "Enhanced",
            "Rust (v2)",
            "CUDA",
            "Enh vs Orig",
            "Rust vs Orig",
            "CUDA vs Orig",
            "CUDA vs Rust",
        ],
        ["─" * 15, "─" * 12, "─" * 12, "─" * 12, "─" * 12, "─" * 15, "─" * 15, "─" * 15, "─" * 15],
        [
            "Execution Time",
            f"{original_time:.2f}s",
            f"{enhanced_time:.2f}s",
            f"{rust_time:.2f}s",
            f"{cuda_time:.2f}s",
            f"{speedup_enh:.2f}x",
            f"{speedup_rust:.2f}x",
            f"{speedup_cuda:.2f}x",
            f"{speedup_cuda_vs_rust:.2f}x",
        ],
        [
            "Peak Memory",
            f"{original_memory:.1f} MB",
            f"{enhanced_memory:.1f} MB",
            f"{rust_memory:.1f} MB",
            f"{cuda_memory:.1f} MB",
            f"{memory_reduction_enh:.1f}%",
            f"{memory_reduction_rust:.1f}%",
            f"{memory_reduction_cuda:.1f}%",
            (f"{(rust_memory - cuda_memory) / rust_memory * 100:.1f}%" if rust_memory > 0 else "-"),
        ],
    ]

    # Signal comparison table
    signal_data = [
        [
            "Signal Comparison",
            "Original vs Enhanced",
            "Original vs Rust",
            "Original vs CUDA",
            "Enhanced vs Rust",
            "Rust vs CUDA",
        ],
        ["─" * 25, "─" * 25, "─" * 25, "─" * 25, "─" * 25, "─" * 25],
        [
            "Match Rate",
            f"{signal_comparison['orig_enh']['match_rate_percent']:.2f}%",
            f"{signal_comparison['orig_rust']['match_rate_percent']:.2f}%",
            f"{signal_comparison['orig_cuda']['match_rate_percent']:.2f}%",
            f"{signal_comparison['enh_rust']['match_rate_percent']:.2f}%",
            f"{signal_comparison['rust_cuda']['match_rate_percent']:.2f}%",
        ],
        [
            "Matching Symbols",
            f"{signal_comparison['orig_enh']['matching_symbols']}/{signal_comparison['total_symbols']}",
            f"{signal_comparison['orig_rust']['matching_symbols']}/{signal_comparison['total_symbols']}",
            f"{signal_comparison['orig_cuda']['matching_symbols']}/{signal_comparison['total_symbols']}",
            f"{signal_comparison['enh_rust']['matching_symbols']}/{signal_comparison['total_symbols']}",
            f"{signal_comparison['rust_cuda']['matching_symbols']}/{signal_comparison['total_symbols']}",
        ],
        [
            "Max Difference",
            f"{signal_comparison['orig_enh']['max_difference']:.2e}",
            f"{signal_comparison['orig_rust']['max_difference']:.2e}",
            f"{signal_comparison['orig_cuda']['max_difference']:.2e}",
            f"{signal_comparison['enh_rust']['max_difference']:.2e}",
            f"{signal_comparison['rust_cuda']['max_difference']:.2e}",
        ],
        [
            "Avg Difference",
            f"{signal_comparison['orig_enh']['avg_difference']:.2e}",
            f"{signal_comparison['orig_rust']['avg_difference']:.2e}",
            f"{signal_comparison['orig_cuda']['avg_difference']:.2e}",
            f"{signal_comparison['enh_rust']['avg_difference']:.2e}",
            f"{signal_comparison['rust_cuda']['avg_difference']:.2e}",
        ],
        [
            "Median Difference",
            f"{signal_comparison['orig_enh']['median_difference']:.2e}",
            f"{signal_comparison['orig_rust']['median_difference']:.2e}",
            f"{signal_comparison['orig_cuda']['median_difference']:.2e}",
            f"{signal_comparison['enh_rust']['median_difference']:.2e}",
            f"{signal_comparison['rust_cuda']['median_difference']:.2e}",
        ],
    ]

    perf_table = tabulate(perf_data, headers="firstrow", tablefmt="grid")
    signal_table = tabulate(signal_data, headers="firstrow", tablefmt="grid")

    return f"\n{perf_table}\n\n{signal_table}\n"


def main():
    """Main benchmark execution."""
    parser = argparse.ArgumentParser(
        description="Benchmark adaptive_trend vs adaptive_trend_enhance vs adaptive_trend_LTS (Rust)"
    )  # noqa: E501
    parser.add_argument("--symbols", type=int, default=1000, help="Number of symbols to test (default: 1000)")
    parser.add_argument("--bars", type=int, default=1500, help="Number of bars per symbol (default: 1000)")
    parser.add_argument("--timeframe", type=str, default="1h", help="Timeframe (default: 1h)")
    parser.add_argument(
        "--clear-cache",
        action="store_true",
        help="Clear MA cache before running benchmark (recommended for accurate comparison)",
    )

    args = parser.parse_args()

    log_info("=" * 60)
    log_info("Benchmark: adaptive_trend vs adaptive_trend_enhance vs adaptive_trend_LTS (Rust) vs CUDA")
    log_info("=" * 60)

    # Clear cache if requested
    if args.clear_cache:
        log_info("Clearing MA cache...")
        import shutil
        from pathlib import Path

        cache_dir = Path(".cache/atc")
        if cache_dir.exists():
            try:
                shutil.rmtree(cache_dir)
                log_success(f"Cache cleared: {cache_dir}")
            except Exception as e:
                log_warn(f"Failed to clear cache: {e}")
        else:
            log_info("Cache directory does not exist, skipping")

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

    rust_config = common_config.copy()
    rust_config.update(
        {
            "parallel_l1": False,
            "parallel_l2": True,
            "precision": "float64",
            "prefer_gpu": True,  # Rust backend will use Rust for MAs
            "use_cache": True,
            "fast_mode": True,
        }
    )

    cuda_config = common_config.copy()
    cuda_config.update(
        {
            "parallel_l1": False,
            "parallel_l2": True,
            "precision": "float64",
            "prefer_gpu": True,
            "use_cache": True,
            "fast_mode": True,
            "use_cuda": True,  # Enable CUDA kernels
        }
    )

    # Step 0: Ensure Rust and CUDA extensions are built
    ensure_rust_extensions_built()
    ensure_cuda_extensions_built()

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

    # Step 4: Run Rust module
    gc.collect()  # Clean memory before benchmark
    rust_results, rust_time, rust_memory = run_rust_module(prices_data, rust_config)

    # Step 5: Run CUDA module
    gc.collect()  # Clean memory before benchmark
    cuda_results, cuda_time, cuda_memory = run_cuda_module(prices_data, cuda_config)

    # Step 6: Compare signals
    signal_comparison = compare_signals(original_results, enhanced_results, rust_results, cuda_results)

    # Step 7: Generate comparison table
    table = generate_comparison_table(
        original_time,
        enhanced_time,
        rust_time,
        cuda_time,
        original_memory,
        enhanced_memory,
        rust_memory,
        cuda_memory,
        signal_comparison,
    )

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
