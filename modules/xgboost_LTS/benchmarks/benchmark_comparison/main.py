"""Main entry point for XGBoost benchmark comparison."""

import argparse
import sys
from datetime import datetime
from pathlib import Path
from typing import TextIO

# Add project root to sys.path
if __file__:
    project_root = Path(__file__).parent.parent.parent.parent.parent
    project_root_str = str(project_root)
    if project_root_str not in sys.path:
        sys.path.insert(0, project_root_str)

from modules.common.utils import log_error, log_info, log_success
from modules.xgboost_LTS.benchmarks.benchmark_comparison.comparison import compare_results, generate_comparison_table
from modules.xgboost_LTS.benchmarks.benchmark_comparison.data import fetch_symbols_data
from modules.xgboost_LTS.benchmarks.benchmark_comparison.html_formatter import ansi_to_html
from modules.xgboost_LTS.benchmarks.benchmark_comparison.runners import (
    run_batch_parallel,
    run_cached,
    run_gpu_accelerated,
    run_original_python,
    run_rust_accelerated,
)
from modules.xgboost_LTS.utils.cache_manager import CacheManager


class TeeOutput:
    """Class to write to both console and file simultaneously."""

    def __init__(self, file: TextIO):
        self.file = file
        self.stdout = sys.stdout

    def write(self, text: str) -> None:
        self.stdout.write(text)
        self.file.write(text)
        self.file.flush()

    def flush(self) -> None:
        self.stdout.flush()
        self.file.flush()

    def isatty(self) -> bool:
        return self.stdout.isatty()


def main():
    parser = argparse.ArgumentParser(description="Benchmark XGBoost LTS Performance")
    parser.add_argument("--symbols", type=int, default=10, help="Number of symbols to test (default: 10)")
    parser.add_argument("--bars", type=int, default=2000, help="Number of bars per symbol (default: 2000)")
    parser.add_argument("--timeframe", type=str, default="1h", help="Timeframe (default: 1h)")
    parser.add_argument("--clear-cache", action="store_true", help="Clear cache before running")

    args = parser.parse_args()

    # Output setup
    results_dir = Path(__file__).parent / "results"
    results_dir.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file = results_dir / f"benchmark_log_{timestamp}.txt"

    log_file_handle = open(str(log_file), "w", encoding="utf-8")
    tee = TeeOutput(log_file_handle)
    original_stdout = sys.stdout
    sys.stdout = tee

    try:
        log_info("=" * 60)
        log_info("XGBoost LTS Benchmark Comparison")
        log_info(f"Symbols: {args.symbols}, Bars: {args.bars}, Timeframe: {args.timeframe}")
        log_info("=" * 60)

        # Clear cache
        if args.clear_cache:
            log_info("Clearing cache...")
            cache_mgr = CacheManager()
            cache_mgr.clear_cache()
            log_success("Cache cleared.")

        # 1. Fetch data
        symbols_data = fetch_symbols_data(num_symbols=args.symbols, bars=args.bars, timeframe=args.timeframe)
        if not symbols_data:
            log_error("No data fetched.")
            return

        # 2. Run Benchmarks
        # Original (Simulated)
        res_orig, time_orig, mem_orig = run_original_python(symbols_data)

        # Rust (CPU)
        res_rust, time_rust, mem_rust = run_rust_accelerated(symbols_data)

        # GPU
        res_gpu, time_gpu, mem_gpu = run_gpu_accelerated(symbols_data)

        # Cached
        res_cached, time_cached, mem_cached = run_cached(symbols_data)

        # Batch Parallel
        res_batch, time_batch, mem_batch = run_batch_parallel(symbols_data)

        # 3. Comparison
        metrics = compare_results(res_orig, res_rust, res_gpu, res_cached, res_batch)

        table = generate_comparison_table(
            time_orig,
            time_rust,
            time_gpu,
            time_cached,
            time_batch,
            mem_orig,
            mem_rust,
            mem_gpu,
            mem_cached,
            mem_batch,
            metrics,
        )

        print("\n" + table)

        # Save results
        with open(results_dir / f"benchmark_results_{timestamp}.txt", "w", encoding="utf-8") as f:
            f.write(table)

        # HTML Report
        input_text = f"XGBoost Benchmark Report\n{table}"
        with open(results_dir / f"benchmark_results_{timestamp}.html", "w", encoding="utf-8") as f:
            f.write(ansi_to_html(input_text))

        log_success(f"Benchmarks completed. Results saved to {results_dir}")

    finally:
        sys.stdout = original_stdout
        log_file_handle.close()


if __name__ == "__main__":
    main()
