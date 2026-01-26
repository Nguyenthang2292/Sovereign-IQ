"""Main entry point for benchmark comparison."""

import argparse
import gc
import sys
from datetime import datetime
from pathlib import Path
from typing import TextIO


class TeeOutput:
    """Class to write to both console and file simultaneously."""

    def __init__(self, file: TextIO):
        self.file = file
        self.stdout = sys.stdout
        self.stderr = sys.stderr

    def write(self, text: str) -> None:
        """Write to both console and file."""
        self.stdout.write(text)
        self.file.write(text)
        self.file.flush()

    def flush(self) -> None:
        """Flush both streams."""
        self.stdout.flush()
        self.file.flush()

    def isatty(self) -> bool:
        """Return True if connected to a TTY device (for colorama compatibility)."""
        return self.stdout.isatty()

# Add project root to sys.path to allow absolute imports when run directly
if __file__:
    project_root = Path(__file__).parent.parent.parent.parent.parent
    project_root_str = str(project_root)
    if project_root_str not in sys.path:
        sys.path.insert(0, project_root_str)

from modules.adaptive_trend_LTS.benchmarks.benchmark_comparison.build import (
    ensure_cuda_extensions_built,
    ensure_rust_extensions_built,
)
from modules.adaptive_trend_LTS.benchmarks.benchmark_comparison.comparison import (
    compare_signals,
    generate_comparison_table,
)
from modules.adaptive_trend_LTS.benchmarks.benchmark_comparison.data import fetch_symbols_data
from modules.adaptive_trend_LTS.benchmarks.benchmark_comparison.html_formatter import ansi_to_html
from modules.adaptive_trend_LTS.benchmarks.benchmark_comparison.runners import (
    run_cuda_dask_module,
    run_cuda_module,
    run_dask_module,
    run_enhanced_module,
    run_original_module,
    run_rust_batch_module,
    run_rust_cuda_dask_module,
    run_rust_dask_module,
    run_rust_module,
)
from modules.common.utils import log_error, log_info, log_success, log_warn


def main():
    """Main benchmark execution."""
    parser = argparse.ArgumentParser(
        description="Benchmark adaptive_trend vs adaptive_trend_enhance vs adaptive_trend_LTS (Rust)"
    )  # noqa: E501
    parser.add_argument("--symbols", type=int, default=20, help="Number of symbols to test (default: 1000)")
    parser.add_argument("--bars", type=int, default=500, help="Number of bars per symbol (default: 1000)")
    parser.add_argument("--timeframe", type=str, default="1h", help="Timeframe (default: 1h)")
    parser.add_argument(
        "--clear-cache",
        action="store_true",
        help="Clear MA cache before running benchmark (recommended for accurate comparison)",
    )

    args = parser.parse_args()

    # Setup log file
    script_dir = Path(__file__).parent.parent
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file = script_dir / f"benchmark_log_{timestamp}.txt"

    # Create TeeOutput to write to both console and file
    log_file_handle = open(str(log_file), "w", encoding="utf-8")
    tee = TeeOutput(log_file_handle)

    # Write header to log file
    log_file_handle.write("=" * 60 + "\n")
    log_file_handle.write(f"Benchmark Execution Log - {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
    log_file_handle.write("=" * 60 + "\n")
    log_file_handle.write(f"Arguments: symbols={args.symbols}, bars={args.bars}, timeframe={args.timeframe}\n")
    log_file_handle.write(f"Clear cache: {args.clear_cache}\n")
    log_file_handle.write("=" * 60 + "\n\n")
    log_file_handle.flush()

    # Save original stdout before redirecting
    original_stdout = sys.stdout

    try:
        # Redirect stdout to TeeOutput (captures print statements and log functions)
        sys.stdout = tee

        log_info("=" * 60)
        log_info("Benchmark: All 8 versions (Original, Enhanced, Rust, CUDA, Dask, Rust+Dask, CUDA+Dask, All Three)")
        log_info("=" * 60)
        log_info("")

        # Rebuild Rust extensions to ensure latest code changes are included
        log_info("=" * 60)
        log_info("STEP 1: Rebuilding Rust & CUDA Extensions")
        log_info("=" * 60)
        log_info("This ensures all code changes (including CUDA kernel fixes) are compiled")
        log_info("")

        ensure_rust_extensions_built()
        log_info("")

        # Note: CUDA extensions are built as part of Rust build (via maturin)
        # Separate CUDA build is only needed for standalone CUDA compilation
        # ensure_cuda_extensions_built()  # Uncomment if using separate CUDA build

        log_info("=" * 60)
        log_info("STEP 2: Running Benchmarks")
        log_info("=" * 60)
        log_info("")

        # Clear cache if requested
        if args.clear_cache:
            log_info("Clearing MA cache...")
            import shutil

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

        # Enhanced uses same config as Original (no special params needed)
        enhanced_config = common_config.copy()

        rust_config = common_config.copy()
        rust_config.update(
            {
                "parallel_l1": False,
                "parallel_l2": True,
                "precision": "float64",
                "use_rust_backend": True,  # Rust backend will use Rust for MAs
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
                "use_rust_backend": True,
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

        # Step 5: Run Rust Rayon module
        gc.collect()  # Clean memory before benchmark
        rust_rayon_results, rust_rayon_time, rust_rayon_memory = run_rust_batch_module(prices_data, rust_config)

        # Step 6: Run CUDA module
        gc.collect()  # Clean memory before benchmark
        cuda_results, cuda_time, cuda_memory = run_cuda_module(prices_data, cuda_config)

        # Step 7: Run Dask module
        gc.collect()  # Clean memory before benchmark
        dask_results, dask_time, dask_memory = run_dask_module(prices_data, common_config)

        # Step 8: Run Rust+Dask hybrid module
        gc.collect()  # Clean memory before benchmark
        rust_dask_results, rust_dask_time, rust_dask_memory = run_rust_dask_module(prices_data, rust_config)

        # Step 9: Run CUDA+Dask hybrid module
        gc.collect()  # Clean memory before benchmark
        cuda_dask_results, cuda_dask_time, cuda_dask_memory = run_cuda_dask_module(prices_data, cuda_config)

        # Step 10: Run Rust+CUDA+Dask hybrid module (all three)
        gc.collect()  # Clean memory before benchmark
        all_three_results, all_three_time, all_three_memory = run_rust_cuda_dask_module(prices_data, cuda_config)

        # Step 11: Compare signals
        signal_comparison = compare_signals(
            original_results,
            enhanced_results,
            rust_results,
            rust_rayon_results,
            cuda_results,
            dask_results,
            rust_dask_results,
            cuda_dask_results,
            all_three_results,
        )

        # Step 12: Generate comparison table
        table = generate_comparison_table(
            original_time,
            enhanced_time,
            rust_time,
            rust_rayon_time,
            cuda_time,
            dask_time,
            rust_dask_time,
            cuda_dask_time,
            all_three_time,
            original_memory,
            enhanced_memory,
            rust_memory,
            rust_rayon_memory,
            cuda_memory,
            dask_memory,
            rust_dask_memory,
            cuda_dask_memory,
            all_three_memory,
            signal_comparison,
        )

        print("\n" + "=" * 60)
        print("BENCHMARK RESULTS")
        print("=" * 60)
        print(table)

        # Save results to file (save in parent benchmarks directory)
        script_dir = Path(__file__).parent.parent
        output_file = script_dir / "benchmark_results.txt"
        with open(str(output_file), "w", encoding="utf-8") as f:
            f.write("Benchmark Results\n")
            f.write("=" * 60 + "\n")
            f.write(f"Symbols: {len(prices_data)}\n")
            f.write(f"Bars per symbol: {args.bars}\n")
            f.write(f"Timeframe: {args.timeframe}\n")
            f.write("\n")
            f.write(table)

        log_success(f"Results saved to {output_file}")

    finally:
        # Restore original stdout before writing final message
        sys.stdout = original_stdout

        # Write footer to log file
        log_file_handle.write("\n" + "=" * 60 + "\n")
        log_file_handle.write(f"Benchmark completed at {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        log_file_handle.write("=" * 60 + "\n")
        log_file_handle.close()

        # Create HTML version with colors for easier debugging
        try:
            # Read the text log file
            with open(str(log_file), "r", encoding="utf-8") as f:
                log_content = f.read()

            # Convert ANSI codes to HTML
            html_content = ansi_to_html(log_content)

            # Save HTML version
            html_file = log_file.with_suffix('.html')
            with open(str(html_file), "w", encoding="utf-8") as f:
                f.write(html_content)

            log_success(f"Logs saved to {log_file}")
            log_success(f"Colored HTML log saved to {html_file}")
        except Exception as e:
            log_warn(f"Failed to create HTML log: {e}")
            log_success(f"Logs saved to {log_file}")


if __name__ == "__main__":
    main()
