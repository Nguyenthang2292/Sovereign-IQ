"""Main entry point for benchmark comparison."""

import argparse
import gc

from modules.common.utils import log_error, log_info, log_success, log_warn

from .build import ensure_cuda_extensions_built, ensure_rust_extensions_built
from .comparison import compare_signals, generate_comparison_table
from .data import fetch_symbols_data
from .runners import (
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


def main():
    """Main benchmark execution."""
    parser = argparse.ArgumentParser(
        description="Benchmark adaptive_trend vs adaptive_trend_enhance vs adaptive_trend_LTS (Rust)"
    )  # noqa: E501
    parser.add_argument("--symbols", type=int, default=200, help="Number of symbols to test (default: 1000)")
    parser.add_argument("--bars", type=int, default=1500, help="Number of bars per symbol (default: 1000)")
    parser.add_argument("--timeframe", type=str, default="1h", help="Timeframe (default: 1h)")
    parser.add_argument(
        "--clear-cache",
        action="store_true",
        help="Clear MA cache before running benchmark (recommended for accurate comparison)",
    )

    args = parser.parse_args()

    log_info("=" * 60)
    log_info("Benchmark: All 8 versions (Original, Enhanced, Rust, CUDA, Dask, Rust+Dask, CUDA+Dask, All Three)")
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
            "use_rust_backend": False,
        }
    )

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


if __name__ == "__main__":
    main()
