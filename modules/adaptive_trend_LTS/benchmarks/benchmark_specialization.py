"""Micro-benchmark for JIT specialization performance.

This module benchmarks the performance difference between specialized
(JIT-compiled) and generic (non-specialized) ATC computation paths.
"""

import time
from typing import Any, Callable, Dict

import numpy as np
import pandas as pd

from modules.adaptive_trend_LTS.core.codegen.specialization import (
    compute_atc_specialized,
    get_specialized_compute_fn,
    is_config_specializable,
)
from modules.adaptive_trend_LTS.core.compute_atc_signals.compute_atc_signals import (
    compute_atc_signals,
)
from modules.adaptive_trend_LTS.utils.config import ATCConfig


def generate_test_data(n: int = 1000) -> pd.Series:
    """Generate synthetic price data for benchmarking.

    Args:
        n: Number of bars

    Returns:
        Price series
    """
    np.random.seed(42)
    prices_arr = 100 + np.cumsum(np.random.randn(n) * 0.1)
    return pd.Series(prices_arr, name="close")


def benchmark_function(
    func: Callable[[pd.Series], dict[str, pd.Series]],
    prices: pd.Series,
    num_runs: int = 100,
    warmup_runs: int = 10,
) -> Dict[str, float]:
    """Benchmark a function with warmup.

    Args:
        func: Function to benchmark
        prices: Input data
        num_runs: Number of runs for timing
        warmup_runs: Number of warmup runs

    Returns:
        Dict with timing statistics
    """
    # Warmup runs
    for _ in range(warmup_runs):
        try:
            _ = func(prices)
        except Exception:
            pass

    # Timing runs
    times = []
    for _ in range(num_runs):
        start = time.perf_counter()
        try:
            _ = func(prices)
        except Exception as e:
            print(f"Error during benchmark: {e}")
            continue
        end = time.perf_counter()
        times.append(end - start)

    times = np.array(times)

    return {
        "mean": float(np.mean(times)),
        "std": float(np.std(times)),
        "min": float(np.min(times)),
        "max": float(np.max(times)),
        "total": float(np.sum(times)),
    }


def benchmark_config(
    config: ATCConfig,
    mode: str = "ema_only",
    num_runs: int = 100,
    warmup_runs: int = 10,
    data_size: int = 1000,
) -> Dict[str, Any]:
    """Benchmark specialized vs generic paths for a configuration.

    Args:
        config: ATC configuration
        mode: Specialization mode
        num_runs: Number of benchmark runs
        warmup_runs: Number of warmup runs
        data_size: Size of test data

    Returns:
        Dict with timing results for both paths
    """
    # Generate test data
    prices = generate_test_data(data_size)

    # Check if specialization is available
    if not is_config_specializable(config, mode):
        print(f"Config not specializable: {mode}")
        return {}

    # Define generic path function
    def generic_path(prices_data: pd.Series) -> dict[str, pd.Series]:
        return compute_atc_signals(
            prices_data,
            ema_len=config.ema_len,
            robustness=config.robustness,
            La=config.lambda_param,
            De=config.decay,
            long_threshold=config.long_threshold,
            short_threshold=config.short_threshold,
            cutout=config.cutout,
            strategy_mode=config.strategy_mode,
        )

    # Define specialized path function
    def specialized_path(prices_data: pd.Series) -> dict[str, pd.Series]:
        return compute_atc_specialized(
            prices_data,
            config,
            mode=mode,
            use_codegen_specialization=True,
            fallback_to_generic=False,
        )

    # Benchmark both paths
    print(f"Benchmarking config: {config.ema_len}, mode: {mode}")
    print(f"Data size: {data_size}, Runs: {num_runs}, Warmup: {warmup_runs}")

    try:
        generic_stats = benchmark_function(generic_path, prices, num_runs, warmup_runs)
        print(f"Generic path: {generic_stats['mean']*1000:.3f} ms (±{generic_stats['std']*1000:.3f})")

        specialized_stats = benchmark_function(specialized_path, prices, num_runs, warmup_runs)
        print(f"Specialized path: {specialized_stats['mean']*1000:.3f} ms (±{specialized_stats['std']*1000:.3f})")

        # Calculate speedup
        if specialized_stats["mean"] > 0:
            speedup = generic_stats["mean"] / specialized_stats["mean"]
            improvement = (1.0 - specialized_stats["mean"] / generic_stats["mean"]) * 100
            print(f"Speedup: {speedup:.2f}x ({improvement:.1f}% improvement)")

            # Check if meets threshold
            if improvement >= 10:
                print(f"✅ Meets target: >=10% improvement ({improvement:.1f}%)")
            else:
                print(f"⚠️ Below target: <10% improvement ({improvement:.1f}%)")

        return {
            "generic": generic_stats,
            "specialized": specialized_stats,
            "config": {
                "ema_len": config.ema_len,
                "robustness": config.robustness,
                "mode": mode,
            },
        }
    except ImportError:
        print("Numba not available, skipping benchmark")
        return {}
    except Exception as e:
        print(f"Benchmark error: {e}")
        return {}


def benchmark_all_configs() -> Dict[str, Dict[str, Dict[str, float]]]:
    """Benchmark multiple configurations.

    Returns:
        Dict with results for each configuration
    """
    configs = [
        (ATCConfig(ema_len=14, robustness="Medium"), "ema_only", "Short Length (14)"),
        (ATCConfig(ema_len=28, robustness="Medium"), "ema_only", "Default Length (28)"),
        (ATCConfig(ema_len=50, robustness="Medium"), "ema_only", "Long Length (50)"),
    ]

    results = {}

    for config, mode, name in configs:
        print(f"\n{'='*60}")
        print(f"Benchmark: {name}")
        print(f"{'='*60}")

        result = benchmark_config(
            config,
            mode=mode,
            num_runs=100,
            warmup_runs=10,
            data_size=1000,
        )

        if result:
            results[name] = result

    # Summary
    print(f"\n{'='*60}")
    print("SUMMARY")
    print(f"{'='*60}")

    for name, result in results.items():
        if "generic" in result and "specialized" in result:
            generic_time = result["generic"]["mean"] * 1000
            specialized_time = result["specialized"]["mean"] * 1000
            speedup = result["generic"]["mean"] / result["specialized"]["mean"]
            improvement = (1.0 - result["specialized"]["mean"] / result["generic"]["mean"]) * 100

            print(
                f"{name:30} | Generic: {generic_time:6.3f} ms | Specialized: {specialized_time:6.3f} ms | Speedup: {speedup:.2f}x ({improvement:.1f}%)"
            )

    return results


if __name__ == "__main__":
    # Run all benchmarks
    results = benchmark_all_configs()

    print(f"\nBenchmark complete!")
    print(f"Results show JIT specialization performance improvements for repeated calls.")
