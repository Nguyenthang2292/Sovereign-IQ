"""
ATC Benchmark: Python vs Rust Implementation
============================================

Benchmark so sánh hiệu năng giữa:
1. modules/adaptive_trend_LTS_mini (Python)
2. modules/adaptive_trend_LTS_serverless (Rust - Scalar)
3. modules/adaptive_trend_LTS_serverless (Rust - SIMD optimized)
4. modules/adaptive_trend_LTS_serverless (Rust - SIMD + Parallel)

Các cặp test: BTCUSDT, ETHUSDT, XMRUSDT
Timeframes: 15m, 1h, 4h

So sánh:
1. Tốc độ chạy (thờigian xử lý)
2. Hiệu suất SIMD optimization (speedup factor)
3. Hiệu suất parallel trên nền SIMD (speedup factor)
4. Đồng nhất tín hiệu (so sánh kết quả LONG/SHORT/NEUTRAL)

Usage:
    python benchmark_atc_comparison.py
    python benchmark_atc_comparison.py --simd  # Include SIMD + SIMD+Parallel benchmark
    python benchmark_atc_comparison.py --no-simd  # Run scalar-only benchmark

Requirements:
    - Rust toolchain installed (nightly for SIMD on Windows)
    - Python dependencies: pandas, numpy
"""

import hashlib
import json
import subprocess
import sys
import time
from pathlib import Path
from typing import cast

import numpy as np
import pandas as pd

# Add project root to path
# Adjust path since this script is now in modules/adaptive_trend_LTS_serverless/benchmarks/
project_root = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(project_root))

from modules.adaptive_trend_LTS_mini import compute_atc_signals
from modules.adaptive_trend_LTS_mini.utils.config import ATCConfig

# =============================================================================
# Configuration
# =============================================================================

SYMBOLS = ["BTCUSDT", "ETHUSDT", "XMRUSDT"]
TIMEFRAMES = ["15m", "1h", "4h"]
BARS_PER_TF = 500  # Số nến cho mỗi timeframe

# ATC Configuration
ATC_CONFIG = {
    "ema_len": 28,
    "hma_len": 28,
    "wma_len": 28,
    "dema_len": 28,
    "lsma_len": 28,
    "kama_len": 28,
    "lambda_param": 0.02,  # La in original
    "decay": 0.03,  # De in original
    "long_threshold": 0.1,
    "short_threshold": -0.1,
    "robustness": "Narrow",
    "cutout": 0,
}

RUST_BINARY_NAME = "atc_benchmark"
# Path to the Rust project root (modules/adaptive_trend_LTS_serverless)
RUST_PROJECT_PATH = Path(__file__).resolve().parent.parent


# =============================================================================
# Data Generation
# =============================================================================


def generate_ohlcv_data(symbol: str, timeframe: str, num_bars: int = 500) -> pd.DataFrame:
    """Generate synthetic OHLCV data for testing."""
    # Use stable hash for deterministic RNG seed across runs
    seed_bytes = hashlib.sha256(f"{symbol}_{timeframe}".encode()).digest()
    seed = int.from_bytes(seed_bytes[:4], byteorder="big")
    np.random.seed(seed)

    # Generate realistic price movements
    returns = np.random.normal(0, 0.01, num_bars)
    trend = np.sin(np.linspace(0, 4 * np.pi, num_bars)) * 0.005

    base_price = 50000 if "BTC" in symbol else (3000 if "ETH" in symbol else 150)
    prices = base_price * (1 + np.cumsum(returns + trend))

    # Map timeframe to pandas frequency
    timeframe_freq_map = {
        "1m": "1min",
        "5m": "5min",
        "15m": "15min",
        "30m": "30min",
        "1h": "1h",
        "2h": "2h",
        "4h": "4h",
        "6h": "6h",
        "8h": "8h",
        "12h": "12h",
        "1d": "1D",
        "3d": "3D",
        "1w": "1W",
    }
    freq = timeframe_freq_map.get(timeframe, "1h")

    # Generate OHLCV from close prices
    df = pd.DataFrame(
        {
            "timestamp": pd.date_range("2024-01-01", periods=num_bars, freq=freq).astype(int) // 10**9,
            "close": prices,
        }
    )

    # Generate realistic OHLC from close while preserving candle invariants:
    # high >= max(open, close) and low <= min(open, close)
    volatility = prices * 0.002  # 0.2% volatility
    df["open"] = df["close"].shift(1)
    first_close = cast(float, df.at[0, "close"])
    df.loc[0, "open"] = first_close * (1 + np.random.normal(0, 0.001))

    max_oc = np.maximum(df["open"].to_numpy(), df["close"].to_numpy())
    min_oc = np.minimum(df["open"].to_numpy(), df["close"].to_numpy())
    upper_wick = np.abs(np.random.normal(0, volatility))
    lower_wick = np.abs(np.random.normal(0, volatility))

    df["high"] = max_oc + upper_wick
    df["low"] = min_oc - lower_wick
    df["volume"] = np.random.uniform(100, 1000, num_bars) * base_price / 1000

    result = df[["timestamp", "open", "high", "low", "close", "volume"]].copy()
    return result


def classify_signal(value: float, threshold: float = 0.1) -> str:
    """Classify signal value into LONG/SHORT/NEUTRAL."""
    if value > threshold:
        return "LONG"
    elif value < -threshold:
        return "SHORT"
    else:
        return "NEUTRAL"


# =============================================================================
# Python Module Benchmark
# =============================================================================


def benchmark_python_module(symbol: str, timeframe: str, data: pd.DataFrame) -> dict:
    """Benchmark Python adaptive_trend_LTS_mini module."""
    config = ATCConfig(**ATC_CONFIG)
    close_prices = pd.Series(data["close"].values)  # Ensure it's a Series

    # Warmup run
    _ = compute_atc_signals(
        prices=close_prices,
        ema_len=config.ema_len,
        hma_len=config.hma_len,
        wma_len=config.wma_len,
        dema_len=config.dema_len,
        lsma_len=config.lsma_len,
        kama_len=config.kama_len,
        lambda_param=config.lambda_param,
        decay=config.decay,
        long_threshold=config.long_threshold,
        short_threshold=config.short_threshold,
        robustness=config.robustness,
        cutout=config.cutout,
        fast_mode=True,
        parallel_l1=False,
        use_rust_backend=False,  # Disable Rust backend to avoid issues
    )

    # Actual benchmark
    times = []
    results = []

    for _ in range(5):  # 5 runs for averaging
        start = time.perf_counter()
        result = compute_atc_signals(
            prices=close_prices,
            ema_len=config.ema_len,
            hma_len=config.hma_len,
            wma_len=config.wma_len,
            dema_len=config.dema_len,
            lsma_len=config.lsma_len,
            kama_len=config.kama_len,
            lambda_param=config.lambda_param,
            decay=config.decay,
            long_threshold=config.long_threshold,
            short_threshold=config.short_threshold,
            robustness=config.robustness,
            cutout=config.cutout,
            fast_mode=True,
            parallel_l1=False,
            use_rust_backend=False,
        )
        end = time.perf_counter()

        times.append((end - start) * 1000)  # Convert to ms
        results.append(result)

    # Get final signal classification
    final_signal = results[-1]["Average_Signal"].iloc[-1]
    signal_type = classify_signal(final_signal)

    return {
        "symbol": symbol,
        "timeframe": timeframe,
        "mean_time_ms": np.mean(times),
        "std_time_ms": np.std(times),
        "min_time_ms": np.min(times),
        "max_time_ms": np.max(times),
        "signal_value": float(final_signal),
        "signal_type": signal_type,
        "module": "Python (mini)",
    }


# =============================================================================
# Rust Module Benchmark
# =============================================================================


def build_rust_benchmark() -> bool:
    """Build Rust benchmark binary."""
    print("Building Rust benchmark binary...")
    print(f"Rust project path: {RUST_PROJECT_PATH}")
    try:
        result = subprocess.run(
            ["cargo", "build", "--release", "--bin", "atc_benchmark"],
            cwd=str(RUST_PROJECT_PATH),
            capture_output=True,
            text=True,
        )
        if result.returncode != 0:
            print(f"Build failed: {result.stderr}")
            return False
        print("✓ Rust binary built successfully")
        return True
    except Exception as e:
        print(f"Build error: {e}")
        return False


def build_rust_benchmark_simd() -> bool:
    """Build Rust benchmark binary with SIMD optimizations."""
    print("Building Rust benchmark binary with SIMD...")
    print(f"Rust project path: {RUST_PROJECT_PATH}")
    print("Note: SIMD requires nightly Rust on some platforms")
    try:
        result = subprocess.run(
            ["cargo", "+nightly", "build", "--release", "--features", "simd", "--bin", "atc_benchmark"],
            cwd=str(RUST_PROJECT_PATH),
            capture_output=True,
            text=True,
        )
        if result.returncode != 0:
            print(f"SIMD build failed: {result.stderr}")
            print("Tip: Install Rust nightly with: rustup install nightly")
            return False
        print("✓ Rust SIMD binary built successfully")
        return True
    except Exception as e:
        print(f"SIMD build error: {e}")
        return False


def benchmark_rust_module(
    symbol: str,
    timeframe: str,
    data: pd.DataFrame,
    parallelism_config: dict | None = None,
    module_name: str = "Rust (serverless)",
) -> dict:
    """Benchmark Rust adaptive_trend_LTS_serverless module."""
    # Prepare JSON input for Rust
    batch_request = {
        "symbols": [
            {
                "symbol": symbol,
                "timeframes": {
                    timeframe: {
                        "timestamp": data["timestamp"].tolist(),
                        "open": data["open"].tolist(),
                        "high": data["high"].tolist(),
                        "low": data["low"].tolist(),
                        "close": data["close"].tolist(),
                        "volume": data["volume"].tolist(),
                    }
                },
            }
        ],
        "config": {
            "weights": {timeframe: 1.0},
            "threshold": 0.1,
            "min_signal": 0.0,
            "use_signal_strength": True,
            "lambda_param": 0.02,
            "decay": 0.03,
            "cutout": 0,
            "ma_configs": [
                {"ma_type": "EMA", "length": 28, "weight": 1.0},
                {"ma_type": "HMA", "length": 28, "weight": 1.0},
                {"ma_type": "WMA", "length": 28, "weight": 1.0},
                {"ma_type": "DEMA", "length": 28, "weight": 1.0},
                {"ma_type": "LSMA", "length": 28, "weight": 1.0},
                {"ma_type": "KAMA", "length": 28, "weight": 1.0},
            ],
        },
    }

    if parallelism_config:
        batch_request["parallelism"] = {
            "threads": parallelism_config.get("threads"),
            "chunk_size": parallelism_config.get("chunk_size"),
            "min_len": parallelism_config.get("min_len"),
        }

    json_input = json.dumps(batch_request)
    binary_path = RUST_PROJECT_PATH / "target" / "release" / RUST_BINARY_NAME

    # Warmup run (using stdin)
    warmup_result = subprocess.run(
        [str(binary_path)],
        input=json_input,
        capture_output=True,
        text=True,
    )
    if warmup_result.returncode != 0:
        warmup_error = warmup_result.stderr.strip() or warmup_result.stdout.strip()
        print(f"Warmup Rust error for {symbol} {timeframe}: " f"{warmup_error}")

    # Actual benchmark
    times = []
    results = []
    rust_errors = []

    for _ in range(5):  # 5 runs for averaging
        start = time.perf_counter()
        result = subprocess.run(
            [str(binary_path)],
            input=json_input,
            capture_output=True,
            text=True,
        )
        end = time.perf_counter()

        if result.returncode == 0:
            try:
                output = json.loads(result.stdout)
                times.append(output["duration_micros"] / 1000.0)  # Convert to ms
                results.append(output)
            except json.JSONDecodeError as err:
                rust_errors.append(f"JSON decode error: {err}; stdout={result.stdout[:500]}")
                times.append((end - start) * 1000)
        else:
            error_message = result.stderr.strip() or result.stdout.strip() or "Unknown Rust subprocess error"
            rust_errors.append(error_message)
            print(f"Rust error: {error_message}")
            times.append((end - start) * 1000)

    # Get signal from last successful result
    if results:
        final_result = results[-1]
        if final_result["results"]:
            signal_value = final_result["results"][0]["score"]
            signal_type = final_result["results"][0]["signal_type"]
        else:
            signal_value = 0.0
            signal_type = "ERROR"
    else:
        signal_value = 0.0
        signal_type = "ERROR"
        if rust_errors:
            print(f"All Rust benchmark runs failed for {symbol} {timeframe}. Last error: {rust_errors[-1]}")

    return {
        "symbol": symbol,
        "timeframe": timeframe,
        "mean_time_ms": np.mean(times),
        "std_time_ms": np.std(times),
        "min_time_ms": np.min(times),
        "max_time_ms": np.max(times),
        "signal_value": signal_value,
        "signal_type": signal_type,
        "module": module_name,
    }


# =============================================================================
# Main Benchmark
# =============================================================================


def run_benchmark(include_simd: bool = False, parallelism_config: dict | None = None):
    """Run complete benchmark comparison.

    Args:
        include_simd: If True, also benchmark SIMD-optimized Rust implementation
        parallelism_config: Optional parallelism config for SIMD+Parallel benchmark
    """
    print("=" * 80)
    print("ATC BENCHMARK: Python (mini) vs Rust (serverless)")
    if include_simd:
        print("              + SIMD + SIMD+Parallel Optimization Comparison")
    print("=" * 80)
    print()
    print(f"Symbols: {', '.join(SYMBOLS)}")
    print(f"Timeframes: {', '.join(TIMEFRAMES)}")
    print(f"Bars per timeframe: {BARS_PER_TF}")
    if include_simd and parallelism_config:
        print(
            "Parallel config: "
            f"threads={parallelism_config.get('threads')}, "
            f"chunk_size={parallelism_config.get('chunk_size')}, "
            f"min_len={parallelism_config.get('min_len')}"
        )
    print()

    # Build Rust binary first
    if not build_rust_benchmark():
        print("Failed to build Rust binary. Exiting.")
        return
    print()

    # Build SIMD binary if requested
    simd_available = False
    if include_simd:
        if build_rust_benchmark_simd():
            simd_available = True
        else:
            print("SIMD build failed, continuing with scalar Rust only...")
        print()

    # Store all results
    python_results = []
    rust_results = []
    rust_simd_results = []
    rust_parallel_results = []

    # Run benchmarks for each symbol and timeframe
    for symbol in SYMBOLS:
        for tf in TIMEFRAMES:
            print(f"\nBenchmarking {symbol} @ {tf}...")

            # Generate data
            data = generate_ohlcv_data(symbol, tf, BARS_PER_TF)

            # Python benchmark
            print("  Running Python module...")
            py_result = benchmark_python_module(symbol, tf, data)
            python_results.append(py_result)
            print(f"    Time: {py_result['mean_time_ms']:.2f} ± {py_result['std_time_ms']:.2f} ms")
            print(f"    Signal: {py_result['signal_type']} ({py_result['signal_value']:.4f})")

            # Rust benchmark (rebuild without SIMD for fair comparison)
            if simd_available:
                # Rebuild without SIMD
                build_rust_benchmark()

            print("  Running Rust module (scalar)...")
            rust_result = benchmark_rust_module(symbol, tf, data)
            rust_results.append(rust_result)
            print(f"    Time: {rust_result['mean_time_ms']:.2f} ± {rust_result['std_time_ms']:.2f} ms")
            print(f"    Signal: {rust_result['signal_type']} ({rust_result['signal_value']:.4f})")

            # SIMD benchmark
            if simd_available:
                # Rebuild with SIMD
                build_rust_benchmark_simd()

                print("  Running Rust module (SIMD)...")
                simd_result = benchmark_rust_module(symbol, tf, data, module_name="Rust (SIMD)")
                rust_simd_results.append(simd_result)
                print(f"    Time: {simd_result['mean_time_ms']:.2f} ± {simd_result['std_time_ms']:.2f} ms")
                print(f"    Signal: {simd_result['signal_type']} ({simd_result['signal_value']:.4f})")

                # Show SIMD speedup
                simd_speedup = rust_result["mean_time_ms"] / simd_result["mean_time_ms"]
                print(f"    SIMD Speedup: {simd_speedup:.2f}x vs scalar")

                print("  Running Rust module (SIMD + Parallel)...")
                parallel_result = benchmark_rust_module(
                    symbol,
                    tf,
                    data,
                    parallelism_config=parallelism_config,
                    module_name="Rust (SIMD+Parallel)",
                )
                rust_parallel_results.append(parallel_result)
                print(f"    Time: {parallel_result['mean_time_ms']:.2f} ± {parallel_result['std_time_ms']:.2f} ms")
                print(f"    Signal: {parallel_result['signal_type']} ({parallel_result['signal_value']:.4f})")

                parallel_speedup = simd_result["mean_time_ms"] / parallel_result["mean_time_ms"]
                print(f"    Parallel Speedup: {parallel_speedup:.2f}x vs SIMD")

    # Print summary
    if simd_available:
        print_summary_with_simd(python_results, rust_results, rust_simd_results, rust_parallel_results)
    else:
        print_summary(python_results, rust_results)


def print_summary(python_results: list, rust_results: list):
    """Print benchmark summary."""
    print()
    print("=" * 80)
    print("BENCHMARK SUMMARY")
    print("=" * 80)
    print()

    # Speed comparison
    print("1. SPEED COMPARISON")
    print("-" * 80)
    print(f"{'Symbol':<12} {'TF':<6} {'Python (ms)':<15} {'Rust (ms)':<15} {'Speedup':<10}")
    print("-" * 80)

    total_py_time = 0
    total_rust_time = 0

    for py, rust in zip(python_results, rust_results):
        speedup = py["mean_time_ms"] / rust["mean_time_ms"] if rust["mean_time_ms"] > 0 else 0
        total_py_time += py["mean_time_ms"]
        total_rust_time += rust["mean_time_ms"]

        print(
            f"{py['symbol']:<12} {py['timeframe']:<6} "
            f"{py['mean_time_ms']:>8.2f} ± {py['std_time_ms']:<4.2f}   "
            f"{rust['mean_time_ms']:>8.2f} ± {rust['std_time_ms']:<4.2f}   "
            f"{speedup:>6.1f}x"
        )

    print("-" * 80)
    avg_speedup = total_py_time / total_rust_time if total_rust_time > 0 else 0
    print(f"{'TOTAL':<12} {'':<6} {total_py_time:>8.2f}          {total_rust_time:>8.2f}          {avg_speedup:>6.1f}x")
    print()

    # Signal consistency comparison
    print("2. SIGNAL CONSISTENCY")
    print("-" * 80)
    print(f"{'Symbol':<12} {'TF':<6} {'Python':<10} {'Rust':<10} {'Match':<8}")
    print("-" * 80)

    matches = 0
    total = 0

    for py, rust in zip(python_results, rust_results):
        match = "YES" if py["signal_type"] == rust["signal_type"] else "NO"
        if py["signal_type"] == rust["signal_type"]:
            matches += 1
        total += 1

        print(f"{py['symbol']:<12} {py['timeframe']:<6} {py['signal_type']:<10} {rust['signal_type']:<10} {match:<8}")

    print("-" * 80)
    print(f"Consistency Rate: {matches}/{total} ({matches / total * 100:.1f}%)")
    print()

    # Signal value difference
    print("3. SIGNAL VALUE DIFFERENCE")
    print("-" * 80)
    print(f"{'Symbol':<12} {'TF':<6} {'Python':<12} {'Rust':<12} {'Diff':<12}")
    print("-" * 80)

    max_diff = 0
    for py, rust in zip(python_results, rust_results):
        diff = abs(py["signal_value"] - rust["signal_value"])
        max_diff = max(max_diff, diff)

        print(
            f"{py['symbol']:<12} {py['timeframe']:<6} "
            f"{py['signal_value']:>10.6f}   "
            f"{rust['signal_value']:>10.6f}   "
            f"{diff:>10.6f}"
        )

    print("-" * 80)
    print(f"Maximum Difference: {max_diff:.6f}")
    print()

    # Overall conclusion
    print("4. CONCLUSION")
    print("-" * 80)
    print(f"Rust implementation is {avg_speedup:.1f}x faster on average")
    print(f"Signal consistency: {matches / total * 100:.1f}% ({matches}/{total})")
    print(f"Maximum signal difference: {max_diff:.6f}")

    if avg_speedup >= 5:
        print("Excellent speedup achieved (>5x)")
    elif avg_speedup >= 2:
        print("Good speedup achieved (2-5x)")
    else:
        print("Moderate speedup (needs optimization)")

    if matches == total:
        print("Perfect signal consistency")
    elif matches / total >= 0.9:
        print("High signal consistency (>90%)")
    else:
        print("Signal inconsistency detected")

    print()
    print("=" * 80)


def print_summary_with_simd(
    python_results: list,
    rust_results: list,
    rust_simd_results: list,
    rust_parallel_results: list,
):
    """Print benchmark summary including SIMD and SIMD+Parallel comparison."""
    print()
    print("=" * 120)
    print("BENCHMARK SUMMARY (WITH SIMD + SIMD+PARALLEL)")
    print("=" * 120)
    print()

    # Speed comparison
    print("1. SPEED COMPARISON")
    print("-" * 130)
    print(
        f"{'Symbol':<12} {'TF':<6} {'Python (ms)':<18} {'Rust (ms)':<18} "
        f"{'Rust SIMD (ms)':<18} {'Rust SIMD+Par (ms)':<20} {'Speedup':<20}"
    )
    print("-" * 130)

    total_py_time = 0
    total_rust_time = 0
    total_simd_time = 0
    total_parallel_time = 0

    for py, rust, simd, parallel in zip(python_results, rust_results, rust_simd_results, rust_parallel_results):
        speedup_rust = py["mean_time_ms"] / rust["mean_time_ms"] if rust["mean_time_ms"] > 0 else 0
        speedup_simd = py["mean_time_ms"] / simd["mean_time_ms"] if simd["mean_time_ms"] > 0 else 0
        speedup_parallel = py["mean_time_ms"] / parallel["mean_time_ms"] if parallel["mean_time_ms"] > 0 else 0
        total_py_time += py["mean_time_ms"]
        total_rust_time += rust["mean_time_ms"]
        total_simd_time += simd["mean_time_ms"]
        total_parallel_time += parallel["mean_time_ms"]

        print(
            f"{py['symbol']:<12} {py['timeframe']:<6} "
            f"{py['mean_time_ms']:>8.2f} ± {py['std_time_ms']:<5.2f}    "
            f"{rust['mean_time_ms']:>8.2f} ± {rust['std_time_ms']:<5.2f}    "
            f"{simd['mean_time_ms']:>8.2f} ± {simd['std_time_ms']:<5.2f}    "
            f"{parallel['mean_time_ms']:>8.2f} ± {parallel['std_time_ms']:<5.2f}      "
            f"{speedup_rust:>5.1f}x / {speedup_simd:>5.1f}x / {speedup_parallel:>5.1f}x"
        )

    print("-" * 130)
    avg_speedup_rust = total_py_time / total_rust_time if total_rust_time > 0 else 0
    avg_speedup_simd = total_py_time / total_simd_time if total_simd_time > 0 else 0
    avg_speedup_parallel = total_py_time / total_parallel_time if total_parallel_time > 0 else 0
    simd_vs_scalar = total_rust_time / total_simd_time if total_simd_time > 0 else 0
    parallel_vs_simd = total_simd_time / total_parallel_time if total_parallel_time > 0 else 0
    parallel_vs_scalar = total_rust_time / total_parallel_time if total_parallel_time > 0 else 0

    print(
        f"{'TOTAL':<12} {'':<6} {total_py_time:>8.2f}            "
        f"{total_rust_time:>8.2f}            {total_simd_time:>8.2f}            "
        f"{total_parallel_time:>8.2f}              "
        f"{avg_speedup_rust:>5.1f}x / {avg_speedup_simd:>5.1f}x / {avg_speedup_parallel:>5.1f}x"
    )
    print()
    print(
        f"Average Speedup: Rust={avg_speedup_rust:.2f}x, SIMD={avg_speedup_simd:.2f}x, "
        f"SIMD+Parallel={avg_speedup_parallel:.2f}x"
    )
    print(f"SIMD vs Scalar Rust: {simd_vs_scalar:.2f}x")
    print(f"SIMD+Parallel vs SIMD: {parallel_vs_simd:.2f}x")
    print(f"SIMD+Parallel vs Scalar Rust: {parallel_vs_scalar:.2f}x")
    print()

    # Optimization impact analysis
    print("2. OPTIMIZATION IMPACT")
    print("-" * 110)
    print(
        f"{'Symbol':<12} {'TF':<6} {'Scalar (ms)':<15} {'SIMD (ms)':<15} {'SIMD+Par (ms)':<15} "
        f"{'SIMD Up':<10} {'Par Up':<10} {'Total Up':<10}"
    )
    print("-" * 110)

    simd_speedups = []
    parallel_speedups = []
    total_speedups = []
    for rust, simd, parallel in zip(rust_results, rust_simd_results, rust_parallel_results):
        simd_speedup = rust["mean_time_ms"] / simd["mean_time_ms"] if simd["mean_time_ms"] > 0 else 0
        parallel_speedup = simd["mean_time_ms"] / parallel["mean_time_ms"] if parallel["mean_time_ms"] > 0 else 0
        total_speedup = rust["mean_time_ms"] / parallel["mean_time_ms"] if parallel["mean_time_ms"] > 0 else 0
        simd_speedups.append(simd_speedup)
        parallel_speedups.append(parallel_speedup)
        total_speedups.append(total_speedup)

        print(
            f"{rust['symbol']:<12} {rust['timeframe']:<6} "
            f"{rust['mean_time_ms']:>8.2f} ± {rust['std_time_ms']:<4.2f}  "
            f"{simd['mean_time_ms']:>8.2f} ± {simd['std_time_ms']:<4.2f}  "
            f"{parallel['mean_time_ms']:>8.2f} ± {parallel['std_time_ms']:<4.2f}  "
            f"{simd_speedup:>6.2f}x   {parallel_speedup:>6.2f}x   {total_speedup:>6.2f}x"
        )

    print("-" * 110)
    avg_simd_speedup = np.mean(simd_speedups)
    min_simd_speedup = np.min(simd_speedups)
    max_simd_speedup = np.max(simd_speedups)
    avg_parallel_speedup = np.mean(parallel_speedups)
    avg_total_speedup = np.mean(total_speedups)

    print("SIMD Speedup Statistics:")
    print(f"  Average: {avg_simd_speedup:.2f}x")
    print(f"  Range: {min_simd_speedup:.2f}x - {max_simd_speedup:.2f}x")
    print(f"Parallel Speedup (vs SIMD) Average: {avg_parallel_speedup:.2f}x")
    print(f"Total Speedup (SIMD+Parallel vs Scalar) Average: {avg_total_speedup:.2f}x")
    print()

    # Signal consistency
    print("3. SIGNAL CONSISTENCY")
    print("-" * 95)
    print(f"{'Symbol':<12} {'TF':<6} {'Python':<10} {'Rust':<10} {'SIMD':<10} {'SIMD+Par':<10} {'Match':<10}")
    print("-" * 95)

    matches = 0
    total = 0

    for py, rust, simd, parallel in zip(python_results, rust_results, rust_simd_results, rust_parallel_results):
        is_consistent = py["signal_type"] == rust["signal_type"] == simd["signal_type"] == parallel["signal_type"]
        match = "YES" if is_consistent else "NO"
        if is_consistent:
            matches += 1
        total += 1

        print(
            f"{py['symbol']:<12} {py['timeframe']:<6} {py['signal_type']:<10} "
            f"{rust['signal_type']:<10} {simd['signal_type']:<10} {parallel['signal_type']:<10} {match:<10}"
        )

    print("-" * 95)
    print(f"Consistency Rate: {matches}/{total} ({matches / total * 100:.1f}%)")
    print()

    # Overall conclusion
    print("4. CONCLUSION")
    print("-" * 80)
    print(f"Python baseline: {total_py_time:.2f}ms total")
    print(f"Rust (scalar): {total_rust_time:.2f}ms total ({avg_speedup_rust:.2f}x faster than Python)")
    print(f"Rust (SIMD): {total_simd_time:.2f}ms total ({avg_speedup_simd:.2f}x faster than Python)")
    print(f"Rust (SIMD+Parallel): {total_parallel_time:.2f}ms total ({avg_speedup_parallel:.2f}x faster than Python)")
    print(f"SIMD optimization: {simd_vs_scalar:.2f}x faster than scalar Rust")
    print(f"Parallel optimization: {parallel_vs_simd:.2f}x faster than SIMD")
    print(f"Total optimization: {parallel_vs_scalar:.2f}x faster than scalar Rust")
    print(f"Signal consistency: {matches / total * 100:.1f}% ({matches}/{total})")
    print()

    if simd_vs_scalar >= 2:
        print("✓ Excellent SIMD speedup (≥2x)")
    elif simd_vs_scalar >= 1.5:
        print("✓ Good SIMD speedup (1.5-2x)")
    elif simd_vs_scalar >= 1.2:
        print("✓ Moderate SIMD speedup (1.2-1.5x)")
    else:
        print("⚠ Minimal SIMD benefit (<1.2x) - may need further optimization")

    if parallel_vs_simd >= 2:
        print("✓ Excellent parallel speedup (≥2x)")
    elif parallel_vs_simd >= 1.5:
        print("✓ Good parallel speedup (1.5-2x)")
    elif parallel_vs_simd >= 1.2:
        print("✓ Moderate parallel speedup (1.2-1.5x)")
    else:
        print("⚠ Minimal parallel benefit (<1.2x) - tune thread/chunk settings")

    if matches == total:
        print("✓ Perfect signal consistency across all implementations")
    elif matches / total >= 0.9:
        print("✓ High signal consistency (>90%)")
    else:
        print("⚠ Signal inconsistency detected")

    print()
    print("=" * 120)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Benchmark ATC implementations")
    parser.set_defaults(simd=True)
    simd_group = parser.add_mutually_exclusive_group()
    simd_group.add_argument("--simd", dest="simd", action="store_true", help="Include SIMD and SIMD+Parallel benchmark")
    simd_group.add_argument("--no-simd", dest="simd", action="store_false", help="Run scalar-only benchmark")
    parser.add_argument("--threads", type=int, default=8, help="Parallel threads for SIMD+Parallel run")
    parser.add_argument("--chunk-size", type=int, default=10, help="Parallel chunk_size for SIMD+Parallel run")
    parser.add_argument("--min-len", type=int, default=5, help="Parallel min_len for SIMD+Parallel run")
    args = parser.parse_args()

    parallelism_config = None
    if args.simd:
        parallelism_config = {
            "threads": args.threads,
            "chunk_size": args.chunk_size,
            "min_len": args.min_len,
        }

    run_benchmark(include_simd=args.simd, parallelism_config=parallelism_config)
