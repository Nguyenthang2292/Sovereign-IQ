#!/usr/bin/env python3
"""
ATC Benchmark: Python vs Rust Implementation
============================================

Benchmark so sánh hiệu năng giữa:
1. modules/adaptive_trend_LTS_mini (Python)
2. modules/adaptive_trend_LTS_serverless (Rust)

Các cặp test: BTCUSDT, ETHUSDT, XMRUSDT
Timeframes: 15m, 1h, 4h

So sánh:
1. Tốc độ chạy (thờigian xử lý)
2. Đồng nhất tín hiệu (so sánh kết quả LONG/SHORT/NEUTRAL)

Usage:
    python benchmark_atc_comparison.py

Requirements:
    - Rust toolchain installed
    - Python dependencies: pandas, numpy
"""

import json
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

# Add project root to path
project_root = Path(__file__).parent
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
RUST_PROJECT_PATH = Path(__file__).parent / "modules" / "adaptive_trend_LTS_serverless"


# =============================================================================
# Data Generation
# =============================================================================


def generate_ohlcv_data(symbol: str, timeframe: str, num_bars: int = 500) -> pd.DataFrame:
    """Generate synthetic OHLCV data for testing."""
    np.random.seed(hash(f"{symbol}_{timeframe}") % 2**32)

    # Generate realistic price movements
    returns = np.random.normal(0, 0.01, num_bars)
    trend = np.sin(np.linspace(0, 4 * np.pi, num_bars)) * 0.005

    base_price = 50000 if "BTC" in symbol else (3000 if "ETH" in symbol else 150)
    prices = base_price * (1 + np.cumsum(returns + trend))

    # Generate OHLCV from close prices
    df = pd.DataFrame(
        {
            "timestamp": pd.date_range("2024-01-01", periods=num_bars, freq="1h").astype(int) // 10**9,
            "close": prices,
        }
    )

    # Generate realistic OHLC from close
    volatility = prices * 0.002  # 0.2% volatility
    df["high"] = df["close"] + np.abs(np.random.normal(0, volatility))
    df["low"] = df["close"] - np.abs(np.random.normal(0, volatility))
    df["open"] = df["close"].shift(1)
    df.loc[0, "open"] = df.loc[0, "close"] * (1 + np.random.normal(0, 0.001))
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
    try:
        result = subprocess.run(
            ["cargo", "build", "--release", "--bin", "atc_benchmark"],
            cwd=RUST_PROJECT_PATH,
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


def benchmark_rust_module(symbol: str, timeframe: str, data: pd.DataFrame) -> dict:
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

    json_input = json.dumps(batch_request)
    binary_path = RUST_PROJECT_PATH / "target" / "release" / RUST_BINARY_NAME

    # Warmup run (using stdin)
    _ = subprocess.run(
        [str(binary_path)],
        input=json_input,
        capture_output=True,
        text=True,
    )

    # Actual benchmark
    times = []
    results = []

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
            output = json.loads(result.stdout)
            times.append(output["duration_micros"] / 1000.0)  # Convert to ms
            results.append(output)
        else:
            print(f"Rust error: {result.stderr}")
            times.append((end - start) * 1000)

    # Get signal from last result
    final_result = results[-1]
    if final_result["results"]:
        signal_value = final_result["results"][0]["score"]
        signal_type = final_result["results"][0]["signal_type"]
    else:
        signal_value = 0.0
        signal_type = "ERROR"

    return {
        "symbol": symbol,
        "timeframe": timeframe,
        "mean_time_ms": np.mean(times),
        "std_time_ms": np.std(times),
        "min_time_ms": np.min(times),
        "max_time_ms": np.max(times),
        "signal_value": signal_value,
        "signal_type": signal_type,
        "module": "Rust (serverless)",
    }


# =============================================================================
# Main Benchmark
# =============================================================================


def run_benchmark():
    """Run complete benchmark comparison."""
    print("=" * 80)
    print("ATC BENCHMARK: Python (mini) vs Rust (serverless)")
    print("=" * 80)
    print()
    print(f"Symbols: {', '.join(SYMBOLS)}")
    print(f"Timeframes: {', '.join(TIMEFRAMES)}")
    print(f"Bars per timeframe: {BARS_PER_TF}")
    print()

    # Build Rust binary first
    if not build_rust_benchmark():
        print("Failed to build Rust binary. Exiting.")
        return
    print()

    # Store all results
    python_results = []
    rust_results = []

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

            # Rust benchmark
            print("  Running Rust module...")
            rust_result = benchmark_rust_module(symbol, tf, data)
            rust_results.append(rust_result)
            print(f"    Time: {rust_result['mean_time_ms']:.2f} ± {rust_result['std_time_ms']:.2f} ms")
            print(f"    Signal: {rust_result['signal_type']} ({rust_result['signal_value']:.4f})")

    # Print summary
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
    print(
        f"{'TOTAL':<12} {'':<6} "
        f"{total_py_time:>8.2f}          "
        f"{total_rust_time:>8.2f}          "
        f"{avg_speedup:>6.1f}x"
    )
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

        print(
            f"{py['symbol']:<12} {py['timeframe']:<6} " f"{py['signal_type']:<10} {rust['signal_type']:<10} {match:<8}"
        )

    print("-" * 80)
    print(f"Consistency Rate: {matches}/{total} ({matches/total*100:.1f}%)")
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
    print(f"Signal consistency: {matches/total*100:.1f}% ({matches}/{total})")
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


if __name__ == "__main__":
    run_benchmark()
