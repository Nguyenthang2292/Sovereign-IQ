"""
Parallelism tuning benchmark for ATC serverless Rust implementation.

This script benchmarks batch processing performance across:
- Batch sizes: 10, 50, 100, 500 symbols
- Thread counts: 1, 2, 4, 8, 16
- Chunk sizes (optional sweep)

It produces:
1. Console summary
2. Markdown report at docs/PARALLELISM_TUNING.md

Usage:
    python benchmarks/benchmark_parallelism_tuning.py
    python benchmarks/benchmark_parallelism_tuning.py --repeats 5
    python benchmarks/benchmark_parallelism_tuning.py --quick
"""

from __future__ import annotations

import argparse
import json
import statistics
import subprocess
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DOC_OUTPUT_PATH = PROJECT_ROOT / "docs" / "PARALLELISM_TUNING.md"
RUST_BINARY_PATH = PROJECT_ROOT / "target" / "release" / "atc_benchmark.exe"

BATCH_SIZES = [10, 50, 100, 500]
THREAD_COUNTS = [1, 2, 4, 8, 16]
CHUNK_SIZE_CANDIDATES = [1, 5, 10, 25, 50]
TIMEFRAMES = ["15m", "1h", "4h"]
BARS_PER_TF = 200

ATC_CONFIG = {
    "weights": {"15m": 0.2, "1h": 0.4, "4h": 0.4},
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
}


@dataclass
class BenchmarkRow:
    batch_size: int
    threads: int
    chunk_size: int
    min_len: int
    mean_ms: float
    std_ms: float
    p95_ms: float
    throughput_symbols_per_sec: float
    speedup_vs_single_thread: float
    parallel_efficiency: float


def build_rust_benchmark() -> None:
    cmd = ["cargo", "build", "--release", "--bin", "atc_benchmark"]
    result = subprocess.run(
        cmd,
        cwd=str(PROJECT_ROOT),
        capture_output=True,
        text=True,
    )
    if result.returncode != 0:
        print(result.stderr)
        raise RuntimeError("Failed to build Rust benchmark binary")



def generate_ohlcv(symbol_index: int, tf_index: int, bars: int) -> dict[str, list[float] | list[int]]:
    seed = symbol_index * 17 + tf_index * 31 + 11
    rng = np.random.default_rng(seed)

    trend = np.linspace(0.0, 1.0, bars) * (0.003 + 0.0005 * tf_index)
    noise = rng.normal(0, 0.0035 + tf_index * 0.0005, bars)
    returns = trend + noise

    base_price = 100.0 + symbol_index * 3.0 + tf_index * 5.0
    close = base_price * np.cumprod(1.0 + returns)

    open_prices = np.empty_like(close)
    open_prices[0] = close[0] * (1.0 + rng.normal(0, 0.001))
    open_prices[1:] = close[:-1]

    spread = np.abs(rng.normal(0, 0.002, bars)) * close
    high = np.maximum(open_prices, close) + spread
    low = np.minimum(open_prices, close) - spread
    volume = rng.uniform(100.0, 800.0, bars)

    return {
        "timestamp": list(range(bars)),
        "open": open_prices.tolist(),
        "high": high.tolist(),
        "low": low.tolist(),
        "close": close.tolist(),
        "volume": volume.tolist(),
    }



def build_symbols(batch_size: int, bars_per_tf: int) -> list[dict[str, Any]]:
    symbols: list[dict[str, Any]] = []

    for symbol_index in range(batch_size):
        tf_data: dict[str, dict[str, list[float] | list[int]]] = {}
        for tf_index, tf_name in enumerate(TIMEFRAMES):
            tf_data[tf_name] = generate_ohlcv(symbol_index, tf_index, bars_per_tf)

        symbols.append(
            {
                "symbol": f"SYM{symbol_index:04d}",
                "timeframes": tf_data,
            }
        )

    return symbols



def run_benchmark_once(payload: dict[str, Any]) -> float:
    started = time.perf_counter()
    result = subprocess.run(
        [str(RUST_BINARY_PATH)],
        input=json.dumps(payload),
        capture_output=True,
        text=True,
    )
    if result.returncode != 0:
        raise RuntimeError(f"Rust benchmark failed: {result.stderr}")

    output = json.loads(result.stdout)
    duration_ms = float(output["duration_ms"])

    finished = time.perf_counter()
    wall_ms = (finished - started) * 1000.0

    if duration_ms <= 0:
        return wall_ms

    return duration_ms



def benchmark_configuration(
    *,
    batch_size: int,
    threads: int,
    chunk_size: int,
    min_len: int,
    repeats: int,
    bars_per_tf: int,
) -> list[float]:
    symbols = build_symbols(batch_size, bars_per_tf)
    payload = {
        "symbols": symbols,
        "config": ATC_CONFIG,
        "parallelism": {
            "threads": threads,
            "chunk_size": chunk_size,
            "min_len": min_len,
        },
    }

    _ = run_benchmark_once(payload)

    measurements: list[float] = []
    for _ in range(repeats):
        measurements.append(run_benchmark_once(payload))
    return measurements



def summarize_measurements(values: list[float]) -> tuple[float, float, float]:
    mean_ms = statistics.fmean(values)
    std_ms = statistics.pstdev(values) if len(values) > 1 else 0.0
    p95_ms = float(np.percentile(values, 95))
    return mean_ms, std_ms, p95_ms



def choose_default_chunk_size(batch_size: int) -> int:
    if batch_size <= 10:
        return 1
    if batch_size <= 50:
        return 5
    if batch_size <= 100:
        return 10
    if batch_size <= 500:
        return 25
    return 50



def format_float(value: float, decimals: int = 2) -> str:
    return f"{value:.{decimals}f}"



def generate_markdown_report(
    rows: list[BenchmarkRow],
    chunk_sweep_rows: list[BenchmarkRow],
    *,
    repeats: int,
    bars_per_tf: int,
    tested_batch_sizes: list[int],
    tested_thread_counts: list[int],
    sweep_batch: int,
    sweep_threads: int,
) -> str:
    by_batch: dict[int, list[BenchmarkRow]] = {}
    for row in rows:
        by_batch.setdefault(row.batch_size, []).append(row)

    lines: list[str] = []
    lines.append("# Parallelism Tuning Report")
    lines.append("")
    lines.append(f"Generated: {time.strftime('%Y-%m-%d %H:%M:%S')}")
    lines.append(f"Benchmark mode: synthetic OHLCV, repeats={repeats}, bars_per_tf={bars_per_tf}")
    lines.append("")
    lines.append("## Scope")
    lines.append("")
    lines.append(
        "- Thread-count sweep: " + ", ".join(str(value) for value in tested_thread_counts)
    )
    lines.append(
        "- Batch sizes: " + ", ".join(str(value) for value in tested_batch_sizes) + " symbols"
    )
    lines.append("- Parallel strategy in code: rayon::into_par_iter + with_min_len + chunks")
    lines.append("- par_bridge vs par_iter: evaluated; par_iter retained because input is an in-memory Vec")
    lines.append("- rayon::scope nested parallelism: evaluated; not adopted to avoid nested scheduling overhead")
    lines.append("")

    lines.append("## Thread Sweep Results")
    lines.append("")
    lines.append("| Batch | Threads | Chunk | Mean (ms) | Std (ms) | P95 (ms) | Throughput (sym/s) | Speedup | Efficiency |")
    lines.append("|---|---:|---:|---:|---:|---:|---:|---:|---:|")

    for batch_size in sorted(by_batch.keys()):
        batch_rows = sorted(by_batch[batch_size], key=lambda row: row.threads)
        for row in batch_rows:
            lines.append(
                "| "
                f"{row.batch_size} | {row.threads} | {row.chunk_size} | "
                f"{format_float(row.mean_ms)} | {format_float(row.std_ms)} | {format_float(row.p95_ms)} | "
                f"{format_float(row.throughput_symbols_per_sec)} | {format_float(row.speedup_vs_single_thread)}x | "
                f"{format_float(row.parallel_efficiency * 100.0)}% |"
            )

    lines.append("")
    lines.append("## Best Configuration by Batch")
    lines.append("")
    lines.append("| Batch | Best Threads | Chunk | Mean (ms) | Throughput (sym/s) | Speedup | Efficiency |")
    lines.append("|---|---:|---:|---:|---:|---:|---:|")

    for batch_size in sorted(by_batch.keys()):
        best_row = min(by_batch[batch_size], key=lambda row: row.mean_ms)
        lines.append(
            "| "
            f"{batch_size} | {best_row.threads} | {best_row.chunk_size} | {format_float(best_row.mean_ms)} | "
            f"{format_float(best_row.throughput_symbols_per_sec)} | {format_float(best_row.speedup_vs_single_thread)}x | "
            f"{format_float(best_row.parallel_efficiency * 100.0)}% |"
        )

    lines.append("")
    lines.append(f"## Chunk Size Sweep (Batch={sweep_batch}, Threads={sweep_threads})")
    lines.append("")
    lines.append("| Chunk | Mean (ms) | Std (ms) | Throughput (sym/s) |")
    lines.append("|---:|---:|---:|---:|")
    for row in sorted(chunk_sweep_rows, key=lambda value: value.chunk_size):
        lines.append(
            f"| {row.chunk_size} | {format_float(row.mean_ms)} | {format_float(row.std_ms)} | {format_float(row.throughput_symbols_per_sec)} |"
        )

    lines.append("")
    lines.append("## Lambda Notes")
    lines.append("")
    lines.append("- Auto-tuning in lambda handler selects thread/chunk by batch size.")
    lines.append("- Validate x86_64 and ARM64 separately in real Lambda runtime for final production tuning.")
    lines.append("- Current report is generated on local host; use as baseline, not absolute capacity planning.")
    lines.append("")

    return "\n".join(lines) + "\n"



def main() -> None:
    parser = argparse.ArgumentParser(description="Parallelism tuning benchmark for ATC serverless")
    parser.add_argument("--repeats", type=int, default=3, help="Runs per configuration")
    parser.add_argument("--bars", type=int, default=BARS_PER_TF, help="Bars per timeframe")
    parser.add_argument("--quick", action="store_true", help="Use smaller sweep for quick validation")
    args = parser.parse_args()

    repeats = args.repeats
    bars_per_tf = args.bars

    batch_sizes = BATCH_SIZES
    thread_counts = THREAD_COUNTS

    if args.quick:
        batch_sizes = [10, 100]
        thread_counts = [1, 4, 8]

    print("Building Rust benchmark binary...")
    build_rust_benchmark()

    rows: list[BenchmarkRow] = []

    for batch_size in batch_sizes:
        chunk_size = choose_default_chunk_size(batch_size)
        min_len = 5

        single_thread_mean = None
        for threads in thread_counts:
            measurements = benchmark_configuration(
                batch_size=batch_size,
                threads=threads,
                chunk_size=chunk_size,
                min_len=min_len,
                repeats=repeats,
                bars_per_tf=bars_per_tf,
            )
            mean_ms, std_ms, p95_ms = summarize_measurements(measurements)

            if threads == 1:
                single_thread_mean = mean_ms

            if single_thread_mean is None or mean_ms <= 0:
                speedup = 1.0
                efficiency = 1.0
            else:
                speedup = single_thread_mean / mean_ms
                efficiency = speedup / threads

            throughput = (batch_size / mean_ms) * 1000.0 if mean_ms > 0 else 0.0

            row = BenchmarkRow(
                batch_size=batch_size,
                threads=threads,
                chunk_size=chunk_size,
                min_len=min_len,
                mean_ms=mean_ms,
                std_ms=std_ms,
                p95_ms=p95_ms,
                throughput_symbols_per_sec=throughput,
                speedup_vs_single_thread=speedup,
                parallel_efficiency=efficiency,
            )
            rows.append(row)
            print(
                f"batch={batch_size:>4}, threads={threads:>2}, chunk={chunk_size:>2}, "
                f"mean={mean_ms:>8.2f}ms, speedup={speedup:>5.2f}x, eff={efficiency*100:>6.2f}%"
            )

    sweep_batch = 100 if not args.quick else 10
    sweep_threads = 8 if not args.quick else 4

    chunk_sweep_rows: list[BenchmarkRow] = []
    single_for_chunk = None

    print(f"\nChunk sweep: batch={sweep_batch}, threads={sweep_threads}")
    for chunk_size in CHUNK_SIZE_CANDIDATES:
        measurements = benchmark_configuration(
            batch_size=sweep_batch,
            threads=sweep_threads,
            chunk_size=chunk_size,
            min_len=5,
            repeats=repeats,
            bars_per_tf=bars_per_tf,
        )
        mean_ms, std_ms, p95_ms = summarize_measurements(measurements)
        throughput = (sweep_batch / mean_ms) * 1000.0 if mean_ms > 0 else 0.0

        if chunk_size == CHUNK_SIZE_CANDIDATES[0]:
            single_for_chunk = mean_ms

        speedup = (single_for_chunk / mean_ms) if single_for_chunk and mean_ms > 0 else 1.0
        efficiency = speedup / sweep_threads

        row = BenchmarkRow(
            batch_size=sweep_batch,
            threads=sweep_threads,
            chunk_size=chunk_size,
            min_len=5,
            mean_ms=mean_ms,
            std_ms=std_ms,
            p95_ms=p95_ms,
            throughput_symbols_per_sec=throughput,
            speedup_vs_single_thread=speedup,
            parallel_efficiency=efficiency,
        )
        chunk_sweep_rows.append(row)
        print(f"chunk={chunk_size:>2}, mean={mean_ms:>8.2f}ms, throughput={throughput:>9.2f} sym/s")

    markdown = generate_markdown_report(
        rows,
        chunk_sweep_rows,
        repeats=repeats,
        bars_per_tf=bars_per_tf,
        tested_batch_sizes=batch_sizes,
        tested_thread_counts=thread_counts,
        sweep_batch=sweep_batch,
        sweep_threads=sweep_threads,
    )
    DOC_OUTPUT_PATH.write_text(markdown, encoding="utf-8")

    print(f"\nReport generated: {DOC_OUTPUT_PATH}")


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        print(f"ERROR: {exc}")
        sys.exit(1)
