"""Profile the benchmark_comparison module using cProfile.

This script does NOT modify any existing code. It simply wraps the existing
benchmark entrypoint:

    modules.adaptive_trend_LTS.benchmarks.benchmark_comparison.main

and saves a cProfile stats file under the `profiles/` directory.
"""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser(description="Profile benchmark_comparison with cProfile (no code changes).")
    parser.add_argument(
        "--symbols",
        type=int,
        default=20,
        help="Number of symbols to test (default: 20)",
    )
    parser.add_argument(
        "--bars",
        type=int,
        default=500,
        help="Number of bars per symbol (default: 500)",
    )
    parser.add_argument(
        "--timeframe",
        type=str,
        default="1h",
        help="Timeframe (default: 1h)",
    )

    args = parser.parse_args()

    profiles_dir = Path("profiles")
    profiles_dir.mkdir(parents=True, exist_ok=True)

    output_file = profiles_dir / "benchmark_comparison.stats"

    cmd = [
        sys.executable,
        "-m",
        "cProfile",
        "-o",
        str(output_file),
        "-m",
        "modules.adaptive_trend_LTS.benchmarks.benchmark_comparison.main",
        "--symbols",
        str(args.symbols),
        "--bars",
        str(args.bars),
        "--timeframe",
        args.timeframe,
    ]

    print(f"[profile] Running: {' '.join(cmd)}")
    subprocess.run(cmd, check=True)
    print(f"[profile] cProfile stats saved to: {output_file}")


if __name__ == "__main__":
    main()
