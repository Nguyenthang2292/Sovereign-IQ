"""Profiling helper script for adaptive_trend_LTS benchmarks.

This script provides a unified interface to run cProfile and py-spy profiling
on the main benchmark comparison pipeline.
"""

import argparse
import os
import subprocess
import sys
from pathlib import Path


def run_cprofile():
    """Run benchmark_comparison with cProfile enabled."""
    print("=" * 60)
    print("Running cProfile on benchmark_comparison/main.py")
    print("=" * 60)

    # Ensure profiles directory exists
    profiles_dir = Path(__file__).parent.parent / "profiles"
    profiles_dir.mkdir(parents=True, exist_ok=True)

    # Build command
    profile_file = profiles_dir / "benchmark_comparison.stats"
    cmd = [
        sys.executable,
        "-m",
        "cProfile",
        "-o",
        str(profile_file),
        "-m",
        "modules.adaptive_trend_LTS.benchmarks.benchmark_comparison.main",
    ] + sys.argv[1:]

    print(f"Command: {' '.join(cmd)}")
    print(f"Output will be saved to: {profile_file}")
    print()

    # Run cProfile
    result = subprocess.run(cmd)

    if result.returncode == 0:
        print()
        print("=" * 60)
        print(f"✅ cProfile completed successfully")
        print(f"✅ Profile stats saved to: {profile_file}")
        print()
        print("To analyze the results, you can use:")
        print(f"  python -m pstats {profile_file}")
        print(f"  gprof2dot -f pstats {profile_file} | dot -Tpng -o profile.png")
        print("=" * 60)
    else:
        print(f"❌ cProfile failed with return code {result.returncode}")
        sys.exit(result.returncode)


def run_pyspy():
    """Run benchmark_comparison with py-spy flamegraph enabled."""
    print("=" * 60)
    print("Running py-spy flamegraph on benchmark_comparison/main.py")
    print("=" * 60)

    # Ensure profiles directory exists
    profiles_dir = Path(__file__).parent.parent / "profiles"
    profiles_dir.mkdir(parents=True, exist_ok=True)

    # Build command
    flamegraph_file = profiles_dir / "benchmark_comparison_flame.svg"
    cmd = [
        "py-spy",
        "record",
        "-o",
        str(flamegraph_file),
        "--",
        sys.executable,
        "-m",
        "modules.adaptive_trend_LTS.benchmarks.benchmark_comparison.main",
    ] + sys.argv[1:]

    print(f"Command: {' '.join(cmd)}")
    print(f"Output will be saved to: {flamegraph_file}")
    print()

    # Run py-spy
    result = subprocess.run(cmd)

    if result.returncode == 0:
        print()
        print("=" * 60)
        print(f"✅ py-spy completed successfully")
        print(f"✅ Flamegraph saved to: {flamegraph_file}")
        print()
        print("To view the flamegraph, open the SVG file in your browser:")
        print(f"  file://{flamegraph_file.absolute()}")
        print("=" * 60)
    else:
        print(f"❌ py-spy failed with return code {result.returncode}")
        sys.exit(result.returncode)


def run_both():
    """Run both cProfile and py-spy profiling."""
    print("=" * 60)
    print("Running FULL PROFILING (cProfile + py-spy)")
    print("=" * 60)

    # First run cProfile
    run_cprofile()

    print()
    print()
    print("=" * 60)
    print("Now running py-spy...")
    print("=" * 60)
    print()

    # Then run py-spy
    run_pyspy()


def main():
    """Main entry point for profiling helper."""
    parser = argparse.ArgumentParser(
        description="Profiling helper for adaptive_trend_LTS benchmarks",
        epilog=(
            "This script wraps cProfile and py-spy to profile "
            "the benchmark_comparison pipeline. Output is saved to profiles/ directory."
        ),
    )
    parser.add_argument("--cprofile", action="store_true", help="Run with cProfile only")
    parser.add_argument("--pyspy", action="store_true", help="Run with py-spy only")
    parser.add_argument(
        "--both",
        action="store_true",
        help="Run both cProfile and py-spy (default if no option specified)",
    )
    parser.add_argument(
        "--symbols",
        type=int,
        default=20,
        help="Number of symbols to benchmark (passed to benchmark_comparison)",
    )
    parser.add_argument(
        "--bars",
        type=int,
        default=500,
        help="Number of bars per symbol (passed to benchmark_comparison)",
    )
    parser.add_argument(
        "--timeframe",
        type=str,
        default="1h",
        help="Timeframe (passed to benchmark_comparison)",
    )
    parser.add_argument(
        "--clear-cache",
        action="store_true",
        help="Clear MA cache before benchmarking",
    )

    args = parser.parse_args()

    # Validate py-spy availability if requested
    if args.pyspy or args.both or (not args.cprofile and not args.pyspy):
        try:
            subprocess.run(["py-spy", "--version"], capture_output=True, check=True)
        except (subprocess.CalledProcessError, FileNotFoundError):
            print("❌ ERROR: py-spy is not installed or not found in PATH")
            print()
            print("To install py-spy:")
            print("  pip install py-spy")
            print()
            print("For more information: https://github.com/benfred/py-spy")
            sys.exit(1)

    # Default to both if no option specified
    if not args.cprofile and not args.pyspy and not args.both:
        args.both = True

    # Build benchmark arguments
    benchmark_args = []
    if args.symbols is not None:
        benchmark_args.extend(["--symbols", str(args.symbols)])
    if args.bars is not None:
        benchmark_args.extend(["--bars", str(args.bars)])
    if args.timeframe is not None:
        benchmark_args.extend(["--timeframe", args.timeframe])
    if args.clear_cache:
        benchmark_args.append("--clear-cache")

    # Replace sys.argv to pass benchmark arguments
    sys.argv = [sys.argv[0]] + benchmark_args

    # Run the requested profiling
    if args.cprofile:
        run_cprofile()
    elif args.pyspy:
        run_pyspy()
    else:  # both
        run_both()


if __name__ == "__main__":
    main()
