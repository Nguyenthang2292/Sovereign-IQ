"""
Performance benchmark script for test optimization comparison.

This script compares test performance before and after optimization.
"""

import subprocess
import sys
import time

import pandas as pd


def run_test_command(command, description):
    """Run a test command and return timing results."""
    print(f"🔄 Running: {description}")

    start_time = time.time()

    try:
        result = subprocess.run(
            command,
            shell=True,
            capture_output=True,
            text=True,
            timeout=60,  # 60 second timeout
        )

        elapsed_time = time.time() - start_time

        # Parse output to extract test timing
        lines = result.stdout.split("\n")
        test_time = None

        for line in lines:
            if "Test execution time:" in line:
                try:
                    test_time = float(line.split(":")[-1].strip().replace("s", ""))
                except Exception:
                    pass
                break

        return {
            "command": description,
            "elapsed": elapsed_time,
            "test_time": test_time,
            "success": result.returncode == 0,
            "output": result.stdout,
        }

    except subprocess.TimeoutExpired:
        return {
            "command": description,
            "elapsed": 60.0,
            "test_time": None,
            "success": False,
            "output": "TIMEOUT after 60s",
        }
    except Exception as e:
        return {"command": description, "elapsed": None, "test_time": None, "success": False, "output": str(e)}


def compare_test_performance():
    """Compare performance of different test approaches."""
    print("🚀 Starting Performance Benchmarking...")
    print("=" * 60)

    tests_to_run = [
        # Original test approach
        {
            "command": "python -m pytest tests/range_oscillator/test_strategy.py::"
            "TestStrategy1::test_strategy1_basic "
            "-v --tb=no",
            "description": "Original slow test",
        },
        # Optimized test approach
        {
            "command": "python -m pytest tests/range_oscillator/test_strategy_optimized.py::"
            "TestRangeOscillatorOptimized::test_strategy_basic_fast "
            "-v --tb=no",
            "description": "Optimized fast test",
        },
        # Multiple optimized tests
        {
            "command": "python -m pytest tests/range_oscillator/test_strategy_optimized.py::"
            "TestRangeOscillatorOptimized::test_strategy_parametrized_fast "
            "-v --tb=no",
            "description": "Parameterized optimized tests",
        },
        # Edge case optimized tests
        {
            "command": "python -m pytest tests/range_oscillator/test_strategy_optimized.py::"
            "TestRangeOscillatorOptimized::test_strategy_edge_cases_fast "
            "-v --tb=no",
            "description": "Edge case optimized tests",
        },
    ]

    results = []
    for test_config in tests_to_run:
        result = run_test_command(test_config["command"], test_config["description"])
        results.append(result)

        status = "✅ PASSED" if result["success"] else "❌ FAILED"
        elapsed_str = f"{result['elapsed']:.2f}s" if result["elapsed"] else "N/A"

        print(f"{status} {test_config['description']}")
        print(f"   Total time: {elapsed_str}")
        if result["test_time"]:
            print(f"   Test time: {result['test_time']:.3f}s")
        print()

    # Performance comparison
    print("📊 Performance Comparison Summary:")
    print("=" * 60)

    successful_results = [r for r in results if r["success"]]

    if successful_results:
        df = pd.DataFrame(successful_results)

        print(f"{'Test Description':<30} {'Total Time (s)':<15} {'Test Time (s)':<15}")
        print("-" * 60)

        for _, row in df.iterrows():
            total_time = f"{row['elapsed']:.2f}" if row["elapsed"] else "N/A"
            test_time = f"{row['test_time']:.3f}" if row["test_time"] else "N/A"
            print(f"{row['command']:<30} {total_time:<15} {test_time:<15}")

        # Calculate improvements
        if len(successful_results) > 1:
            baseline_time = successful_results[0]["elapsed"]
            fastest_time = min(r["elapsed"] for r in successful_results)
            speedup = baseline_time / fastest_time if fastest_time > 0 else 1

            print()
            print(f"🚀 Speedup Achieved: {speedup:.2f}x")
            print(f"🕒 Baseline Time: {baseline_time:.2f}s")
            print(f"⚡ Fastest Time: {fastest_time:.2f}s")
            print(f"📈 Time Saved: {baseline_time - fastest_time:.2f}s")
            print(f"   ({(1 - fastest_time / baseline_time) * 100:.1f}% saved)")

    # Recommendations
    print("\n💡 Recommendations:")
    print("=" * 60)

    if any(r for r in results if not r["success"]):
        print("❌ Some tests failed - check test implementation")

    if successful_results:
        fastest = min(successful_results, key=lambda x: x["elapsed"])
        print(f"✅ Fastest approach: {fastest['command']}")
        print("✅ Consider adopting this pattern for other tests")

        if fastest["test_time"]:
            if fastest["test_time"] < 1.0:
                print("✅ Excellent performance - tests under 1 second")
            elif fastest["test_time"] < 5.0:
                print("✅ Good performance - tests under 5 seconds")
            else:
                print("⚠️  Consider further optimization")

    # Failed tests details
    failed_results = [r for r in results if not r["success"]]
    if failed_results:
        print("\n❌ Failed Tests Details:")
        print("=" * 60)
        for result in failed_results:
            print(f"Command: {result['command']}")
            print(f"Output: {result['output'][:200]}...")


def run_parallel_tests():
    """Test parallel execution performance."""
    print("\n🔄 Testing Parallel Execution...")
    print("=" * 60)

    parallel_tests = [
        {
            "command": "python -m pytest tests/range_oscillator/test_strategy_optimized.py -v --tb=no",
            "description": "Optimized tests (sequential)",
        },
        {
            "command": "python -m pytest tests/range_oscillator/test_strategy_optimized.py -v --tb=no -n auto",
            "description": "Optimized tests (parallel)",
        },
    ]

    for test_config in parallel_tests:
        result = run_test_command(test_config["command"], test_config["description"])

        status = "✅ PASSED" if result["success"] else "❌ FAILED"
        elapsed_str = f"{result['elapsed']:.2f}s" if result["elapsed"] else "N/A"

        print(f"{status} {test_config['description']}")
        print(f"   Time: {elapsed_str}")
        print()


def main():
    """Main benchmark execution."""
    try:
        compare_test_performance()
        run_parallel_tests()

        print("🎯 Benchmarking Complete!")
        print("Next steps:")
        print("1. Adopt fast test patterns across the suite")
        print("2. Use pytest -m 'not slow' for daily development")
        print("3. Use pytest -n auto for CI/CD pipelines")

    except Exception as e:
        print(f"❌ Benchmark failed: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
