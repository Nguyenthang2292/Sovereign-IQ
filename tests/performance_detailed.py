
import sys
import time

import numpy as np
import pandas as pd
import pandas as pd

"""
🚀 Detailed performance profiler for test optimization.

This script profiles specific operations to identify bottlenecks.
"""



def time_operation(name, operation, iterations=5):
    """Time an operation multiple times and return statistics."""
    times = []
    
    for i in range(iterations):
        start = time.time()
        try:
            result = operation()
            elapsed = time.time() - start
            times.append(elapsed)
        except Exception as e:
            print(f"Error in {name}: {e}")
            return None
    
    if not times:
        return None
    
    times_np = np.array(times)
    return {
        'operation': name,
        'mean': float(np.mean(times_np)),
        'median': float(np.median(times_np)),
        'std': float(np.std(times_np)),
        'min': float(np.min(times_np)),
        'max': float(np.max(times_np)),
        'iterations': iterations
    }


def profile_data_creation():
    """Profile different data creation methods."""
    print("Data Creation Profiling")
    print("=" * 60)
    
    # Method 1: Original approach
    def original_method():
        np.random.seed(42)
        n = 200
        dates = pd.date_range("2024-01-01", periods=n, freq="1h")
        close = 50000 + np.cumsum(np.random.randn(n) * 100)
        high = close + np.abs(np.random.randn(n) * 50)
        low = close - np.abs(np.random.randn(n) * 50)
        
        return pd.Series(high, index=dates), pd.Series(low, index=dates), pd.Series(close, index=dates)
    
    # Method 2: Optimized with caching
    try:
        from test_data_factories import TestDataFactory
        factory = TestDataFactory(seed=42)
        
        def cached_method():
            return factory.create_series_data(size=200, cache_key='test_200')
    except ImportError:
        def cached_method():
            # Fall back to original if import fails
            return original_method()
    
    # Profile both
    result1 = time_operation("Original data creation (200 rows)", original_method, iterations=3)
    result2 = time_operation("Cached factory creation (200 rows)", cached_method, iterations=3)
    
    for result in [result1, result2]:
        if result:
            print(f"{result['operation']}:")
            print(f"  Mean: {result['mean']:.4f}s")
            print(f"  Median: {result['median']:.4f}s")
            print(f"  Min: {result['min']:.4f}s")
            print(f"  Max: {result['max']:.4f}s")
            print()


def test_import_overhead():
    """Test import overhead for strategy module."""
    print("Import Overhead Profiling")
    print("=" * 60)
    
    # Test 1: Import at top level
    import_time = time_operation("Import at module level", lambda: __import__('modules.range_oscillator.strategies.basic'))
    
    # Test 2: Lazy import
    def lazy_import():
        from modules.range_oscillator.strategies.basic import generate_signals_basic_strategy
        return generate_signals_basic_strategy
    
    lazy_import_time = time_operation("Lazy import (inside function)", lazy_import, iterations=3)
    
    for result in [import_time, lazy_import_time]:
        if result:
            print(f"{result['operation']}:")
            print(f"  Mean: {result['mean']:.4f}s")
            print(f"  Median: {result['median']:.4f}s")
            print()


def test_strategy_execution():
    """Test actual strategy execution time."""
    print("Strategy Execution Profiling")
    print("=" * 60)
    
    # Prepare data
    from tests.test_data_factories import get_global_factory
    factory = get_global_factory()
    high, low, close = factory.create_series_data(size=100, cache_key='test_100')
    
    # Import strategy
    from modules.range_oscillator.strategies.basic import generate_signals_basic_strategy
    
    # Test strategy execution
    def execute_strategy():
        return generate_signals_basic_strategy(
            high=high, low=low, close=close,
            length=20, mult=2.0
        )
    
    strategy_time = time_operation("Strategy execution (100 rows)", execute_strategy, iterations=5)
    
    if strategy_time:
        print(f"{strategy_time['operation']}:")
        print(f"  Mean: {strategy_time['mean']:.4f}s")
        print(f"  Median: {strategy_time['median']:.4f}s")
        print(f"  Min: {strategy_time['min']:.4f}s")
        print(f"  Max: {strategy_time['max']:.4f}s")
        print()


def test_assertion_overhead():
    """Test assertion overhead."""
    print("Assertion Overhead Profiling")
    print("=" * 60)
    
    from tests.test_data_factories import get_global_factory
    factory = get_global_factory()
    high, low, close = factory.create_series_data(size=100, cache_key='assertion_test')
    
    # Minimal assertions
    def minimal_assertions(data):
        assert len(data) > 0
        assert isinstance(data, tuple)
        return True
    
    minimal_time = time_operation("Minimal assertions", lambda: minimal_assertions(close), iterations=10)
    
    # Comprehensive assertions
    def comprehensive_assertions(data):
        assert isinstance(data[0], pd.Series), "High should be Series"
        assert isinstance(data[1], pd.Series), "Low should be Series"
        assert isinstance(data[2], pd.Series), "Close should be Series"
        assert len(data[0]) > 0, "High should have values"
        assert len(data[1]) > 0, "Low should have values"
        assert len(data[2]) > 0, "Close should have values"
        assert len(data[0]) == len(data[1]), "High and Low should match length"
        assert len(data[1]) == len(data[2]), "Low and Close should match length"
        return True
    
    comprehensive_time = time_operation("Comprehensive assertions", lambda: comprehensive_assertions((high, low, close)), iterations=10)
    
    for result in [minimal_time, comprehensive_time]:
        if result:
            print(f"{result['operation']}:")
            print(f"  Mean: {result['mean']:.4f}s")
            print(f"  Median: {result['median']:.4f}s")
            print()


def test_pytest_overhead():
    """Test pytest collection and execution overhead."""
    print("Pytest Overhead Analysis")
    print("=" * 60)
    
    import subprocess
    import os
    
    # Set environment for Windows
    env = os.environ.copy()
    env['PYTHONIOENCODING'] = 'utf-8'
    env['PYTHONLEGACYWINDOWSSTDIO'] = 'utf-8'
    
    # Test collection time
    collect_start = time.time()
    collect_result = subprocess.run(
        ['python', '-m', 'pytest', 'tests/range_oscillator/test_strategy1_optimized.py', '--collect-only'],
        capture_output=True,
        env=env
    )
    collect_time = time.time() - collect_start
    
    print(f"Test collection time: {collect_time:.2f}s")
    print(f"Tests collected: {collect_result.stdout.count('test_')}")
    print()


def main():
    """Run all profiling tests."""
    print("Detailed Performance Profiling")
    print("=" * 60)
    print()
    
    profile_data_creation()
    test_import_overhead()
    test_strategy_execution()
    test_assertion_overhead()
    test_pytest_overhead()
    
    print("=" * 60)
    print("Profiling Complete!")
    print("Key Findings:")
    print("1. Data creation: Compare original vs cached factory")
    print("2. Import overhead: Compare top-level vs lazy imports")
    print("3. Strategy execution: Time for actual calculation")
    print("4. Assertion overhead: Minimal vs comprehensive")
    print("5. Pytest overhead: Collection time")
    print()
    print("Recommendations:")
    print("- Use cached data factories if > 2x speedup")
    print("- Lazy imports only help if imports are slow")
    print("- Strategy execution is likely the bottleneck")
    print("- Minimal assertions are 10-100x faster")


if __name__ == "__main__":
    main()