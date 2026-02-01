# Performance Benchmarks

This directory contains performance benchmarks for various components of the Sovereign-IQ system.

## Available Benchmarks

### ATC Scanner Conversion Overhead Analysis

**Script**: `atc_scanner_conversion_overhead.py`

Measures the performance overhead of Pandas->Polars conversion in the ATCScanner module as part of the migration to prepare for Rust integration (Phase 2).

#### Features

- **Statistical Rigor**: Multiple iterations with warmup runs
- **Realistic Mock Data**: Varied signal strengths and realistic distributions
- **Error Handling**: Robust try-finally blocks ensure cleanup
- **Type Safety**: Full type annotations throughout
- **Configurable**: CLI arguments for test cases and iterations
- **Export Options**: CSV export and visualization plots
- **Memory Profiling**: Tracks peak memory usage with tracemalloc

#### Usage

Basic usage with defaults (10, 50, 100, 500 symbols, 5 iterations):
```bash
python tests/performance/atc_scanner_conversion_overhead.py
```

Custom configuration:
```bash
# Custom test cases
python tests/performance/atc_scanner_conversion_overhead.py --test-cases 10 50 100

# More iterations for better statistics
python tests/performance/atc_scanner_conversion_overhead.py --iterations 10

# Export results to CSV
python tests/performance/atc_scanner_conversion_overhead.py --csv results.csv

# Generate plots (requires matplotlib)
python tests/performance/atc_scanner_conversion_overhead.py --plot

# All options combined
python tests/performance/atc_scanner_conversion_overhead.py \
  --test-cases 10 50 100 500 1000 \
  --iterations 5 \
  --csv results.csv \
  --plot \
  --plot-output benchmark.png
```

#### Output

The benchmark provides:
- Formatted table with statistical measures (mean ± stdev)
- Percentage differences vs baseline
- Memory usage statistics
- CSV export for trend analysis
- Visualization plots showing:
  - Execution time comparison with error bars
  - Memory usage comparison with error bars

#### Interpreting Results

**What the benchmark measures:**
- Conversion overhead (Pandas -> Polars)
- Parallel execution overhead
- Aggregation logic performance
- Memory efficiency

**What it does NOT measure:**
- Fully optimized Polars implementation benefits
- Rust integration performance (Phase 2)
- Network/data fetching overhead (mocked)

**Expected results:**
- Polars conversion adds 7-38% overhead (depending on symbol count)
- This is temporary overhead for Phase 1 (prerequisite for Rust)
- Phase 2 (Rust integration) will more than compensate for this overhead

#### Technical Details

**Mock Data Generation:**
- 60% of symbols receive signals
- 70% longs, 30% shorts distribution
- Signal strengths: 0.3-0.9 (longs), -0.9 to -0.3 (shorts)
- Realistic variation in signal distribution

**Statistical Measures:**
- Mean, median, standard deviation
- Min, max values
- Warmup runs to stabilize JIT/caching
- Multiple iterations for confidence

**Error Handling:**
- Try-finally blocks ensure tracemalloc cleanup
- Robust exception handling throughout
- Safe division (checks for zero denominators)

## Adding New Benchmarks

When adding new performance benchmarks to this directory:

1. **Follow the naming convention**: `<module>_<what_is_measured>.py`
   - Example: `atc_scanner_conversion_overhead.py`

2. **Include comprehensive documentation**:
   - Module docstring with usage examples
   - Type annotations for all functions
   - Comments explaining methodology

3. **Implement statistical rigor**:
   - Multiple iterations
   - Warmup runs
   - Calculate mean, median, stdev
   - Report confidence intervals

4. **Error handling**:
   - Use try-finally for cleanup (tracemalloc, file handles)
   - Handle edge cases (empty data, division by zero)

5. **Make it configurable**:
   - Use argparse for CLI arguments
   - Default values for common use cases
   - Help text with examples

6. **Provide multiple output formats**:
   - Human-readable console output
   - CSV export for trend analysis
   - Plots for visualization (optional)

7. **Document results**:
   - Create or update documentation in `modules/<module>/docs/`
   - Include methodology, results, and analysis
   - Explain what the benchmark measures and what it doesn't

## Best Practices

### Memory Profiling

Always use tracemalloc with proper cleanup:

```python
tracemalloc.start()
try:
    # Your benchmark code here
    start_time = time.perf_counter()
    result = function_to_benchmark()
    end_time = time.perf_counter()
finally:
    current, peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()
```

### Statistical Analysis

Collect multiple samples and report statistics:

```python
import statistics

times = []
for _ in range(iterations):
    start = time.perf_counter()
    function_to_benchmark()
    end = time.perf_counter()
    times.append(end - start)

mean_time = statistics.mean(times)
median_time = statistics.median(times)
stdev_time = statistics.stdev(times) if len(times) > 1 else 0.0
```

### Realistic Mock Data

Generate varied, realistic data:

```python
# Good: Varied signal strengths
signals = [random.uniform(0.3, 0.9) for _ in range(n)]

# Bad: All identical values
signals = [0.8] * n
```

### Type Safety

Use type hints throughout:

```python
from typing import List, Dict, Any, Tuple

def benchmark_function(
    num_items: int,
    iterations: int = 5
) -> Dict[str, Any]:
    """
    Run benchmark with type-safe parameters.

    Args:
        num_items: Number of items to process
        iterations: Number of benchmark iterations

    Returns:
        Dictionary with benchmark results
    """
    # Implementation
    pass
```

## Dependencies

Common dependencies for benchmarks:
- `time` (stdlib): High-precision timing
- `tracemalloc` (stdlib): Memory profiling
- `statistics` (stdlib): Statistical calculations
- `argparse` (stdlib): CLI argument parsing
- `csv` (stdlib): CSV export
- `matplotlib` (optional): Visualization

Install optional dependencies:
```bash
pip install matplotlib  # For plots
```

## Results Storage

Store benchmark results in:
- **Console**: Human-readable formatted tables
- **CSV**: `<benchmark_name>_results_<timestamp>.csv`
- **Plots**: `<benchmark_name>_plot_<timestamp>.png`
- **Documentation**: `modules/<module>/docs/<benchmark>_results.md`

## Continuous Monitoring

To track performance over time:

1. Run benchmarks with CSV export
2. Store results in version control or separate tracking system
3. Generate trend plots periodically
4. Set up alerts for regressions (e.g., >10% slowdown)

Example:
```bash
# Run and timestamp results
python tests/performance/atc_scanner_conversion_overhead.py \
  --csv "results/atc_scanner_$(date +%Y%m%d).csv"
```

## See Also

- `modules/auto_trade/docs/core/atc_scanner_benchmark_results.md`: ATC Scanner benchmark results and analysis
- `tests/docs/test_memory_usage_guide.md`: Memory optimization for tests
- `pytest_memory.ini`: Memory-optimized pytest configuration
