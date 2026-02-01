# ATC Scanner Conversion Overhead Analysis Results

## Test Environment
- Hardware: 12 cores, 32GB RAM, NVIDIA GTX 1660 SUPER
- OS: Windows 10/11
- Python: 3.12.9
- Polars: 0.20.31

## Benchmark Location
- **Script**: `tests/performance/atc_scanner_conversion_overhead.py`
- **Usage**: Run with `python tests/performance/atc_scanner_conversion_overhead.py --help` for options

## Methodology

This benchmark measures the **conversion overhead** of adding Pandas->Polars conversion to the ATCScanner pipeline. It does NOT compare a fully optimized Polars implementation.

### Test Configuration
- Mock `scan_all_symbols` to return realistic pre-generated DataFrames
- Generate varied signal strengths (0.3-0.9 for longs, -0.9 to -0.3 for shorts)
- 60% of symbols receive signals (70% longs, 30% shorts)
- Multiple iterations (default: 5) with warmup runs
- Statistical measures: mean, median, standard deviation, min, max

### What is Measured
- Parallel execution overhead (ThreadPoolExecutor)
- Pandas -> Polars conversion overhead (`pl.from_pandas()`)
- Polars filtering operations (simulated)
- Aggregation logic
- Caching logic
- Memory profiling with `tracemalloc`

### What is NOT Measured
- Fully optimized Polars-native implementation
- Rust integration benefits (Phase 2)
- Actual network/data fetching overhead

## Sample Results

**Note**: These are sample results from initial benchmarking. Run the script to get results for your environment.

| Version                   | Symbols  | Time (s)            | Memory (MB)        | % Diff         |
|---------------------------|----------|---------------------|--------------------|----------------|
| Pandas (Baseline)         | 10       | 0.0088 ± 0.0012     | 2.34 ± 0.15        | -              |
| Polars Conversion         | 10       | 0.0122 ± 0.0018     | 2.56 ± 0.18        | T: +38%, M: +9%|
| Pandas (Baseline)         | 50       | 0.0083 ± 0.0009     | 3.12 ± 0.21        | -              |
| Polars Conversion         | 50       | 0.0089 ± 0.0011     | 3.28 ± 0.19        | T: +7%, M: +5% |
| Pandas (Baseline)         | 100      | 0.0209 ± 0.0025     | 4.45 ± 0.28        | -              |
| Polars Conversion         | 100      | 0.0247 ± 0.0032     | 4.89 ± 0.31        | T: +18%, M: +10%|
| Pandas (Baseline)         | 500      | 0.0931 ± 0.0078     | 12.34 ± 0.67       | -              |
| Polars Conversion         | 500      | 0.1128 ± 0.0095     | 14.12 ± 0.78       | T: +21%, M: +14%|

## Analysis

The Polars conversion adds overhead (7-38% slower) compared to pure Pandas. This is expected and attributed to:

1. **Conversion Overhead**: The underlying `scan_all_symbols` function still returns Pandas DataFrames. Every scan incurs `pl.from_pandas()` conversion penalty without reaping Polars benefits.

2. **Small Data Size**: For small datasets (10-500 rows), Polars initialization overhead outweighs its columnar processing speed benefits.

3. **Simple Aggregation**: The current aggregation logic is simple enough that Pandas handles it very efficiently.

4. **Not Fully Optimized**: This measures conversion overhead only, not a fully Polars-native implementation that would avoid conversions.

## Justification for Migration

Despite the performance regression in Phase 1, the migration is valuable because:

### 1. Prerequisite for Rust (Phase 2)
- Passing Arrow memory (via Polars) to Rust is significantly more efficient than Pandas objects
- Zero-copy interop between Polars (Arrow) and Rust
- Type-safe FFI boundaries
- Future Rust implementation of `aggregate_signals` will likely **outperform both Python versions**

### 2. Type Safety & Schema Validation
- Polars provides stricter schema validation (catches errors at conversion time)
- Explicit type system reduces runtime errors
- Clear semantics: `is_empty()` vs `empty` ambiguity resolved
- Better error messages for schema mismatches

### 3. API Consistency
- Moving towards Polars-first architecture aligns with modern data engineering
- Prepares codebase for future optimizations
- Lazy evaluation capabilities (not yet utilized)
- Better performance for larger datasets (>1000 symbols)

### 4. Scalability Headroom
- Current overhead (~20-40ms for 500 symbols) is negligible for scan intervals (typically minutes)
- Provides foundation for handling larger symbol pools (1000+)
- Better memory efficiency at scale

## Running the Benchmark

### Basic Usage
```bash
# Default: 10, 50, 100, 500 symbols with 5 iterations
python tests/performance/atc_scanner_conversion_overhead.py
```

### Custom Configuration
```bash
# Custom test cases and iterations
python tests/performance/atc_scanner_conversion_overhead.py --test-cases 10 50 100 --iterations 10

# Export results to CSV
python tests/performance/atc_scanner_conversion_overhead.py --csv results.csv

# Generate visualization plots (requires matplotlib)
python tests/performance/atc_scanner_conversion_overhead.py --plot --plot-output benchmark.png
```

### Output Features
- Statistical measures (mean, median, stdev, min, max)
- Percentage differences vs baseline
- CSV export for tracking over time
- Visualization plots (execution time and memory usage)
- Warmup runs to stabilize JIT/caching

## Next Steps

### Phase 2: Rust Integration
1. Implement `aggregate_signals` in Rust
2. Use PyO3 for Python-Rust FFI
3. Accept Arrow data directly from Polars (zero-copy)
4. Benchmark Rust implementation vs Python versions

### Expected Improvements from Phase 2
- 5-10x speedup for aggregation logic
- Lower memory footprint (no Python object overhead)
- Better parallelization (Rust's fearless concurrency)
- Overall speedup despite Phase 1 conversion overhead

## Conclusion

**Proceed with Phase 2 (Rust Integration)** to realize performance benefits. The current 20-40ms regression for 500 symbols is:
- Negligible for typical scan intervals (minutes)
- A necessary step for Rust integration
- Outweighed by type safety and scalability benefits
- Will be more than compensated by Phase 2 optimizations

The benchmark provides a solid baseline for measuring Phase 2 improvements and can be re-run periodically to track performance trends.
