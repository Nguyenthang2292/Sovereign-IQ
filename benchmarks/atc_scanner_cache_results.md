# ATCScanner Cache Benchmark Results

**Date**: 2026-02-01 23:30:44

## Performance Comparison

| Operation | Python (탎) | Rust (탎) | Speedup |
|-----------|-------------|-----------|----------|
| SET (mean) | 0.39 | 2.98 | **0.1x** |
| GET (mean) | 0.26 | 3.55 | **0.1x** |
| MISS (mean) | 0.19 | 0.41 | **0.5x** |

## Memory Usage

| Implementation | Entries | Memory (KB) |
|----------------|---------|-------------|
| Python Cache | 1000 | ~25.42 |
| Rust ScanCache | 1000 | ~270.00 |

## Detailed Metrics

### Python Cache

- **SET**: 0.39 탎 (mean), 0.40 탎 (median), 0.50 탎 (p95)
- **GET**: 0.26 탎 (mean), 0.30 탎 (median), 0.30 탎 (p95)
- **MISS**: 0.19 탎 (mean)

### Rust ScanCache

- **SET**: 2.98 탎 (mean), 2.70 탎 (median), 4.90 탎 (p95)
- **GET**: 3.55 탎 (mean), 3.30 탎 (median), 4.80 탎 (p95)
- **MISS**: 0.41 탎 (mean)

## Recommendations

