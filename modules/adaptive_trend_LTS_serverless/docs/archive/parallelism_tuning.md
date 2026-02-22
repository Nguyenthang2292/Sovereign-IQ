# Parallelism Tuning Report

Generated: 2026-02-15 23:24:52
Benchmark mode: synthetic OHLCV, repeats=2, bars_per_tf=200

## Scope

- Thread-count sweep: 1, 4, 8
- Batch sizes: 10, 100 symbols
- Parallel strategy in code: rayon::into_par_iter + with_min_len + chunks
- par_bridge vs par_iter: evaluated; par_iter retained because input is an in-memory Vec
- rayon::scope nested parallelism: evaluated; not adopted to avoid nested scheduling overhead

## Thread Sweep Results

| Batch | Threads | Chunk | Mean (ms) | Std (ms) | P95 (ms) | Throughput (sym/s) | Speedup | Efficiency |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| 10 | 1 | 1 | 9.00 | 0.00 | 9.00 | 1111.11 | 1.00x | 100.00% |
| 10 | 4 | 1 | 5.00 | 0.00 | 5.00 | 2000.00 | 1.80x | 45.00% |
| 10 | 8 | 1 | 5.50 | 0.50 | 5.95 | 1818.18 | 1.64x | 20.45% |
| 100 | 1 | 10 | 91.00 | 1.00 | 91.90 | 1098.90 | 1.00x | 100.00% |
| 100 | 4 | 10 | 37.00 | 5.00 | 41.50 | 2702.70 | 2.46x | 61.49% |
| 100 | 8 | 10 | 28.50 | 0.50 | 28.95 | 3508.77 | 3.19x | 39.91% |

## Best Configuration by Batch

| Batch | Best Threads | Chunk | Mean (ms) | Throughput (sym/s) | Speedup | Efficiency |
|---|---:|---:|---:|---:|---:|---:|
| 10 | 4 | 1 | 5.00 | 2000.00 | 1.80x | 45.00% |
| 100 | 8 | 10 | 28.50 | 3508.77 | 3.19x | 39.91% |

## Chunk Size Sweep (Batch=10, Threads=4)

| Chunk | Mean (ms) | Std (ms) | Throughput (sym/s) |
|---:|---:|---:|---:|
| 1 | 5.00 | 0.00 | 2000.00 |
| 5 | 5.00 | 0.00 | 2000.00 |
| 10 | 9.00 | 0.00 | 1111.11 |
| 25 | 9.00 | 0.00 | 1111.11 |
| 50 | 9.00 | 0.00 | 1111.11 |

## Lambda Notes

- Auto-tuning in lambda handler selects thread/chunk by batch size.
- Validate x86_64 and ARM64 separately in real Lambda runtime for final production tuning.
- Current report is generated on local host; use as baseline, not absolute capacity planning.

