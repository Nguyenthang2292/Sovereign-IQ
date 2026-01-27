# Add Approximate MAs Benchmark

## Goal
Add benchmark tests for Approximate MAs (basic and adaptive) to `benchmark_comparison` to measure 2-3x speedup and ~95% accuracy compared to full precision.

## Tasks

- [x] Task 1: Create `run_approximate_module()` in `runners.py` with `use_approximate=True` → Verify: Function runs ATC with approximate MAs and returns (results, time, memory)
- [x] Task 2: Create `run_adaptive_approximate_module()` in `runners.py` with `use_adaptive_approximate=True` → Verify: Function runs ATC with adaptive approximate MAs and returns (results, time, memory)
- [x] Task 3: Add approximate configs to `main.py` (copy rust_config, add `use_approximate=True` and `use_adaptive_approximate=True`) → Verify: Two new config dicts exist with approximate flags
- [x] Task 4: Add benchmark runs in `main.py` after rust_rayon (Step 5.5 and 5.6) → Verify: Both approximate benchmarks execute and log completion
- [x] Task 5: Update `compare_signals()` to include approximate results → Verify: Function accepts and compares approximate_results and adaptive_approximate_results
- [x] Task 6: Update `generate_comparison_table()` to include approximate columns → Verify: Table shows "Approximate" and "Adaptive Approx" rows with speedup vs Rust
- [x] Task 7: Test benchmark with `--symbols 20 --bars 500` → Verify: Approximate shows ~2-3x speedup and ~95% signal match vs full precision

## Done When

- [x] Benchmark runs both approximate modes (basic and adaptive)
- [x] Comparison table shows speedup metrics (2-3x expected)
- [x] Signal accuracy comparison shows ~95% match vs full precision
- [x] Results saved to `benchmark_results.txt` with approximate data


## Notes

- Approximate MAs should be compared against Rust (full precision) as baseline
- Expected: 2-3x faster, ~95% accuracy (signals may differ slightly by design)
- Use same config as Rust but with `use_approximate=True` or `use_adaptive_approximate=True`
