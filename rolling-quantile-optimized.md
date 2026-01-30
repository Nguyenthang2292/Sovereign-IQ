# Optimized Rolling Quantile Algorithm (O(n log w) vs O(n·w·log w))

## Goal
Replace the sort-per-window rolling quantile with an incremental BTreeMap-based algorithm to achieve O(n log w) instead of O(n × w × log w), giving 5–10x speedup for large windows (500+).

## Tasks

- [ ] **Task 1**: Add `ordered-float = "4.2"` to `rust_extensions/Cargo.toml` [dependencies] → Verify: `cargo build --release` in rust_extensions succeeds

- [ ] **Task 2**: Implement `get_quantile_from_btree(btree: &BTreeMap<OrderedFloat<f64>, usize>, q: f64, window: usize) -> f64` in `labeling.rs` (iterate sorted entries, accumulate count until rank ≥ (window-1)*q) → Verify: Unit test for known quantiles passes

- [ ] **Task 3**: Implement `rolling_quantile_btree()` using BTreeMap sliding window (add new, remove old, call `get_quantile_from_btree`) in `labeling.rs` → Verify: Results match current `rolling_quantile_rust` on sample data

- [ ] **Task 4**: Replace `rolling_quantile_rust` body with BTreeMap implementation (or call `rolling_quantile_btree` internally), keep same `#[pyfunction]` signature → Verify: Python `rolling_quantile_rust()` unchanged; `pytest tests/xgboost_LTS/` passes

- [ ] **Task 5**: Handle NaN/Inf in input (skip or use `OrderedFloat::from` with NaN handling) → Verify: Input with NaN does not panic; output matches pandas semantics where applicable

- [ ] **Task 6**: Add benchmark with window=500 to `labeling_benchmark.rs`, run `cargo bench` → Verify: New impl faster than old (or document if parallel rayon still wins for small n)

- [ ] **Task 7**: Run full test suite and labeling integration → Verify: `pytest tests/xgboost_LTS/ -v` and `pytest modules/xgboost_LTS/benchmarks/ -v` pass

## Done When

- [ ] BTreeMap-based rolling quantile implemented and wired to Python
- [ ] All tests pass, no regression in labeling output
- [ ] Benchmark shows improvement for window ≥ 500

## Notes

- **BTreeMap multiset**: `BTreeMap<OrderedFloat<f64>, usize>` — key = value, count = multiplicity (handles duplicates).
- **Quantile rank**: `rank = ((window - 1) as f64 * q).floor() as usize` (0-indexed, matches current behavior).
- **Sequential only**: BTreeMap sliding window is inherently sequential; parallel version would need a different approach (e.g. parallel chunks + merge). Expect 5–10x for large w even vs parallel sort.
- **NaN**: `OrderedFloat` from `ordered-float` handles NaN (NaN compares greater than all); consider filtering NaNs or matching pandas behavior (NaN in window → NaN out).
