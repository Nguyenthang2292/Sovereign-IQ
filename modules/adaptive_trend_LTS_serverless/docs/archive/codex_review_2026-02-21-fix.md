# ATC Serverless — Codex Review Fixes

## Goal

Fix all **Critical** and **High** issues identified in `docs/codex_review_2026-02-21.md`, then address selected **Medium** issues. Estimated total effort: ~12h.

---

## Phase 1 — Critical Fixes (Block release)

- [x] **C3** Fix KAMA SIMD silent data corruption  
  File: `src/ma_simd.rs` — In `calculate_kama_simd()` inner SIMD loop (line ~369), replace the `else { 0.0 }` sentinel fallback with a `continue` skip so out-of-bounds positions are excluded from volatility sum instead of zeroed.  
  → Verify: `cargo test --features simd test_kama_simd_vs_scalar` passes with tolerance `< 1e-10`

- [x] **C1** Fix SMA SIMD O(n×length) performance regression  
  File: `src/ma_simd.rs` — Rewrite `calculate_sma_simd()` to use sliding window: compute first window with SIMD `reduce_sum`, then slide `O(n)` using scalar `window_sum += from - out`.  
  → Verify: Benchmark `atc_benchmark` shows SIMD SMA ≥ scalar SMA speed; `test_sma_simd_vs_scalar` still passes

- [x] **C2** Replace `eprintln!` with `tracing` / `log` in core library  
  Files: `src/signal_detection.rs`, `src/aggregation.rs`, `src/buffer_pool.rs` — Add `tracing = { version = "0.1", optional = true }` to root `Cargo.toml`. Wrap calls: `tracing::warn!(...)` behind `#[cfg(feature = "tracing")]`, keeping `eprintln!` as fallback so the library compiles without the feature.  
  → Verify: Lambda CloudWatch output shows structured JSON for warn/error messages from core lib; `cargo test` (no feature) still compiles

---

## Phase 2 — High Fixes (Before next iteration)

- [x] **H3** Make Rayon thread pool static (Lambda warm reuse)  
  File: `src/aggregation.rs` — Replace `create_custom_thread_pool(num_threads)` call with a `static THREAD_POOL: OnceLock<rayon::ThreadPool>` initialized on first call. Use `num_threads` from environment variable `ATC_FORCE_THREADS` or default.  
  → Verify: Add test asserting `process_batch` called twice reuses the same pool (check thread count stays stable); confirm no thread-spawn overhead in warm Lambda benchmark

- [x] **H2** Replace `ma_type: String` with `MAType` enum  
  Files: `src/lib.rs`, `src/signal_detection.rs` — Add `pub enum MAType { Ema, Hma, Wma, Dema, Lsma, Kama }` with `#[derive(Serialize, Deserialize)]` + `#[serde(rename_all = "UPPERCASE")]`. Update `MAConfig.ma_type`, `calculate_ma_variation()` match arm, and `validate_config()`. Remove silent EMA fallback — return `Err` on unknown type.  
  → Verify: `cargo test` passes; deserializing `{"ma_type": "BAD"}` returns a serde error; existing JSON payloads with `"EMA"` still deserialize correctly

- [x] **H1** Replace wildcard re-exports with explicit public API  
  File: `src/lib.rs` (lines 85–94) — Replace `pub use aggregation::*;` etc. with explicit named re-exports of: `process_batch`, `ATCConfig`, `MAConfig`, `MAType`, `BatchRequest`, `SymbolData`, `OHLCVData`, `ScanResult`, `SignalResult`, `SymbolError`, `validate_batch_request`, `validate_config`, `validate_ohlcv_data`, `ParallelismConfig`, `Robustness`, `calculate_diflen`, `compute_symbol_score`, `get_memory_usage_mb`.  
  → Verify: `cargo doc` compiles clean; existing tests still compile without importing from submodules directly

- [x] **H4** Fix `Robustness` parse error type and remove silent unwrap  
  File: `src/signal_detection.rs` — Change `type Err = ()` → `type Err = String`, return `Err(format!("Unknown robustness: '{}'", s))`. In `compute_symbol_score()`, change `.unwrap_or(Robustness::Medium)` → `validate_config` already rejects bad values, so use `.expect("robustness already validated")` or propagate the error.  
  → Verify: Passing `robustness: "bad"` through `validate_batch_request` returns `ValidationError::Config`; no silent fallback in prod path

- [x] **H5** Improve buffer pool to accept buffers ≥ requested size  
  File: `src/buffer_pool.rs` — In `get_buffer()`, change exact-size match `pool[i].len() == size` to `pool[i].len() >= size`, call `buf.slice_mut(s![..size])` before returning (or `buf.truncate(size)` equivalent). Pool cap stays at 16.  
  → Verify: `test_buffer_pool()` passes; add a new test `test_buffer_pool_reuse_larger()` confirming a size-200 buffer is returned for a size-150 request

---

## Phase 3 — Medium Fixes (Quality/Correctness)

- [x] **M1** Extract shared ROC helper to remove duplication  
  File: `src/signal_detection.rs` — Extract the identical ROC block (lines 179–184 and 296–301) into `fn calculate_roc(prices: ArrayView1<f64>, n: usize) -> Array1<f64>`. Call from both `calculate_layer1_signal` and `calculate_layer1_signal_single`.  
  → Verify: `cargo test` passes; no logic change

- [x] **M2** Fix double-counted memory estimate in `estimate_batch_memory_mb`  
  File: `src/aggregation.rs` (lines 49–57) — Replace the two identical `bars * 6 * 8` lines with one OHLCV line + one working-buffer line: `total_bytes += bars * 6 * 8` (raw data) + `total_bytes += bars * 3 * 8` (3× working buffers: roc, r_adjusted, sig_shifted).  
  → Verify: Unit test asserting `estimate_batch_memory_mb(&[1 symbol, 200 bars, 2 timeframes])` ≈ expected MB value

- [x] **M3** Add validation guard in `process_batch` for direct callers  
  File: `src/aggregation.rs` — Add optional param or doc-comment clearly stating validation is the caller's responsibility. At minimum, add a `debug_assert_eq!(validate_config(&config).is_ok(), true)` so tests surface invalid configs immediately.  
  → Verify: `cargo test` with a deliberately bad config triggers the assertion in debug mode

- [x] **M4** Clamp adaptive threshold to prevent 0 over-sensitivity  
  File: `src/multi_tf_voting.rs` (line 41) — Change to:  

  ```rust
  let adaptive_threshold = (config.threshold * weight_ratio).max(config.threshold * 0.1);
  ```  

  → Verify: Add unit test `test_aggregate_no_matching_timeframe()` asserting result is `NEUTRAL` when no timeframe matches config weights

- [x] **M5** Fix `BehaviorVersion` import path  
  File: `lambda/src/main.rs` (line 6) — Change `use aws_sdk_sqs::config::BehaviorVersion;` → `use aws_config::BehaviorVersion;`  
  → Verify: `cargo build -p atc_lambda` compiles without warnings

---

## Phase 4 — Low / Style Fixes

- [x] **L2** Remove `#[allow(dead_code)]` from public MA functions  
  File: `src/ma_calculations.rs` — Remove the attribute from `calculate_wma`, `calculate_dema`, `calculate_lsma`, `calculate_kama`, `calculate_hma`. These are public API functions.  
  → Verify: `cargo build` compiles without new dead-code warnings

- [x] **L4** Normalize `equity.rs` line endings to LF  
  File: `src/equity.rs` — Convert CRLF → LF (use editor "Change End of Line Sequence" or `sed -i 's/\r//' equity.rs`)  
  → Verify: `git diff --check` shows no whitespace warnings for this file

- [x] **L1** Add large test data files to `.gitignore`  
  File: `.gitignore` — Append `test_data_120.json` and `test_data_500.json`  
  → Verify: `git status` no longer lists those files as tracked

---

## Phase 5 — New Tests (Close coverage gaps)

- [x] Add unit tests for `multi_tf_voting::aggregate_timeframes()` directly  
  File: `tests/atc_tests.rs` — Add tests: all-LONG timeframes → LONG, mixed signals, no matching weights → NEUTRAL, `use_signal_strength = false` path  
  → Verify: `cargo test test_aggregate` passes

- [x] Add Lambda handler integration test (mock SQS)  
  File: `lambda/src/handler.rs` — Add `#[cfg(test)]` module with `MockSqsClient`, test `handle_request` with valid payload returns `Ok(())`, invalid payload returns `Err`.  
  → Verify: `cargo test -p atc_lambda` passes

---

## Done When

- [x] `cargo test` passes (all existing + new tests)
- [x] `cargo clippy -- -D warnings` produces zero errors
- [x] `cargo build --features simd --release` compiles
- [x] All 3 Critical issues resolved
- [x] All 5 High issues resolved
- [x] CHANGELOG updated under `[Unreleased]` with each fix

---

## Notes

- Work in order: C3 → C1 → C2 → H3 → H2 → H1 → H4 → H5 → Medium → Low
- C2 (`eprintln` → `tracing`) can be done in parallel with H2 (enum) if needed
- H2 is a **breaking API change** — bump version to `0.2.0` when merged
- Do not start Phase 2 tasks until Phase 1 is green (`cargo test` passes)
