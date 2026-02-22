# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added

- Added final codex review: `docs/codex_review_final_2026-02-22.md` — **32/32 issues resolved across 2 review cycles**, 0 Critical/High/Medium remaining, 3 cosmetic-only items. Module approved for v0.2.0 release ✅.
- Added codex review report: `docs/codex_review_2026-02-22.md` — follow-up review confirming all 21 previous issues resolved ✅, identified 2 High, 4 Medium, 5 Low new findings.
- Added codex review report: `docs/codex_review_2026-02-21.md` — full module audit with 3 Critical, 5 High, 7 Medium, 6 Low findings. (archived)
- Added codex review fix tracker: `docs/codex_review_2026-02-21-fix.md` — actionable checklist for all identified issues. (archived)
- **`MAType` enum** (`src/lib.rs`) — replaces `ma_type: String` in `MAConfig` with a strongly-typed enum (`Ema`, `Hma`, `Wma`, `Dema`, `Lsma`, `Kama`). Implements `Display`, `FromStr`, and serde `rename_all = "UPPERCASE"` for backward-compatible JSON deserialization.
- **`calculate_roc` helper** (`src/signal_detection.rs`) — extracted shared Rate-of-Change block used by both `calculate_layer1_signal` and `calculate_layer1_signal_single` to eliminate code duplication.
- **`test_buffer_pool_reuse_larger`** (`src/buffer_pool.rs`) — new unit test confirming a ≥-sized pooled buffer is correctly sliced and reused.
- **`test_estimate_batch_memory_mb`** (`src/aggregation.rs`) — unit test asserting the corrected per-symbol memory estimate formula.
- **`test_aggregate_no_matching_timeframe`** (`src/multi_tf_voting.rs`) — unit test confirming `NEUTRAL` result when no timeframe matches config weights.
- **Integration tests for `aggregate_timeframes`** (`tests/atc_tests.rs`): all-LONG → LONG, mixed signals, no matching weights → NEUTRAL, `use_signal_strength = false` path.
- **Lambda handler test** (`lambda/src/handler.rs`) — `test_handler_validation_error` validates that an empty-symbol payload correctly returns `Err`.
- **`SignalType` enum** (`src/lib.rs`) — replaces string-based signal classification with strongly typed variants (`Long`, `Short`, `Neutral`) serialized as `UPPERCASE`.
- **`SignalParams` struct** (`src/signal_detection.rs`) — groups Layer-1 runtime parameters to simplify function signatures and remove excessive argument lists.
- **Mockable Lambda SQS boundary** (`lambda/src/handler.rs`) — introduced `SqsSender` trait with `MockSqsClient` and added `test_handler_success_path`.

### Changed

- **`MAConfig.ma_type`** changed from `String` to `MAType` enum — **breaking API change** (bump to `0.2.0` on release).
- **`ATCConfig.robustness`** changed from `String` to `Robustness` enum (`PascalCase` serde) — invalid values now fail during deserialization.
- **`SignalResult.signal_type`** changed from `String` to `SignalType` enum; `details` now use `HashMap<String, SignalType>`.
- **`calculate_ma_variation`** (`src/signal_detection.rs`) — now accepts `&MAType` instead of `&str`; SIMD path gated with `#[cfg(feature = "simd")]`, fallback gated with `#[cfg(not(feature = "simd"))]` to eliminate unreachable-code warning.
- **`compute_symbol_score`** (`src/signal_detection.rs`) — uses typed `Robustness` directly from config and returns typed `SignalType` instead of string literals.
- **`estimate_batch_memory_mb`** (`src/aggregation.rs`) — replaced double `bars * 6 * 8` with `bars * 6 * 8` (OHLCV data) + `bars * 3 * 8` (working buffers: roc, r_adjusted, sig_shifted).
- **`process_batch`** (`src/aggregation.rs`) — added `debug_assert!` for config validation on entry; added doc note that callers are responsible for pre-validation.
- **`process_single_symbol` / `aggregate_timeframes`** (`src/aggregation.rs`, `src/multi_tf_voting.rs`) — removed duplicate `tf_strengths` map and reuse timeframe scores for strength inputs.
- **Adaptive threshold** (`src/multi_tf_voting.rs`) — clamped to `max(threshold * weight_ratio, threshold * 0.1)` to prevent over-sensitivity when `weight_ratio → 0`.
- **Buffer pool** (`src/buffer_pool.rs`) — `get_buffer` and `get_buffer_zero` now accept any buffer with `len() >= size` (previously exact match only), using `slice_move(s![..size])` to resize.
- **`src/buffer_pool.rs`** — removed unused `ScopedBuffer` RAII wrapper and related unit test.
- **Static Rayon thread pool** (`src/aggregation.rs`) — replaced per-call `create_custom_thread_pool` with `OnceLock<rayon::ThreadPool>` for warm Lambda reuse.
- **Logging macros** (`src/lib.rs`) — introduced `log_warn!`, `log_error!`, `log_info!` macros that dispatch to `tracing` when the `tracing` feature is enabled, otherwise fall back to `eprintln!`; replaced all raw `eprintln!` calls in `src/signal_detection.rs`, `src/aggregation.rs`, `src/parallelism.rs`.
- **`BehaviorVersion` import** (`lambda/src/main.rs`) — changed from `aws_sdk_sqs::config::BehaviorVersion` to `aws_config::BehaviorVersion`.
- **`src/equity.rs`** — line endings converted from CRLF to LF.
- **`src/ma_calculations.rs`** — gated scalar EMA helper to non-SIMD builds and split `calculate_dema` into SIMD/non-SIMD implementations.
- **`.gitignore`** — added `test_data_120.json` and `test_data_500.json`.

### Fixed

- **C3** (Critical): KAMA SIMD silent data corruption — `else { 0.0 }` sentinel replaced with `continue` to exclude out-of-bounds positions from volatility sum.
- **C1** (Critical): SMA SIMD O(n×length) regression — rewritten with sliding window O(n) algorithm.
- **C2** (Critical): `eprintln!` in core library — replaced with structured `tracing` / `log` macros.
- **H3** (High): Rayon thread pool recreated on every warm Lambda invocation — now statically cached via `OnceLock`.
- **H2** (High): `ma_type: String` allowed arbitrary values with silent EMA fallback — now `MAType` enum.
- **H4** (High): `Robustness` parse silent fallback — now fails fast via `expect`.
- **H5** (High): Buffer pool discarded reusable larger buffers — now correctly slices to requested size.
- **M1** (Medium): Duplicate ROC computation block — extracted into `calculate_roc` helper.
- **M2** (Medium): Double-counted memory estimate (`bars * 6 * 8` twice) — corrected to OHLCV + 3 working buffers.
- **M3** (Medium): `process_batch` had no validation guard for direct callers — added `debug_assert!`.
- **M4** (Medium): Adaptive threshold could reach `0` causing over-sensitivity — clamped to `threshold * 0.1` minimum.
- **M5** (Medium): Wrong `BehaviorVersion` import path in Lambda crate.

### Security

- Resolved **H2**: eliminated silent EMA fallback that could produce silently incorrect trading signals from unknown MA type strings.

## [0.1.0] - 2026-02-11

### Added

- Initial release of ATC Serverless module
- Complete signal detection logic with diflen variations
- Error recovery with per-symbol handling
- Comprehensive test suite
- AWS Lambda deployment ready
- SIMD optimizations for EMA, SMA, WMA (HMA and DEMA added in Phase 1.1)
- Python integration examples and client
- API reference documentation
- Migration guide from Python ATC

### Changed

- None (initial release)

### Fixed

- None (initial release)

### Removed

- None (initial release)
