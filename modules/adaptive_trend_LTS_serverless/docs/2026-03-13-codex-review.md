# Code Review — `adaptive_trend_LTS_serverless`

**Date:** 2026-03-13 (updated 2026-03-14 — production-code-audit pass)
**Reviewer:** Antigravity / Claude Code (codex-review skill + production-code-audit skill)
**Scope:** Full module review — `modules/adaptive_trend_LTS_serverless`
**Overall Rating: 97/100 — Production-ready**

---

## Architecture Overview

```
src/                    # Core Rust library
├── lib.rs             # Public API, types, enums
├── constants.rs       # Algorithm constants + docs
├── ma_calculations.rs # 6 MA types (EMA, WMA, DEMA, LSMA, HMA, KAMA)
├── signal_detection.rs# ATC layer 1/2 signal + diflen
├── equity.rs          # Equity curve with Pine Script parity
├── validation.rs      # OHLCV + config + schema validation
├── buffer_pool.rs     # Thread-local memory pooling
├── aggregation.rs     # Batch processing + error recovery
├── multi_tf_voting.rs # Multi-timeframe signal aggregation
└── parallelism.rs     # Rayon thread pool tuning
lambda/src/
├── main.rs            # Lambda entrypoint (tokio + tracing)
└── handler.rs         # Request handling + CloudWatch EMF metrics
lambda_client.py       # Python boto3 client
scripts/               # Demo, deploy, benchmark scripts
tests/atc_tests.rs     # Comprehensive Rust test suite
```

---

## What Changed Since Previous Review (2026-02-22)

The following issues from the previous review cycle have been fully resolved:

| Count | Previous Issues | Status |
|-------|----------------|--------|
| 32 | Resolved across 2 prior review cycles | ✅ All done |
| M1 | Parity call-site comment in `aggregation.rs` | ✅ Fixed |
| M2 | Pool miss-rate telemetry in `buffer_pool.rs` | ✅ Fixed |
| M3 | `use_signal_strength` in `DEFAULT_ATC_CONFIG` | ✅ Was already present |
| L1 | `parallel_efficiency` always `None` | ✅ Documented intentionally |
| L2 | `sqs.rs` removed without compile break  | ✅ Confirmed compiles |
| L3 | SIMD not benchmarked | ✅ Benchmark exists |

---

## Strengths

### 1. Type Safety — Enums Everywhere
`MAType`, `SignalType`, `Robustness` are all strongly-typed enums with `serde` via `rename_all`. Silent fallbacks on bad input strings are completely eliminated. Tests `test_ma_type_deserialize_invalid_value` and `test_atc_config_deserialize_invalid_robustness_value` enforce this.

### 2. Thread Pool Caching (`aggregation.rs`)
`OnceLock<Mutex<HashMap<usize, Arc<rayon::ThreadPool>>>>` — pools are keyed by thread count, created once per thread configuration, and reused across Lambda warm invocations. Test `test_process_batch_reuses_custom_thread_pool` validates this by comparing pointer addresses.

### 3. Buffer Pool with Miss-Rate Telemetry (`buffer_pool.rs`)
`AtomicU64` counters for checkouts and misses. Sample-window based warning (every 64 checkouts) fires when miss rate ≥ 30%. Larger buffers are properly down-sized via `truncate()` then restored to full capacity on return via `resize(capacity)`. Three unit tests including the regression guard for mixed-size reuse.

### 4. Layer 1 Parity Contract (`signal_detection.rs`)
`compute_symbol_score` has an explicit multi-line parity contract comment explaining the 3-step contract: Layer 1 signal → discretize to `{-1,0,1}` → weighted mean. `discretize_layer1_vote` is unit-tested directly. The call-site comment in `aggregation.rs::process_single_symbol` cross-references `signal_detection.rs`.

### 5. Lambda CloudWatch EMF (`handler.rs`)
Five custom metrics emitted per invocation: `MemoryUsageMB`, `MemoryDeltaMB`, `SymbolsPerSecond`, `ThreadCount`, `ErrorRate`. Dual threshold alert (68% warning / 85% critical). `test_build_cloudwatch_metrics_log_uses_emf_namespace` validates the JSON structure. `emit_cloudwatch_metrics` uses `println!` to ensure CloudWatch picks up the raw EMF line.

### 6. Lambda Defensive Deprecation (`lambda_client.py`)
`ATCLambdaClient.__init__` emits `DeprecationWarning` for all three SQS parameters only when the caller explicitly passes a non-default value — avoiding spam. The stored attributes remain for backward-compat object inspection.

### 7. Single-Symbol Memory Check in Handler
`handler.rs::estimate_batch_memory_mb_rough` uses a 55 KB/symbol heuristic for pre-validation checks before parsing full OHLCV. The accurate per-symbol estimate in `aggregation.rs` is used post-parse. The handler doc comment explicitly notes this distinction.

### 8. Validation Completeness (`validation.rs`)
Full OHLCV checks: timestamp monotonicity, `high >= open && high >= close`, `low <= open && low <= close`, `high >= low`, price bounds `(MIN_PRICE, MAX_PRICE]`, non-finite rejection, volume non-negative. MA weight must have at least one positive entry. Weight sum within ±0.001 tolerance.

---

## New Issues Found

### Low

#### L1 — `aggregation.rs`: unused `MEMORY_WARNING_THRESHOLD_MB` constant confusingly named
Line 68 defines `MEMORY_WARNING_THRESHOLD_MB = 80` as a flat value. The handler uses `MEMORY_WARNING_RATIO = 0.68` of configured Lambda memory. These are two independent memory check mechanisms: the aggregation module checks against a hard 80 MB baseline; the handler checks against Lambda-aware ratio thresholds. This duality is not documented.

**Recommendation:** Add a doc comment to `MEMORY_WARNING_THRESHOLD_MB` clarifying it is the core batch library's own static guard (independent of the Lambda handler's ratio-based thresholds).

#### L2 — `lambda_client.py`: `_mock_invoke` sets detail values to `"MOCK"` (non-standard signal)
Line 267: `"details": {tf: "MOCK" for tf in ...}`. The real payload would contain `"LONG"` / `"SHORT"` / `"NEUTRAL"`. If caller code inspects `details` values and does string comparison in mock mode, it will silently succeed but with unexpected values.

**Recommendation:** Use `"NEUTRAL"` instead of `"MOCK"` to maintain strict protocol compatibility in mock mode.

#### L3 — `parallelism.rs`: `optimal_for_batch_size` reads env vars on every call
`ATC_FORCE_THREADS` and `ATC_FORCE_CHUNK_SIZE` are read via `std::env::var` on every call to `optimal_for_batch_size`. These env vars are typically set once at process startup. On Lambda warm invocations this adds two system calls per batch.

**Recommendation:** Cache the parsed env-var override in a `OnceLock<Option<usize>>` at module level. Low-priority since the cost is minimal, but worth noting for high-frequency usage.

#### L4 — `handler.rs`: `estimate_batch_memory_mb_rough` estimate (55 KB/symbol heuristic) not tested
The rough estimate in `handler.rs` is a hand-wave. There is no test checking this estimate against `aggregation::estimate_batch_memory_mb` for sanity. If the constants diverge in future (e.g., working buffer count changes), the pre-check will silently become stale.

**Recommendation:** Add a unit test that computes `estimate_batch_memory_mb_rough(1)` and `aggregation::estimate_batch_memory_mb(...)` for a 1-symbol, 1-bar dataset and checks the rough estimate is at least as large as the accurate estimate.

---

## Security & Safety

| Check | Status |
|-------|--------|
| No `unsafe` blocks in core logic | ✓ |
| All numeric inputs bounds-checked | ✓ |
| Per-symbol panic isolation (`catch_unwind`) | ✓ |
| No PII in logs | ✓ |
| Schema version validation | ✓ |
| Timestamp monotonicity enforced | ✓ |
| `Box<[f64]>` prevents accidental mutation | ✓ |
| Strong enum types — no silent string fallbacks | ✓ |
| `DeprecationWarning` for deprecated `__init__` params | ✓ |
| Buffer miss-rate monitored | ✓ |
| Thread pool poisoning handled gracefully | ✓ |

---

## Deployment Checklist

- [x] Unit + integration tests in `atc_tests.rs`
- [x] CloudWatch alarms in `template.yaml`
- [x] Python mock mode in `lambda_client.py`
- [x] Memory monitoring (warning 68%, critical 85%)
- [x] SAM deploy script
- [x] `sqs.rs` removal compile verified
- [x] `parallel_efficiency` metric documented as intentionally `None`
- [x] Buffer pool miss-rate telemetry active
- [x] Parity call-site comment present in `aggregation.rs`
- [ ] Load test results not present — still not provided in this review context

---

## Action Items (Priority Order) — 2026-03-13

1. **[LOW]** Add doc comment to `MEMORY_WARNING_THRESHOLD_MB` in `aggregation.rs` clarifying it is an independent static guard (L1) — ✅ Fixed
2. **[LOW]** Replace `"MOCK"` with `"NEUTRAL"` in `_mock_invoke` details dict (L2) — ✅ Fixed
3. **[LOW]** Cache `ATC_FORCE_THREADS` / `ATC_FORCE_CHUNK_SIZE` in `OnceLock` (L3) — ✅ Fixed
4. **[LOW]** Add cross-validation test for `estimate_batch_memory_mb_rough` vs `estimate_batch_memory_mb` (L4) — ✅ Fixed (2026-03-14)

## New Findings — 2026-03-14 Production Audit

| # | File | Issue | Action |
|---|------|-------|--------|
| N1 | `constants.rs:144` | `MAX_BATCH_SIZE` doc referenced SQS 256 KB limit — stale since v0.2.0 synchronous invoke (6 MB limit) | ✅ Updated to Lambda payload limit |
| N2 | `lambda_client.py:180` | Docstring typo `OHLVC` → `OHLCV` | ✅ Fixed |
| N3 | `ma_simd.rs:128` | `calculate_wma_simd` is `O(n*length)` vs scalar `O(n)` sliding window — undocumented complexity trade-off | ✅ Documented in function doc comment |

---

## Verdict

**Module is production-ready at 97/100.** All 4 previous Low issues and all 3 new Low findings are now resolved. The architecture is sound, type-safe, memory-efficient, and well-tested. No blocking changes required before deployment.
