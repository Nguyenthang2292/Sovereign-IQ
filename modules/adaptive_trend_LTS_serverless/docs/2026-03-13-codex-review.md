# Code Review — `adaptive_trend_LTS_serverless`

**Date:** 2026-03-13
**Reviewer:** Claude Code (codex-review skill)
**Scope:** Full module review — `modules/adaptive_trend_LTS_serverless`
**Overall Rating: 92/100 — Production-ready**

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

## Strengths

### 1. MA Implementations (`ma_calculations.rs`)
All 6 types (EMA, WMA, DEMA, LSMA, HMA, KAMA) use sliding-window O(n) algorithms. KAMA has explicit recovery when `prev_kama` goes non-finite — reseeds from the prior price. All handle NaN correctly.

### 2. Memory Management (`buffer_pool.rs`)
Thread-local `RefCell<Vec<Vec<f64>>>` pool. On return of a larger-than-requested buffer, capacity is restored via `raw.resize(target_len, NaN)` before reinsertion — prevents capacity degradation under mixed-size workloads. `MAX_POOL_BUFFERS = 16` keeps memory bounded.

### 3. Error Isolation (`aggregation.rs`)
Each symbol wrapped in `catch_unwind` — a panic in one symbol does not fail the batch. Returns `SymbolError` per failed symbol while successfully processed symbols are returned normally. Per-symbol wall-clock timing is measured and logged.

### 4. Lambda Observability (`handler.rs`)
CloudWatch EMF metrics emitted per invocation: `MemoryUsageMB`, `SymbolsPerSecond`, `ErrorRate`. Memory thresholds: warning at 68%, critical at 85% of allocated Lambda memory. SAM template (`template.yaml`) wires SNS alarms for all four key signals.

### 5. Parallelism Tuning (`parallelism.rs`)
Tiered defaults by batch size (2→4→6→8 threads for 0→10→50→100→500 symbols). Lambda-aware: reads `AWS_LAMBDA_FUNCTION_MEMORY_SIZE` and maps `memory / 1769 * 2` vCPU-equivalent threads. Override via `ATC_FORCE_THREADS`.

### 6. Validation (`validation.rs`)
Four-layer validation: OHLCV bounds/monotonicity, config (threshold ∈ [0,1], MA lengths, weights sum to 1.0 ± 0.001), symbol-level, schema version. Returns structured `ValidationError` variants with field names and symbol context.

### 7. Pine Script Parity (`equity.rs`)
`exp_growth` preserves `bar_index = 1` for `i=0` — intentional double-count of the first bar to match Pine Script semantics. Documented explicitly with a "DO NOT SIMPLIFY without parity validation" contract.

---

## Issues Found

### Medium

#### M1 — `signal_detection.rs`: discretization contract undocumented at call-site
The Layer 1 output is discretized to `{-1, 0, 1}` at aggregation time using threshold. The contract comment lives in `signal_detection.rs` but not in `aggregation.rs` where `compute_layer1_signal` is called. A new developer could unknowingly change the aggregation logic and break parity.

**Recommendation:** Add a `// PARITY: see signal_detection.rs L1 contract` comment at the aggregation call-site.

#### M2 — `buffer_pool.rs`: pool size not validated against batch size
`MAX_POOL_BUFFERS = 16` is a fixed cap, but nothing warns if `num_threads * buffers_per_thread > 16`. Under high parallelism (8 threads), each thread could request multiple buffers simultaneously, causing pool misses and fresh allocations silently.

**Recommendation:** Add a debug-mode assertion or log when pool miss rate exceeds a threshold.

#### M3 — `multi_tf_voting.rs`: `use_signal_strength` default is undocumented in Python client
`lambda_client.py` `DEFAULT_ATC_CONFIG` does not include `use_signal_strength`. The Rust default is `false`, but this is not stated in the Python default config or `__init__.py` docstring. Users may not know they can enable it.

**Recommendation:** Add `"use_signal_strength": false` explicitly to `DEFAULT_ATC_CONFIG` with a comment.

### Low

#### L1 — `parallelism.rs`: `parallel_efficiency` always `None`
`ParallelMetrics.parallel_efficiency` is a placeholder `Option<f64>` always set to `None`. It's included in metrics logging but never computed. Either compute it or remove the field to avoid confusion.

#### L2 — `handler.rs`: `sqs.rs` deleted but imports may linger
Git status shows `lambda/src/sqs.rs` was deleted. Verify no `mod sqs` or `use crate::sqs` references remain in `main.rs` or `handler.rs` — otherwise it will fail to compile.

#### L3 — SIMD feature gate not benchmarked
`src/ma_simd.rs` exists behind `#[cfg(feature = "simd")]` (nightly only). No benchmark comparing SIMD vs non-SIMD paths. Either add a benchmark or document it as experimental.

---

## Security & Safety

| Check | Status |
|-------|--------|
| No `unsafe` blocks in core logic | ✓ |
| All numeric inputs bounds-checked | ✓ |
| Per-symbol panic isolation | ✓ |
| No PII in logs | ✓ |
| Schema version validation | ✓ |
| Timestamp monotonicity enforced | ✓ |
| `Box<[f64]>` prevents accidental mutation | ✓ |

---

## Deployment Checklist

- [x] Unit + integration tests in `atc_tests.rs`
- [x] CloudWatch alarms in `template.yaml`
- [x] Python mock mode in `lambda_client.py`
- [x] Memory monitoring (warning 68%, critical 85%)
- [x] SAM deploy script
- [x] **L2: Confirm `sqs.rs` removal doesn't break compile** (verified via `cargo build -p atc_lambda`)
- [ ] Load test results not present
- [x] `parallel_efficiency` metric is a no-op (known intentional)

---

## Action Items (Priority Order)

1. [DONE] Verify `sqs.rs` removal compiles (verified via `cargo build -p atc_lambda`) (L2)
2. [DONE] Add call-site comment in `aggregation.rs` referencing Layer 1 discretization contract (M1)
3. [OBSOLETE] Add `use_signal_strength` to Python default config (already present) (M3)
4. [KNOWN] Remove or implement `parallel_efficiency` in `ParallelMetrics` (baseline-dependent placeholder) (L1)
5. [DONE] Add pool miss logging in `buffer_pool.rs` for high-parallelism scenarios (M2)
---

## Verification Update (2026-03-13, follow-up)

- `M1` - **Partially valid (intent correct, location outdated)**  
  `compute_layer1_signal` is not called from `aggregation.rs`; parity contract now lives in `signal_detection.rs` (`compute_symbol_score`).  
  **Fix applied:** added explicit parity call-site comment in `src/aggregation.rs`.

- `M2` - **Valid (observability gap)**  
  Pool miss behavior was not surfaced.  
  **Fix applied:** added miss-rate telemetry warning in `src/buffer_pool.rs` (sample-window based warning when miss rate is high).

- `M3` - **Invalid/outdated**  
  `lambda_client.py` already contains `use_signal_strength` in `DEFAULT_ATC_CONFIG`.

- `L1` - **Known/intentional (not changed)**  
  `parallel_efficiency` is intentionally `None` and documented as requiring a sequential baseline.

- `L2` - **Invalid/outdated**  
  Re-verified by compile: `cargo build -p atc_lambda` passes; no lingering `sqs` module compile break.

- `L3` - **Invalid/outdated**  
  SIMD benchmarking exists in `benchmarks/benchmark_atc_comparison.py` (including `--simd` flow and SIMD vs scalar reporting).

---

## Completion Marking (2026-03-13, consolidated)

### Issue Status

| Issue | Marking |
|------|---------|
| `M1` | **DONE** |
| `M2` | **DONE** |
| `M3` | **OBSOLETE / NOT APPLICABLE** |
| `L1` | **KNOWN INTENTIONAL** |
| `L2` | **OBSOLETE / NOT APPLICABLE** |
| `L3` | **OBSOLETE / NOT APPLICABLE** |

### Deployment Checklist Status

- [x] L2 compile verification completed (`cargo build -p atc_lambda` passes)
- [x] `parallel_efficiency` no-op explicitly treated as known intentional
- [ ] Load test results: still not provided in this review context

### Action Items Status

1. [x] Verify `sqs.rs` removal compiles
2. [x] Add parity call-site comment in `aggregation.rs`
3. [x] M3 identified as obsolete (no code change needed)
4. [x] L1 identified as known intentional (no code change needed)
5. [x] Add pool miss logging in `buffer_pool.rs`

