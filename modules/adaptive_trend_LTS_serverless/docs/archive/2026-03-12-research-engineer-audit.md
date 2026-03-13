# Technical Audit: `modules/adaptive_trend_LTS_serverless` (v0.2.0)

**Date**: 2026-03-12
**Scope**: Full static analysis of core algorithm, data pipeline, performance architecture, and correctness defects.
**Auditor**: Research Engineer (via `/research-engineer` skill)
**AWS deployment code**: Analyzed as secondary scope.

## Status Update (Post-Fix Tracking)

Updated on: **2026-03-12**

- **F1 (shared SQS queue race)**: **Resolved in current synchronous path**.
  - `lambda_client.py` now uses direct Lambda `RequestResponse` payload.
- **F2 (CloudWatch metrics non-emission)**: **Resolved in code**.
  - `lambda/src/handler.rs` emits EMF payloads for `ATC/Serverless`.
  - Deployed AWS runtime verification remains pending.
- **F3 (adaptive threshold inversion)**: **Resolved**.
  - Stable configured-weight normalization and fixed threshold logic.
  - Required configured timeframes validated in request validation.
- **H5-bug + H5-test**: **Resolved**.
  - Buffer pool reuses larger buffers via slicing.
  - Tests now verify real reuse behavior.
- **KAMA-O / WMA-O / LSMA-O**: **Resolved**.
  - Sliding-window O(n) implementations added.
  - Parity regression tests added against baseline formulas.
- **DBL-TH**: **Resolved** (continuous aggregation, single final threshold).
- **GROWTH**: **Resolved as legacy-parity contract**.
  - `exp_growth` intentionally preserves Pine/legacy first-bar mapping (`i=0 -> bar_index=1`).
  - Contract is now explicitly documented and covered by regression tests.
- **DECAY**: **Resolved as contract/documentation fix**.
  - `decay` scaling contract (`DECAY_SCALE = 100`) documented in public config docs.
- **III.2 (Rayon vs Lambda vCPU)**: **Resolved in code**.
  - Thread cap now derives from Lambda memory (`ceil(memory_mb / 1769) * 2`), matching operational targets.
  - Large batches (`>500`) now also honor memory-derived cap instead of always falling back to uncapped Rayon default.

---

## I. Algorithmic Complexity Analysis

### 1.1 [DONE] Moving Average Implementations

**SMA** (`ma_calculations.rs:166`): Correctly O(n) via sliding window. This was the C1 fix from v0.1.0.

**WMA** (`ma_calculations.rs:80-88`): **O(n × length)** — not optimal. A rolling-sum formulation reduces this to O(n):

```
WMA(i) = WMA(i-1) + (length * price[i] - sum_window[i]) / denominator
where sum_window[i] = sum_window[i-1] + price[i] - price[i - length]
```

This requires O(length) space for the rolling window but achieves O(n) time. For `length = 28` (default) over `n = 1000` bars: 28,000 operations instead of 1,000. Tolerable at current scale; becomes a bottleneck if `length` scales up toward `MAX_MA_LENGTH = 10,000`.

**LSMA** (`ma_calculations.rs:130-158`): **O(n × length)** — the inner loop recomputes `sum_y` and `sum_xy` from scratch each bar. The x-axis constants are precomputed (lines 124-128 — correct), but y-sums are not incremental. An incremental formulation using running moments reduces to O(n):

```
sum_y[i]  = sum_y[i-1]  + price[i] - price[i - length]
sum_xy[i] = sum_xy[i-1] + (length - 1) * price[i] - sum_y[i-1] + price[i - length]
```

**KAMA** (`ma_calculations.rs:248-251`): **O(n × length)** — the volatility inner loop recomputes `|p[i-j] - p[i-j-1]|` for all `j ∈ [0, length)` per bar. A sliding window over absolute differences reduces to O(n):

```
volatility_window[i] = volatility_window[i-1]
                     + |price[i]   - price[i-1]|
                     - |price[i-length] - price[i-length-1]|
```

**Throughput impact**: At `length = 28`, `n = 1000`, KAMA requires 28,000 inner-loop iterations per call. With 8 diflen variations × 6 MA types × 2 timeframes × 100 symbols = 1,600 KAMA calls per batch → **44.8M inner-loop iterations** from KAMA alone, before any parallelism benefit.

---

### 1.2 [DONE] Buffer Pool Implementation (`buffer_pool.rs`)

The CHANGELOG (v0.2.0) claims H5 was fixed: *"Buffer pool discarded larger buffers → now correctly slices"*. **This fix is absent from the current code.** Lines 15-18:

```rust
for i in 0..pool.len() {
    if pool[i].len() == size {  // exact match ONLY — no slicing
```

No slicing of larger buffers is implemented.

**False-positive test**: `test_buffer_pool_reuse_larger` (line 67-75) does not verify reuse. It returns a 200-element buffer, then requests a 150-element buffer. The pool lookup finds no exact match, allocates a new 150-element array, and the test passes vacuously. The test name is misleading — it proves nothing about the claimed H5 fix.

**Consequence**: For diflen with 8 length variations (e.g., [22, 24, 26, 28, 30, 32, 34, 36] for Medium robustness), each variation requests differently-sized buffers. Since none match exactly, pool reuse across diflen iterations is less frequent than intended.

---

### 1.3 [DONE] Double-Threshold Signal Discretization

In `compute_symbol_score` (`signal_detection.rs:390-401`), two thresholding operations are applied with the same parameter:

```rust
// Step 1: threshold the diflen-average per MA type → discrete {-1, 0, +1}
let discrete_signal = if last_signal > config.threshold { SIGNAL_LONG }
                      else if last_signal < -config.threshold { SIGNAL_SHORT }
                      else { SIGNAL_NEUTRAL };

// Step 2: weighted sum of discrete signals
weighted_score_sum += discrete_signal * combined_weight;

// Step 3: threshold the weighted sum again with the same boundary
let signal_type = if final_score > config.threshold { Long } ...
```

**Mathematical consequence**: Consider 6 MA configs with equal weight; 5 yield `last_signal = 0.29` (just below `threshold = 0.3`), 1 yields `last_signal = 0.35`. After step 1, the 5 signals become 0, one becomes +1. `final_score = 1/6 = 0.167 < 0.3` → **Neutral**. But the raw continuous average would be `(5×0.29 + 0.35)/6 = 0.298 ≈ threshold`. The double threshold artificially suppresses near-boundary signals. A single threshold applied once to the weighted continuous average is mathematically cleaner and more sensitive.

---

## II. Correctness Defects

### II.1 [DONE] F3: Adaptive Threshold Inversion (P0 — Incorrect Trading Signals)

**Location**: `multi_tf_voting.rs:40`

```rust
let adaptive_threshold = (config.threshold * weight_ratio).max(config.threshold * 0.1);
```

**Concrete failure case**: Config has `weights = {"1h": 0.6, "4h": 0.4}`. Only `"1h"` is present in the payload.

| Variable | Value |
|----------|-------|
| `active_weight` | 0.6 |
| `total_config_weight` | 1.0 |
| `weight_ratio` | 0.6 |
| `adaptive_threshold` | `max(0.3 × 0.6, 0.3 × 0.1)` = **0.18** |
| `"1h"` normalized weight | 0.6/0.6 = **1.0** |

A "1h" score of 0.20 → `total_weighted_score = 0.20 > 0.18` → **LONG**.
With both timeframes, the same "1h" score contributes only `0.6 × 0.20 = 0.12 < 0.30` → **Neutral**.

A partial payload that *should* be weaker produces a *stronger* classification. This violates monotonicity: adding more information (completing the timeframe set) should not reverse a signal. The `.max(config.threshold * 0.1)` floor only prevents the threshold from dropping below 3% of original — offering no meaningful protection.

**Required fix**: Either:
- **Option A (fail-closed)**: Reject symbols with missing configured timeframes in `validate_batch_request`.
- **Option B (stable threshold)**: Remove `adaptive_threshold` entirely; use `config.threshold` unconditionally regardless of which timeframes are present.

The current "forgive and lower threshold" behavior is the worst possible design choice: it appears to handle partial data gracefully while producing results with opposite reliability properties.

---

### II.2 [DONE] F1: Shared SQS Queue Race Condition (P0 — Message Loss + DLQ Contamination)

**Location**: `lambda/src/sqs.rs`, `lambda_client.py`, `template.yaml`

Lambda publishes all `ScanResult` messages to a single shared SQS queue. The Python client polls this queue and temporarily sets `VisibilityTimeout=0` for non-matching messages, re-exposing them.

**Deterministic failure path under concurrent load**:
1. Client A receives Client B's result → sets visibility=0 (re-exposes)
2. Client B receives Client A's result → sets visibility=0 (re-exposes)
3. Each message has now been received twice. With `maxReceiveCount = 3` in the DLQ redrive policy, one more misrouted receive sends a valid result to the DLQ permanently.

This is both a **liveness failure** (messages permanently lost to DLQ) and a **safety failure** (wrong client can consume results if `batch_id` check is bypassed or collides).

**Option A is unambiguously correct** for synchronous invocations. Lambda already returns the result synchronously in the HTTP response payload (up to 6MB), which exceeds any plausible `ScanResult` JSON for ≤1000 symbols. SQS adds minimum 1-second poll latency, per-message cost, and the above failure modes, with zero benefit for request/response patterns. SQS is appropriate for asynchronous fan-out (`InvocationType::Event`), not for return values.

---

### II.3 [DONE as legacy-parity contract] `exp_growth` First-Bar Mapping (`equity.rs`)

Current code intentionally keeps:

```rust
let bar_index = if i == 0 { 1.0 } else { i as f64 };
```

This is now treated as a **compatibility contract**, not an active defect:
- Required for Pine/legacy parity (`bar_index == 0 ? 1 : bar_index`)
- Explicitly documented in `src/equity.rs` with a "do not simplify" warning
- Protected by regression tests:
  - `test_exp_growth_matches_legacy_first_bar_contract`
  - `test_exp_growth_cutout_prefix_remains_one`

So the previous recommendation "make growth strictly monotonic from index 0" is superseded by parity requirements.

---

### II.4 [DONE] Decay Parameter Hidden Double-Scaling (`signal_detection.rs:371-372`)

```rust
decay_scaled: config.decay / DECAY_SCALE,  // 0.03 / 100 = 0.0003
// ...
STARTING_EQUITY - params.decay_scaled       // decay_multiplier = 0.9997
```

The user-facing `decay = 0.03` is labeled "equity decay rate" but the actual per-bar decay applied is **0.03%**, not 3%. With `n = 1000` bars: `0.9997^1000 ≈ 0.741`. This is reasonable behavior, but the `DECAY_SCALE = 100` divisor is undocumented in the public API. A user setting `decay = 1.0` expecting "1% decay per bar" receives 0.01% actual decay. The parameter contract should be documented explicitly, or the scale factor removed and the expected input range changed.

---

## III. Architecture Assessment

### III.1 SIMD Feature Stability Risk

The `simd` feature requires `nightly-2026-02-01`. `std::simd` is an unstable API subject to breaking changes between nightly versions. Bug C3 from v0.1.0 was a **silent data corruption** in the SIMD KAMA path caused by incorrect sentinel handling. Any SIMD path for stateful algorithms (KAMA, DEMA, EMA-based) must be verified with property-based tests that compare SIMD vs. scalar output element-by-element, not just final equity values.

The scalar default path is the correct production choice for Lambda deployments — stability outweighs throughput optimization at `n ≤ 1000`.

### III.2 [DONE] Rayon Thread Count vs. Lambda vCPU Allocation

Lambda assigns CPU proportionally to memory. The previous static tiers could oversubscribe low-memory runtimes.

**Current implementation status (DONE):**
- `parallelism.rs` now computes a Lambda memory thread cap as:

```
max_threads = ceil(memory_mb / 1769) * 2
```

- This yields the intended operational targets:
  - 1769MB -> 2 threads
  - 3008MB -> 4 threads
- For large batches where default tiering previously returned `None`, the memory cap is now applied directly.

### III.3 [DONE in code] F2: CloudWatch Alarms Reference Non-Emitted Metrics (P3 — No Operational Alerting)

`template.yaml` defines alarms for `ATC/Serverless` metrics (`MemoryUsageMB`, `SymbolsPerSecond`, `ErrorRate`, etc.). `lambda/src/handler.rs` logs these fields as plain JSON strings without emitting Embedded Metric Format (EMF) payloads.

**Result**: All alarms defined in `template.yaml` are permanently in `INSUFFICIENT_DATA` state. The system has no automated alerting despite the appearance of having it.

**Fix**: Emit EMF-formatted JSON to stdout. CloudWatch Logs automatically extracts metrics from EMF payloads with zero additional API calls:

```json
{
  "_aws": {
    "Timestamp": 1234567890,
    "CloudWatchMetrics": [{
      "Namespace": "ATC/Serverless",
      "Dimensions": [["FunctionName"]],
      "Metrics": [{"Name": "SymbolsPerSecond", "Unit": "Count/Second"}]
    }]
  },
  "FunctionName": "atc-serverless",
  "SymbolsPerSecond": 31.4
}
```

---

## IV. Prioritized Defect Table

| ID | Priority | Location | Defect | Impact |
|----|----------|----------|--------|--------|
| F3 | **P0** | `multi_tf_voting.rs:40` | Adaptive threshold lowers when timeframes are missing — partial payload produces stronger signal than complete payload | Incorrect trading signals |
| F1 | **P0** | `sqs.rs`, `lambda_client.py` | Shared SQS queue race: concurrent callers can redirect each other's results to DLQ | Message loss, false DLQ contamination |
| H5-bug | **P1** | `buffer_pool.rs:15-18` | CHANGELOG claims larger-buffer slicing; code does exact-match only | Higher allocation rate than intended |
| H5-test | **P1** | `buffer_pool_test.rs:67-75` | `test_buffer_pool_reuse_larger` is vacuous — passes without verifying reuse | False confidence in claimed fix |
| KAMA-O | **P2** | `ma_calculations.rs:248-251` | KAMA volatility loop O(n×length) — not O(n) | Throughput bottleneck at large `length` |
| WMA-O | **P2** | `ma_calculations.rs:80-88` | WMA inner loop O(n×length) | Same |
| LSMA-O | **P2** | `ma_calculations.rs:130-158` | LSMA sum recomputed per bar O(n×length) | Same |
| DBL-TH | **P3** | `signal_detection.rs:390-401` | Double thresholding with same parameter suppresses near-boundary signals | Reduced signal sensitivity |
| GROWTH | **P3** | `equity.rs` | First-bar mapping intentionally follows Pine/legacy parity (`i=0 -> bar_index=1`) and is regression-tested | Closed by design |
| DECAY | **P3** | `signal_detection.rs:371` | `decay` parameter has hidden 100× scale factor undocumented in API | Interface confusion |
| F2 | **P3** | `template.yaml`, `handler.rs` | Alarms reference metrics that are never emitted | No operational alerting |

---

## V. Recommended Execution Order

1. **F3 (P0)**: Fix `multi_tf_voting.rs` threshold logic. Decide: reject partial payloads, or fix threshold to be invariant. Add regression test proving a partial payload cannot produce a stronger signal than the corresponding complete payload.

2. **F1 (P0)**: Remove SQS from synchronous result path. Switch `lambda_client.py` to direct Lambda invocation (`InvocationType::RequestResponse`). Retain SQS only for async fire-and-forget triggers.

3. **H5-bug + H5-test (P1)**: Either implement actual larger-buffer slicing in `get_buffer`, or revert the CHANGELOG entry for H5 and fix the test to accurately reflect behavior.

4. **F2 (P3)**: Add EMF metric emission to `handler.rs`. Update `docs/aws/cloudwatch_monitoring.md` to reflect actual metric emission mechanism.

5. **Complexity (P2)**: Apply O(n) incremental updates to KAMA, WMA, LSMA only if profiling confirms throughput bottleneck at production batch sizes. Do not optimize prematurely.

---

## Appendix: Key Constants Reference

| Constant | Value | Effective Meaning |
|----------|-------|-------------------|
| `DEFAULT_THRESHOLD` | 0.3 | Signal boundary for Long/Short classification |
| `DEFAULT_LAMBDA_PARAM` | 0.02 | Growth exponent (actual: 0.02/1000 = 0.00002 per bar) |
| `DEFAULT_DECAY` | 0.03 | Decay rate (actual: 0.03/100 = 0.0003 per bar) |
| `LAMBDA_SCALE` | 1000 | Hidden divisor for lambda parameter |
| `DECAY_SCALE` | 100 | Hidden divisor for decay parameter |
| `MIN_DATA_LENGTH` | 10 | Minimum bars for valid calculation |
| `MAX_BARS_PER_TIMEFRAME` | 1000 | Upper bound on input data size |
| `WEIGHT_SUM_TOLERANCE` | 0.001 | Tolerance for timeframe weight normalization |

---

## VI. Execution Status Matrix

| ID | Status | Notes |
|----|--------|-------|
| F1 | Done (sync path) | Direct `RequestResponse` result flow prevents shared-queue polling race in synchronous client mode. |
| F2 | Done in code / Pending in AWS runtime | EMF emission implemented; production CloudWatch verification remains operational follow-up. |
| F3 | Done | Threshold behavior stabilized and missing configured timeframes handled by explicit validation policy. |
| H5-bug | Done | Buffer pool now slices reusable larger buffers. |
| H5-test | Done | Reuse test now validates actual allocation reuse, not just output length. |
| KAMA-O | Done | O(n) volatility rolling window applied. |
| WMA-O | Done | O(n) rolling weighted sum applied. |
| LSMA-O | Done | O(n) rolling moment updates applied. |
| DBL-TH | Done | Removed double-threshold suppression pattern. |
| GROWTH | Done (legacy parity contract) | First-bar mapping intentionally preserved for Pine/legacy compatibility; behavior documented and regression-tested. |
| DECAY | Done (contract doc) | Public parameter semantics for scaled decay are now documented. |
| III.2 (Rayon vs Lambda vCPU) | Done | Lambda memory-based thread cap implemented and applied to large-batch path. |
