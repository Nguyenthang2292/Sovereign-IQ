# Error Tracking — adaptive_trend_LTS_serverless

> Last updated: 2026-03-12
> Audit scope: `modules/adaptive_trend_LTS_serverless/`
> Sources: `/production-code-audit` (2026-03-12), comprehensive_review_report.md (2026-02-16)

---

## Legend

| Symbol | Meaning |
|--------|---------|
| ✅ FIXED | Verified fixed in code |
| ✅ DONE | Closed after verification/fix completion |
| ❌ OPEN | Not yet addressed |
| ⚠️ PARTIAL | Partially addressed |

---

## CRITICAL — Security

### [✅ FIXED] #1 — Lambda URL with no authentication
- **File**: `scripts/fix_lambda_url.py`
- **Problem**: `AuthType="NONE"` + `Principal="*"` made Lambda endpoint publicly accessible.
- **Fix applied**: File **deleted** entirely. `lambda_client.py` already uses boto3 IAM-signed invocation.

---

### [✅ FIXED] #2 — Overly broad IAM policy (`AmazonSQSFullAccess`)
- **File**: `scripts/deploy_lambda.py` (was line 149)
- **Problem**: Full access to ALL SQS queues in the account.
- **Fix applied**: Replaced with inline policy `ATCSQSSendOnly` — `sqs:SendMessage` on specific queue ARN only.

---

## CRITICAL — Bug Risk

### [✅ FIXED] #3 — Deprecated `np.random.seed()` global state
- **File**: `benchmarks/benchmark_atc_comparison.py` (line 88)
- **Problem**: Global random state causes non-deterministic results in parallel test execution.
- **Fix applied**: Replaced with `rng = np.random.default_rng(seed)`, scoped instance used throughout.

---

## HIGH — Bugs

### [✅ FIXED] #4 — Shallow copy of nested config dict
- **Files**: `lambda_client.py:170`, `scripts/binance_lambda_demo.py:214`, `examples/python_client.py:72`
- **Problem**: `.copy()` shallow-copies `ma_configs` (list of dicts); mutations corrupt shared default.
- **Fix applied**: `copy.deepcopy(DEFAULT_ATC_CONFIG)` in all 3 files.

---

### [✅ FIXED] #5 — `ATCLambdaClient` duplicated in 3 files
- **Files**: `scripts/binance_lambda_demo.py`, `examples/python_client.py`
- **Problem**: Full class copy in each file; bug fixes to main client don't propagate.
- **Fix applied**: Duplicate classes removed; both files now import from `lambda_client.py`.

---

## HIGH — Test Coverage

### [✅ FIXED] #9 — No unit tests for `lambda_client.py`
- **Problem**: Core exported module had zero test coverage.
- **Fix applied**: `tests/adaptive_trend_LTS_serverless/test_lambda_client.py` created with 6 test cases:
  - Mock mode invoke — returns valid result
  - Mock mode batch invoke — all symbols returned
  - `_poll_sqs_for_batch` timeout handling — partial results
  - Lambda invocation error returns error dict
  - Malformed SQS message skipped gracefully
  - `DEFAULT_ATC_CONFIG` deepcopy mutation isolation

---

## MEDIUM — Bugs

### [✅ FIXED] #6 — Division by zero in benchmark display
- **File**: `benchmarks/benchmark_atc_comparison.py` (lines 566, 751, 596)
- **Problem**: `matches / total * 100` crashes with `ZeroDivisionError` on empty result lists.
- **Fix applied**: `(matches / total * 100) if total > 0 else 0.0` at all occurrences.

---

## MEDIUM — Security Hygiene

### [✅ FIXED] #8 — XSS in HTML report generation
- **File**: `scripts/benchmark_tracking.py` (lines 143–149)
- **Problem**: Benchmark names and values interpolated directly into HTML without escaping.
- **Fix applied**: `html.escape()` applied to all user-derived values in the HTML template.

---

## MEDIUM — Code Quality

### [✅ FIXED] #7 — Missing type annotations in `generate_test_data.py`
- **File**: `generate_test_data.py` (lines 7, 47, 163)
- **Problem**: Public functions had no type hints.
- **Fix applied**:
  ```python
  def generate_ohlcv(num_bars: int = 200) -> dict[str, list[float] | list[int]]:
  def generate_symbol_data() -> dict[str, Any]:
  def generate_test_data(num_symbols: int = 120) -> dict[str, Any]:
  ```

---

## LOW — Code Style

### [✅ FIXED] #10 — Legacy `typing.Dict`/`typing.List` imports
- **Files**: `examples/python_client.py`, `scripts/binance_lambda_demo.py`
- **Problem**: Python 3.9+ built-in generics (`list[...]`, `dict[...]`) preferred over `typing.List`/`typing.Dict`.
- **Fix applied**: All occurrences updated to use built-in generic syntax.

---

## DONE — From Comprehensive Review (2026-02-16)

### [✅ DONE] #11 — No input size validation (DoS vector)
- **File**: `lambda/src/handler.rs`
- **Severity**: HIGH — Security / Stability
- **Problem**: No limit on `num_symbols` per batch or `num_bars` per timeframe. Malformed or oversized requests can exhaust Lambda memory (3GB limit).
- **Fix verified/applied**:
  - `src/constants.rs`: `MAX_BATCH_SIZE` + `MAX_BARS_PER_TIMEFRAME`
  - `src/validation.rs`: rejects batches over symbol limit and OHLCV history over bar limit
  - Added test: `test_validate_ohlcv_too_long`

---

### [✅ DONE] #12 — `debug = true` in release profile
- **File**: `Cargo.toml`
- **Severity**: LOW — Build artifact size
- **Problem**: `debug = true` in `[profile.release]` adds ~2–3 MB to Lambda binary.
- **Fix verified**:
  - `[profile.release]` now uses `debug = false`
  - profiling kept in separate `[profile.release-debug]`

---

### [✅ DONE] #13 — SQS failure not handled in Lambda handler
- **File**: `lambda/src/handler.rs` / `src/sqs.rs`
- **Severity**: MEDIUM — Reliability
- **Problem**: If `sqs:SendMessage` fails, results are silently lost. No retry, no DLQ documented.
- **Fix verified**:
  - `lambda/src/sqs.rs`: exponential backoff retry + DLQ fallback
  - `template.yaml`: DLQ resource + redrive policy + `DLQ_URL` environment wiring
  - `handler.rs`: propagates send failure if all attempts fail

---

### [✅ DONE] #14 — No AWS X-Ray distributed tracing
- **File**: `template.yaml`, `lambda/src/handler.rs`
- **Severity**: LOW — Observability
- **Problem**: No end-to-end request tracing; debugging latency spikes requires log trawling.
- **Fix applied**:
  - `template.yaml`: enabled `Tracing: Active`
  - `template.yaml`: added `AWSXrayWriteOnlyAccess` policy

---

### [✅ DONE] #15 — No benchmark regression CI
- **File**: `.github/workflows/` (missing)
- **Severity**: LOW — Quality
- **Problem**: Benchmarks exist but no automated detection of performance regressions between commits.
- **Fix applied**:
  - Added root CI workflow: `.github/workflows/adaptive_trend_benchmark.yml`
  - Runs `benchmark_atc_comparison.py --no-simd`
  - Fails pipeline if average speedup `< 70x`

---

### [✅ DONE] #16 — SIMD not implemented for HMA, DEMA, LSMA, KAMA
- **File**: `src/ma_calculations.rs` / `src/ma_simd.rs`
- **Severity**: LOW — Performance
- **Problem**: Only EMA, SMA, WMA are SIMD-optimized; remaining 4 MA types use scalar path.
- **Note**: LSMA/KAMA may have recursive dependencies that prevent SIMD; this should be documented.
- **Expected gain**: ~10–15% additional speedup if implemented.
- **Fix verified**:
  - `src/ma_simd.rs` now contains SIMD implementations/tests for DEMA, HMA, LSMA, KAMA
  - `src/ma_calculations.rs` routes to SIMD versions when `simd` feature is enabled

---

### [✅ DONE] #17 — Cold start optimization not documented
- **File**: `docs/`, `template.yaml`
- **Severity**: LOW — Performance
- **Problem**: No mention of SnapStart (not applicable to custom runtimes) or Provisioned Concurrency for latency-sensitive use cases.
- **Fix applied**:
  - `README.md`: added `Cold Start Strategy` section
  - Documents SnapStart limitation for `provided.al2`
  - Documents when to use Provisioned Concurrency + cost/latency trade-off

---

## Summary

| Status | Count | Items |
|--------|-------|-------|
| ✅ Done | 17 | #1–#17 |
| ❌ Open | 0 | - |
| **Total** | **17** | |

### Final status

`DONE` — all tracked findings (#1–#17) are now resolved or verified as resolved in code.
