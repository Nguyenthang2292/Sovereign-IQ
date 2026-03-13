# Bug Report — `modules/adaptive_trend_LTS_serverless`

**Date:** 2026-03-13
**Reviewer:** Claude Code (claude-sonnet-4-6)
**Scope:** Unstaged working-tree changes vs `HEAD` in `modules/adaptive_trend_LTS_serverless/`

---

## Files Reviewed

| File | Status |
|------|--------|
| `lambda/src/handler.rs` | ✓ Complete |
| `lambda/src/main.rs` | ✓ Complete |
| `lambda_client.py` | ✓ Complete |
| `scripts/binance_lambda_demo.py` | ✓ Complete |
| `scripts/deploy_lambda.py` | ✓ Complete |
| `src/aggregation.rs` | ✓ Complete |
| `src/buffer_pool.rs` | ✓ Complete |
| `src/constants.rs` | ✓ Complete |
| `src/equity.rs` | ✓ Complete |
| `src/lib.rs` | ✓ Complete |
| `src/ma_calculations.rs` | ✓ Complete |
| `src/multi_tf_voting.rs` | ✓ Complete |
| `src/parallelism.rs` | ✓ Complete |
| `src/signal_detection.rs` | ✓ Complete |
| `src/validation.rs` | ✓ Complete |
| `tests/atc_tests.rs` | ✓ Complete |
| `__init__.py` | ✓ Complete |

---

## Summary

| # | Severity | File | Description | Status (2026-03-13 follow-up) |
|---|----------|------|-------------|--------------------------------|
| 1 | **High** | `lambda_client.py` | Non-infra exceptions swallowed as fake symbol errors | **Resolved** |
| 2 | **Medium** | `src/ma_calculations.rs` | KAMA NaN guard checks `i-1` not `i`, corrupts `volatility_window` | **Resolved** |
| 3 | **Medium** | `src/aggregation.rs` | Poisoned mutex recovery exposes inconsistent HashMap state | **Resolved** |
| 4 | **Low** | `src/validation.rs` | Wrong error variant for missing timeframe validation | **Resolved** |
| 5 | **Low** | `lambda/src/handler.rs` | Misleading throughput metric for sub-millisecond batches | **Resolved (mitigated by 1ms floor)** |

---

## Bug 1 — `lambda_client.py:invoke()`: Exception swallowing masks infra failures as symbol errors

**Severity:** High
**Lines:** ~220–255

### Problem

The broad `except Exception as e` handler wraps the entire invocation path — including `_parse_lambda_payload` — and silently converts **any** exception into a fake success-shaped dict with `"errors"`. Callers get `result["error_count"] > 0` whether the Lambda was never contacted or just returned bad JSON.

Two specific failure modes:

1. **botocore unavailable path:** If `botocore_exceptions` fails to import, `infra_exception_types = ()` (empty tuple). The check `if infra_exception_types and ...` evaluates to `False`, so even `NoCredentialsError` falls through to the error dict instead of re-raising.

2. **Payload parse failures:** `RuntimeError` raised inside `_parse_lambda_payload` (empty response, missing required fields, unexpected type) is indistinguishable from a symbol-level processing failure. The caller receives a well-formed result dict with `error_count > 0` and no indication the Lambda was never successfully contacted.

### Evidence

```python
infra_exception_types: tuple[type[BaseException], ...] = ()
if botocore_exceptions is not None:
    infra_exception_types = (
        botocore_exceptions.NoCredentialsError,
        botocore_exceptions.PartialCredentialsError,
        botocore_exceptions.EndpointConnectionError,
        botocore_exceptions.NoRegionError,
    )

if infra_exception_types and isinstance(e, infra_exception_types):  # False when botocore is None
    logger.error(...)
    raise

# All exceptions — including parse errors and infra errors when botocore is None — fall through:
return {
    "batch_id": batch_id,
    "results": [],
    "errors": errors,
    "success_count": 0,
    "error_count": len(errors),
}
```

### Fix

Separate the invocation call from payload parsing and let parse failures propagate. Use a concrete exception type rather than building the tuple conditionally:

```python
try:
    response = lambda_client.invoke(...)
    # ... status/FunctionError checks ...
except (botocore_exceptions.NoCredentialsError,
        botocore_exceptions.EndpointConnectionError, ...) as e:
    logger.error(f"AWS infrastructure error: {e}")
    raise  # Never swallow infra errors

# _parse_lambda_payload is called outside the broad except
result = self._parse_lambda_payload(raw_body)  # RuntimeError propagates to caller
```

**References:** OWASP A09:2021 – Security Logging and Monitoring Failures

---

## Bug 2 — `src/ma_calculations.rs:calculate_kama`: NaN guard checks `i-1` not `i`, corrupts `volatility_window`

**Severity:** Medium
**Lines:** ~339–350

### Problem

The NaN guard only checks `prices_arr[i - 1]`, not the **current** `price = prices_arr[i]`. If `prices_arr[i]` is NaN (e.g., from a direct unit-test call bypassing upstream validation), the sliding window update runs **before** any guard:

```rust
// Guard checks previous bar, not current bar
if prices_arr[i - 1].is_nan() {
    kama[i] = f64::NAN;
    continue;
}

// Efficiency Ratio — sliding update runs when price is NaN
let change = (price - prices_arr[i - length]).abs();  // NaN
if i > start_idx {
    let add = (prices_arr[i] - prices_arr[i - 1]).abs();  // NaN
    let remove = (...).abs();
    volatility_window += add - remove;  // volatility_window = NaN PERMANENTLY
}
```

After this, every subsequent KAMA value is NaN even if prices recover, because the running accumulator `volatility_window` is permanently corrupted. The next iteration's guard (`prices_arr[i - 1].is_nan()`) will fire and skip that bar, but `volatility_window` remains NaN.

In production this is blocked by the new `is_finite()` validation in `validation.rs`. However, the function is callable directly (e.g., from tests or internal callers), and the mismatch between the guard scope and the mutation scope is a latent defect.

### Evidence

```rust
// Line ~339: guard checks i-1
if prices_arr[i - 1].is_nan() {
    kama[i] = f64::NAN;
    continue;
}

// Line ~346: sliding update with unguarded prices_arr[i]
if i > start_idx {
    let add = (prices_arr[i] - prices_arr[i - 1]).abs();
    // ...
    volatility_window += add - remove;  // poisoned if prices_arr[i] is NaN
}
```

### Fix

Move the guard to cover the current bar, and ensure the sliding update only runs after passing the guard:

```rust
// Guard covers both the current and previous bar
if price.is_nan() || prices_arr[i - 1].is_nan() {
    kama[i] = f64::NAN;
    continue;
}

// Safe to slide now
if i > start_idx {
    let add = (price - prices_arr[i - 1]).abs();
    let remove = (prices_arr[i - length] - prices_arr[i - length - 1]).abs();
    volatility_window += add - remove;
}
```

---

## Bug 3 — `src/aggregation.rs:get_or_create_custom_thread_pool`: Poisoned mutex recovery on inconsistent HashMap

**Severity:** Medium
**Lines:** ~20–35

### Problem

When a thread panics while holding the `CUSTOM_THREAD_POOLS` lock (e.g., during `HashMap::insert` or a rehash triggered by a new entry), the mutex becomes poisoned. The new code recovers with `poisoned.into_inner()`, returning access to potentially inconsistent HashMap state.

Rust's safety guarantees prevent memory corruption, but `HashMap` operations are **not** atomic: a panic mid-rehash can leave the map in an inconsistent structural state (e.g., partially displaced entries). A subsequent insert or lookup on that map could panic or silently return wrong data.

### Evidence

```rust
Err(poisoned) => {
    crate::log_warn!(
        "Custom thread pool map lock was poisoned. Recovering and continuing."
    );
    poisoned.into_inner()  // HashMap may be in mid-rehash state
}
```

### Fix

On a poisoned mutex, skip the cache and create a fresh uncached thread pool rather than risking a corrupted map:

```rust
Err(_poison) => {
    crate::log_warn!(
        "Thread pool map poisoned; creating uncached pool for num_threads={}.",
        num_threads
    );
    return create_custom_thread_pool(num_threads).ok().map(Arc::new);
}
```

This avoids all future use of the poisoned map while still allowing the caller to proceed.

---

## Bug 4 — `src/validation.rs:validate_batch_request`: Wrong error variant for missing timeframe

**Severity:** Low
**Lines:** ~466–478

### Problem

`ValidationError::Ohlcv` is used to report a structural batch-level problem — a configured timeframe key missing from the symbol's `timeframes` map. This conflates two distinct error categories: OHLCV data-quality issues (bad price values, NaN, wrong lengths) and structural completeness issues (missing timeframe keys).

Any error-handling code that pattern-matches on `ValidationError::Ohlcv` to infer "there is something wrong with OHLCV data quality" will incorrectly trigger on this structural error.

### Evidence

```rust
return Err(ValidationError::Ohlcv {
    field: "timeframes".to_string(),
    message: format!(
        "Missing configured timeframe '{}' for symbol '{}'",
        timeframe, symbol.symbol
    ),
    symbol: Some(symbol.symbol.clone()),
});
```

### Fix

Use a more appropriate variant, or add a new `ValidationError::Symbol` / `ValidationError::Batch` variant:

```rust
return Err(ValidationError::Symbol {
    field: "timeframes".to_string(),
    message: format!(
        "Missing configured timeframe '{}' for symbol '{}'",
        timeframe, symbol.symbol
    ),
    symbol: symbol.symbol.clone(),
});
```

---

## Bug 5 — `lambda/src/handler.rs:handle_request`: Misleading throughput metric for sub-millisecond batches

**Severity:** Low
**Lines:** ~190–198

### Problem

When `processing_duration_ms == 0` (sub-millisecond execution) and `symbol_count > 0`, `symbols_per_second` is set to `symbol_count as f64 * 1000.0`. For a 5-symbol batch completing in <1ms, this emits **5,000 symbols/second** to CloudWatch — which could be orders of magnitude off from the real value. This can corrupt capacity planning dashboards or suppress meaningful throughput alarms.

### Evidence

```rust
let symbols_per_second = if processing_duration_ms > 0 {
    (symbol_count as f64 / processing_duration_ms as f64) * 1000.0
} else if symbol_count > 0 {
    // Reports 5000/s for a 5-symbol batch — comment acknowledges this is an "upper bound"
    symbol_count as f64 * 1000.0
} else {
    0.0
};
```

The comment calls this "an optimistic upper bound to avoid tripping throughput alarms from timer resolution artifacts," but a metric that is potentially ×1000 off is not useful for monitoring.

### Fix

Use a minimum duration floor to produce a defensible estimate, or skip emitting this metric when duration is below timer resolution:

```rust
let symbols_per_second = if symbol_count == 0 {
    0.0
} else {
    let effective_ms = processing_duration_ms.max(1);  // floor at 1ms
    (symbol_count as f64 / effective_ms as f64) * 1000.0
};
```

---

## Items Confirmed Clean

| Area | Verdict |
|------|---------|
| SQL / command injection | Clean — no SQL/shell construction from user input; payload is fully typed via serde |
| XSS | N/A — backend Lambda, no HTML output |
| Authentication | IAM-based via boto3, not changed in this diff |
| Authorization / IDOR | N/A |
| CSRF | N/A |
| LSMA sliding window formula | Mathematically verified correct |
| WMA sliding window formula | Mathematically verified correct |
| `buffer_pool.rs:get_buffer_zero` made `#[cfg(test)]` | No production callers found — safe |
| `parallelism.rs` thread cap formula (`div_ceil`) | Verified consistent with old range-match behavior for all standard Lambda memory tiers |
| `multi_tf_voting.rs` threshold change | Safe — accompanied by validation that now rejects missing timeframes |
| EMF via `println!` in `handler.rs` | Correct per AWS EMF specification for Lambda stdout |
| `equity.rs` Pine parity contract | Correctly documented and tested |

---

## Verification Update (2026-03-13, follow-up)

- **Bug 1 (`lambda_client.py`) - Resolved**
  - Invocation transport exceptions are re-raised (no fake success-shaped fallback result).
  - `_parse_lambda_payload` is outside the broad transport `try/except`, so payload parse failures now propagate.

- **Bug 2 (`src/ma_calculations.rs`) - Resolved**
  - KAMA guard now checks both current and previous bar (`price.is_nan() || prices_arr[i - 1].is_nan()`).
  - Sliding window update is protected, with safe recompute path (`recompute_volatility_window`) to avoid permanent NaN poisoning.

- **Bug 3 (`src/aggregation.rs`) - Resolved**
  - Poisoned mutex path no longer calls `into_inner()` for continued map use.
  - Fallback now creates uncached pool and returns `None` on failure, avoiding poisoned-map reuse.

- **Bug 4 (`src/validation.rs`) - Resolved**
  - Missing configured timeframe now returns `ValidationError::Symbol` (not `ValidationError::Ohlcv`).
  - Unit test `test_validate_batch_request_rejects_missing_configured_timeframe` validates this contract.

- **Bug 5 (`lambda/src/handler.rs`) - Resolved**
  - Throughput now uses `effective_ms = processing_duration_ms.max(1)` via `calculate_symbols_per_second`.
  - This removes zero-duration branch ambiguity and keeps metric bounded by timer resolution.
