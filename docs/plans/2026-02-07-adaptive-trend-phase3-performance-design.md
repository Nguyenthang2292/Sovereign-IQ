# Adaptive Trend Phase 3 Performance – Design

**Date:** 2026-02-07  
**Status:** Approved  
**Scope:** Cache monitoring, parallel data fetching, async API for incremental updates — verification + small features, auto-integrated into existing code.

---

## 1. Scope and integration strategy

**Goal:** Close the three Phase 3 performance items by (1) verifying existing behaviour where it already satisfies the plan, and (2) adding only what’s missing, wired into current modules (no separate one-off scripts or duplicate entry points).

| Item | Verify | Add & integrate |
|------|--------|------------------|
| **Cache monitoring** | `cache_manager` exposes hit/miss and hit rate; script and HTTP metrics endpoint exist | Optional periodic hit-rate logging from `CacheManager` (e.g. every N requests or time interval), so production runs get hit rate in logs without extra scripts |
| **Parallel data fetching** | Batch path is faster than strict sequential for 100 symbols | Ensure main batch/scan path uses parallel execution by default or is clearly recommended; ensure data fetching is parallelized inside workers; add/confirm 100-symbol benchmark |
| **Async API for incremental** | `asyncio` works with incremental update; WebSocket demo exists | Single async API surface (re-export) and doc update so the async incremental entry point is discoverable and documented |

**Principles:** Reuse existing `CacheManager`, `CacheMonitor`, `CacheMetricsService`, `scan_all_symbols`, `AsyncIncrementalATC`, and benchmarks. Add only hooks or small options inside these; no new top-level apps or duplicate logic.

---

## 2. Cache monitoring

**Verify (no code change):**
- `CacheManager.get_stats()` / `get_detailed_metrics()` provide hit/miss and hit rate.
- Script: `cache_monitor.py`; HTTP: `cache_metrics_api.py`. Checklist: cache_manager can log hit rate via `log_stats()`; script and metrics endpoint exist.

**Add and integrate:**

1. **Periodic hit-rate logging in CacheManager**
   - In `cache_manager.py`: optional periodic logging so long-running runs get hit rate in logs without running the script.
   - **Behaviour:** After every N requests (e.g. 500 or 1000) or at most once per 60s, call existing `log_stats()` or a one-line `log_info` with hit rate. **Opt-in** via constructor: `periodic_log_interval_requests: Optional[int] = None` and/or `periodic_log_interval_seconds: Optional[float] = None` (default `None` = disabled).
   - If logging raises, catch and log the exception once; do not affect cache path. Validate in `__init__` (e.g. reject negative values).

2. **Documentation**
   - In `CACHE_MONITORING.md`: add a short “Periodic logging” subsection (how to enable, parameters).
   - In the action-items plan: verification note that cache monitoring is satisfied.

3. **Tests**
   - In existing cache test file: with `periodic_log_interval_requests=50`, perform ≥50 cache operations and assert logging was triggered (mock/capture logs). One test with `None` to ensure no logging when disabled.

---

## 3. Parallel data fetching

**Verify:** `scan_all_symbols(execution_mode="threadpool"|"asyncio")` and `run_batch_atc_async()` exist; confirm a benchmark shows batch 100 symbols faster when parallel.

**Add and integrate:**

1. **Default and docs**
   - Keep default `execution_mode="threadpool"`. In scanner/README or `setting_guides_speed_optimization.md`: for batch scanning (50–100+ symbols), use `threadpool` or `asyncio`; `sequential` for rate-limit or debugging.

2. **Data fetching parallelism**
   - Ensure in the scan path, fetch happens inside the parallel unit (per symbol or per batch). If any path fetches all symbols sequentially before parallel compute, change so fetch is inside the worker.

3. **Benchmark**
   - One runnable benchmark: 100 symbols sequential vs parallel; assert parallel is faster (or both complete and results consistent). Reuse `benchmark_cache_parallel.py` or add a small script in `benchmarks/`. Document how to run it.

4. **Action-items plan**
   - Note: “Parallel data fetching: batch 100 symbols benchmark runnable; parallel mode faster than sequential.”

---

## 4. Async API for incremental updates

**Verify:** `AsyncIncrementalATC` works with asyncio; `websocket_incremental_live.py` demonstrates WebSocket; existing tests cover async incremental.

**Add and integrate:**

1. **Single async API surface**
   - In `core/compute_atc_signals/incremental/`: add `async_api.py` (or equivalent) that re-exports `AsyncIncrementalATC` and main coroutines. Package `__init__.py` re-exports so one documented import path exists (e.g. `from ...incremental import AsyncIncrementalATC`).

2. **Documentation**
   - In `ASYNC_API.md`: canonical async API is `AsyncIncrementalATC`; add “Quick import” and “Verification” (asyncio + WebSocket demo path).

3. **Verification**
   - At least one test uses `asyncio.run(...)` with `AsyncIncrementalATC.initialize` and `.update`. Optionally one test that imports from the new async surface.

4. **Action-items plan**
   - Note: “Async API for incremental: asyncio usable; WebSocket demo: `websocket_incremental_live.py`.”

---

## 5. Error handling, testing, rollout

- **Cache:** Logging failures caught once; no retry; don’t affect cache. Validate periodic-log parameters in `__init__`.
- **Parallel:** Preserve existing per-symbol error handling in scan path.
- **Async:** No change to `AsyncIncrementalATC`; document that callers handle errors in their event loop.

**Testing:** Cache periodic-log tests; parallel 100-symbol benchmark or test; existing async incremental tests as verification. Run full `adaptive_trend_LTS_mini` test suite after changes.

**Rollout:** (1) CacheManager periodic logging + cache tests, (2) docs and benchmark for parallel, (3) async surface + ASYNC_API.md. Then run tests, run 100-symbol benchmark, run WebSocket example if feasible. Mark the three Phase 3 items done in `adaptive-trend-action-items-plan.md` with short “Done” notes.

---

## Implementation plan (checklist)

1. **Cache**
   - [ ] Add `periodic_log_interval_requests` and `periodic_log_interval_seconds` to `CacheManager.__init__`; implement periodic call to `log_stats()` (or one-line hit rate) in `get()`; catch logging exceptions.
   - [ ] Update `CACHE_MONITORING.md` with “Periodic logging” subsection.
   - [ ] Add tests for periodic logging (enabled + disabled) in existing cache test file.
   - [ ] Update `adaptive-trend-action-items-plan.md`: mark cache monitoring done, add verification note.

2. **Parallel**
   - [ ] Confirm scan path: fetch inside workers (no sequential fetch-all); fix if needed.
   - [ ] Add or extend benchmark (e.g. in `benchmark_cache_parallel.py`) for 100 symbols sequential vs parallel; document run command.
   - [ ] Add short doc note in scanner/setting_guides for batch execution mode.
   - [ ] Update action-items plan: mark parallel data fetching done, add verification note.

3. **Async API**
   - [ ] Add `async_api.py` (or incremental `__init__.py` section) re-exporting `AsyncIncrementalATC`; ensure package exposes it.
   - [ ] Update `ASYNC_API.md` with Quick import and Verification.
   - [ ] Confirm existing test for asyncio + incremental; optionally add import-from-surface test.
   - [ ] Update action-items plan: mark async API for incremental done, add verification note.

4. **Final**
   - [ ] Run `pytest tests/adaptive_trend_LTS_mini/`; fix regressions.
   - [ ] Run 100-symbol benchmark once; run WebSocket example once if feasible.
