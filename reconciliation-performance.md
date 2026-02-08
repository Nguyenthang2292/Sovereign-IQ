# Reconciliation Optimization + Performance Testing

## Goal
Finish remaining Day 3 (reconcile optimization) and Day 4-5 (performance testing) tasks: batch DB writes, concurrency safety, stale-order optimization, profiling/optimization, and documentation.

## Tasks
- [x] **1. Batch DB inserts in reconcile** — In `modules/auto_trade/database/reconcile.py`, collect new orders and insert using SQLAlchemy bulk (e.g., `session.bulk_insert_mappings`) instead of `create_order` in a loop. Keep validation for required fields and record skipped/failed IDs.  
  → Verify: reconcile runs with same inputs and reports inserted/skipped counts; no per-order DB writes in the hot path.

- [x] **2. Add write-locking for concurrent updates** — Implement an explicit write lock for reconcile updates (SQLite: `BEGIN IMMEDIATE` or a dedicated `threading.Lock` around reconcile DB writes) to prevent concurrent reconcile runs from double-inserting or conflicting updates.  
  → Verify: two concurrent reconcile calls cannot run DB writes at the same time (lock acquired), and no duplicate inserts occur in a simulated parallel run.

- [x] **3. Optimize stale order detection** — Reduce per-order `fetch_order` calls by batching where possible: prefer `fetch_orders` / `fetch_closed_orders` for a symbol/time window and map results by `clientOrderId`, falling back to `fetch_order` only if missing.  
  → Verify: stale handling updates the same final statuses with fewer exchange calls (log fewer `fetch_order` calls).

- [x] **4. Profile reconcile + stats queries** — Add a small profiling script or toggle (e.g., `scripts/profile_reconcile.py` or `AUTO_TRADE_PROFILE_RECONCILE=true`) to capture timing for reconcile, `get_overall_stats`, and cursor pagination.  
  → Verify: running the profiler produces a timing report (stdout or file) with top functions.

- [x] **5. Optimize bottlenecks from profile** — Apply targeted improvements based on profiling results (e.g., batch size tuning, query index check, reduce per-order updates). Keep changes scoped to reconcile + stats.  
  → Verify: profiling shows reduced time for the identified hot paths.

- [x] **6. Document performance improvements** — Add a concise doc (e.g., `modules/auto_trade/docs/performance_improvements.md`) with before/after timings, profiling summary, and any behavior changes.  
  → Verify: doc includes measured numbers and references the profiling run and benchmarks.

- [X] **7. Verification** — Re-run `tests/auto_trade/test_performance_10k_orders.py` and a reconcile run in demo/test mode to confirm no regressions.  
  → Verify: tests pass and reconcile output looks correct; update REFACTORING_RECOMMENDATIONS Day 3 + Day 4-5 to DONE.

## Done When
- [x] Reconcile uses batch DB inserts and is concurrency-safe.
- [x] Stale order detection uses fewer exchange calls with same outcomes.
- [x] Profiling + improvements documented with measurable results.
- [x] Performance tests pass; REFACTORING_RECOMMENDATIONS updated.

## Notes
- Relevant files: `modules/auto_trade/database/reconcile.py`, `modules/auto_trade/database/utils.py`, `tests/auto_trade/test_performance_10k_orders.py`.
