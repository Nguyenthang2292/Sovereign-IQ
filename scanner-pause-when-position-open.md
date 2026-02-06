# Scanner pause when position open

## Goal
Implement design: skip expensive scan (Gemini) when DB already has enough open positions; when position closes, next cycle runs full scan automatically. Save CPU and API cost.

## Tasks

- [x] **Gate in `_scanner_cycle`** — Before `_run_signal_scan()`, call `get_open_positions(session)` and `risk.max_open_positions`; if `count >= max` set `scan_skipped = True`, skip pipeline, log clearly. → Verify: Run scanner with 1 OPEN order in DB (max=1), log shows "Scanner cycle skipped... no Gemini call".
- [x] **Pass skip state to GUI** — Put `("scanner_done", {"skipped": True, "count": N})` when skipped, `None` when full run. → Verify: Queue receives correct payload.
- [x] **GUI status label** — In `updaters._drain_update_queue`, on `scanner_done` with `data.get("skipped")` set `scanner_status_label` to "RUNNING (scan skipped – N open position)"; else "RUNNING". → Verify: Start scanner, add OPEN order in DB, wait next cycle or manual scan, label shows "scan skipped – 1 open position".
- [x] **Fallback on DB error (design §4)** — If `get_open_positions` / `session_scope` raises, skip scan (do not run Gemini) and log warning. → Verify: Temporarily break DB in test, run cycle, no Gemini call and log contains warning.
- [x] **Unit test** — Test gate logic: e.g. `test_scanner_skips_when_open_count_ge_max` with mock session (1 open, max=1 → skip) and `test_scanner_runs_when_open_count_lt_max` (0 open, max=1 → run). → Verify: `pytest tests/auto_trade/gui/test_scanner_pause_when_position_open.py -v` passes.

## Done when

- [x] Gate skips full scan when `open_count >= max_open_positions` (already done).
- [x] GUI shows "scan skipped – N open position" when cycle skipped (already done).
- [x] On DB error, cycle skips scan and logs (no Gemini).
- [x] At least one unit test covers skip vs run by open count.

## Notes

- Manual scan uses same `_scanner_cycle()`, so gate applies automatically.
- When position closes (Reconcile/WS updates DB), next timer cycle sees fewer open → full scan runs; no extra "resume" logic.
- Plan file in project root per plan-writing skill; design lives in `modules/auto_trade/docs/plans/2026-02-06-scanner-pause-when-position-open-design.md`.
