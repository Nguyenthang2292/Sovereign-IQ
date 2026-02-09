# Scanner Guard and Scan Duration Logging — Design

**Date:** 2026-02-09  
**Status:** Validated (v2 — updated with review fixes)  
**Summary:** Add a single guard to prevent overlapping scan cycles (auto or manual) and log each cycle's duration. When a new cycle would start while one is still running, skip and reschedule (next attempt after interval).

---

## 1. Goal and scope

**Goals**
- Only one scan (auto or manual) runs at a time.
- If the auto timer fires while a scan is still running: skip the new cycle, log clearly, and the next auto run remains 5 minutes after the skip (no backlog).
- Log how long each scan takes (start/end and total duration).

**In scope**
- **ScannerManager** (`gui/main_window/scanner.py`): add guard and duration logging in `_scanner_cycle`; no change to the contract with ScannerControl.
- **PeriodicUpdater** unchanged: it keeps calling the callback every interval; the callback (e.g. `_scanner_cycle`) does the guard and returns immediately when skipping, so the next run is still 5 minutes later.

**Out of scope**
- No dynamic interval based on scan duration.
- No queue of pending scans.

---

## 2. Guard: single flag and where to check

**Single flag for pipeline guard**
- Add one flag on `ScannerManager`: `_scan_running` (bool).
- The overlap guard sits at the **very top** of `_scanner_cycle`, **before** the banner log (`"Running scanner cycle..."`). This is distinct from the existing `scan_skipped` variable (which gates on max open positions and happens later in the method).
- At the start of `_scanner_cycle`: under lock, if `_scan_running` is True → log `"Scan skipped: previous cycle still running"` and **return** immediately. Do **not** put `scanner_done` on the queue (the in-progress cycle will send its own when it finishes).
- If `_scan_running` is False: set `_scan_running = True` (under lock), then proceed with the existing logic.
- In a `finally` in `_scanner_cycle`: set `_scan_running = False` (under lock).

**Two flags, two purposes**
- **`_scan_running`** (new): pipeline-level guard in `_scanner_cycle`. Prevents auto-vs-manual and manual-vs-auto overlap.
- **`_manual_scan_running`** (existing, keep): UI-level guard in `_manual_scan()`. Prevents spawning duplicate manual-scan threads and manages UI state ("Scanning..." text, timestamp update, progress clear). These are separate concerns.

**Thread-safety**
- Use a short-held `threading.Lock` (`_scan_lock`) when reading/writing `_scan_running` (check-and-set at start, and set False in `finally`) so that two threads (e.g. auto timer thread and manual scan thread) cannot both pass the guard.

---

## 3. Scan duration logging

**What to measure**
- Inside `_scanner_cycle`: from **after** the guard (when we actually start the pipeline) to the end of the cycle.
- When the cycle is skipped (overlap guard early return), do not log a duration; only log the skip message.

**How**
- Right after setting `_scan_running = True`, set `start_time = time.perf_counter()`.
- Compute duration and log **entirely in `finally`** (so it always runs, even on exception):

```python
finally:
    with self._scan_lock:
        self._scan_running = False
    if start_time is not None:
        duration = time.perf_counter() - start_time
        logger.info("Scanner cycle completed in %.1fs", duration)
```

**Format**
- Single line: `"Scanner cycle completed in X.Xs"` (one decimal place).
- On exception: same line still fires (duration up to the failure point), plus the exception log from the existing `except` block.

**Where**
- Use the existing scanner logger (`logging.getLogger("auto_trade.scanner")`); no separate file. Visible in GUI log / console.

**Edge cases**
- If an exception is raised, `finally` still clears `_scan_running` and logs duration (because `start_time` was already set).
- If guard skips (early return), `start_time` is `None` → no duration log, no flag clear needed (flag was never set to True).

---

## 4. File changes and verification

**Files to change**

1. **`modules/auto_trade/gui/main_window/scanner.py`**
   - Imports: add `threading` and `time` (top-level, not lazy).
   - In `__init__`: add `self._scan_running = False` and `self._scan_lock = threading.Lock()`.
   - At the **very top** of `_scanner_cycle` (before the banner `"Running scanner cycle..."`):
     - `start_time = None`
     - With `self._scan_lock`: if `self._scan_running` → log skip, return immediately (no `scanner_done`).
     - Else: set `self._scan_running = True`.
     - `start_time = time.perf_counter()`
   - Wrap the existing try/except body in try/finally:
     - `finally`: clear `self._scan_running` under lock; if `start_time is not None`, compute and log duration.
   - **Keep** `_manual_scan_running` as-is in `_manual_scan()` for UI state.

2. **`modules/auto_trade/gui/utils/threading_utils.py`**
   - No change.

**Verification**
- **Unit test:** mock `_run_signal_scan` to sleep; invoke `_scanner_cycle` concurrently from two threads → second returns immediately, log contains "skipped", and `_scan_running` is False when both are done.
- **Manual:** Enable auto scan at 5 min; click manual scan while auto is running → log shows "Scan skipped: previous cycle still running"; each completed cycle shows "Scanner cycle completed in X.Xs".

**Risk**
- Any code path that runs the pipeline without going through `_scanner_cycle` will not be guarded; ensure all scan entry points (auto and manual) go through `_scanner_cycle`.
