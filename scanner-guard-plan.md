# Scanner Guard & Duration Logging

## Goal
Prevent overlapping scan cycles (auto/manual) and log each cycle's wall-clock duration. Single file change: `modules/auto_trade/gui/main_window/scanner.py`.

## Tasks
- [x] Task 1: Add `import threading` and `import time` to top-level imports (line 2-3) → Verify: no `ImportError`
- [x] Task 2: Add `self._scan_running = False` and `self._scan_lock = threading.Lock()` in `__init__` (after line 26) → Verify: attrs exist on instance
- [x] Task 3: Insert overlap guard at very top of `_scanner_cycle` (before line 291 banner) — `start_time = None`, lock check, early return with log, else set flag + `start_time` → Verify: guard block present before `"Running scanner cycle..."`
- [x] Task 4: Wrap existing try/except body in try/finally — `finally` clears `_scan_running` under lock, computes+logs duration if `start_time is not None` → Verify: `_scan_running` always cleared, duration always logged
- [x] Task 5: Keep `_manual_scan_running` in `_manual_scan()` untouched → Verify: no diff in `_manual_scan` method
- [x] Task 6: Write unit test `tests/auto_trade/gui/test_scanner_overlap_guard.py` — two threads call `_scanner_cycle`, second gets skipped, log has "skipped", duration logged for first → Verify: `pytest tests/auto_trade/gui/test_scanner_overlap_guard.py` passes
- [x] Task 7: Run linter on changed files → Verify: no new errors

## Done When
- [x] Two concurrent `_scanner_cycle` calls → only one runs, other logs skip
- [x] Each completed cycle logs `"Scanner cycle completed in X.Xs"`
- [x] All existing tests still pass
