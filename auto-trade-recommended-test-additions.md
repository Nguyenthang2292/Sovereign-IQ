# Auto Trade – Recommended Test Additions

## Goal

Add the missing tests from REFACTORING_RECOMMENDATIONS.md (Recommended Test Additions): database query tests, then fill gaps in existing pipeline/websocket/selector/GUI tests and improve assertion quality.

## Tasks

- [x] **Task 1: Add `tests/auto_trade/database/` and pagination tests**  
  Create `tests/auto_trade/database/test_queries_pagination.py` with tests for `get_orders_cursor` / `get_all_programmatic_orders` (or equivalent): first page, second page, no overlap, empty DB, symbol filter.  
  → Verify: `pytest tests/auto_trade/database/test_queries_pagination.py -v` passes.

- [x] **Task 2: Add database stats/edge-case tests**  
  In the same file or a new `test_queries_stats.py`, add tests for `get_overall_stats` with empty DB, single order, and multiple orders (win/loss counts, total_pnl).  
  → Verify: `pytest tests/auto_trade/database/ -v` passes.

- [x] **Task 3: Review and add missing signal pipeline health tests**  
  In `tests/auto_trade/core/test_signal_pipeline_health.py`, ensure coverage for: degraded health continues, circuit breaker OPEN skips Gemini, metrics recorded (increment/gauge/histogram). Add any missing tests from the doc’s examples.  
  → Verify: `pytest tests/auto_trade/core/test_signal_pipeline_health.py -v` passes.

- [x] **Task 4: Review and add missing WebSocket service tests**  
  In `tests/auto_trade/gui/test_websocket_service.py`, ensure coverage for: init/credential loading, start/stop lifecycle, position callback registration/invocation, connection failure handling. Add any missing tests from the doc.  
  → Verify: `pytest tests/auto_trade/gui/test_websocket_service.py -v` passes.

- [x] **Task 5: Review and add missing signal selector scoring tests**  
  In `tests/auto_trade/core/test_signal_selector_scoring.py`, ensure coverage for: score components (confidence, R/R, consistency), R/R capping at 3.0, invalid xgboost_conf fallback. Add any missing tests from the doc.  
  → Verify: `pytest tests/auto_trade/core/test_signal_selector_scoring.py -v` passes.

- [x] **Task 6: Review and add missing database panel UI tests**  
  In `tests/auto_trade/gui/test_database_panel.py`, ensure coverage for: data viewer refresh (e.g. table switch), export to CSV (success path). Add mocks as in the doc’s examples if missing.  
  → Verify: `pytest tests/auto_trade/gui/test_database_panel.py -v` passes.

- [x] **Task 7: Add shared fixtures to `tests/auto_trade/conftest.py`**  
  Move or add common fixtures (e.g. pipeline mocks, session_scope, test_db) used by multiple test files to avoid duplication.  
  → Verify: Full suite still passes: `pytest tests/auto_trade/ -v --tb=short`.

- [x] **Task 8: Improve assertion messages in new/updated tests**  
  Where new or edited tests use bare `assert result` or `assert mock.called`, add messages (e.g. `assert x, f"Expected ... got {x}"`) and use `mock.assert_called_once_with(...)` where appropriate.  
  → Verify: Intentionally break one test and confirm the failure message is clear.

## Done When

- [x] `tests/auto_trade/database/` exists with pagination and stats tests.
- [x] Pipeline health, WebSocket, signal selector scoring, and database panel tests include the scenarios from REFACTORING_RECOMMENDATIONS.md.
- [X] `pytest tests/auto_trade/ -v` passes and new tests are covered by the above verification steps.

## Notes

- Target is ~65–83 new tests total across the five areas; implement in the order above (database first, then fill gaps in existing files).
- Use existing `tests/auto_trade/conftest.py` and DB helpers (e.g. `test_database.py` / `test_reconcile.py`) for session and schema.
- For GUI tests, mock `session_scope`, `filedialog`, and CTk where needed to avoid real UI.
