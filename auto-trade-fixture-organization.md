# Auto Trade – Fixture Organization

## Goal

Move duplicated pytest fixtures from per-test-class definitions into `tests/auto_trade/conftest.py` so tests share one place for common setup and duplication is removed.

## Tasks

- [x] **Task 1: Add shared DB fixtures to conftest**  
  In `tests/auto_trade/conftest.py`, add `test_db` (session or function scope, temp DB path + cleanup) and `sample_order_data` (factory returning a dict of order fields). Reuse the same DB creation/teardown pattern used in `test_database.py` and `test_reconcile.py`.  
  → Verify: `pytest tests/auto_trade/test_database.py tests/auto_trade/test_reconcile.py -v` still passes (no change to test logic yet).

- [x] **Task 2: Switch test_database.py and test_queries_pagination.py to conftest test_db + sample_order_data**  
  Remove local `test_db` and `sample_order_data` from both files; use fixtures from conftest. Adjust any scope (e.g. tmp_path) if needed.  
  → Verify: `pytest tests/auto_trade/test_database.py tests/auto_trade/database/test_queries_pagination.py -v` passes.

- [x] **Task 3: Switch test_reconcile.py to conftest test_db**  
  Remove the two class-level `test_db` fixtures from `test_reconcile.py` and use the conftest `test_db`. Keep `mock_exchange` and order mocks in-file (test-specific).  
  → Verify: `pytest tests/auto_trade/test_reconcile.py -v` passes.

- [x] **Task 4: Add shared pipeline + mock_components to conftest**  
  In conftest, add a factory fixture that builds `mock_components` (gemini, metrics, health, etc.) and a `pipeline` fixture that takes it, matching the pattern in `test_signal_pipeline_health.py` / `test_signal_pipeline.py`.  
  → Verify: One of the pipeline tests still passes using the new fixture (e.g. `pytest tests/auto_trade/core/test_signal_pipeline_health.py -v`).

- [x] **Task 5: Refactor test_signal_pipeline_health.py and test_signal_pipeline.py to use conftest pipeline**  
  Remove repeated class-level `mock_components` and `pipeline` from both files; inject conftest fixtures. Keep only test-specific mocks (e.g. psutil, circuit breaker state) in the test.  
  → Verify: `pytest tests/auto_trade/core/test_signal_pipeline_health.py tests/auto_trade/core/test_signal_pipeline.py -v` passes.

- [x] **Task 6: Add shared selector fixture to conftest and use in signal_selector tests**  
  Add a `selector` fixture in conftest (or a small factory) that builds the SignalSelector under test. Remove the three duplicate `selector` fixtures from `test_signal_selector_scoring.py` and the one from `test_signal_selector.py`; use conftest.  
  → Verify: `pytest tests/auto_trade/core/test_signal_selector_scoring.py tests/auto_trade/core/test_signal_selector.py -v` passes.

- [x] **Task 7: Optional – move mock_data_fetcher / mock_scan_all_symbols to conftest**  
  If `test_atc_scanner.py` and `test_atc_scanner_enhancements.py` use identical or near-identical mocks, add them to conftest and switch both files to use them. If mocks differ a lot, leave them in-file.  
  → Verify: `pytest tests/auto_trade/core/test_atc_scanner.py tests/auto_trade/core/test_atc_scanner_enhancements.py -v` passes.

- [X] **Task 8: Full suite check**  
  Run the full auto_trade test suite and fix any regressions from fixture renames or scope changes.  
  → Verify: `pytest tests/auto_trade/ -v --tb=short` passes.

## Done When

- [x] `tests/auto_trade/conftest.py` provides shared `test_db`, `sample_order_data`, pipeline-related fixtures, and selector fixture.
- [x] No duplicate definitions of those fixtures remain in test_database, test_reconcile, test_queries_pagination, test_signal_pipeline_health, test_signal_pipeline, test_signal_selector_scoring, test_signal_selector.
- [ ] `pytest tests/auto_trade/ -v` passes.

## Notes

- Keep `gui/utils/conftest.py` as-is for GUI/utils-specific fixtures; only promote fixtures that are shared across multiple test files under `tests/auto_trade/`.
- Prefer function-scoped `test_db` if tests mutate DB; use session scope only if DB is read-only and creating it is slow.
- If a test class needs a slightly different pipeline/selector setup, use a thin local fixture that depends on the conftest one and overrides only what’s needed.
