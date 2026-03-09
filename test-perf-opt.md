# Test Performance Optimization

## Goal
Reduce auto_trade test suite runtime using 3-layer architecture: `unit_fast` / `integration_smoke` / `integration_slow`.

## Tasks

### Phase A — Non-integration (fast-first)

- [ ] **T1 – pytest.ini: add 3 new markers, bump `--durations`**
  In `pytest.ini`: add `unit_fast`, `integration_smoke`, `integration_slow` to the `markers` block;
  change `--durations=10` → `--durations=20`.
  → Verify: `python -m pytest --markers | findstr unit_fast` prints the marker.

- [ ] **T2 – `test_health.py::test_check_timeout` — remove real `time.sleep(5)`**
  In `slow_check()` replace `time.sleep(5)` with `threading.Event().wait()` (blocks indefinitely, no real sleep).
  Registry timeout at 0.1 s will still fire and mark it UNHEALTHY.
  Add `@pytest.mark.unit_fast` to the test.
  → Verify: `pytest tests/auto_trade/core/test_health.py::test_check_timeout -v` passes in < 1 s.

- [ ] **T3 – `test_scan_cache.py` TTL tests — eliminate `time.sleep` sleeps up to 2.1 s total**
  `ScanCache` is a Rust extension — cannot freeze its clock.
  In the 4 TTL tests (`test_cache_ttl_expiration`, `test_cache_remove_expired`, `test_cache_mixed_expiration`,
  `test_cache_mixed_expiration`): change `ttl_seconds=1.0` / `ttl_seconds=2.0` → `ttl_seconds=0.05`;
  change all `time.sleep(1.5)` / `time.sleep(0.6)` → `time.sleep(0.1)`.
  Add `@pytest.mark.unit_fast` to each.
  → Verify: `pytest tests/auto_trade/core/test_scan_cache.py -v` finishes < 3 s total.

- [ ] **T4 – `test_adaptive_close.py::test_fallback_no_data` — patch `_fetch_ohlcv`**
  Add `@patch("modules.auto_trade.execution.adaptive_close_calculator.AdaptiveCloseCalculator._fetch_ohlcv", return_value=None)`
  decorator so the test returns `None` without hitting real ccxt.
  Add `@pytest.mark.unit_fast`.
  → Verify: test passes; `ccxt` is never imported during the test run.

- [ ] **T5 – `test_data_service.py` — fix fallback test + mark non-integration**
  In `test_get_signals_fallback`: change `service.database_manager = None` → `service.repo_context = None`
  (matches the actual attribute name on `DataService`).
  Add `@pytest.mark.unit_fast` to all tests that do not call a real network/DB
  (`test_init_dry_run_mode`, `test_get_current_price_dry_run`, `test_get_signals`, `test_get_signals_fallback`,
  `test_get_quick_stats_dry_run`, `test_get_positions_dry_run`, `test_mock_price_feed_centralization`,
  `test_error_handling_in_price_fetch`).
  → Verify: `pytest tests/auto_trade/gui/utils/test_data_service.py -m unit_fast -v` passes with no network call.

### Phase B — Integration smoke/slow

- [ ] **T6 – `test_backtest_phase6.py` — smoke profile + layer markers**
  Add module-level constants:
  ```python
  SMOKE_LOOKBACK_DAYS = 2
  SMOKE_SKIP_INDICATORS = {"xgboost", "hmm", "random_forest"}
  ```
  Create `test_basic_backtest_smoke` and `test_martingale_backtest_smoke` that reuse the same logic
  with `lookback_days = SMOKE_LOOKBACK_DAYS` and skip heavy indicators.
  Mark smoke variants `@pytest.mark.integration_smoke`; mark originals `@pytest.mark.integration_slow`.
  → Verify: `pytest tests/auto_trade/integration/test_backtest_phase6.py -m integration_smoke -v` completes < 30 s.

- [ ] **T7 – `test_gemini_integration.py` — module-scoped heavy init fixture**
  Extract `ChartGenerator` + `GeminiChartAnalyzer` patches used repeatedly across
  `TestGeminiIntegrationConfiguration` into a `@pytest.fixture(scope="module")` in the same file.
  → Verify: second-run of the module (warm) is measurably faster than first run; `test_init_with_defaults` passes.

### Phase C — CI wiring

- [ ] **T8 – Add PR & nightly commands to `run_tests.ps1` / `run_tests.bat`**
  Add two clearly commented sections (not replacing existing defaults):
  - PR fast: `python -m pytest -m "unit_fast or integration_smoke" --durations=20`
  - Nightly:  `python -m pytest -m integration_slow --durations=20`
  → Verify: `python -m pytest -m "unit_fast or integration_smoke" --collect-only` collects > 0 tests,
    `python -m pytest -m integration_slow --collect-only` collects > 0 tests.

## Done When
- [ ] `python -m pytest -m unit_fast --durations=20` — no item takes > 1 s wall time
- [ ] `python -m pytest -m integration_smoke` — suite finishes < 120 s
- [ ] `python -m pytest --durations=20` baseline comparison shows ≥ 50 % reduction in top-10 slow items
