# Changelog

All notable changes and review findings for the `modules/auto_trade` module.

## [Implementation Update] - 2026-02-22

### Current status after Phase 7

- Implementation status in this pass: **completed through Phase 7**.
- Any unresolved findings should be interpreted from the historical review section below, not as current implementation state.

### Completed in this implementation pass

#### ✅ Phase 1 — Event System Unification

- Unified on `monitoring/event_system.py`; removed legacy `monitoring/events.py`.
- Added thread-safety to `EventSystem` with `RLock` around subscriber/history access.
- Replaced event history list trimming with `deque(maxlen=1000)`.
- Migrated imports and usage from legacy events module to unified `event_system` in production paths and related monitoring tests.
- Added backward-compatible support in `EventSystem` (`EventBus` alias and dual publish style) to avoid regressions during migration.

#### ✅ Phase 2 — BinanceClient Caching

- Implemented `DataService._binance_client` cache and `_get_or_create_client()` factory usage.
- Invalidated client cache when credentials change.
- Routed `WebSocketHandler._get_binance_client()` through `DataService` instead of constructing clients inline.

#### ✅ Phase 3 — TP/SL Sync Decoupling from WS Tick

- Added `DataService` TP/SL cache (`_tpsl_cache`, `_tpsl_cache_time`) with TTL.
- Replaced WS inline TP/SL sync with cached lookup via `DataService.get_cached_tpsl(...)`.
- Replaced inline TP/SL sync in `DataService.get_positions()` with the same cache path.
- Updated TP/SL cache API signature to `get_cached_tpsl(symbol: str, ttl_seconds: int = 30)`.

#### ✅ Phase 4 — Replace `print()` with Structured Logging

- Replaced `print()` calls in:
  - `gui/main_window/auto_trade.py`
  - `gui/main_window/websocket_handler.py`
  - `gui/utils/data_service.py`
  - `gui/utils/settings_manager.py`
- Applied `log_info/log_debug/log_warn/log_error` based on message severity and context.

#### ✅ Phase 5 — P1 Code Fixes

- **5.1**: Fixed `SignalPipeline` logging setup import path (`setup_logging` is now explicitly imported and import check passes).
- **5.2**: Added credential reload caching in `DataService` (`_credentials_loaded`) and event-driven invalidation.
  - Added `EventType.SETTINGS_SAVED`.
  - `SettingsManager.save()` publishes `SETTINGS_SAVED`.
  - `DataService` subscribes and invalidates credential/client cache on save events.
- **5.3**: Implemented module-level singleton cache for `RepositoryContext.from_env()`.
- **5.4**: Removed deprecated `modules/auto_trade/main.py`; repurposed `test_monitoring_mode.py` to a deprecation-safe stub path.

### Verification snapshot

- `SignalPipeline` import check passes.
- `RepositoryContext.from_env()` singleton identity check passes (`a is b is c`).
- DataService credential reload cache check passes (repeated reload calls hit credential loader once until invalidation).
- `SETTINGS_SAVED` event invalidates `DataService` credential cache as expected.
- No remaining `print(` in the 4 targeted Phase 4 files.
- No remaining imports of `modules.auto_trade.main` under `modules/auto_trade/**`.

### ✅ Phase 6–7 Status Update

- **Phase 6 (Tests): Completed** — dedicated tests for `CircuitBreaker`, `SignalPipeline`, and `DataService` have been implemented.
- **Phase 7 (Verification): Completed** — broader/full verification and final sweep have been executed.
- This implementation pass is now considered complete through Phase 7.

### Resolved since Review v3

- **A1 / #9**: Deprecated `modules/auto_trade/main.py` removed; compatibility path repurposed.
- **A2 / #3**: Event system unified on `monitoring/event_system.py`; legacy `monitoring/events.py` removed.
- **A3 / #8**: `SignalPipeline` logging setup import fixed.
- **A4 / #12**: `RepositoryContext.from_env()` now uses module-level singleton caching.
- **S2 / #4 / Q1**: Targeted GUI production paths migrated from `print()` to structured logging.
- **R1 / R2 / P1 / #1 / #2**: `BinanceClient` reuse + TP/SL cache flow implemented for GUI update paths.
- **P2 / #10**: `DataService._reload_credentials()` cached with event-driven invalidation.
- **P3 / #14**: Event history backed by `deque(maxlen=1000)`.
- **T2 / T3 / T4 / #5 / #6**: Dedicated tests for `CircuitBreaker`, `SignalPipeline`, and `DataService` completed (per implementation status).

## [Review v3 — Final (Historical Snapshot)] - 2026-02-22

### Code Review Summary

Final comprehensive code review by Antigravity Codex Review.
**Scope:** Full `modules/auto_trade` module (core, execution, strategies, monitoring, database, websocket, gui) — 50+ source files, ~20,000 LoC.
**Focus areas:** Architecture, security, reliability, performance, testing, code quality, GUI-specific concerns.

> Note: This section captures findings at review time. Current implementation status is tracked in the **Implementation Update** section above.

---

### Architecture ⭐⭐⭐⭐ (4/5)

**Strengths:**
- **Clean layered architecture**: `core → execution → strategies → monitoring → database → gui` with well-defined boundaries between signal generation, order execution, strategy management, and presentation.
- **Protocol-based decoupling**: `SignalPipeline` uses `XGBoostFilterLike` and `ATCScannerLike` protocols for dependency inversion — excellent for testability and swapping implementations.
- **Repository pattern**: `RepositoryContext` provides backend-agnostic DynamoDB access with clean query functions exposed via `database/__init__.py`.
- **Event-driven monitoring**: `EventSystem` provides pub-sub decoupling between `RecoveryManager`, `ScannerManager`, `AutoTradeManager`, and WebSocket handlers.
- **Dual ATC backend**: `ATCScanner` (local) and `ATCServerlessScanner` (Lambda) with seamless fallback.
- **GUI decomposition**: `AutoTradeDashboard` properly delegates to 7+ manager classes (`LayoutManager`, `UpdaterManager`, `ScannerManager`, etc.) avoiding a monolithic window class.
- **Passthrough filter pattern**: `PassthroughXGBoostFilter` elegantly handles missing XGBoost models without conditional logic throughout the pipeline.

**Issues:**

| # | Severity | Issue | Location |
|---|----------|-------|----------|
| A1 | P1 | `main.py` is a deprecated scaffold — all its methods return empty/False. It logs "DEPRECATION NOTICE" at init but still has a full event loop and signal handlers. Should be removed or converted to a thin CLI entry point delegating to `run_gui.py`. | `main.py:103-107` |
| A2 | P1 | **Two incompatible Event/EventBus implementations**: `monitoring/events.py` defines `Event(namedtuple)` + `EventBus` while `monitoring/event_system.py` defines `Event(dataclass)` + `EventSystem`. `SignalPipeline` imports the first; `RecoveryManager`, `AutoTradeManager`, and GUI import the second. These cannot interoperate — events published from the pipeline never reach GUI subscribers. | `events.py` vs `event_system.py` |
| A3 | P2 | `signal_pipeline.py` calls `setup_logging()` at line 150 but this function is never imported — relies on a wildcard or monkey-patching that may fail silently. Uses `log_info`/`log_error`/`log_warn` without importing them (likely from a star import of `modules.common.ui.logging`). | `signal_pipeline.py:150` |
| A4 | P2 | `RepositoryContext.from_env()` creates a new context on every call — no singleton/caching. Multiple components (`DataService`, `OrderManager`, `OrderExecutor`, `ScannerManager`) each call `from_env()`, creating redundant DynamoDB connections. | Multiple files |
| A5 | P2 | Symbol normalization spread across 5+ locations: `normalize_symbol()`, `normalize_symbol_key()`, manual `replace("/", "")`, and `sym.replace('USDT', '')}/USDT` conversion. No single source of truth. | `order_executor.py`, `order_manager.py`, `auto_trade.py`, `data_service.py` |
| A6 | P3 | `LayoutManager` is 601 lines with scanner configuration, Current Settings panel, system logs, and live stream logs all built inline. The Scanner tab UI should be extracted to its own component for maintainability. | `layout.py` |

---

### Security ⭐⭐⭐ (3/5)

**Strengths:**
- `SecretString` wrapper used in `OrderExecutor` and `OrderManager` to avoid accidental credential logging.
- Config JSON export redacts API keys (`***REDACTED***`).
- `CircuitBreaker.sanitize_errors` option prevents exception details from leaking.
- `CredentialManager` properly adds `.env` to `.gitignore` and uses `python-dotenv` for storage.
- Binance time sync error detection with `-1021` code provides actionable user guidance.

**Issues:**

| # | Severity | Issue | Location |
|---|----------|-------|----------|
| S1 | P0 | **API credentials passed as plain strings** through multiple call chains: `OrderManager.__init__`, `RiskManager.calculate_position_size()`, `DataService._reload_credentials()`. The `SecretString` wrapper is only used at the top level but credentials are immediately `.get_secret_value()`'d and passed as raw strings. | Multiple files |
| S2 | P1 | **76+ `print()` statements** across production code bypass structured logging. These include credentials context (`[AutoTrade] Checking for signals`), position data, and order details. In `websocket_handler.py` alone there are 15+ print statements logging position P&L, TP/SL values, and sync results. | `auto_trade.py`, `websocket_handler.py`, `scanner.py`, `data_service.py`, `settings_manager.py` |
| S3 | P1 | `CredentialManager.test_connection()` creates an unauthenticated CCXT exchange instance that calls `fetch_balance()`. If credentials are wrong, the `ccxt.AuthenticationError` is caught but the `except Exception` catch-all returns the raw exception message which may contain partial credential info. | `credential_manager.py:220-231` |
| S4 | P2 | `settings.yaml` stores `api_key` and `api_secret` in the `api` section alongside mode/exchange config. While defaults are empty strings, if a user enters credentials in the Settings panel, they're persisted to a YAML file that has `.backup` copies made automatically. | `settings_manager.py:30, 264-274` |
| S5 | P2 | Full stack traces are printed via `traceback.print_exc()` in production code paths in `scanner.py:263`, `main_window.py:376`. These can expose internal paths, module names, and data structures. | Multiple files |

---

### Reliability ⭐⭐⭐½ (3.5/5)

**Strengths:**
- **Circuit Breaker** is well-implemented: thread-safe with `RLock`, proper CLOSED→OPEN→HALF_OPEN state machine, metrics tracking, and decorator support.
- **Retry with tenacity**: `OrderManager` and `OrderExecutor` use `@retry(stop=stop_after_attempt(3), wait=wait_exponential())` on all exchange API calls.
- **Fallback persistence**: `OrderManager` writes to `fallback_orders.jsonl` if DynamoDB fails (line 321-327).
- **Emergency stop** persists to DynamoDB via `set_system_state` and loads on startup via `get_system_state` — fixed from previous review.
- **Scanner gate**: Before running expensive Gemini analysis, scanner checks both DB and Binance for open positions using `max()` of both counts (most conservative).
- **Lock-protected trading**: `_trading_lock` prevents concurrent auto-trade cycles, and `_scan_lock` prevents concurrent scanner runs.

**Issues:**

| # | Severity | Issue | Location |
|---|----------|-------|----------|
| R1 | P0 | **New `BinanceClient` instance created per request** in `WebSocketHandler._convert_positions_to_dicts()` — called on every single WebSocket position update (~every few seconds). Each creates a new CCXT exchange instance with full auth handshake. | `websocket_handler.py:170-175` |
| R2 | P0 | **TP/SL sync runs synchronously inside every position update** via `TPSLSyncService.sync_position_tp_sl()`. This makes HTTP calls to Binance (fetch open orders) on every WebSocket tick, blocking the GUI update thread. | `websocket_handler.py:179-185`, `data_service.py:444-451` |
| R3 | P1 | **Race condition in `_auto_trade_cycle`**: After checking `_trading_running` under lock, the actual trading logic runs outside the lock (try/finally). If the method raises during the `OrderExecutor` constructor, `_trading_running` is properly reset in `finally`, but if the thread is killed, it stays `True` forever. | `auto_trade.py:247-332` |
| R4 | P1 | `DataService.get_positions()` creates a fresh `BinanceClient` on every call (line 383-388) to fetch mark prices. Combined with periodic refresh, this creates 2-3 new exchange instances per minute. | `data_service.py:380-388` |
| R5 | P2 | `EventSystem.publish()` calls subscribers synchronously in the publishing thread. If a subscriber blocks (e.g., DB write in `RecoveryManager._on_position_closed`), it blocks the WebSocket handler thread. | `event_system.py:114-121` |
| R6 | P2 | `SettingsManager._create_backup()` runs on every `.save()` call — settings are saved frequently (on every config change, every Apply Settings, on exit). This creates cascading `.yaml.backup` writes. | `settings_manager.py:264-274` |

---

### Performance ⭐⭐⭐⭐ (4/5)

**Strengths:**
- Rust-optimized signal aggregation, scoring, and caching in `ATCScanner`.
- Parallel multi-timeframe scanning with `ThreadPoolExecutor`.
- Async Gemini analysis (`analyze_candidates_batch_async`) with `max_concurrency=3`.
- XGBoost per-symbol training is isolated to scan cycles, not blocking GUI.
- `circuit_breaker` prevents hammering failed APIs.
- Scanner skips expensive Gemini calls when max positions are reached.

**Issues:**

| # | Severity | Issue | Location |
|---|----------|-------|----------|
| P1 | P1 | Same as R1/R2 — `BinanceClient` instantiation and TP/SL sync per WebSocket tick is the #1 performance bottleneck in production. | `websocket_handler.py` |
| P2 | P2 | `DataService.get_account_data()` calls `_reload_credentials()` on every invocation (line 135), which creates a new `CredentialManager`, reads `.env` from disk, and calls `load_dotenv`. This happens on every periodic account refresh. | `data_service.py:135` |
| P3 | P2 | `EventSystem._event_history` grows to 1000 events, then trims by slicing (creating new list copy every time). With position updates arriving every few seconds, this is costly. A `collections.deque(maxlen=1000)` would be O(1). | `event_system.py:107-111` |
| P4 | P3 | `ScannerManager._find_xgboost_model()` scans the filesystem (glob + stat + sort) on every pipeline initialization. Result should be cached since models don't change during runtime. | `scanner.py:267-290` |

---

### Testing ⭐⭐⭐ (3/5) — Improved from Previous Review

**Strengths:**
- Test suite now has 10 dedicated test files in `tests/` covering:
  - `test_order_builder.py`, `test_order_validator.py`, `test_risk_manager.py`
  - `test_trailing_stop.py`, `test_negative_breakeven.py`
  - `test_gradual_recovery.py`, `test_martingale.py`, `test_recovery_manager.py`
  - `test_resilience_and_fallback.py`, `test_secret_string.py`
- Plus 3 integration test files at the module root:
  - `test_execution_phase3.py`, `test_monitoring_mode.py`, `test_pipeline.py`

**Issues:**

| # | Severity | Issue | Location |
|---|----------|-------|----------|
| T1 | P1 | **No tests for the GUI layer** — the entire `gui/` directory (17 components + 21 utility files + main_window with 10 files) has zero test coverage. Side effects of settings changes, mode transitions, and WebSocket callback handling are untested. | `gui/` |
| T2 | P1 | **No tests for `CircuitBreaker`** — a critical reliability component with complex state machine (CLOSED/OPEN/HALF_OPEN transitions, metrics, thread safety). | `core/circuit_breaker.py` |
| T3 | P1 | **No tests for `SignalPipeline`** orchestration — the core business logic that chains ATC→XGBoost→Gemini→Selection. `test_pipeline.py` at root level is an integration test. | `core/signal_pipeline.py` |
| T4 | P2 | **No tests for `DataService`** — the data access layer that handles 3 different modes (DRY_RUN, DEMO, PRODUCTION), price fetching, position conversion, and TP/SL sync. | `gui/utils/data_service.py` |
| T5 | P2 | Test files in `tests/` are thin (850-2200 bytes each vs 6000-17000 byte source files). Need to verify they test more than just instantiation. | `tests/` |

---

### Code Quality ⭐⭐⭐⭐ (4/5)

**Strengths:**
- Comprehensive docstrings with `Args:`, `Returns:`, `Example:`, `Raises:`, and `Formula:` sections throughout.
- Consistent type annotations with `Optional`, `Dict`, `List`, `Literal`, `TypedDict`.
- Clean dataclass usage (`OrderTicket`, `CircuitBreakerMetrics`, `Event`, `RecoveryState`).
- Protocol classes (`XGBoostFilterLike`, `ATCScannerLike`) for structural subtyping.
- `TypedDict` for configuration (`PipelineConfig`, `RecoveryConfig`).
- Proper enum usage (`CircuitState`, `EventType`, `TradingMode`).
- Configuration validation in `AutoTradeConfig.__post_init__()` and `SettingsManager._validate_settings()`.

**Issues:**

| # | Severity | Issue | Location |
|---|----------|-------|----------|
| Q1 | P2 | **Inconsistent logging**: Some files use `log_info`/`log_error` from `modules.common.ui.logging`, others use bare `print()`. The GUI files are especially inconsistent — `main_window.py` uses `log_info` but `auto_trade.py`, `websocket_handler.py`, and `data_service.py` use `print()`. | Multiple files |
| Q2 | P2 | **Duplicate code in `_convert_positions_to_dicts`**: The TP/SL sync logic with Binance client creation and fallback to DB is nearly identical in `websocket_handler.py:163-219` and `data_service.py:439-470`. Should be extracted to a shared service. | `websocket_handler.py`, `data_service.py` |
| Q3 | P2 | **`main_window.py:672` lines**: Despite delegation to managers, the main window still handles settings application, recovery config changes, position sync, and theme refreshing inline. `on_apply_settings()` alone is 60 lines of settings extraction and propagation. | `main_window.py` |
| Q4 | P3 | `ScanningConfig.enabled_scanners` typed as bare `list` — should be `List[str]` or `list[str]`. | `auto_trade_config.py:35` |
| Q5 | P3 | `AutoTradeSystem.stats` is a generic `Dict[str, Any]` mixing `datetime`, `int`, and `None` values. Should be a proper dataclass. | `main.py:68-74` |
| Q6 | P3 | Dead code: `_scan_for_signals()` returns `[]` twice — after the deprecation warning (line 245) and in the unreachable code below (line 248). | `main.py:244-248` |
| Q7 | P3 | `LayoutManager._populate_scanner_tab()` at 340+ lines defines a `ScannerControlAdapter` inner class, a `_push_scanner_config` closure, and builds 4 grid columns inline. This is the GUI's most complex layout method and should be decomposed. | `layout.py:146-486` |
| Q8 | P3 | `recovery_config` serialization logic is duplicated 3 times: in `on_apply_settings()`, `reload_current_settings()`, and `on_recovery_config_change()`. | `main_window.py:446-466, 510-531, 549-587` |

---

### GUI-Specific Findings

**Strengths:**
- **5-tab organization** (Dashboard, Scanner, Trading, Settings, Database) with keyboard shortcuts (F1, Ctrl+1-5, Ctrl+R, Ctrl+S, Ctrl+M).
- **Mode-aware UI**: Color-coded mode indicators (PRODUCTION/DEMO/DRY_RUN) in header, stats, and status bar.
- **WebSocket-driven updates**: Real-time position, balance, and order updates via `WebSocketDataService` with debounced trailing stop and negative breakeven handlers.
- **Shortcuts help dialog**: F1 shows all available keyboard shortcuts.
- **Context-aware shortcuts**: Ctrl+M only triggers scan when Scanner tab is active; Ctrl+Enter only trades when Trading tab is active.
- **StatusBar** with connection status, mode indicator, and last update timestamp.

**Issues:**

| # | Severity | Issue | Location |
|---|----------|-------|----------|
| G1 | P1 | **Thread safety**: `_thread_refresh_positions()` calls `data_service.get_positions()` which creates `BinanceClient` instances, but results are pushed to `_update_queue` which is processed on the main thread. However, `refresh_positions()` (called from keyboard shortcut) calls `data_service.get_positions()` directly on the main thread, potentially blocking the GUI for seconds during API calls. | `main_window.py:278-283 vs 307-311` |
| G2 | P1 | **WebSocket callback creates new `BinanceClient` per position** in `_convert_positions_to_dicts()`. With 3 open positions, this creates 3 new CCXT instances per WebSocket tick, each making HTTP calls. This will cause visible GUI lag. | `websocket_handler.py:170-175` |
| G3 | P2 | `_update_mode_display()` destroys and recreates the `ModeIndicator` widget on every call. If called frequently (e.g., during settings changes), this causes a visible flicker. Should update text/color rather than rebuild. | `main_window.py:253-257` |
| G4 | P2 | `on_apply_settings()` accesses `config_panel.recovery_panel.get_config()` with nested hasattr checks but no try/except around widget access. If the recovery panel hasn't been created yet (race during startup), it will raise. | `main_window.py:445-466` |
| G5 | P2 | `_handle_confirm_trade()` catches all exceptions silently (`except Exception: pass`). If the trade form's `_confirm_trade()` fails, user gets no feedback. | `main_window.py:222-226` |
| G6 | P3 | `_handle_escape()` only closes the first `CTkToplevel` found. If multiple dialogs are open, only one closes. | `main_window.py:184-189` |

---

### Positive Changes Since Last Review

1. ✅ **Emergency stop now persists to DynamoDB** — `RiskManager` loads `emergency_stop` from `get_system_state()` on init and writes via `set_system_state()` on trigger/reset.
2. ✅ **Retry logic added** — `OrderManager` and `OrderExecutor` use `@retry` with `stop_after_attempt(3)` and `wait_exponential` for all exchange API calls.
3. ✅ **Test suite expanded** — From 3 test files to 13 (10 in `tests/` + 3 at root).
4. ✅ **Fallback persistence** — `OrderManager` writes to `fallback_orders.jsonl` when DynamoDB persistence fails.
5. ✅ **SecretString wrapper** — Credentials in `OrderExecutor` and `OrderManager` use `SecretString` to prevent accidental logging.
6. ✅ **WebSocket trailing stop/negative BE handlers** — Debounced WebSocket-driven handlers replace pure timer-based polling.
7. ✅ **Scanner position gate** — Scanner checks both DB and Binance positions before running expensive Gemini calls.

---

### Priority Action Items

#### P0 — Critical (Fix Before Next Production Deploy)

| # | Action | Impact |
|---|--------|--------|
| 1 | **Cache `BinanceClient` instances** — Create once per mode change, reuse everywhere. Eliminate per-request instantiation in `WebSocketHandler._convert_positions_to_dicts()`, `DataService.get_positions()`, and `DataService.get_account_data()`. | Performance, Reliability |
| 2 | **Decouple TP/SL sync from WebSocket tick** — Move `TPSLSyncService.sync_position_tp_sl()` to a periodic background job (e.g., every 30s) instead of calling it on every position update. Cache results and serve from cache during WebSocket updates. | Performance, GUI responsiveness |
| 3 | **Unify Event systems** — Merge `monitoring/events.py` and `monitoring/event_system.py` into a single implementation. Currently pipeline events never reach GUI subscribers because they use different classes. | Architecture, Reliability |
| 4 | **Replace all `print()` with structured logging** — 76+ print statements across production code. Create a logging facade or use `log_info`/`log_error` consistently. | Security, Observability |

#### P1 — High (Fix Within Sprint)

| # | Action | Impact |
|---|--------|--------|
| 5 | **Write `CircuitBreaker` tests** — Cover all state transitions, thread safety, metrics, decorator, and context manager. | Testing |
| 6 | **Write `SignalPipeline` tests** — Mock all protocol dependencies, test pipeline orchestration, error handling, and circuit breaker integration. | Testing |
| 7 | **Write GUI unit tests** — At minimum: `DataService` mode switching, `SettingsManager` load/save/merge, `CredentialManager` credential lifecycle. | Testing |
| 8 | **Fix `signal_pipeline.py` missing import** — `setup_logging()` is called but never imported. Add explicit import or remove the call. | Reliability |
| 9 | **Remove or repurpose `main.py`** — Deprecated scaffold with TODO comments. Either wire it as a CLI entry point or remove it. | Code Quality |
| 10 | **Cache `DataService._reload_credentials()`** — Called on every `get_account_data()`. Cache result and only reload when settings change (listen to event). | Performance |

#### P2 — Medium (Fix in Backlog)

| # | Action | Impact |
|---|--------|--------|
| 11 | **Extract TP/SL sync to shared service** — Deduplicate between `websocket_handler.py` and `data_service.py`. | Code Quality |
| 12 | **Cache `RepositoryContext.from_env()`** — Use a module-level singleton or `functools.lru_cache`. | Performance |
| 13 | **Centralize symbol normalization** — Single `normalize_symbol()` function used everywhere. | Code Quality |
| 14 | **Use `deque(maxlen=N)` for event history** — Replace list slicing in `EventSystem`. | Performance |
| 15 | **Extract Scanner tab layout** — Move `_populate_scanner_tab()` to its own component class. | Code Quality |
| 16 | **Deduplicate recovery config serialization** — Extract to a helper in `RecoveryManager` or `SettingsHandler`. | Code Quality |
| 17 | **Stop storing API credentials in `settings.yaml`** — Only use `.env` file via `CredentialManager`. | Security |

#### P3 — Low (Improve When Touching)

| # | Action | Impact |
|---|--------|--------|
| 18 | Fix `ScanningConfig.enabled_scanners` type annotation. | Code Quality |
| 19 | Convert `AutoTradeSystem.stats` dict to a dataclass. | Code Quality |
| 20 | Remove dead code in `main.py: _scan_for_signals()`. | Code Quality |
| 21 | Update `_update_mode_display()` to update in-place instead of destroy/recreate. | GUI UX |
| 22 | Show error feedback in `_handle_confirm_trade()` instead of silently catching. | GUI UX |
| 23 | Handle multiple open dialogs in `_handle_escape()`. | GUI UX |

---

### Overall Score

| Category | Score | Trend |
|----------|-------|-------|
| Architecture | ⭐⭐⭐⭐ (4/5) | → Stable |
| Security | ⭐⭐⭐ (3/5) | → Stable |
| Performance | ⭐⭐⭐⭐ (4/5) | → Stable |
| Reliability | ⭐⭐⭐½ (3.5/5) | ↑ Improved |
| Testing | ⭐⭐⭐ (3/5) | ↑ Improved |
| Code Quality | ⭐⭐⭐⭐ (4/5) | → Stable |
| **Overall** | **⭐⭐⭐½ (3.6/5)** | **↑ Up from 3.2** |

**Bottom line:** The module has matured significantly since the first review. Emergency stop persistence, retry logic, expanded test suite, and WebSocket-driven real-time handlers are all solid additions. The two most impactful improvements remaining are: (1) caching `BinanceClient` instances to eliminate per-request instantiation, and (2) unifying the two Event systems so pipeline events reach the GUI. These two fixes alone would bring the system to a ⭐⭐⭐⭐ (4/5) overall rating.
