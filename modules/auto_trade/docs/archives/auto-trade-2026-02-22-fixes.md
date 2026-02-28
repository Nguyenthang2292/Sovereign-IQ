# Auto Trade — P0/P1 Fix Plan

## Goal

Fix 10 critical and high-severity issues identified in the CHANGELOG review. Ordered by impact and dependency.

---

## Phase 1: Event System Unification (Blocks everything else)

### Context

Two incompatible systems exist:

- `monitoring/events.py` → `EventBus` + `EventType` (uppercase string values, thread-safe RLock, `publish(Event)`)
- `monitoring/event_system.py` → `EventSystem` + `EventType` (lowercase string values, no lock, `publish(type, data)`)

**Winner: Keep `event_system.py` (`EventSystem`)** — it's used by GUI, `RecoveryManager`, `AutoTradeManager`, `ScannerManager`, and `WebSocketDataService`. Add thread-safety (RLock) to it.

- [x] **Task 1.1** — Add `RLock` to `EventSystem` in `monitoring/event_system.py`
  - Wrap `_subscribers` access in `subscribe()`, `unsubscribe()`, and `publish()` with `self._lock`
  - Replace `list` trim with `collections.deque(maxlen=1000)` in `_event_history`
  - Verify: existing `EventSystem` tests pass

- [x] **Task 1.2** — Migrate all imports from `events.py` to `event_system.py`
  - Search: `from modules.auto_trade.monitoring.events import`
  - Affected files: `signal_pipeline.py`, `lifecycle_handler.py`, `breakeven_manager.py`, `audit.py`, `health.py`
  - Map old `EventBus` → `EventSystem`, old `EventType` members → new ones (add missing: `PIPELINE_START`, `PIPELINE_COMPLETE`, `PIPELINE_ERROR`, `ORDER_CREATED`, `ORDER_FILLED`, `BREAK_EVEN_MOVED`, `CIRCUIT_OPEN`, `HEALTH_CHECK_FAILED`, `SIGNAL_REJECTED`)
  - Verify: `grep -r "from modules.auto_trade.monitoring.events" .` returns no results

- [x] **Task 1.3** — Delete `monitoring/events.py` after migration
  - Verify: no `ImportError` on `python -c "from modules.auto_trade.monitoring import event_system"`

---

## Phase 2: Cache BinanceClient (P0 — GUI Performance)

### Context

`BinanceClient` is created per-request in 4 hot paths:

- `WebSocketHandler._convert_positions_to_dicts()` — per position per WS tick
- `DataService.get_positions()` — per periodic refresh
- `DataService.get_account_data()` — per periodic refresh
- `WebSocketHandler._get_binance_client()` — per WS callback registration

- [x] **Task 2.1** — Add `_binance_client: Optional[BinanceClient]` cache to `DataService`
  - Create `DataService._get_or_create_client() -> Optional[BinanceClient]` that returns cached instance
  - Invalidate cache (set to `None`) in `_reload_credentials()` when credentials change
  - Replace all inline `BinanceClient(...)` constructions in `data_service.py` with `_get_or_create_client()`
  - Verify: `DataService.get_positions()` called 5 times creates only 1 `BinanceClient`

- [x] **Task 2.2** — Route `WebSocketHandler._get_binance_client()` through `DataService`
  - Replace the fallback `BinanceClient(...)` construction in `websocket_handler.py:62-79` with `self.parent.data_service._get_or_create_client()`
  - Remove the duplicate credential extraction in `WebSocketHandler._get_binance_client()`
  - Verify: `WebSocketHandler` no longer imports `BinanceClient` directly (or has a single import at top-level only)

---

## Phase 3: Decouple TP/SL Sync from WebSocket Tick (P0 — GUI Responsiveness)

### Context

`TPSLSyncService.sync_position_tp_sl()` (HTTP call) runs on every single WS position update tick inside `_convert_positions_to_dicts()`. With 3 positions open and updates every 2s, this is 1.5 HTTP calls/second blocking the GUI thread.

- [x] **Task 3.1** — Create a TP/SL cache in `DataService`
  - Add `_tpsl_cache: Dict[str, dict]` and `_tpsl_cache_time: Dict[str, float]` to `DataService.__init__()`
  - Add `DataService.get_cached_tpsl(symbol: str, ttl_seconds: int = 30) -> dict` — returns cache or fetches via `TPSLSyncService` and stores result
  - Verify: two calls within 30s with same symbol produce only 1 HTTP request

- [x] **Task 3.2** — Replace inline TP/SL sync in `websocket_handler.py` with cache lookup
  - In `_convert_positions_to_dicts()`, replace `TPSLSyncService.sync_position_tp_sl(...)` call with `self.parent.data_service.get_cached_tpsl(p.symbol)`
  - Remove the duplicate `BinanceClient` creation for TP/SL sync in `websocket_handler.py:170-175`
  - Verify: GUI position update no longer shows network latency spike on each WS tick

- [x] **Task 3.3** — Replace inline TP/SL sync in `data_service.get_positions()` with same cache
  - `data_service.py:444` — replace `TPSLSyncService.sync_position_tp_sl(...)` with `self.get_cached_tpsl(symbol)`
  - Verify: `get_positions()` called back-to-back produces a single TP/SL fetch (not two)

---

## Phase 4: Replace print() with Structured Logging (P0 — Security/Observability)

- [x] **Task 4.1** — Replace `print()` in `gui/main_window/auto_trade.py`
  - `from modules.common.ui.logging import log_info, log_error, log_warn`
  - Replace all `print(f"[AutoTrade] ...")` with `log_info(...)` etc.
  - Verify: running a dry-run cycle shows `[AutoTrade]` lines in the structured log pane, not stdout

- [x] **Task 4.2** — Replace `print()` in `gui/main_window/websocket_handler.py`
  - Replace all `print(f"[WebSocket] ...")` with proper log calls
  - Specifically: TP/SL sync debug prints (lines ~161, 191, 198, 201, 217) → `log_debug()`; errors → `log_error()`
  - Verify: no `print(` remains in the file (grep check)

- [x] **Task 4.3** — Replace `print()` in `gui/utils/data_service.py` and `gui/utils/settings_manager.py`
  - `data_service.py`: 12 print statements → `log_*` calls
  - `settings_manager.py`: 10 print statements → `log_*` calls  
  - Verify: `grep -n "^\s*print(" gui/utils/data_service.py gui/utils/settings_manager.py` returns 0 results

---

## Phase 5: P1 Code Fixes (Quick wins)

- [x] **Task 5.1** — Fix `signal_pipeline.py` missing `setup_logging` import
  - Check line 150: if `setup_logging()` is unused cruft, remove it; if needed, add `from modules.common.ui.logging import setup_logging`
  - Verify: `python -c "from modules.auto_trade.core.signal_pipeline import SignalPipeline"` produces no `NameError`

- [x] **Task 5.2** — Cache `DataService._reload_credentials()` result
  - Add `_credentials_loaded: bool = False` flag; only call `CredentialManager().load_credentials()` once
  - Listen for settings-save events (via `event_bus`) to invalidate and reload
  - Verify: `get_account_data()` called 10 times triggers only 1 filesystem `.env` read

- [x] **Task 5.3** — Cache `RepositoryContext.from_env()` as a module singleton
  - In `database/repository/context.py`, add a `_INSTANCE: Optional[RepositoryContext] = None` module-level var
  - `RepositoryContext.from_env()` → return `_INSTANCE` if not None, else create and cache
  - Verify: 3 separate calls to `from_env()` return the same object (`is` check)

- [x] **Task 5.4** — Remove/repurpose deprecated `main.py`
  - If keeping as CLI: wire `AutoTradeSystem.__init__` to real components, remove all TODO/scaffold methods
  - If removing: delete `main.py`, confirm no other file imports from it
  - Verify: `grep -r "from modules.auto_trade.main import\|from .main import" .` returns nothing

---

## Phase 6: Tests (P1)

- [x] **Task 6.1** — Write `tests/test_circuit_breaker.py`
  - Test CLOSED→OPEN on N failures, OPEN→HALF_OPEN after timeout, HALF_OPEN→CLOSED on success
  - Test thread safety: 10 concurrent threads calling `call()` while circuit opens
  - Test decorator and context manager interfaces
  - Test `sanitize_errors=True` hides exception details
  - Verify: `pytest tests/test_circuit_breaker.py -v` — all pass

- [x] **Task 6.2** — Write `tests/test_signal_pipeline.py`
  - Mock `ATCScannerLike`, `XGBoostFilterLike`, `GeminiIntegration`, `SignalSelector`, `SymbolManager`
  - Test: full pipeline run returns `FinalSignal`
  - Test: pipeline skips Gemini when no XGBoost candidates remain
  - Test: pipeline handles scanner exception gracefully (returns None)
  - Verify: `pytest tests/test_signal_pipeline.py -v` — all pass

- [x] **Task 6.3** — Write `tests/test_data_service.py`
  - Test `DataService(mode="DRY_RUN")` returns mock data without credentials
  - Test `DataService._get_or_create_client()` caching (mock `BinanceClient`)
  - Test `DataService.get_cached_tpsl()` TTL expiry
  - Verify: `pytest tests/test_data_service.py -v` — all pass

---

## Phase 7: Verification

- [x] Run full test suite → `pytest modules/auto_trade/tests/ -v --tb=short`
- [x] Run existing integration tests → `pytest test_execution_phase3.py test_pipeline.py -v`
- [x] Verify no orphan `print(` in production paths: `grep -rn "^\s*print(" modules/auto_trade --include="*.py" | grep -v test | grep -v __pycache__`
- [x] Verify single event system: `grep -rn "from modules.auto_trade.monitoring.events import" . | grep -v __pycache__` → 0 results
- [x] Verify no `BinanceClient` constructed inline in WS handler: check `websocket_handler.py` has no `BinanceClient(` outside of `_get_binance_client()`

---

## Done When

- [x] `monitoring/events.py` deleted, all code on `event_system.py`
- [x] `BinanceClient` created once per `DataService` lifecycle (cached)
- [x] TP/SL sync served from 30s cache, not per-WS-tick
- [x] `print()` → `log_*` in all non-test production files
- [x] `CircuitBreaker`, `SignalPipeline`, `DataService` have pytest coverage
- [x] Full test suite green

## Notes

- Run tests after each Phase — don't batch all 7 phases before testing
- Phase 1 (Event unification) must complete before Phase 6 tests (pipeline test imports EventType)
- Phase 2 (BinanceClient cache) must complete before Phase 3 (TP/SL cache uses the shared client)
- P2/P3 items (symbol normalization, scanner tab extraction, etc.) are tracked in CHANGELOG but NOT in this plan — tackle in a separate sprint
