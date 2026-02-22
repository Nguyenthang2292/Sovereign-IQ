Kieemr# Codex Review Fixes — `modules/auto_trade`

## Goal

Address all issues from `codex-review-2026-02-22.md` in priority order (P0 → P1 → P2).

---

## P0 — Critical (Do Now)

- [x] **T1: Safety-critical tests** — Write pytest tests for the 5 untested financial modules
  - `test_trailing_stop.py` → cover long/short step triggers, BE level, multi-step escalation
  - `test_negative_breakeven.py` → cover trigger thresholds for both sides
  - `test_order_builder.py` → cover quantity rounding, TP/SL price construction
  - `test_order_validator.py` → cover rejection cases (leverage, size, symbol)
  - `test_risk_manager.py` → cover position sizing, emergency stop gate
  - Verify: `pytest modules/auto_trade/tests/` all pass

- [x] **T2: Add retry logic for exchange API calls** — Wrap `fetch_ticker`, `fetch_balance`, `create_order` in `order_manager.py` and `order_executor.py` with `tenacity` exponential backoff (3 attempts, max 10s)
  - Verify: unit test simulating network error recovers on 2nd attempt

- [x] **T3: Replace print() with logger** — Remove all 13+ `print()` / `traceback.print_exc()` calls in `order_executor.py`; use `logger.info()` / `logger.error()` / `logger.exception()`
  - Verify: `grep -n "print(" modules/auto_trade/execution/order_executor.py` returns 0 results

- [x] **T4: Persist emergency_stop flag** — In `risk_manager.py`, store the flag via `set_system_state("emergency_stop", True)` on trigger and load it in `__init__` via `get_system_state("emergency_stop")`
  - Verify: unit test proves flag survives a fresh `RiskManager()` instantiation after being set

---

## P1 — Important (This Sprint)

- [x] **T5: Wire or deprecate `main.py`** — Connect `SignalPipeline` + `OrderExecutor` inside `main.py`'s `_scan_for_signals()` and `_execute_signal()` stubs, or add a deprecation notice + pointer to the real entrypoint
  - Verify: running `python -m modules.auto_trade.main` executes one scan loop without errors

- [x] **T6: Deduplicate symbol normalization** — Create/extend `modules.common.domain.symbols.normalize_symbol()` and replace inline normalization in `order_executor.py` (lines 61–70, 190–195) and `order_manager.py` (line 221)
  - Verify: `grep -rn "replace.*USDT\|split.*/" modules/auto_trade/execution/` returns 0 results

- [x] **T7: Reuse BinanceClient instance** — Move `BinanceClient(...)` construction from `execute_from_signal()` into `OrderExecutor.__init__()` as `self._client`; update all call sites
  - Verify: `grep -n "BinanceClient(" modules/auto_trade/execution/order_executor.py` shows exactly 1 match (in `__init__`)

- [x] **T8: DB persistence fallback** — In `order_manager.py`'s DB persistence except-block, write the failed order JSON to `~/.auto_trade/fallback_orders.jsonl` for manual reconciliation
  - Verify: unit test that mocks DB failure finds a `.jsonl` entry written to disk

- [x] **T9: Recovery/strategy tests** — Write pytest tests for:
  - `test_recovery_manager.py` → cover activation, step escalation, deactivation
  - `test_gradual_recovery.py` → cover lot-size progression
  - `test_martingale.py` → cover multiplier calculation, max-step capping
  - Verify: `pytest modules/auto_trade/tests/` all pass

---

## P2 — Nice to Have (Next Sprint)

- [x] **T10: SecretString wrapper** — Create `modules/auto_trade/security/secret_string.py` with a `SecretString` class that redacts `__repr__`/`__str__`; apply to `_api_key` and `_api_secret` in `OrderExecutor`, `OrderManager`, `BinanceClient`
  - Verify: printing `OrderExecutor._api_key` explicitly returns `***`
- [x] **T11: Update `.env.example`** — Remove `DB_BACKEND=sqlite` line; replace with DynamoDB env-var comment block
  - Verify: `DB_BACKEND` string no longer appears in `.env.example`
- [x] **T12: Remove backup files** — Delete `core/atc_scanner.py.backup` and `settings.yaml.backup` from version control (`git rm`)
  - Verify: `git status` shows them as deleted
- [x] **T13: Fix deprecated UTC datetime calls** — Replace all deprecated UTC calls in `recovery_manager.py` (and any other files) with `datetime.now(timezone.utc)`
  - Verify: no deprecated UTC datetime call patterns remain in `modules/auto_trade/`
- [x] **T14: WebSocket heartbeat** — Add an application-level staleness check in `websocket/client.py` that flags the stream if no message is received for N seconds
  - Verify: simulate a 5-minute silence and verify a warning log is generated
- [x] **T15: Hash cache keys** — Replace the long comma-separated symbol string in `atc_scanner.py` cache key with `hashlib.md5(symbol_key.encode()).hexdigest()`
  - Verify: log output shows short hashes instead of 30+ symbol tickers in cache statements

---

## Done When

- [x] All P0 tests written and passing (`pytest` green)
- [x] No `print()` calls remain in `order_executor.py`
- [x] `emergency_stop` survives process restart
- [x] `BinanceClient` created once per `OrderExecutor` instance
- [x] All P1 tests written and passing
