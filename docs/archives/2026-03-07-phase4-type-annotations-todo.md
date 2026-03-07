# Phase 4 — Symbol Type Annotations To-Do

**Date:** 2026-03-07  
**Parent plan:** [2026-03-07-symbol-ordertype-codec-todo.md](../plans/2026-03-07-symbol-ordertype-codec-todo.md)  
**Status:** Completed  
**Approach:** Annotate bottom-up (private CCXT-leaf → converters → public API), run pyright after each batch.

---

## Convention

```python
from modules.common.domain.symbol_types import DbSymbol, CcxtSymbol, FuturesSymbol
```

| Type            | Format example           | When to use                                   |
|-----------------|--------------------------|-----------------------------------------------|
| `DbSymbol`      | `"BTCUSDT"`              | DB keys, position records stored in DynamoDB  |
| `CcxtSymbol`    | `"BTC/USDT"`             | CCXT spot / reconcile side                    |
| `FuturesSymbol` | `"BTC/USDT:USDT"`        | Passed directly to `exchange.*()` CCXT calls  |

---

## Batch 1 — Private CCXT-leaf helpers (no callers to update yet)

These functions already receive a converted symbol; adding the annotation costs nothing and immediately starts catching misuse.

### 1.1 `order_management.py` — `_fetch_all_open_orders`

```python
# BEFORE
def _fetch_all_open_orders(exchange: ccxt.binance, symbol: str) -> list:

# AFTER
def _fetch_all_open_orders(exchange: ccxt.binance, symbol: FuturesSymbol) -> list:
```

File: `modules/auto_trade/execution/binance/order_management.py` line 69  
Docstring already says `'BAND/USDT:USDT'` — annotation confirms it.

---

### 1.2 `order_management.py` — `_get_mark_price_from_exchange`

```python
# BEFORE
def _get_mark_price_from_exchange(exchange: ccxt.binance, symbol: str) -> Optional[float]:

# AFTER
def _get_mark_price_from_exchange(exchange: ccxt.binance, symbol: FuturesSymbol) -> Optional[float]:
```

File: line 37  
Calls `exchange.fetch_ticker(symbol)` directly — must be FuturesSymbol.

---

### 1.3 `order_management.py` — private method `_get_mark_price`

```python
# BEFORE
def _get_mark_price(self, symbol: str) -> Optional[float]:

# AFTER
def _get_mark_price(self, symbol: FuturesSymbol) -> Optional[float]:
```

File: line 138  
Thin wrapper over `_get_mark_price_from_exchange`.

---

### 1.4 `order_management.py` — `_ccxt_futures_symbol` return type

```python
# BEFORE
def _ccxt_futures_symbol(exchange: ccxt.binance, symbol: str) -> str:

# AFTER
def _ccxt_futures_symbol(exchange: ccxt.binance, symbol: str) -> FuturesSymbol:
```

File: line 52  
Return value is always `codec.to_futures(symbol)` → `FuturesSymbol`.  
All `ccxt_symbol_tp: str = _ccxt_futures_symbol(...)` call-sites should become `FuturesSymbol`.

---

### 1.5 `tp_sl_sync.py` — `_get_mark_price`

```python
# BEFORE
def _get_mark_price(client, symbol: str) -> Optional[float]:

# AFTER
def _get_mark_price(client, symbol: FuturesSymbol) -> Optional[float]:
```

File: `modules/auto_trade/gui/services/tp_sl_sync.py` line 170  
Calls `exchange.fetch_ticker(symbol)` directly.

---

### 1.6 `reconcile.py` — `_normalize_symbol` return type

```python
# BEFORE
def _normalize_symbol(symbol: str | None) -> str:

# AFTER
def _normalize_symbol(symbol: str | None) -> CcxtSymbol:
```

File: `modules/auto_trade/database/reconcile.py` line 16  
Returns `_SYMBOL_CODEC.to_ccxt(s)` → `CcxtSymbol`.

---

**Verify Batch 1:**
```powershell
pyright modules/auto_trade/execution/binance/order_management.py `
         modules/auto_trade/gui/services/tp_sl_sync.py `
         modules/auto_trade/database/reconcile.py
```
Expected: zero new errors (call-sites pass already-converted values).

---

## Batch 2 — Converters & callers of private helpers

### 2.1 Local variables holding `_ccxt_futures_symbol()` result

After 1.4 the return is `FuturesSymbol`. Update the 4 call-sites to reflect that:

| File | Line (approx) | Variable | Change |
|------|---------------|----------|--------|
| `order_management.py` | 193 | `ccxt_symbol_tp: str` | `ccxt_symbol_tp: FuturesSymbol` |
| `order_management.py` | 270 | `ccxt_symbol: str` | `ccxt_symbol: FuturesSymbol` |
| `order_management.py` | 413 | `ccxt_sym: str` | `ccxt_sym: FuturesSymbol` |
| `ensure_tp_sl_job.py` | ~215 | `ccxt_sym = _ccxt_futures_symbol(...)` | add `: FuturesSymbol` |

These are **local** variable annotations — no signature changes yet.

---

### 2.2 `order_management.py` — public `OrderManagement` methods

Public contract: callers always pass a DB key coming from DynamoDB records.

```python
# BEFORE
def modify_take_profit(self, symbol: str, ...) -> Optional[dict]:
def modify_stop_loss(self, symbol: str, ...) -> Optional[dict]:
def modify_tp_sl(self, symbol: str, ...) -> Optional[dict]:
def cancel_open_orders(self, symbol: str) -> Optional[dict]:

# AFTER
def modify_take_profit(self, symbol: DbSymbol, ...) -> Optional[dict]:
def modify_stop_loss(self, symbol: DbSymbol, ...) -> Optional[dict]:
def modify_tp_sl(self, symbol: DbSymbol, ...) -> Optional[dict]:
def cancel_open_orders(self, symbol: DbSymbol) -> Optional[dict]:
```

---

**Verify Batch 2:**
```powershell
pyright modules/auto_trade/execution/binance/order_management.py `
         modules/auto_trade/execution/ensure_tp_sl_job.py
```
Fix any new errors before proceeding. Expected pattern: callers passing a plain `str` where `DbSymbol` is now required → wrap with `DbSymbol(...)` at the call-site.

---

## Batch 3 — Mid-layer services

### 3.1 `ensure_tp_sl_job.py` — methods that receive DB keys

```python
# BEFORE
def _is_position_closed_on_binance(self, symbol: str) -> bool:
def _cleanup_closed_position(self, symbol: str, db_order: ...) -> None:

# AFTER
def _is_position_closed_on_binance(self, symbol: DbSymbol) -> bool:
def _cleanup_closed_position(self, symbol: DbSymbol, db_order: ...) -> None:
```

File: `modules/auto_trade/execution/ensure_tp_sl_job.py` lines 184, 203  
Both receive `symbol` straight from `db_order["symbol"]` — already a `DbSymbol`.

---

### 3.2 `tp_sl_sync.py` — public methods

```python
# BEFORE
def _filter_orders_for_symbol(open_orders: list, symbol: str) -> list:
def fetch_tp_sl_from_binance(client, symbol: str) -> Tuple[...]:
def sync_to_database(repo_context, symbol: str, ...) -> bool:

# AFTER
def _filter_orders_for_symbol(open_orders: list, symbol: FuturesSymbol) -> list:
def fetch_tp_sl_from_binance(client, symbol: DbSymbol) -> Tuple[...]:
def sync_to_database(repo_context, symbol: DbSymbol, ...) -> bool:
```

Notes:
- `_filter_orders_for_symbol` receives an already-converted `ccxt_symbol` → `FuturesSymbol`
- `fetch_tp_sl_from_binance` is called from the GUI with a DB key → `DbSymbol`
- `sync_to_database` writes to DB after calling `_codec.to_db()` internally → `DbSymbol`

---

### 3.3 `position_sync_service.py` — `_fetch_tp_sl_orders`

```python
# BEFORE
def _fetch_tp_sl_orders(client, symbol: str) -> tuple[Optional[float], Optional[float]]:

# AFTER
def _fetch_tp_sl_orders(client, symbol: DbSymbol) -> tuple[Optional[float], Optional[float]]:
```

File: `modules/auto_trade/gui/services/position_sync_service.py` line 94  
Delegates to `TPSLSyncService.fetch_tp_sl_from_binance(client, symbol)`.

---

**Verify Batch 3:**
```powershell
pyright modules/auto_trade/execution/ensure_tp_sl_job.py `
         modules/auto_trade/gui/services/tp_sl_sync.py `
         modules/auto_trade/gui/services/position_sync_service.py
```

---

## Batch 4 — order_manager.py

### 4.1 `_fetch_ticker` (private helper)

```python
# BEFORE
def _fetch_ticker(self, symbol: str) -> dict:

# AFTER
def _fetch_ticker(self, symbol: FuturesSymbol) -> dict:
```

File: `modules/auto_trade/execution/order_manager.py` line 115

### 4.2 Local variable at line ~298

The `symbol_db` variable is already assigned from `_codec.to_db(...)` in the existing code. Add the type annotation:

```python
# BEFORE
symbol_db: str = _codec.to_db(order.symbol or "")

# AFTER
symbol_db: DbSymbol = _codec.to_db(order.symbol or "")
```

---

**Verify Batch 4:**
```powershell
pyright modules/auto_trade/execution/order_manager.py
```

---

## Final verification

```powershell
# Full module check
pyright modules/auto_trade
```

Expected outcome: no `DbSymbol`/`FuturesSymbol`/`CcxtSymbol` mismatches remain at the annotated boundaries.

---

## Done When

- [x] All 4 batches annotated and pyright-clean for the annotated files.
- [x] Zero regression in existing tests:
  ```powershell
  pytest tests/common/domain/test_symbol_codec.py `
         tests/common/domain/test_order_type_codec.py `
         tests/auto_trade/execution/test_ensure_tp_sl_job.py `
         tests/auto_trade/execution/test_order_management.py `
         modules/auto_trade/tests/test_data_service.py -q
  ```
- [x] `DeprecationWarning` added to `normalize_symbol()` and `normalize_symbol_key()` in `symbols.py` (callers redirected to `SymbolCodec`).
