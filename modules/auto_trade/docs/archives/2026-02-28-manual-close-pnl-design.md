# Design: Manual Close PnL Fix + Legacy Cleanup

**Date:** 2026-02-28  
**Status:** Approved  
**Scope:** `modules/auto_trade`

---

## Problem

Khi user đóng position thủ công (không qua TP/SL), `RecoveryManager` nhận PnL = 0 thay vì giá trị thực.

**Root cause:** `_handle_position_update()` trong `WebSocketDataService` fallback về `db_order.get("pnl", 0.0)`,
field này luôn `NULL` vì PnL chỉ được set khi TP/SL fill.

**Impact:**

- Recovery không được kích hoạt sau manual close lỗ
- DB không lưu PnL thực cho closed positions (cả TP/SL path cũng bị)

---

## Task A: Fix Manual Close PnL

### Approach

Sau khi detect `position_amt == 0`, delay 1s rồi gọi Binance Income API
(`GET /fapi/v1/income?incomeType=REALIZED_PNL`) để lấy realized PnL chính xác.

### Fallback chain

```
1. Binance Income API  →  pnl_from_api       (ưu tiên: chính xác nhất)
2. position.unrealized_pnl                   (fallback tốt: snapshot ngay trước close)
3. 0.0                                       (worst case: behavior hiện tại)
```

### New method: `_fetch_realized_pnl_from_binance()`

```python
def _fetch_realized_pnl_from_binance(
    self,
    symbol: str,              # e.g. "BTCUSDT" (no slash)
    delay_ms: int = 1000,     # wait before calling API
    lookback_seconds: int = 30,  # time window to match income entry
) -> Optional[float]:
    """
    Fetch realized PnL from Binance REALIZED_PNL income history.
    Returns PnL in USDT or None if fetch fails.
    """
    import time
    try:
        time.sleep(delay_ms / 1000)

        from modules.auto_trade.execution.binance import BinanceClient
        _client = BinanceClient(
            api_key=self.api_key,
            api_secret=self.api_secret,
            testnet=self.testnet,
            dry_run=False,
        )

        since_ms = int((time.time() - 300) * 1000)  # last 5 min
        response = _client.exchange.fapiPrivateGetIncome({
            "symbol": symbol,
            "incomeType": "REALIZED_PNL",
            "startTime": since_ms,
            "limit": 5,
        })

        if not response:
            return None

        now_ms = time.time() * 1000
        recent = [
            e for e in response
            if (now_ms - float(e.get("time", 0))) < (lookback_seconds * 1000)
        ]

        if not recent:
            return None

        return float(recent[-1].get("income", 0.0))

    except Exception as e:
        log_error(f"Failed to fetch realized PnL from income API for {symbol}: {e}")
        return None
```

### Changes in `_handle_position_update()`

```python
# BEFORE (line 376):
pnl_value = float(db_order.get("pnl", 0.0) or 0.0)

# AFTER:
pnl_from_api = self._fetch_realized_pnl_from_binance(symbol_normalized)
pnl_value = (
    pnl_from_api
    if pnl_from_api is not None
    else position.unrealized_pnl  # fallback: last snapshot before close
)
```

### Changes in `update_order_status()` calls — both paths

**Manual close path** (`_handle_position_update`, line 369):

```python
# BEFORE:
ctx.orders.update_order_status(order_id, "CLOSED")

# AFTER:
ctx.orders.update_order_status(order_id, "CLOSED", pnl=pnl_value)
```

**TP/SL fill path** (`_handle_order_update`, line 495):

```python
# BEFORE:
ctx.orders.update_order_status(order_id, "CLOSED")

# AFTER:
ctx.orders.update_order_status(order_id, "CLOSED", pnl=effective_pnl)
```

### Files changed

- `modules/auto_trade/gui/utils/websocket_data_service.py`
  - Add `_fetch_realized_pnl_from_binance()` method
  - Fix `_handle_position_update()` — PnL fetch + DB sync
  - Fix `_handle_order_update()` — DB sync with PnL

---

## Task B: Remove `PositionLifecycleHandler` (Dead Code)

### Why safe to delete

- **No callers:** Not imported anywhere in `gui/`, `run_gui.py`, or any active module
- **Replaced by:** `WebSocketDataService` + `RecoveryManager` + `EventBus` architecture
- **Dependency on Martingale:** `MartingaleStrategy` which itself is superseded by `GradualRecoveryStrategy`

### Files changed

1. **Delete:** `modules/auto_trade/monitoring/lifecycle_handler.py`
2. **Edit:** `modules/auto_trade/monitoring/__init__.py`
   - Remove: `from modules.auto_trade.monitoring.lifecycle_handler import PositionLifecycleHandler`
   - Remove: `"PositionLifecycleHandler"` from `__all__`

---

## Testing

### Task A

- Unit test: `_fetch_realized_pnl_from_binance()` với mock exchange response
- Unit test: fallback to `position.unrealized_pnl` khi API returns None
- Integration check: Sau manual close, DB record có `pnl != NULL`
- Integration check: `RecoveryManager` nhận đúng PnL và kích hoạt recovery

### Task B

- `pytest --collect-only` — verify không có test nào import `PositionLifecycleHandler`
- `grep -r "PositionLifecycleHandler"` — verify 0 references còn lại sau xóa

---

## Implementation Order

1. Task B (xóa legacy) — đơn giản, không risk
2. Task A (PnL fix) — implement + test
