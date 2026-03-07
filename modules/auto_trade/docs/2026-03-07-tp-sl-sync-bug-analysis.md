# Bug Analysis: TP/SL Conditional Orders Không Đóng Đồng Bộ Sau Khi Position Closed

> **Created:** 2026-03-07  
> **Phạm vi:** `run_auto_trade_gui.py` → toàn bộ pipeline WebSocket + DB sync  
> **Severity:** HIGH — các lệnh TP/SL orphan vẫn mở trên Binance, tốn margin, có thể trigger unexpected  
> **Status:** Open — chưa fix

---

## 1. Kiến Trúc Tổng Quan — Luồng Đóng Lệnh

```
Binance Event
     │
     ├─ [Path A] TP/SL order fills
     │       └─ WebSocket ORDER update (type=TAKE_PROFIT_MARKET/STOP_MARKET, status=closed)
     │               └─ _handle_order_update()   ← xử lý chính
     │                       ├─ cancel_open_orders() ← hủy sibling (TP hoặc SL còn lại)
     │                       └─ DB update_order_status(CLOSED) + emit POSITION_CLOSED
     │
     └─ [Path B] Manual close (không qua TP/SL)
             └─ WebSocket POSITION update (position_amt → 0)
                     └─ _handle_position_update()  ← safety net
                             ├─ cancel_open_orders() ← hủy cả TP lẫn SL
                             └─ DB update_order_status(CLOSED) + emit POSITION_CLOSED
```

Ngoài ra còn có 2 cơ chế defensive backup:

- **`EnsureTPSLJob.run()`** — chạy định kỳ, detect stale position và gọi `_cleanup_closed_position()`
- **`PositionSyncService.sync_all_positions()`** — chạy khi startup / manual sync, Phase 2 đóng stale DB orders

---

## 2. Các Bug Đã Xác Định

### 🔴 BUG #1 (CRITICAL) — Race Condition: Path A vs Path B

**File:** `modules/auto_trade/gui/services/websocket_data_service.py` dòng 377–461 và 507–582

**Mô tả:**
Khi TP fill, Binance gửi **đồng thời** 2 events:

1. `ORDER_TRADE_UPDATE` → `_handle_order_update()` (Path A)
2. `ACCOUNT_UPDATE` với `positions[].positionAmt = 0` → `_handle_position_update()` (Path B)

Thứ tự arrive không đảm bảo. Nếu **Path B arrive trước Path A**:

- Path B thấy DB còn `OPEN` → chạy `cancel_open_orders()` OK
- Path B update DB → `CLOSED`
- Path A arrive → `get_open_positions()` trả về `[]` (đã CLOSED)
- Path A **bỏ qua** cancel sibling orders vì vòng lặp `for db_order in db_orders` rỗng

**Hậu quả:** Sibling order (TP hoặc SL còn lại) **KHÔNG được cancel**.

**Code gốc (lỗi):**

```python
# websocket_data_service.py dòng 538-544
db_orders = ctx.orders.get_open_positions(symbol=symbol_normalized)

for db_order in db_orders:    # ← nếu Path B đã CLOSED trước, danh sách này rỗng
    order_id = db_order.get("order_id")
    ...
    ctx.orders.update_order_status(order_id, "CLOSED", pnl=effective_pnl)
    # cancel_open_orders() chỉ được gọi trước vòng này (dòng 527), không bị ảnh hưởng
    # nhưng context là: cancel chỉ được gọi khi is_tp_sl_fill = True
    # Tức là nếu Path B đã cancel trước, Path A skip cancel luôn
```

**Điều kiện kích hoạt bug:**

- `ACCOUNT_UPDATE` arrive trước `ORDER_TRADE_UPDATE` (phổ biến trên Binance WS, thứ tự này hay xảy ra)
- DynamoDB write nhanh (< 50ms) — thực tế thường < 10ms trong ap-southeast-1

---

### 🔴 BUG #2 (HIGH) — `cancel_open_orders` Fail Silently

**File:** `websocket_data_service.py` dòng 404–418 và 526–530

```python
try:
    cancel_res = _client.cancel_open_orders(symbol_normalized)
except Exception as _ce:
    log_warn(f"[WS Data] cancel_open_orders non-fatal: {_ce}")
    # ← exception swallowed, flow tiếp tục như thể không có gì xảy ra
```

Khi cả 2 path cùng chạy, một path sẽ nhận lỗi `"Order does not exist"` từ Binance (order đã bị cancel bởi path kia). Exception bị `log_warn` → không retry, không escalate. Nếu `cancel_open_orders` partial fail (cancel được 1 trong 2 conditional orders rồi exception), **1 conditional order vẫn còn mở**.

---

### 🔴 BUG #3 (HIGH) — `EnsureTPSLJob._cleanup_closed_position` Key Lookup Sai

**File:** `modules/auto_trade/execution/ensure_tp_sl_job.py` dòng 244–248

```python
order_id = db_order.get("order_id") or db_order.get("pk") or db_order.get("id")
if order_id:
    repo.orders.update_order_status(order_id, "CLOSED")
else:
    log_warn(f"[EnsureTPSL] Could not determine order_id for {symbol} to update DB")
    # ← DB record không được update CLOSED, job bị skip silently
```

`db_order` là dict từ DynamoDB. Nếu item dùng key field khác với `"order_id"`, `"pk"`, `"id"` (e.g., DynamoDB partition key uppercase `"PK"`), tất cả `get()` đều trả về `None` → **DB record không bao giờ được đánh dấu CLOSED**.

Consequence: Mỗi lần `EnsureTPSLJob` chạy, nó vẫn thấy position "OPEN" trong DB, gọi `_is_position_closed_on_binance()`, xác nhận là closed, nhưng không thể update DB → infinite loop cleanup attempts.

---

### 🟡 BUG #4 (MEDIUM) — `PositionSyncService` Cancel với Sai Symbol Format

**File:** `modules/auto_trade/gui/services/position_sync_service.py` dòng 248–256

```python
ccxt_sym = db_symbol   # "BTCUSDT" ← plain format, WRONG
try:
    client.exchange.cancel_all_orders(ccxt_sym)  # CCXT Binance Futures cần "BTC/USDT:USDT"
except Exception:
    pass  # ← exception swallowed hoàn toàn
# Fallback: cancel_open_orders(db_symbol) → có gọi _ccxt_futures_symbol() → đúng format
```

`cancel_all_orders` được gọi với `BTCUSDT` thay vì `BTC/USDT:USDT`. Nếu CCXT chưa load market data cho symbol này, nó raise `BadSymbol` exception → bị `pass` → **conditional orders KHÔNG bị cancel trong path này**. Fallback `cancel_open_orders` có sử dụng `_ccxt_futures_symbol()` nên thực ra đúng, nhưng:

- Sub-bug: Nếu `cancel_all_orders` cancel được một số orders trước khi raise, sau đó `cancel_open_orders` gọi `_fetch_all_open_orders` và thấy một phần đã gone → inconsistent log

---

### 🟡 BUG #5 (MEDIUM) — `POSITION_CLOSED` Event Không Publish Khi `client_order_id = None`

**File:** `websocket_data_service.py` dòng 436 và 561

```python
if self.event_bus and client_order_id and client_order_id not in self._published_closed_events:
    self.event_bus.publish(EventType.POSITION_CLOSED, {...})
```

**Điều kiện bug:** `client_order_id` là `None` (các DB record cũ tạo trước khi có AT_ prefix scheme, hoặc SYNCED records từ `PositionSyncService` có `client_order_id = "SYNC_BTCUSDT_1234567890"`).

**Hậu quả:** DB được update `CLOSED` thành công, nhưng `RecoveryManager` và các subscriber khác **không nhận được event** → Gradual Recovery không trigger → người dùng không thấy notification.

---

### 🟡 BUG #6 (LOW) — `_published_closed_events` Memory Leak

**File:** `websocket_data_service.py` dòng 83

```python
self._published_closed_events: set[str] = set()  # không có TTL, không clear
```

In-memory set accumulate `client_order_id` strings mãi mãi. App chạy liên tục nhiều ngày sẽ tích lũy. Không gây bug logic nhưng là memory leak nhỏ và gây khó debug.

---

## 3. Call Flow Diagram — Scenario "TP Fill" (Bug #1 kích hoạt)

```
Binance WS fires 2 events simultaneously:
  [E1] ORDER_TRADE_UPDATE { type=TAKE_PROFIT_MARKET, status=closed, symbol=BTCUSDT }
  [E2] ACCOUNT_UPDATE { positions: [{ symbol=BTCUSDT, positionAmt=0 }] }

asyncio event loop (single thread) trong WebSocket background thread:
  │
  ├── [E2 arrives first — common on Binance]
  │   PositionMonitor._handle_ws_position_update([E2])
  │       └── WebSocketDataService._handle_position_update(PositionSnapshot{amt=0})
  │               ├── get_open_positions("BTCUSDT") → [{ status: "OPEN", order_id: "123" }]
  │               ├── pending_orders = [{ "OPEN" record }]
  │               ├── BinanceClient.cancel_open_orders("BTCUSDT") ← cancel SL conditional order
  │               │       └── OrderManagement.cancel_open_orders()
  │               │               └── _fetch_all_open_orders() [basic + conditional]
  │               │                       → cancel TP + SL both
  │               ├── update_order_status("123", "CLOSED", pnl=X) ← DB now CLOSED
  │               └── event_bus.publish(POSITION_CLOSED, ...)
  │
  └── [E1 arrives 20ms later]
      OrderMonitor._handle_ws_order_update([E1])
          └── WebSocketDataService._handle_order_update(OrderSnapshot)
                  ├── is_tp_sl_fill = True
                  ├── BinanceClient.cancel_open_orders("BTCUSDT")
                  │       → log_warn "Order does not exist" (already cancelled by Path B)
                  │       → exception swallowed ← BUG #2
                  └── ctx.orders.get_open_positions("BTCUSDT")
                          → [] (already CLOSED by Path B)  ← BUG #1 consequence
                          → for loop is EMPTY
                          → NO DB update, NO event publish
                          → [sibling cancel already done by Path B, so OK here]
                          → [but if Path B's cancel partially failed, sibling still open]
```

**Kết luận race condition:**

- Nếu Path B cancel thành công toàn bộ → OK (dù Path A skip)
- Nếu Path B cancel partial fail → **sibling conditional order còn trên Binance**
- Path A sẽ không retry cancel vì nó thấy DB đã CLOSED

---

## 4. Root Causes Theo Mức Độ Nghiêm Trọng

| # | Root Cause | Severity | File | Dòng |
|---|---|---|---|---|
| 1 | Race condition Path A vs B — cancel sibling bị bỏ qua khi DB đã CLOSED | **CRITICAL** | `websocket_data_service.py` | 507–582 |
| 2 | `cancel_open_orders` exception bị swallow, không retry | **HIGH** | `websocket_data_service.py` | 527, 415 |
| 3 | `EnsureTPSLJob._cleanup` dùng sai key để lookup order_id trong DB | **HIGH** | `ensure_tp_sl_job.py` | 244–251 |
| 4 | `PositionSyncService` gọi `cancel_all_orders` với plain symbol format | **MEDIUM** | `position_sync_service.py` | 248–253 |
| 5 | `POSITION_CLOSED` event không publish khi `client_order_id = None` | **MEDIUM** | `websocket_data_service.py` | 436, 561 |
| 6 | `_published_closed_events` không clear, memory leak | **LOW** | `websocket_data_service.py` | 83 |

---

## 5. Fix Recommendations

### Fix #1 + #2 (CRITICAL+HIGH) — Tách Cancel Khỏi DB Check + Per-Symbol Lock

**Strategy:** Gộp cancel + DB update vào 1 method dùng chung cho cả Path A và B, bảo vệ bằng `threading.Lock` per symbol.

```python
# websocket_data_service.py

import threading
from typing import Dict

class WebSocketDataService:
    def __init__(self, ...):
        ...
        self._close_locks: Dict[str, threading.Lock] = {}

    def _get_close_lock(self, symbol: str) -> threading.Lock:
        """Per-symbol lock để eliminate race giữa Path A và Path B."""
        if symbol not in self._close_locks:
            self._close_locks[symbol] = threading.Lock()
        return self._close_locks[symbol]

    def _cancel_and_close_position(
        self,
        symbol: str,
        pnl: Optional[float],
        exit_price: float,
        entry_price: float,
        leverage: int,
        source: str,
    ) -> None:
        """
        Idempotent: cancel Binance conditional orders + update DB to CLOSED.
        Dùng chung cho cả Path A (TP/SL fill) và Path B (manual close).
        """
        # STEP 1: Cancel ALL conditional orders trên Binance (unconditional, không cần DB)
        if self.api_key and self.mode != "DRY_RUN":
            try:
                _client = BinanceClient(
                    api_key=self.api_key,
                    api_secret=self.api_secret,
                    testnet=self.testnet,
                    dry_run=False,
                )
                cancel_res = _client.cancel_open_orders(symbol)
                log_info(f"[WS Data] Cancelled conditional orders for {symbol}: {cancel_res}")
            except Exception as _ce:
                log_warn(f"[WS Data] cancel_open_orders({symbol}) non-fatal: {_ce}")
                # Note: log_warn vẫn OK ở đây vì EnsureTPSLJob sẽ catch up nếu cancel fail

        # STEP 2: Update DB (chỉ update OPEN records — idempotent)
        try:
            from modules.auto_trade.database.repository.context import RepositoryContext
            from modules.auto_trade.monitoring.event_system import EventType

            ctx = RepositoryContext.from_env()
            db_orders = ctx.orders.get_open_positions(symbol=symbol)

            for db_order in db_orders:  # Chỉ có OPEN records
                order_id = db_order.get("order_id")
                client_order_id = db_order.get("client_order_id")
                if not order_id:
                    continue

                ctx.orders.update_order_status(order_id, "CLOSED", pnl=pnl)
                log_info(f"[WS Data] DB CLOSED for {symbol} (order={order_id}, pnl={pnl:+.2f}, src={source})")

                # Publish event với fallback dedup key
                dedup_key = client_order_id or f"order_{order_id}"
                if self.event_bus and dedup_key not in self._published_closed_events:
                    self.event_bus.publish(
                        EventType.POSITION_CLOSED,
                        {
                            "symbol": symbol,
                            "pnl": pnl or 0.0,
                            "is_profit": (pnl or 0.0) >= 0,
                            "exit_price": exit_price,
                            "entry_price": entry_price,
                            "leverage": leverage,
                            "duration_seconds": 0,
                            "is_programmatic": True,
                        },
                        source=f"WebSocketDataService ({source})",
                    )
                    self._published_closed_events.add(dedup_key)
                    log_info(f"[WS Data] POSITION_CLOSED published for {symbol} (pnl={pnl:+.2f})")

        except Exception as _db_err:
            log_error(f"[WS Data] DB/event error for {symbol}: {_db_err}", exc_info=True)

    def _handle_order_update(self, order: OrderSnapshot) -> None:
        ...
        if is_tp_sl_fill:
            symbol_normalized = _SYMBOL_CODEC.to_db(order.symbol)
            with self._get_close_lock(symbol_normalized):  # ← Lock per symbol
                self._cancel_and_close_position(
                    symbol=symbol_normalized,
                    pnl=order.realized_pnl,
                    exit_price=order.price,
                    entry_price=0.0,  # sẽ lấy từ db_order trong method
                    leverage=1,
                    source="TP/SL fill",
                )

    def _handle_position_update(self, position: PositionSnapshot) -> None:
        if position.position_amt == 0:
            symbol_normalized = _SYMBOL_CODEC.to_db(position.symbol)
            with self._get_close_lock(symbol_normalized):  # ← Cùng lock
                self._cancel_and_close_position(
                    symbol=symbol_normalized,
                    pnl=position.unrealized_pnl,
                    exit_price=position.mark_price,
                    entry_price=position.entry_price,
                    leverage=position.leverage,
                    source="manual close",
                )
        # Pass to GUI callbacks (always)
        for callback in self._position_callbacks:
            ...
```

---

### Fix #3 (HIGH) — Fix Key Lookup trong `EnsureTPSLJob._cleanup_closed_position`

```python
# ensure_tp_sl_job.py dòng 244–248

# TRƯỚC:
order_id = db_order.get("order_id") or db_order.get("pk") or db_order.get("id")

# SAU — thêm DynamoDB common key patterns và log diagnostic:
order_id = (
    db_order.get("order_id")     # standard field
    or db_order.get("OrderId")   # PascalCase variant
    or db_order.get("pk")        # DynamoDB PK alias lowercase
    or db_order.get("PK")        # DynamoDB PK alias uppercase
    or db_order.get("id")        # generic id
)
if not order_id:
    log_error(
        f"[EnsureTPSL] Cannot determine order_id for {symbol}. "
        f"Available keys: {list(db_order.keys())}. "
        f"Check DynamoDB schema in RepositoryContext."
    )
    return  # Không thể fix mà không biết key, cần manual investigation
```

**Lâu dài:** Inspect `RepositoryContext.orders` DynamoDB implementation để xác định chính xác field name và hardcode đúng.

---

### Fix #4 (MEDIUM) — Fix Symbol Format trong `PositionSyncService`

```python
# position_sync_service.py dòng 246–260

# TRƯỚC:
ccxt_sym = db_symbol   # "BTCUSDT" ← WRONG
try:
    client.exchange.cancel_all_orders(ccxt_sym)
except Exception:
    pass

# SAU:
try:
    from modules.auto_trade.execution.binance.order_management import _ccxt_futures_symbol
    ccxt_sym = _ccxt_futures_symbol(client.exchange, db_symbol)  # "BTC/USDT:USDT"
    client.exchange.cancel_all_orders(ccxt_sym)
    log_info(f"[PositionSync] cancel_all_orders succeeded for {db_symbol} (ccxt={ccxt_sym})")
except Exception as cancel_all_err:
    log_warn(f"[PositionSync] cancel_all_orders failed for {db_symbol}: {cancel_all_err}. Trying per-order cancel...")
    # Fallback: cancel_open_orders đã đúng vì dùng _ccxt_futures_symbol() bên trong
    try:
        cancel_res = client.cancel_open_orders(db_symbol)
        log_info(f"[PositionSync] Fallback cancel_open_orders for {db_symbol}: {cancel_res}")
    except Exception as fallback_err:
        log_error(f"[PositionSync] Both cancel methods failed for {db_symbol}: {fallback_err}")
```

---

### Fix #5 (MEDIUM) — Handle `client_order_id = None` khi Publish Event

```python
# websocket_data_service.py dòng 436 và 561

# TRƯỚC:
if self.event_bus and client_order_id and client_order_id not in self._published_closed_events:
    self.event_bus.publish(EventType.POSITION_CLOSED, {...})
    self._published_closed_events.add(client_order_id)

# SAU — dùng order_id làm fallback dedup key:
dedup_key = client_order_id or (f"binance_{order_id}" if order_id else None)
if self.event_bus:
    if dedup_key and dedup_key not in self._published_closed_events:
        self.event_bus.publish(EventType.POSITION_CLOSED, {...})
        self._published_closed_events.add(dedup_key)
    elif not dedup_key:
        # Không có dedup key — publish nhưng không track (chấp nhận potential duplicate)
        log_warn(f"[WS Data] POSITION_CLOSED published without dedup key for {symbol_normalized}")
        self.event_bus.publish(EventType.POSITION_CLOSED, {...})
```

---

## 6. Files Cần Sửa (Theo Mức Độ Ưu Tiên)

| File | Priority | Bugs Fixed | Estimated Effort |
|---|---|---|---|
| `modules/auto_trade/gui/services/websocket_data_service.py` | **P0** | #1, #2, #5, #6 | 2–3h |
| `modules/auto_trade/execution/ensure_tp_sl_job.py` | **P1** | #3 | 30m |
| `modules/auto_trade/gui/services/position_sync_service.py` | **P1** | #4 | 30m |

---

## 7. Test Cases Cần Viết

```python
# modules/auto_trade/tests/test_tp_sl_sync.py
import pytest
from unittest.mock import MagicMock, patch

def test_tp_fill_cancels_sibling_sl_even_when_db_already_closed():
    """
    BUG #1: Path A (TP fill) phải cancel sibling SL kể cả khi DB đã CLOSED bởi Path B.
    """
    # Setup: DB đã CLOSED (mô phỏng Path B đã chạy trước)
    mock_ctx = MagicMock()
    mock_ctx.orders.get_open_positions.return_value = []  # đã CLOSED rồi

    mock_binance = MagicMock()

    with patch("modules.auto_trade.database.repository.context.RepositoryContext.from_env", return_value=mock_ctx):
        service = WebSocketDataService(mode="PRODUCTION", event_bus=None)
        service.api_key = "test_key"
        service._binance_client = mock_binance

        # Simulate TP fill order snapshot
        order = MagicMock()
        order.status = "closed"
        order.raw_info = {"type": "TAKE_PROFIT_MARKET", "stopPrice": "50000"}
        order.symbol = "BTC/USDT:USDT"

        service._handle_order_update(order)

    # Verify: cancel PHẢI được gọi dù DB đã CLOSED
    mock_binance.cancel_open_orders.assert_called_once()


def test_manual_close_always_cancels_before_db_check():
    """Path B phải cancel conditional orders trước khi check DB state."""
    call_order = []

    mock_cancel = MagicMock(side_effect=lambda *a, **kw: call_order.append("cancel"))
    mock_db_check = MagicMock(side_effect=lambda *a, **kw: call_order.append("db_check") or [])

    # ... assert call_order == ["cancel", "db_check"]


def test_cleanup_position_logs_error_when_order_id_missing():
    """
    BUG #3: EnsureTPSLJob không được fail silently khi order_id không tìm được.
    """
    job = EnsureTPSLJob(settings_manager=MagicMock(), binance_client=MagicMock())
    db_order = {"symbol": "BTCUSDT", "unknown_key": "value"}  # không có order_id

    with patch("modules.auto_trade.common.ui.logging.log_error") as mock_log_error:
        job._cleanup_closed_position(DbSymbol("BTCUSDT"), db_order)
        mock_log_error.assert_called_once()
        assert "Cannot determine order_id" in mock_log_error.call_args[0][0]


def test_position_sync_uses_ccxt_format_for_cancel():
    """
    BUG #4: cancel_all_orders phải nhận CCXT futures format (BTC/USDT:USDT), không phải plain (BTCUSDT).
    """
    mock_client = MagicMock()
    mock_client.exchange.cancel_all_orders = MagicMock()

    with patch("modules.auto_trade.execution.binance.order_management._ccxt_futures_symbol",
               return_value="BTC/USDT:USDT") as mock_converter:
        PositionSyncService.sync_all_positions(mock_client)

    # Verify conversion được gọi trước cancel_all_orders
    mock_converter.assert_called()
    call_args = mock_client.exchange.cancel_all_orders.call_args[0][0]
    assert call_args == "BTC/USDT:USDT", f"Expected CCXT format, got: {call_args}"


def test_position_closed_event_published_without_client_order_id():
    """
    BUG #5: POSITION_CLOSED event phải được publish kể cả khi client_order_id = None.
    """
    mock_event_bus = MagicMock()
    mock_ctx = MagicMock()
    mock_ctx.orders.get_open_positions.return_value = [
        {"order_id": "123456", "client_order_id": None, "entry_price": 50000}
    ]

    service = WebSocketDataService(mode="PRODUCTION", event_bus=mock_event_bus)
    service.api_key = ""  # skip cancel

    with patch("modules.auto_trade.database.repository.context.RepositoryContext.from_env", return_value=mock_ctx):
        service._cancel_and_close_position("BTCUSDT", pnl=-10.0, exit_price=49000, entry_price=50000, leverage=5, source="test")

    # Verify event được publish dù không có client_order_id
    mock_event_bus.publish.assert_called_once()


def test_no_race_condition_between_path_a_and_path_b(tmp_path):
    """
    BUG #1+#2: Per-symbol lock đảm bảo _cancel_and_close_position chỉ chạy 1 lần,
    không overlap giữa Path A và Path B.
    """
    import threading
    call_count = 0

    def mock_cancel_close(*args, **kwargs):
        nonlocal call_count
        call_count += 1

    service = WebSocketDataService(mode="PRODUCTION", event_bus=None)
    service._cancel_and_close_position = mock_cancel_close

    # Simulate concurrent calls from Path A and Path B
    threads = [
        threading.Thread(target=service._handle_order_update, args=(mock_tp_order,)),
        threading.Thread(target=service._handle_position_update, args=(mock_zero_position,)),
    ]
    for t in threads:
        t.start()
    for t in threads:
        t.join()

    # Với lock, chỉ 1 trong 2 được cancel + update DB
    assert call_count == 1, f"Expected 1 call, got {call_count} (race condition detected)"
```

---

## 8. Checklist Verification Sau Fix

- [ ] Mở 1 LONG position trên testnet
- [ ] Để TP hit tự nhiên → verify SL conditional order bị cancel trên Binance
- [ ] Verify DB record status = `CLOSED` với pnl đúng
- [ ] Verify `POSITION_CLOSED` event được publish (check RecoveryManager log)
- [ ] Close 1 position manually → verify cả TP lẫn SL bị cancel
- [ ] Chạy `EnsureTPSLJob` khi có stale position → verify DB được update CLOSED
- [ ] Chạy `PositionSyncService.sync_all_positions()` → verify orphaned conditional orders bị cancel

---

*Tác giả: Antigravity Analysis Engine — 2026-03-07*
