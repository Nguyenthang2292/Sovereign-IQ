# Order Book Imbalance Gate — Implementation To-do

**Date**: 2026-03-02  
**Source**: `2026-03-01-order-book-imbalance-gate-design.md`  
**Status**: ✅ Completed (Phase 1-8 Complete)

---

## Goal

Implement `modules/order_book/` — một **Order Book Imbalance Gate** tích hợp vào `OrderExecutor` để xác nhận dòng tiền real-time (OBI + Cumulative Delta) trước khi đặt lệnh.

---

## Tasks

### Phase 1: Data Models

- [x] **Task 1**: Tạo `modules/order_book/__init__.py` (empty)  
  → Verify: `from modules.order_book import ...` không báo lỗi import

- [x] **Task 2**: Tạo `modules/order_book/models.py` với 3 dataclass + 1 enum:
  - `OrderBookSnapshot` (symbol, bids, asks, timestamp)
  - `AggTrade` (price, quantity, timestamp, is_buyer_maker)
  - `CombinedResult` (obi_score, delta_score, combined_score, obi_raw, delta_raw, weighted_mid)
  - `OBIDecision(str, Enum)`: `PASS`, `SKIP`  
  → Verify: `python -c "from modules.order_book.models import CombinedResult, OBIDecision"` OK

---

### Phase 2: Calculator

- [x] **Task 3**: Tạo `modules/order_book/order_book_imbalance_calculator.py`  
  Implement hàm `calculate_combined_score(snapshot, trades, bin_step_pct=0.001) -> CombinedResult`:
  - Aggregated OBI với bins (0.1% step) → `I_bins ∈ [-1, +1]`
  - Cumulative Delta (`is_buyer_maker=False` → buy aggressive) → normalize bằng `tanh(Δ_cum / V_avg_5m)`
  - `S_combined = 0.4 × I_bins + 0.6 × Δ_norm`  
  → Verify: unit test pass (xem Phase 5)

---

### Phase 3: Market Data Fetcher

- [x] **Task 4**: Tạo `modules/order_book/market_data_fetcher.py`  
  Implement 2 hàm (`async` hoặc sync, dùng `requests`):
  - `fetch_depth(symbol, limit=100, testnet=False) -> OrderBookSnapshot | None`  
    → Endpoint: `GET /fapi/v1/depth`
  - `fetch_agg_trades(symbol, window_minutes=5, testnet=False) -> List[AggTrade] | None`  
    → Endpoint: `GET /fapi/v1/aggTrades?startTime=<now-5m>&endTime=<now>`
  - **Fail-open**: exception → return `None` + log warning  
  → Verify: ✅ Verified - fetch_depth và fetch_agg_trades hoạt động với testnet

---

### Phase 4: OBI Gate

- [x] **Task 5**: Tạo `modules/order_book/order_book_imbalance_gate.py`  
  Implement class `OrderBookImbalanceGate`:
  - `__init__`: `threshold=0.15`, `retry_wait_seconds=30`, `max_retries=2`, `delta_window_minutes=5`, `testnet=False`, `enabled=True`
  - `check(symbol, signal_type) -> tuple[OBIDecision, CombinedResult | None]`
    - Fetch depth + trades → tính `combined_score`
    - Alignment logic (xem bảng Section 7 của design doc)
    - Nếu conflict: sleep → retry tối đa `max_retries` lần
    - Sau `max_retries` vẫn conflict → `SKIP`
    - API trả `None` → `PASS` (fail-open)  
  → Verify: ✅ Verified - class khởi tạo đúng, check() method hoạt động

---

### Phase 5: Tests

- [x] **Task 6**: Tạo `tests/modules/order_book/__init__.py` (empty)

- [x] **Task 7**: Tạo `tests/modules/order_book/test_models.py`  
  → Verify: ✅ PASSED - các dataclass khởi tạo đúng, `OBIDecision.PASS == "PASS"`

- [x] **Task 8**: Tạo `tests/modules/order_book/test_order_book_imbalance_calculator.py`  
  Test các công thức:
  - OBI bins tính đúng (all bid → +1.0, all ask → -1.0)
  - Cumulative Delta normalize đúng (`tanh`)
  - Combined Score tỷ lệ 40-60 (OBI âm mạnh -0.5 + Delta dương mạnh +0.8 → ~+0.28)  
  → Verify: ✅ `pytest tests/modules/order_book/test_order_book_imbalance_calculator.py -v` 16/16 GREEN

- [x] **Task 9**: Tạo `tests/modules/order_book/test_market_data_fetcher.py`  
  Mock HTTP via `pytest-mock` / `unittest.mock`:
  - Normal response → parse đúng `OrderBookSnapshot`, `List[AggTrade]`
  - Timeout / ConnectionError → return `None`
  - HTTP 4xx/5xx → return `None`  
  → Verify: ✅ `pytest tests/modules/order_book/test_market_data_fetcher.py -v` 12/12 GREEN

- [x] **Task 10**: Tạo `tests/modules/order_book/test_order_book_imbalance_gate.py`  
  Test các scenario quan trọng (mock `market_data_fetcher`):

  | Scenario | Score | Expected |
  |----------|-------|----------|
  | LONG, score dương lần 1 | +0.4 | PASS |
  | LONG, score neutral | +0.05 | PASS |
  | LONG, lần 1 âm → lần 2 dương | -0.3 → +0.2 | PASS sau retry |
  | LONG, âm cả 3 lần | -0.4 → -0.3 → -0.2 | SKIP |
  | SHORT, score âm ngay | -0.5 | PASS |
  | REST API lỗi → None | N/A | PASS (fail-open) |
  | `enabled=False` | N/A | PASS ngay, không fetch |

  → Verify: ✅ `pytest tests/modules/order_book/test_order_book_imbalance_gate.py -v` 11/11 GREEN

---

### Phase 6: Tích hợp OrderExecutor

- [x] **Task 11**: Sửa `OrderExecutor.__init__` — thêm param `order_book_imbalance_config: Optional[Dict[str, Any]] = None`  
  - Nếu config `is not None` → khởi tạo `OrderBookImbalanceGate`  
  - Nếu `None` → `self._order_book_imbalance_gate = None`  
  → Verify: ✅ PASS - `OrderExecutor()` không config khởi tạo bình thường, gate là `None`

- [x] **Task 12**: Sửa `OrderExecutor.execute_from_signal` — sau `fetch_ticker`, trước tạo `FinalSignal`:

  ```python
  if self._order_book_imbalance_gate is not None:
      decision, combined_result = self._order_book_imbalance_gate.check(symbol, signal_type)
      if decision == OBIDecision.SKIP:
          return {"success": False, "skipped": True, "reason": "ORDER_BOOK_IMBALANCE_CONFLICT"}
  ```

  → Verify: ✅ PASS - test `dry_run=True` mock: SKIP trả đúng dict conflict, PASS tiếp tục execution

---

### Phase 7: Configuration

- [x] **Task 13**: Thêm section vào `settings.yaml` (hoặc `auto_trade_config.py`):

  ```yaml
  order_book_imbalance:
    enabled: true
    threshold: 0.15
    retry_wait_seconds: 30
    max_retries: 2
    delta_window_minutes: 5
  ```

  → Verify: ✅ PASS - `settings.yaml` + `SettingsManager` có section `order_book_imbalance` và `AutoTradeManager` truyền config vào `OrderExecutor`

---

### Phase 8: End-to-End Verification

- [x] **Task 14**: Chạy toàn bộ test suite `order_book`:

  ```bash
  pytest tests/modules/order_book/ -v --tb=short
  ```

  → Verify: ✅ **47 passed, 0 failures, 0 errors**

- [x] **Task 15**: Chạy bot với `dry_run=True` + OBI gate enabled, quan sát log:
  - Thấy `[OrderBookImbalanceGate] BTCUSDT PASSED (Combined Score=...)` khi PASS
  - Thấy `[OrderBookImbalanceGate] SKIPPED after retry` khi SKIP  
  → Verify: ✅ PASS - test `dry_run` với real gate ghi nhận log PASSED/SKIPPED after retry, không exception

---

## Done When

- [x] Tất cả 4 files trong `modules/order_book/` đã tạo và import sạch
- [x] Tất cả pytest trong `tests/modules/order_book/` đều GREEN
- [x] `OrderExecutor` nhận `order_book_imbalance_config` và gate hoạt động đúng
- [x] Bot chạy `dry_run` có log OBI gate rõ ràng
- [x] Config trong `settings.yaml` có thể toggle `enabled: false` để bypass hoàn toàn

---

## Notes

- **Fail-open policy**: API lỗi mạng → PASS, không bao giờ block trade do lỗi kỹ thuật.
- **Threshold mặc định 0.15** — có thể hạ xuống 0.1 nếu quá nhiều PASS giả, hoặc tăng lên 0.2 nếu block quá nhiều.
- **Retry tối đa 60s** (`max_retries=2 × retry_wait_seconds=30s`) — đủ thời gian orderbook reset nhưng không quá trễ với MFT.
- **WebSocket là Out of Scope** — dùng REST đủ cho gate validation.
