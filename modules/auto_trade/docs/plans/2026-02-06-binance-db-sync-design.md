# Binance ↔ DB Sync: Close Stale Open Orders

**Date:** 2026-02-06  
**Status:** Design  
**Scope:** Auto-trade GUI — sync giữa Binance account và DB; tự động đóng (cập nhật status) các order OPEN trong DB khi Binance không còn order đó.

---

## 1. Mục tiêu và phạm vi

**Vấn đề:** Reconcile hiện tại chỉ **một chiều**: lấy closed orders từ Binance và **thêm** vào DB nếu chưa có. Không có bước nào kiểm tra ngược: order đang **OPEN** trong DB nhưng trên Binance đã đóng (đóng tay, TP/SL chạy, cancel, v.v.) → DB vẫn giữ OPEN → scanner / trailing stop / negative breakeven vẫn coi là còn position.

**Mục tiêu:** Bổ sung **sync chiều ngược**: với mỗi order **OPEN** trong DB (chỉ PROGRAMMATIC, AT_*), nếu trên Binance **không còn** trong danh sách open orders thì coi là “stale” → cập nhật DB: đặt status = CLOSED hoặc CANCELLED, điền **closed_at** và **pnl** từ Binance nếu API trả về, **giữ nguyên row** (không xóa) để lịch sử và audit.

**Phạm vi:** Chỉ xử lý order có `order_source = PROGRAMMATIC` và `client_order_id` dạng AT_*; không đụng manual trades. Logic nằm trong luồng reconcile hiện có (gọi định kỳ khi auto-trade bật + nút Reconcile trên Database tab) và trong WebSocket order handler (primary path).

---

## 2. Cách làm: Hybrid (WebSocket trước, fallback Reconcile)

**Primary – WebSocket (Cách 3):** Khi nhận event order FILLED / CANCELED từ Binance WS → cập nhật DB ngay (status, closed_at, pnl). Real-time, không cần poll.

**Fallback – Reconcile mở rộng (Cách 1):** Định kỳ (và nút Reconcile) chạy reconcile: ngoài bước “insert closed từ Binance” hiện có, thêm bước “close stale OPEN”: lấy OPEN từ DB → lấy open orders từ Binance → diff → với mỗi OPEN trong DB không còn trên Binance thì lấy chi tiết (closed list hoặc fetch_order) rồi update DB. Bù khi: WS không gửi event, app offline khi order đóng, đóng thủ công trên sàn.

**Thứ tự:** WebSocket xử lý trước khi có event; nếu WS miss hoặc handler lỗi thì reconcile sẽ dọn sau.

---

## 3. Luồng chi tiết

### 3.1. Luồng WebSocket (primary)

- Khi Binance WS gửi event order (ví dụ `executionReport` với `orderStatus = FILLED` hoặc `CANCELED` / `EXPIRED`):
  - Handler (lifecycle / order monitor) nhận payload; lấy `clientOrderId`, `orderId`, `status`, `lastFilledPrice`, `realizedPnl`, `timestamp` (hoặc field tương đương).
  - Chỉ xử lý nếu `clientOrderId` là AT_* (programmatic).
  - Trong DB: tìm row theo `client_order_id` (hoặc `order_id` Binance). Nếu tìm thấy và `status` đang OPEN → cập nhật: `status` = CLOSED hoặc CANCELLED (map từ Binance status), `closed_at` = thời điểm từ event, `pnl` = realizedPnl nếu có; commit.
  - Nếu không tìm thấy row (order manual hoặc chưa sync) → bỏ qua; không insert từ WS để tránh trùng logic reconcile.

### 3.2. Fallback – Reconcile “close stale OPEN” (Cách 1)

Chạy trong `reconcile_orders_with_binance`, **sau** bước insert closed orders hiện tại:

1. **Lấy OPEN từ DB:** Query orders với `status = 'OPEN'`, `order_source = 'PROGRAMMATIC'`, chỉ AT_* (hoặc dùng `get_open_positions` tương đương). Gom danh sách symbol (unique).
2. **Lấy open orders từ Binance:** Với mỗi symbol, gọi `exchange.fetch_open_orders(symbol)`. Build set `binance_open_client_ids = { client_order_id }` (chuẩn hóa symbol theo CCXT).
3. **Xác định stale:** Với mỗi order OPEN trong DB, nếu `order.client_order_id` không nằm trong `binance_open_client_ids` (cùng symbol) → đánh dấu stale.
4. **Lấy chi tiết cho từng stale:**
   - Ưu tiên: tìm trong batch **closed orders** đã fetch ở bước reconcile (cùng symbol, `since_hours`) theo `order_id` hoặc `client_order_id`. Lấy `status`, `lastTradeTimestamp` / closed time, `info.realizedPnl` (nếu có).
   - Nếu không có trong closed batch: gọi `exchange.fetch_order(order_id, symbol)` để lấy trạng thái cuối, closed_at, pnl.
   - Nếu API lỗi / order quá cũ không trả về: vẫn update DB: `status = 'CLOSED'` (hoặc CANCELLED nếu biết), `closed_at = now()` hoặc `NULL`, `pnl = NULL`.
5. **Update DB:** Một lần per row: update `status`, `closed_at`, `pnl`; commit. Không xóa row.

### 3.3. API / CCXT

- `fetch_open_orders(symbol)` → danh sách order đang open trên sàn.
- Closed orders: đã có trong reconcile qua `fetch_closed_orders(symbol, since=..., limit=...)`.
- `fetch_order(id, symbol)` để lấy một order khi không có trong closed batch.
- WebSocket: dùng event executionReport (hoặc equivalent) từ kết nối WS hiện có; không cần thêm REST call trong WS path.

### 3.4. Thứ tự và tần suất

- **WS:** real-time khi có event → cập nhật DB ngay khi nhận FILLED / CANCELED.
- **Reconcile:** chạy định kỳ (ví dụ 1h như hiện tại) + nút “Reconcile”; trong mỗi lần chạy: insert closed (cũ) rồi close stale OPEN (mới). Nếu WS không gửi event (mất kết nối, đóng ngoài app trước khi bật WS) thì reconcile sẽ dọn sau; nếu WS handler lỗi (DB lock, v.v.) thì log và để reconcile bù.

---

## 4. Error handling

**WebSocket path:** Bọc handler trong try/except. Nếu lỗi (DB lock, validation, v.v.): log, không crash WS; order đó sẽ được reconcile fallback xử lý sau. Không retry ngay trong handler.

**Reconcile path:**

- `fetch_open_orders(symbol)` lỗi mạng/API: log, bỏ qua symbol đó trong bước “close stale”; vẫn chạy tiếp các symbol khác.
- `fetch_order(id, symbol)` lỗi hoặc 404 (order quá cũ): vẫn update DB với `status = CLOSED` (hoặc CANCELLED nếu có thông tin), `closed_at = None` hoặc `datetime.utcnow()`, `pnl = None`; log “stale closed without Binance details”.
- Rate limit Binance: giữ `enableRateLimit: True`; nếu cần, thêm sleep nhẹ giữa các symbol hoặc giữa batch fetch_order khi có nhiều stale.

**Chung:** Không xóa/cập nhật hàng loạt mà không kiểm tra từng order; mỗi update một row, commit (hoặc batch commit nhỏ) để tránh rollback toàn bộ khi lỗi giữa chừng.

---

## 5. Edge cases

- **DB OPEN nhưng Binance không trả order (đã quá cũ):** Reconcile không tìm thấy trong closed batch và `fetch_order` 404 hoặc lỗi → vẫn đánh dấu CLOSED trong DB với closed_at/pnl = None (hoặc closed_at = now) để scanner không còn coi là position mở.
- **Cùng client_order_id xuất hiện nhiều lần (ví dụ nhiều symbol):** So sánh theo (symbol, client_order_id); mỗi row DB map một order Binance duy nhất.
- **WS nhận FILLED nhưng DB chưa có row (reconcile chưa chạy / order từ nguồn khác):** Bỏ qua trong WS handler; reconcile sẽ insert khi fetch closed orders (logic cũ) hoặc không insert nếu không phải AT_*.
- **Reconcile và WS cùng lúc cập nhật một order:** Update theo primary key (id hoặc order_id); “last write wins”. Nếu WS đã đóng row rồi, reconcile thấy không còn trong open list và tìm thấy trong closed → update lại closed_at/pnl là idempotent.

---

## 6. Testing

- **Unit (reconcile “close stale”):** Mock exchange: `fetch_open_orders` trả về rỗng; `fetch_closed_orders` (hoặc `fetch_order`) trả về một order FILLED với client_order_id = AT_xxx. DB có một row OPEN cùng client_order_id. Gọi reconcile (hoặc hàm con close_stale_open_orders). Assert: row đó có status = CLOSED, closed_at và pnl được set theo mock.
- **Unit (WS handler):** Mock event executionReport (FILLED, client_order_id = AT_xxx). DB có row OPEN tương ứng. Gọi handler. Assert: row cập nhật CLOSED, closed_at/pnl đúng. Trường hợp DB không có row → handler không insert, không crash.
- **Integration (tùy chọn):** Testnet Binance: tạo một order rồi đóng/cancel trên sàn; chạy reconcile; kiểm tra DB đã chuyển OPEN → CLOSED và (nếu có) closed_at/pnl.

---

## 7. Kết quả trả về reconcile

Mở rộng `reconcile_orders_with_binance` return: thêm key ví dụ `closed_stale: int` (số row OPEN đã được cập nhật thành CLOSED/CANCELLED). Giữ `inserted`, `skipped`, `errors` như hiện tại để GUI/log có thể hiển thị: “Reconcile: inserted=X, closed_stale=Y, errors=Z”.
