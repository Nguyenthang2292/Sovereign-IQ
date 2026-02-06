# Trailing Stop Step – Timer rồi WebSocket

## Goal
Triển khai trailing stop theo bước (BE → +step% → +2*step% …) theo design `modules/auto_trade/docs/plans/2026-02-06-trailing-stop-step-design.md`: trước dùng Cách 1 (timer polling), sau nối tiếp Cách 2 (WebSocket-driven).

## Tasks

### Phase 1 – Timer (polling)
- [x] **DB:** Thêm cột `trailing_step_index` (Integer, default 0) vào bảng orders; thêm migration trong `modules/auto_trade/database/migrations/` và chạy migrate. → Verify: mở DB, bảng orders có cột `trailing_step_index`.
- [x] **Settings GUI:** Trong `config_panel.py` (TP/SL), thêm input "Trailing step (%)" (mặc định 2.0), checkbox "Limit trailing steps" (khi tick hiện field "Max steps", mặc định 5). Trong `get_settings()`/`load_settings()` đọc/ghi `trailing_step_pct`, `trailing_limit_steps`, `trailing_max_steps` vào `tp_sl`. Cập nhật `settings_manager` default `tp_sl` với 3 key trên. → Verify: Apply Settings, mở lại tab Settings thấy giá trị đã lưu; Current Settings (Trading) có thể hiển thị Trailing step nếu cần.
- [x] **Logic thuần:** Tạo module `modules/auto_trade/execution/trailing_stop.py` (hoặc tên tương đương) với hàm: tính `next_threshold` và `new_sl_price` từ (entry, side, step_index, step_pct); kiểm tra `next_step_index <= max_steps` khi limit_steps=True; trả về (should_step: bool, new_sl_price hoặc None). Bước 1 = BE (entry). → Verify: pytest unit test gọi hàm với entry=100, side=LONG, step_index=0/1, step_pct=2, assert threshold và new_sl đúng; assert max_steps chặn đúng.
- [x] **Timer job:** Hàm (hoặc class) nhận settings_manager, db session_scope, BinanceClient (hoặc factory): lấy danh sách order OPEN, order_source=PROGRAMMATIC; với mỗi symbol lấy mark price (REST `fetch_ticker` hoặc từ cache nếu có); nếu tp_sl.trailing_stop và step_pct>0 thì với mỗi order gọi logic trailing, nếu should_step thì gọi `client.modify_stop_loss(symbol, new_sl_price)`, thành công thì update order `trailing_step_index` và `stop_loss` trong DB. → Verify: pytest integration (mock client.modify_stop_loss, mock fetch_ticker trả mark; order OPEN với trailing_step_index=0; assert sau 1 lần gọi job thì trailing_step_index=1, stop_loss=entry khi profit_pct>=0).
- [x] **Gắn job vào app:** Khi start auto-trade (hoặc khi app có GUI đã load), tạo updater "trailing_stop" interval 30s gọi job trên (đọc settings từ parent.settings_manager; lấy client/credentials từ data_service hoặc env). Khi stop auto-trade thì dừng updater. → Verify: Bật Auto-Trade, mở position test; sau vài chu kỳ 30s (hoặc giảm interval test) kiểm tra log hoặc DB trailing_step_index thay đổi khi giá đạt ngưỡng.
- [x] **Query DB:** Thêm/ dùng query lấy open orders PROGRAMMATIC (có sẵn `get_open_positions` hoặc tương đương); đảm bảo trả về order có `trailing_step_index`. → Verify: Gọi query trong test hoặc từ job, thấy order OPEN với trailing_step_index.

### Phase 2 – WebSocket-driven
- [x] **Hàm dùng chung:** Đảm bảo logic “có nên nhảy bước + new_sl_price” nằm trong 1 hàm (trailing_stop module) để timer và WS đều gọi. → Verify: Timer job và WS handler đều import cùng hàm.
- [x] **WS hook:** Trong PositionMonitor hoặc lifecycle handler nhận event position update (mark price): lấy tp_sl từ settings; lấy open orders PROGRAMMATIC từ DB theo symbol; với mỗi order gọi hàm trailing chung với mark từ WS; nếu should_step thì gọi BinanceClient.modify_stop_loss và update order trong DB. → Verify: Unit test với mock position update (mark), mock client, assert modify_stop_loss được gọi đúng khi profit_pct đạt ngưỡng.
- [x] **Bật WS trailing:** Khi WebSocket đã kết nối và có position stream, gọi trailing sau mỗi position update (tránh gọi quá dày). Có thể debounce 2–5s theo symbol. → Verify: Chạy app với WS, có position; thay đổi mark (mock hoặc thị trường) thấy SL cập nhật khi đạt bước (có thể so sánh với timer: tắt timer chỉ dùng WS vẫn thấy cập nhật).

### Verification (cuối)
- [x] Chạy `pytest` cho `tests/auto_trade/` (module trailing_stop + integration job); không lint error trên file đã sửa. → Verify: `pytest -q tests/auto_trade/...` pass. *(Unit tests trong `modules/auto_trade/tests/test_trailing_stop.py`.)*

## Done When
- [x] Timer job chạy mỗi 30s, cập nhật SL theo bước (BE → +step% …) khi settings bật trailing_stop và giá đạt ngưỡng; max_steps được tôn trọng.
- [x] WebSocket khi có position update gọi cùng logic trailing và cập nhật SL; có debounce tránh spam API.
- [x] Pytest cho logic và job pass; DB migration và Settings GUI đã áp dụng.

## Notes
- BinanceClient.modify_stop_loss đã có sẵn; cần cancel SL cũ và place SL mới (hoặc modify nếu API hỗ trợ). Kiểm tra `binance_client.modify_stop_loss` / `modify_stop_loss` signature.
- Map position (symbol) ↔ order: đơn giản hóa 1 order OPEN per symbol; nếu nhiều position cùng symbol cần map qua client_order_id hoặc position_side sau.
