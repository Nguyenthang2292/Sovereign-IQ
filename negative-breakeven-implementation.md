# Negative Break-Even – Timer rồi WebSocket

## Goal
Triển khai negative-breakeven theo design `modules/auto_trade/docs/plans/2026-02-06-negative-breakeven-design.md`: khi lệnh âm x% (so với entry) chưa chạm SL thì move TP về entry. Nơi chạy: Cách 1 (timer) trước, sau đó Cách 2 (WebSocket).

## Tasks

### Cách 1 – Timer (polling)
- [x] **Settings GUI:** Trong `config_panel.py` (tab TP/SL), thêm checkbox "Enable Negative Breakeven" (mặc định off), input "Negative BE threshold (%)" (mặc định 2.0). Trong `get_settings()`/`load_settings()` đọc/ghi `negative_be_enabled`, `negative_be_threshold_pct` vào `tp_sl`. `settings_manager` default `tp_sl`: thêm 2 key trên. Validate threshold > 0 và ≤ 100 khi Apply. → Verify: Apply Settings, mở lại tab thấy giá trị đã lưu.
- [x] **DB/API:** Đảm bảo khi trigger: cập nhật order `take_profit = entry_price`, `be_moved = True`, `be_moved_at = now`. Mở rộng `mark_be_moved` (hoặc hàm tương tự trong `database/queries.py`) nếu hiện chỉ cập nhật SL — thêm cập nhật `take_profit`. → Verify: Gọi hàm với order_id, assert DB take_profit và be_moved đúng.
- [x] **Logic thuần:** Tạo module `modules/auto_trade/execution/negative_breakeven.py`: hàm tính `profit_pct` từ (entry, mark, side); hàm `should_trigger(profit_pct, threshold, mark, sl, side, be_moved)` — True khi profit_pct ≤ -threshold, mark chưa vượt SL, not be_moved. → Verify: pytest unit (entry=100, mark=97 LONG → profit_pct=-3; should_trigger với threshold=2 → True).
- [x] **Timer job:** Hàm/class nhận settings_manager, session_scope, BinanceClient: lấy open orders PROGRAMMATIC chưa be_moved; với mỗi symbol lấy mark price (REST hoặc cache); nếu tp_sl.negative_be_enabled và threshold > 0, với mỗi order gọi logic should_trigger, nếu True thì `client.modify_take_profit(symbol, entry_price)`, thành công thì update order (take_profit, be_moved, be_moved_at), commit. → Verify: pytest integration mock modify_take_profit và mark; order OPEN be_moved=False, entry=100, mark=97, threshold=2 → sau 1 lần job: be_moved=True, take_profit=100.
- [x] **Gắn job:** Trong `auto_trade.py` (AutoTradeManager), khi start thêm updater "negative_breakeven" interval 30s gọi job; khi stop thì dừng updater. → Verify: Bật Auto-Trade, log hoặc DB thấy negative_breakeven chạy khi có order thỏa điều kiện.

### Cách 2 – WebSocket-driven
- [x] **Hàm dùng chung:** Timer job và WS handler đều gọi cùng logic (negative_breakeven module: should_trigger + profit_pct). → Verify: Job và handler import cùng hàm.
- [x] **WS handler:** Tạo handler (ví dụ `negative_breakeven_ws_handler.py`): on_position_update nhận position_snapshot; lấy tp_sl từ settings; open orders PROGRAMMATIC theo symbol chưa be_moved; mark từ position; nếu should_trigger thì modify_take_profit và cập nhật DB. Có thể debounce 2s theo symbol. → Verify: Unit test mock position (mark), mock client, assert modify_take_profit gọi đúng khi profit_pct ≤ -threshold.
- [x] **Đăng ký WS:** Trong `websocket_handler.py` register_callbacks: tạo negative_breakeven handler (settings_manager, binance_client=None), đăng ký `ws_data_service.on_position_update(handler.on_position_update)`. → Verify: Chạy app với WS, có position âm ≥ threshold, thấy TP cập nhật (hoặc log) khi handler chạy.

### Verification (cuối)
- [x] Chạy pytest cho module negative_breakeven và job/handler; không lint error. → Verify: `pytest -q tests/auto_trade/...` (hoặc path tới test negative_breakeven) pass.

## Done When
- [x] Timer job 30s: khi enabled và lệnh âm ≥ threshold chưa chạm SL thì move TP về entry, DB be_moved=True.
- [x] WebSocket: position update gọi cùng logic, modify TP và cập nhật DB (có debounce).
- [x] Settings GUI và pytest pass.

## Notes
- Dùng sẵn `BinanceClient.modify_take_profit`. `get_open_positions(session)` đã có; filter be_moved trong job/handler.
- Chưa chạm SL: LONG mark > stop_loss; SHORT mark < stop_loss.
