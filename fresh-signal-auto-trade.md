# Fresh-signal auto-trade ( < 60s ) using Settings TP/SL

## Goal
Auto-trade chỉ vào **1 lệnh/chu kỳ** khi DB có signal “fresh” (created_at < 60s), chọn **symbol có score cao nhất**, và đặt market order với **TP/SL lấy từ Settings** (default_tp/default_sl). Không phụ thuộc Gemini levels cho path này.

## Tasks
- [x] Update `modules/auto_trade/gui/utils/data_service.py:get_signals()` để trả thêm `created_at` + `created_at_ts` (epoch seconds) trong mỗi signal dict → Verify: chạy unit test `tests/auto_trade/gui/utils/test_data_service.py` và thấy dict có key mới, giá trị hợp lệ.
- [x] Implement “fresh filter + max score” trong `modules/auto_trade/gui/main_window/auto_trade.py:_auto_trade_cycle()` (lọc `(now - created_at_ts) < 60`, sort theo score desc, chọn 1) và **bypass SignalSelector** cho path này → Verify: chạy pytest cho test mới (xem task dưới) và thấy chỉ chọn đúng signal max score trong nhóm fresh.
- [x] Plumb TP/SL settings vào execution: thêm tham số `tp_sl_settings` (hoặc tương đương) cho `modules/auto_trade/execution/order_executor.py:execute_from_signal()` để tính TP/SL từ `default_tp/default_sl` thay vì hardcode → Verify: test unit cho OrderExecutor tính đúng TP/SL LONG/SHORT với entry giả lập.
- [x] Wire Settings vào auto-trade execution: trong `_auto_trade_cycle()` lấy `tp_sl = settings_manager.get("tp_sl", {})` và truyền xuống `OrderExecutor.execute_from_signal(..., tp_sl_settings=tp_sl)` → Verify: test auto-trade manager assert OrderExecutor được gọi với đúng tp_sl từ settings.
- [x] Add/Update pytest tests:
  - `tests/auto_trade/gui/utils/test_data_service.py`: assert `created_at_ts` có và là number, gần với `created_at` (hoặc parseable).
  - New: `tests/auto_trade/gui/main_window/test_auto_trade_fresh_signal.py`: stub `data_service.get_signals()` trả list signals (fresh + stale), assert chọn đúng signal score cao nhất trong nhóm fresh và skip khi không có fresh.
  - New: `tests/auto_trade/test_order_executor_tp_sl_settings.py`: mock ticker/entry và assert TP/SL từ settings được dùng.
  → Verify: `pytest -q tests/auto_trade/...` pass.
- [x] (Optional) Logging: in `_auto_trade_cycle()` log ngắn “No fresh signals (<60s)” khi skip; log symbol/score khi chọn → Verify: chạy auto-trade dry-run, thấy log rõ ràng, không spam.

## Done when
- [x] Auto-trade không còn bị “Gemini analysis required …” chặn; có signal fresh (<60s) thì vào lệnh market với TP/SL theo Settings.
- [x] Pytest cho các test mới/đã sửa chạy pass, không phát sinh lint errors ở các file đã chạm.

## Notes
- Break-even hiện chưa có setting riêng trong Settings tab; scope bản này chỉ đảm bảo TP/SL theo Settings. Break-even (move SL to entry) có thể thêm sau như một setting hoặc hành vi monitor/position action.
