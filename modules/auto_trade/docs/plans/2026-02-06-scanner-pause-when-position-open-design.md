# Scanner Pause When Position Open – Design

**Date:** 2026-02-06  
**Context:** Chỉ 1 position mở tại một thời điểm; khi đã có lệnh open, Scanner vẫn chạy → lãng phí CPU và gọi Gemini (~$0.05/5 phút). Cải tiến: dùng DB (và double-check với luồng hiện có) để tạm bỏ qua bước scan tốn kém khi đã đủ số position.

---

## 1. Mục tiêu và ràng buộc

- **Mục tiêu:** Khi đã có đủ open position (theo DB), không chạy pipeline tốn kém (ATC → XGBoost → **Gemini**); vẫn giữ timer Scanner và cập nhật UI (signals từ DB).
- **Nguồn sự thật “có position”:**
  - **DB:** Bảng orders, `status = OPEN`, `order_source = PROGRAMMATIC`. Dùng để quyết định “có đủ position → skip scan”.
  - **Double-check:** Luồng hiện tại đã đúng: lệnh gửi qua API Binance → nhận response → cập nhật status vào DB; Reconcile/WebSocket đồng bộ Binance → DB khi đóng position thủ công. Không cần gọi exchange thêm chỉ để quyết định skip scan.
- **Ràng buộc:** Giữ nguyên kiến trúc Scanner loop + Auto-trade (signal &lt;60s từ DB); chỉ thêm một “gate” trước bước gọi Gemini.

---

## 2. Luồng hiện tại (tóm tắt)

- **Scanner:** Timer mỗi `scan_interval` (mặc định 5 phút) → `_scanner_cycle()` → PRODUCTION/DEMO: `_run_signal_scan()` (SignalPipeline, có Gemini) → signal ghi DB → `get_signals()` → đẩy lên UI.
- **Auto-trade:** Timer 60s → lấy signal “fresh” (&lt;60s) từ DB → risk (max_open_positions từ exchange positions qua `data_service.get_positions()`) → execute qua API → cập nhật DB.
- **Reconcile:** Định kỳ đồng bộ orders từ Binance vào DB (status, v.v.) → đảm bảo đóng position thủ công trên sàn cũng phản ánh vào DB.

---

## 3. Thay đổi đề xuất

### 3.1 Gate trong Scanner cycle (backend)

- **Vị trí:** Đầu `_scanner_cycle()`, trước khi gọi `_run_signal_scan()` (chỉ khi `parent.mode in ["PRODUCTION", "DEMO"]`).
- **Logic:**
  1. Lấy `max_open_positions` từ `settings_manager.get("risk.max_open_positions", 1)` (hoặc 3 nếu giữ default hiện tại; có thể thêm setting “pause scanner when at max” = true).
  2. Trong DB: `get_open_positions(session)` (programmatic, status=OPEN), đếm số bản ghi.
  3. Nếu `count >= max_open_positions`:
     - **Không** gọi `_run_signal_scan()` (không ATC/XGBoost/Gemini).
     - Log rõ: ví dụ `"Scanner cycle skipped: open position(s) present ({count}/{max}), no Gemini call."`
     - Vẫn lấy signals từ DB (`get_signals()`), vẫn `_update_queue.put(("signals", signals))` và `("scanner_done", None)` để UI không “đơ”.
  4. Nếu `count < max_open_positions`: chạy như hiện tại (gọi `_run_signal_scan()` rồi lấy signals từ DB và cập nhật UI).

- **Manual scan:** Có thể áp dụng cùng gate: khi user bấm “Scan ngay”, nếu đã đủ open position thì không gọi Gemini, chỉ refresh từ DB (hoặc cho phép “force” scan nếu cần sau này).

### 3.2 GUI

- **Trạng thái Scanner:** Khi Scanner đang chạy (timer bật) nhưng cycle vừa qua bị skip vì đủ position:
  - Có thể hiển thị dạng: `"Scanner: RUNNING (scan skipped – 1 open position)"` hoặc `"Scanner: PAUSED – 1 open position"` tùy copy.
  - Cập nhật trạng thái này trong callback xử lý `scanner_done` (ví dụ nhận thêm flag “skipped_reason” hoặc đọc lại open count từ DB/stats).
- **Cách đơn giản:** Trong `_scanner_cycle()` khi skip, put thêm một message kiểu `("scanner_skipped", {"reason": "open_position", "count": count})`; GUI (layout/status label) xử lý message này và đổi text thành “Scan skipped (open position)” cho đến cycle tiếp theo chạy full.

### 3.3 Double-check và đồng bộ

- **Không thêm API call** chỉ để quyết định skip: nguồn “có position” để gate là DB.
- **Đồng bộ DB ↔ Binance** giữ nguyên như hiện tại:
  - Mở lệnh: Auto-trade → API Binance → response → ghi/update order trong DB (status OPEN).
  - Đóng position (TP/SL hoặc thủ công trên sàn): Reconcile hoặc WebSocket cập nhật status trong DB → lần sau `get_open_positions()` giảm → Scanner cycle tiếp theo sẽ chạy full lại (gọi Gemini).

---

## 4. Lỗi và biên

- **DB không có hoặc lỗi:** Nếu `get_open_positions` ném lỗi hoặc không lấy được session, fallback an toàn: **không** gọi Gemini (coi như “đang có position”) hoặc log và skip cycle; tránh fallback “luôn chạy Gemini” khi lỗi DB.
- **max_open_positions = 0:** Coi như tắt gate (luôn cho phép scan) hoặc cấu hình không hợp lệ, dùng default 1.
- **Manual close ngoài app:** Reconcile định kỳ sẽ cập nhật DB; nếu cần nhanh hơn, có thể trigger reconcile sau khi nhận event đóng position từ WebSocket (nếu đã có).

---

## 5. Testing gợi ý

- Unit: Hàm helper “should_skip_scan(db_session, settings) → bool” với mock session (0 open, 1 open, 2 open) và max_open_positions = 1, 3.
- Integration: Scanner cycle với DB có 1 order OPEN → không gọi SignalPipeline / Gemini (mock pipeline); sau khi “đóng” order trong DB (status CLOSED), cycle tiếp theo gọi pipeline.
- GUI: Khi put `scanner_skipped`, label đổi đúng text; khi put `scanner_done` sau full scan, label trở lại “Scanner: RUNNING” (hoặc tương đương).

---

## 6. Tóm tắt

- **Gate:** Trước khi gọi Gemini trong Scanner, đọc `get_open_positions(session)` và `risk.max_open_positions`; nếu `count >= max` thì skip `_run_signal_scan()`, vẫn refresh signals từ DB và cập nhật UI.
- **Double-check:** DB là nguồn quyết định; lệnh vẫn đi API Binance → response → update DB; Reconcile/WS giữ DB khớp với sàn khi đóng position.
- **GUI:** Hiển thị rõ khi “scan skipped (open position)” để user hiểu không tốn Gemini và vì sao không có signal mới từ pipeline.

Sau khi validate design này, bước tiếp theo có thể là: implement gate trong `_scanner_cycle`, thêm message `scanner_skipped` và cập nhật GUI, rồi bổ sung test như trên.
