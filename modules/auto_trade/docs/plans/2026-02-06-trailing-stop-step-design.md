# Step-Based Trailing Stop Design

**Date:** 2026-02-06  
**Goal:** Nối `trailing_stop` từ Settings xuống execution; khi bật, mỗi khi giá đạt thêm x% (bước cấu hình) thì SL nhảy 1 bước (BE → +2% → +4% …). Hỗ trợ tùy chọn C: giới hạn số bước (checkbox + max steps) trong GUI Settings.

---

## 1. Hành vi mong muốn

- **Trailing step (%)** — cấu hình trong Settings (A): ví dụ 2%. **Bước 0 luôn là BE (entry):** bước đầu tiên ta áp dụng là đưa SL lên entry; các bước sau là +step%, +2*step%, … (LONG: entry, entry+step%, entry+2*step%; SHORT: entry, entry−step%, entry−2*step%).
- **Chỉ số bước trong DB:** `trailing_step_index` = 0 nghĩa là chưa áp dụng trailing (vẫn SL ban đầu). `trailing_step_index` = 1 = đã lên BE; 2 = đã lên +step%; 3 = đã lên +2*step%; …
- **Điều kiện nhảy:** Giá (mark) đạt ≥ ngưỡng bước tiếp theo (so với entry) thì mới cập nhật SL. LONG: profit_pct = (mark - entry) / entry * 100. SHORT: profit_pct = (entry - mark) / entry * 100. Ngưỡng bước 1 (BE) = 0%; bước 2 = step_pct%; bước 3 = 2*step_pct%; …
- **Ràng buộc max_steps (C):** Checkbox "Limit trailing steps" + field "Max steps" (số nguyên). **max_steps** = số bước trailing tối đa (bao gồm cả bước BE). Chỉ được nhảy nếu `next_step_index <= max_steps`. Ví dụ max_steps = 3 ⇒ tối đa 3 bước: BE → +step% → +2*step% rồi dừng (không nhảy bước 4).
- **Nguồn giá:** Mark price từ WebSocket hoặc REST. Chỉ áp dụng cho lệnh PROGRAMMATIC và trạng thái OPEN.

---

## 2. Cách tiếp cận và trade-off

**Cách 1 – Timer (polling):** Job định kỳ (ví dụ mỗi 15–30s) lấy danh sách position OPEN từ DB/API, lấy mark price (REST hoặc cache từ WS), với mỗi order có `trailing_stop=True` trong settings thì tính profit %, so sánh với bước hiện tại (lưu trong Order), nếu đạt ngưỡng bước tiếp theo và (nếu bật) step < max_steps thì gọi `BinanceClient.modify_stop_loss(symbol, new_sl_price)`, cập nhật DB (trailing_step_index, stop_loss). **Ưu:** Đơn giản, không phụ thuộc WS. **Nhược:** Độ trễ theo chu kỳ polling.

**Cách 2 – WebSocket-driven:** Mở rộng PositionMonitor (hoặc lifecycle handler nhận event từ WS): mỗi khi nhận cập nhật position (mark price), kiểm tra cùng logic trailing; nếu đạt bước tiếp theo thì gọi modify_stop_loss và cập nhật DB. **Ưu:** Phản ứng nhanh theo giá real-time. **Nhược:** Phải inject settings và client vào monitor/lifecycle; cần map position (symbol/side) với order trong DB để biết entry và trailing_step_index.

**Cách 3 – Binance TRAILING_STOP_MARKET:** Khi trailing_stop bật, không đặt stop_market cố định mà đặt lệnh TRAILING_STOP_MARKET với callbackRate (%). **Ưu:** Binance tự kéo SL. **Nhược:** Không đúng mô hình "bước" (x% nhảy 1 bước); không có "limit steps" theo từng bước; khó đồng bộ với DB (bước hiện tại).

**Đề xuất:** Cách 1 (timer) cho MVP: dễ triển khai, đúng logic bước; sau có thể chuyển sang Cách 2 (WebSocket) để giảm độ trễ. Không dùng Cách 3 cho yêu cầu "step-based".

---

## 3. Luồng dữ liệu và thành phần

**3.1 Settings (GUI + schema)**

- **Config panel (TP/SL):** Giữ "Enable Trailing Stop". Thêm:
  - **Trailing step (%)**: input số (mặc định 2.0).
  - **Limit trailing steps**: checkbox (mặc định unchecked). Khi tick: hiện **Max steps** (số nguyên, mặc định 5).
- **settings_manager / tp_sl:** Thêm key `trailing_step_pct` (float), `trailing_limit_steps` (bool), `trailing_max_steps` (int, chỉ dùng khi limit_steps=True). Khi load/save settings, đọc/ghi các field này cùng `trailing_stop`.

**3.2 Execution và lưu trữ**

- **Đặt lệnh (OrderManager/BinanceClient):** Không đổi: vẫn đặt TP/SL cố định lúc vào lệnh. Không cần truyền trailing_stop xuống lúc place order; trailing chỉ chạy **sau** khi có position.
- **DB – Order:** Thêm cột `trailing_step_index` (Integer, default 0): 0 = chưa áp dụng trailing; 1 = đã lên BE (bước 0 luôn là BE); 2 = +step%; 3 = +2*step%; … Khi gọi modify_stop_loss thành công thì tăng trailing_step_index và cập nhật stop_loss. **max_steps** (khi bật Limit trailing steps) giới hạn số bước tối đa: chỉ nhảy khi next_step_index <= max_steps.
- **Consumer settings:** Module chạy trailing (timer job hoặc WebSocket handler) cần đọc tp_sl: `trailing_stop`, `trailing_step_pct`, `trailing_limit_steps`, `trailing_max_steps`. Có thể lấy từ settings_manager singleton hoặc inject từ main_window khi khởi tạo job/monitor.

**3.3 Logic trailing (timer job)**

- Chu kỳ (ví dụ 30s): Lấy danh sách order OPEN, order_source=PROGRAMMATIC, có client_order_id (để map với position nếu cần). Với mỗi symbol có position, lấy mark price (từ cache WebSocket hoặc REST fetch_ticker).
- Với mỗi order tương ứng: entry = order.entry_price, side = order.side, step_index = order.trailing_step_index, step_pct = settings.trailing_step_pct, limit = settings.trailing_limit_steps, max_steps = settings.trailing_max_steps.
- Tính profit_pct (LONG: (mark - entry)/entry*100; SHORT: (entry - mark)/entry*100).
- Bước tiếp theo: next_step_index = step_index + 1. **Ràng buộc max_steps:** nếu limit_steps=True và next_step_index > max_steps thì bỏ qua (không nhảy nữa). Ngưỡng lợi nhuận: next_threshold = (next_step_index - 1) * step_pct (bước 1 = BE = 0%, bước 2 = step_pct%, bước 3 = 2*step_pct%, …).
- Nếu profit_pct >= next_threshold: tính new_sl_price — bước 1 luôn BE (entry); bước 2 trở đi: LONG entry * (1 + (next_step_index - 1) * step_pct / 100), SHORT entry * (1 - (next_step_index - 1) * step_pct / 100). Gọi BinanceClient.modify_stop_loss(symbol, new_sl_price). Thành công thì update order (trailing_step_index = next_step_index, stop_loss = new_sl_price), commit.

**3.4 Nơi chạy job**

- **Option A:** Updater riêng trong UpdaterManager (ví dụ `trailing_stop`), interval 30s, callback đọc settings và chạy logic trên; chỉ chạy khi auto_trade đang bật (hoặc luôn chạy khi app mở, tùy product).
- **Option B:** Gắn vào lifecycle/PositionMonitor: mỗi khi có position update từ WebSocket, gọi hàm `maybe_apply_trailing(positions, settings)`; hàm này lấy open orders từ DB, map symbol, tính toán và gọi modify_stop_loss khi đủ điều kiện.

Thiết kế đề xuất: **Option A** (timer) để tách biệt, dễ test; sau có thể thêm Option B nếu cần latency thấp hơn.

---

## 4. Lỗi và biên

- **API modify_stop_loss lỗi:** Log, không tăng trailing_step_index; lần sau sẽ thử lại (ngưỡng vẫn thỏa).
- **Mark price không lấy được:** Bỏ qua chu kỳ đó cho symbol đó.
- **Nhiều order cùng symbol:** Chỉ áp dụng trailing cho order PROGRAMMATIC, OPEN; nếu có nhiều position cùng symbol (hedge mode) cần map đúng order–position (position_side hoặc client_order_id). Đơn giản hóa: one position per symbol (BOTH), một order OPEN per symbol.
- **Trailing step 0 hoặc âm:** Coi như tắt trailing (hoặc validate trong Settings: step_pct > 0).
- **max_steps:** Khi Limit trailing steps bật, validate max_steps >= 1 (ít nhất 1 bước = BE). Nếu max_steps = 1 thì chỉ cho phép lên BE rồi dừng.

---

## 5. Kiểm thử

- **Unit:** Hàm tính next_threshold và new_sl_price từ (entry, side, step_index, step_pct); kiểm tra điều kiện profit_pct >= threshold và limit_steps / max_steps.
- **Integration:** Mock BinanceClient.modify_stop_loss; mock mark price; gọi job một lần với order OPEN và settings trailing_stop=True, step_pct=2, kiểm tra DB trailing_step_index và stop_loss được cập nhật đúng sau khi "đạt ngưỡng".

---

## 6. Tóm tắt

- **Settings:** Thêm Trailing step (%), checkbox Limit trailing steps + Max steps; lưu trong tp_sl.
- **DB:** Order thêm trailing_step_index (default 0).
- **Logic:** Job định kỳ (hoặc WebSocket handler) đọc mark price và settings, với mỗi order OPEN có trailing_stop bật: nếu profit_pct đạt ngưỡng bước tiếp theo và (nếu bật) step < max_steps thì gọi modify_stop_loss, cập nhật order.
- **Executor/place order:** Không cần truyền trailing_stop lúc đặt lệnh; trailing chỉ chạy sau khi position đã mở, do job/monitor đọc settings và áp dụng.

File này là bản thiết kế đã xác nhận (A + C: step % cấu hình + limit steps tùy chọn trong GUI).
