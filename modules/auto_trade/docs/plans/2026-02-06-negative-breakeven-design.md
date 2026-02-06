# Negative Break-Even Design

**Date:** 2026-02-06  
**Goal:** Khi lệnh đang âm x% (unrealized PnL % so với entry) nhưng chưa chạm SL, tự động move **TP** (Take Profit) về **entry** (break-even) để gồng lệnh — hy vọng giá quay lại BE để đóng không lỗ.

**Khác với Break-Even Manager hiện tại:** `monitoring/breakeven_manager.py` trigger theo **30% drawdown của tài khoản**. Ở đây trigger theo **% lỗ của chính lệnh** (position PnL % so với entry). Đây là yêu cầu negative-breakeven đúng nghĩa.

---

## 1. Hành vi mong muốn

- **Trigger:** Unrealized PnL % của **position** so với entry ≤ **-x%** (x cấu hình, ví dụ 2%, 3%). LONG: `profit_pct = (mark - entry) / entry * 100`. SHORT: `profit_pct = (entry - mark) / entry * 100`. Khi `profit_pct <= -x` thì điều kiện lỗ đạt.
- **Chưa chạm SL:** Giá (mark) chưa vượt qua SL. LONG: mark > stop_loss. SHORT: mark < stop_loss. Nếu đã chạm SL thì không move TP (hoặc tùy chính sách: có thể vẫn cho phép move TP về BE trước khi SL fill).
- **Duplicate prevention:** Chỉ move một lần cho mỗi order. Dùng flag `be_moved` (và `be_moved_at`) trên bảng orders: khi đã move TP về BE thì set `be_moved = True`, không trigger lại.
- **Hành động:** Gọi API chỉnh TP về **entry** (modify take profit). Cập nhật DB: `take_profit = entry_price`, `be_moved = True`, `be_moved_at = now`.
- **Cấu hình Settings:** Trong TP/SL (hoặc Risk): **Negative breakeven threshold (%)** — khi lệnh âm ≥ ngưỡng này thì move TP về BE (ví dụ 2.0 → lỗ ≥ 2% so với entry thì trigger). Checkbox **Enable Negative Breakeven** (ví dụ `negative_be_enabled`).

---

## 2. Nguồn giá, phạm vi, nơi chạy

- **Nguồn giá:** Mark price từ WebSocket (position update) hoặc REST (fetch ticker). Công thức PnL % dùng mark và entry như trên.
- **Phạm vi:** Chỉ lệnh **OPEN**, **order_source = PROGRAMMATIC**. Map position (symbol, side) với order trong DB (1 order OPEN per symbol hoặc client_order_id / position_side nếu có nhiều).
- **Nơi chạy:**
  - **Cách 1 – Timer (polling):** Job định kỳ (ví dụ 30s): lấy open orders PROGRAMMATIC, mark price (REST hoặc cache WS), với mỗi order chưa `be_moved`: nếu profit_pct ≤ -threshold và chưa chạm SL thì `modify_take_profit(symbol, entry_price)`, cập nhật DB. Ưu: đơn giản. Nhược: trễ theo chu kỳ.
  - **Cách 2 – WebSocket-driven:** Handler nhận position update: kiểm tra cùng điều kiện, modify TP và cập nhật DB. Ưu: phản ứng nhanh. Nhược: cần inject settings + client.
- **Đề xuất:** Cách 1 (timer) cho MVP; sau thêm Cách 2 (WS) nếu cần giảm độ trễ.
- **Cấu hình đọc từ Settings:** `tp_sl.negative_be_enabled` (bool), `tp_sl.negative_be_threshold_pct` (float, ví dụ 2.0). Chỉ chạy khi enabled và threshold > 0.

**2.1 Settings (GUI + schema)**

- **Config panel (TP/SL):** Trong tab TP/SL (cùng khu vực Trailing Stop hoặc ngay dưới), thêm:
  - **Enable Negative Breakeven**: checkbox (mặc định unchecked). Khi bật: khi lệnh âm ≥ ngưỡng thì move TP về BE.
  - **Negative BE threshold (%)**: input số (mặc định 2.0). Chỉ hiện hoặc chỉ enable khi checkbox trên được tick. Ý nghĩa: khi unrealized PnL % của lệnh ≤ -threshold thì trigger (ví dụ 2.0 → lỗ ≥ 2% so với entry).
- **settings_manager / tp_sl:** Thêm key `negative_be_enabled` (bool), `negative_be_threshold_pct` (float, mặc định 2.0). Khi load/save settings (get_settings / load_settings trong config_panel), đọc/ghi hai field này cùng các TP/SL khác.
- **Validation:** threshold > 0 và ≤ 100 (hoặc giới hạn hợp lý); nếu invalid khi Apply thì revert về mặc định 2.0 hoặc báo lỗi.
- **Current Settings (Trading):** Nếu có panel hiển thị cấu hình đang dùng, có thể thêm dòng "Negative BE: On/Off", "Negative BE threshold: x%".

---

## 3. DB, API, lỗi và kiểm thử

**DB**  
Dùng sẵn cột `be_moved`, `be_moved_at` trên bảng orders. Khi trigger: set `be_moved = True`, `be_moved_at = now`, `take_profit = entry_price`. Có thể dùng hàm `mark_be_moved(session, order_id, ...)` nếu đã có; mở rộng hoặc thêm cập nhật `take_profit` nếu hiện chỉ cập nhật SL.

**API**  
`BinanceClient.modify_take_profit(symbol, position_id=None, take_profit_price=entry_price)`. Nếu API yêu cầu cancel TP cũ rồi place TP mới thì làm theo flow đó.

**Xử lý lỗi**  
- Gọi `modify_take_profit` lỗi: log, không set `be_moved`; lần sau job/WS thử lại.  
- Không lấy được mark price: bỏ qua chu kỳ đó cho symbol đó.  
- Nhiều order cùng symbol: chỉ xử lý PROGRAMMATIC, OPEN; map đúng order–position.  
- Threshold ≤ 0 hoặc không cấu hình: coi như tắt (không trigger).

**Kiểm thử**  
- **Unit:** Hàm tính profit_pct từ (entry, mark, side); điều kiện trigger: profit_pct ≤ -threshold, mark chưa vượt SL, not be_moved.  
- **Integration:** Mock modify_take_profit, mock mark; order OPEN chưa be_moved, entry=100, mark=97 (LONG, -3%), threshold=2 → assert modify_take_profit(TP=100), DB be_moved=True, take_profit=100.  
- **Biên:** Đã be_moved → không gọi modify; mark đã chạm SL → không move TP (hoặc cho phép tùy chính sách).

---

## 4. Tóm tắt

- **Trigger:** Lệnh âm ≥ x% so với entry (position PnL %), chưa chạm SL, chưa be_moved.  
- **Action:** Move TP về entry; cập nhật DB (take_profit, be_moved, be_moved_at).  
- **Settings (GUI + schema):** Tab TP/SL: checkbox "Enable Negative Breakeven", input "Negative BE threshold (%)" (mặc định 2.0). settings_manager / tp_sl: `negative_be_enabled`, `negative_be_threshold_pct`; load/save trong config_panel.  
- **Chạy:** Timer job (MVP) rồi có thể thêm WebSocket handler.  
- **Khác với breakeven_manager hiện tại:** Trigger theo % lỗ của lệnh, không phải 30% drawdown của tài khoản.

File này là bản thiết kế negative-breakeven đã xác nhận.
