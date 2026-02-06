# Fresh-Signal Auto-Trade Design

**Date:** 2026-02-06  
**Goal:** Auto-trade vào lệnh market khi có signal “mới” (< 60s) với TP/SL từ Settings; chọn symbol có **score cao nhất** trong số các signal mới.

---

## 1. Context and problem

**Current behavior:** Auto-trade cycle gọi `SignalSelector.select_best_signal(xb_signals, gemini_signals={})`. Vì `gemini_signals` luôn rỗng, mọi candidate có `entry=0, stop_loss=0, take_profit=0` và bị reject (“Gemini analysis required for accurate levels”). Kết quả: auto-trade không bao giờ vào lệnh.

**Desired behavior:**

- Xem **thời gian tạo signal** trên DB (created_at).
- So sánh với thời gian hiện tại: **nếu signal được tạo < 60 giây** thì coi là “fresh”.
- Với các signal fresh: chọn **một** signal — symbol có **score cao nhất**.
- Vào lệnh **market trực tiếp** với **stop-loss, take-profit (và break-even) lấy từ tab Settings** (default_tp %, default_sl %).

**Break-even:** Hiện Settings chưa có trường break-even. Trong phiên bản đầu: dùng TP/SL từ Settings; break-even (move SL về entry) có thể thêm sau như một setting hoặc hành vi mặc định khi đặt lệnh.

---

## 2. Approach and trade-offs

**Approach A (chosen): Logic trong auto-trade cycle, không phụ thuộc Signal Selector cho path “fresh”.**

- Mở rộng `get_signals` để trả về `created_at` (hoặc timestamp) cho mỗi signal.
- Trong `_auto_trade_cycle`: lọc signals có `(now - created_at) < 60` giây → trong tập đó sort theo score giảm dần → lấy signal đầu tiên (score cao nhất).
- Nếu không có signal nào fresh → bỏ qua chu kỳ (không vào lệnh).
- Gọi execution với **TP/SL lấy từ Settings** (default_tp %, default_sl %); entry = giá hiện tại (fetch ticker như hiện tại).
- **Trade-off:** Hai path rõ ràng: (1) “fresh + Settings” — dùng trong auto_trade; (2) Signal Selector + Gemini vẫn giữ cho các flow khác (GUI, manual). Không sửa logic bên trong Signal Selector.

**Approach B (alternative):** Signal Selector nhận thêm `settings_tp_sl` và `created_at`; khi không có Gemini levels, nếu signal fresh thì build FinalSignal với entry = current price, TP/SL từ settings %. **Trade-off:** Một điểm chọn signal duy nhất nhưng selector phức tạp hơn và cần inject settings + thời gian.

**Approach C (alternative):** Module riêng `FreshSignalSelector.select_fresh_signal(signals_with_created_at, max_age_seconds=60)` trả về signal score cao nhất trong tập fresh. **Trade-off:** Tách biệt, dễ test; vẫn cần mở rộng get_signals và execution với Settings.

**Recommendation:** Approach A — ít thay đổi, dễ triển khai, đúng YAGNI. Có thể tách helper “chọn best fresh signal” trong auto_trade nếu cần tái sử dụng.

---

## 3. Data flow and components

**3.1 Signals với thời gian**

- **Nguồn:** `get_recent_signals(session, limit=100)` đã trả về `Signal` có `created_at`.
- **Thay đổi:** `DataService.get_signals(min_score, signal_types)` khi build dict cho mỗi signal, thêm:
  - `created_at`: datetime (hoặc ISO string) để log/debug.
  - `created_at_ts`: số giây (epoch) hoặc timestamp để so sánh với `time.time()` (thống nhất đơn vị: giây).
- API giữ nguyên: `get_signals()` vẫn trả về `List[Dict]` với keys hiện có + `created_at`, `created_at_ts`.

**3.2 Auto-trade cycle**

- **Luồng:**
  1. `signals = self.parent.data_service.get_signals(min_score=0.7)` (đã có `created_at_ts`).
  2. `now = time.time()`.
  3. `fresh = [s for s in signals if (now - s.get("created_at_ts", 0)) < 60]`.
  4. Nếu `not fresh` → log “No fresh signals (< 60s)”, return (không vào lệnh).
  5. Sort `fresh` theo `score` giảm dần; lấy `best = fresh[0]`.
  6. Risk check như hiện tại (RiskManager.check_limits với symbol, position_size, leverage từ settings).
  7. Build `sig_dict`: symbol, signal, score (và nếu cần created_at cho log).
  8. Lấy `tp_sl = self.parent.settings_manager.get("tp_sl", {})` (default_tp, default_sl, mode).
  9. Gọi `OrderExecutor().execute_from_signal(sig_dict, tp_sl_settings=tp_sl)` (hoặc cơ chế tương đương để executor nhận TP/SL từ Settings).
  10. Nếu success → refresh positions/account như hiện tại.

- **Không gọi** `SignalSelector.select_best_signal` cho path này (tránh reject do thiếu Gemini).

**3.3 Execution với TP/SL từ Settings**

- **OrderExecutor.execute_from_signal(signal_dict, tp_sl_settings=None):**
  - Nếu `tp_sl_settings` có:
    - `default_tp = float(tp_sl_settings.get("default_tp", 5.0))`
    - `default_sl = float(tp_sl_settings.get("default_sl", 2.5))`
    - Entry = fetch_ticker(symbol) như hiện tại.
    - LONG: `take_profit = entry * (1 + default_tp/100)`, `stop_loss = entry * (1 - default_sl/100)`.
    - SHORT: `take_profit = entry * (1 - default_tp/100)`, `stop_loss = entry * (1 + default_sl/100)`.
  - Nếu `tp_sl_settings` None → giữ hành vi cũ (hardcode 5.0 / 2.0 hoặc từ nơi khác).
- **OrderManager / BinanceClient:** Không bắt buộc thay đổi nếu đã nhận FinalSignal với entry/tp/sl đúng; chỉ cần executor build FinalSignal từ Settings %.

**3.4 Break-even**

- Tab Settings hiện không có trường break-even. Phiên bản đầu: chỉ dùng TP/SL từ Settings. Break-even (move SL to entry) có thể:
  - Thêm sau như option trong Settings (ví dụ “Move SL to entry at X% profit”), hoặc
  - Giữ là thao tác thủ công trên position (Modify TP/SL / Breakeven) như hiện tại.

---

## 4. Edge cases and error handling

- **Không có signal nào < 60s:** Bỏ qua chu kỳ, không vào lệnh; có thể log ngắn “No fresh signals”.
- **Nhiều signal cùng score tối đa:** Chọn phần tử đầu tiên sau khi sort (hoặc quy ước thêm: cùng score thì chọn created_at mới nhất).
- **Thiếu created_at / created_at_ts trong dict:** Coi signal đó “không fresh” (ví dụ `created_at_ts = 0` → `(now - 0) > 60`), hoặc bỏ qua trong filter fresh.
- **Settings thiếu default_tp / default_sl:** Dùng giá trị mặc định 5.0 và 2.5 tương ứng.
- **Risk limits exceeded:** Giữ hành vi hiện tại (skip trade, log).

---

## 5. Testing

- **Unit:**
  - Filter “fresh”: list signals với created_at_ts, assert chỉ những signal có (now - created_at_ts) < 60 được giữ.
  - Chọn best: sau khi sort theo score desc, assert phần tử đầu đúng symbol và score cao nhất trong tập fresh.
  - OrderExecutor với `tp_sl_settings`: assert TP/SL tính đúng từ default_tp / default_sl và entry (LONG/SHORT).
- **Integration:** Auto-trade cycle với mock `get_signals` trả về 1 signal có created_at_ts = now - 30; assert gọi execute với đúng symbol và tp_sl từ settings_manager (có thể mock settings_manager.get("tp_sl")).

---

## 6. Summary

- **get_signals:** Trả về thêm `created_at`, `created_at_ts` cho mỗi signal.
- **Auto-trade cycle:** Lọc signal < 60s → chọn score cao nhất → nếu có thì execute với TP/SL từ Settings; không dùng Signal Selector cho path này.
- **OrderExecutor:** Nhận `tp_sl_settings` (default_tp, default_sl) và tính TP/SL theo % từ entry.
- **Break-even:** Chỉ dùng TP/SL từ Settings trong bản đầu; break-even có thể bổ sung sau.

File này là bản thiết kế đã xác nhận (option A: chọn symbol có score cao nhất trong các signal < 60s).
