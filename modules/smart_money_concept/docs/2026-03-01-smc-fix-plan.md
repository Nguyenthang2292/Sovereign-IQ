# SMC: Fix PineScript Translation Bugs

## Goal
Sửa 5 lỗi Critical + 2 lỗi Medium được phát hiện trong audit `2026-03-01-pinescript-translation-audit.md`, đảm bảo logic Python khớp với PineScript gốc của LuxAlgo.

---

## Tasks

- [x] **Task 1: Sửa `swing.py` — default external_order**
  - Đổi `external_order=30` → `external_order=50` trong `detect_swings()` và `SMCAnalyzer.__init__()`
  - Verify: `SMCAnalyzer().external_order == 50`

- [x] **Task 2: Sửa `equal_hl.py` — detect pivot riêng với order=3**
  - Thêm tham số `equal_length: int = 3` vào `identify_equal_hl()`
  - Gọi `_detect_swing_pivots(df, order=equal_length)` riêng thay vì nhận `internal_highs/lows` từ bên ngoài
  - Verify: EQH/EQL xuất hiện nhiều hơn (nhạy hơn) so với trước

- [x] **Task 3: Refactor `bos.py` — dùng close crossover thay vì OR trên OHLC**
  - Thay đổi breakout condition: chỉ dùng `close` cross qua level (`close > level` AND `prev_close <= level`)
  - Duyệt từng bar trong `df_range` theo thứ tự thời gian, lấy bar đầu tiên thỏa mãn
  - Mỗi pivot chỉ được break một lần (set `crossed = True` sau khi detect)
  - Verify: Số lượng BOS giảm so với trước (chặt hơn)

- [x] **Task 4: Refactor `choch.py` — phân loại BOS vs CHoCH dựa trên trend state**
  - Xóa logic "tìm swing giữa 2 BOS timestamps"
  - Track `current_trend` (khởi tạo từ `detect_trend`)
  - Trong mỗi crossover event từ task 3: nếu cross **ngược** chiều `current_trend` → CHoCH, cùng chiều → BOS; sau đó flip `current_trend`
  - Trả về `BosChochResult` gộp chung cả BOS lẫn CHoCH với field `event_type`
  - Verify: CHoCH chỉ xảy ra khi trend đổi chiều

- [x] **Task 5: Sửa `trend.py` — dùng last structure break thay vì HH/HL pattern**
  - Thay `detect_trend()` nhận thêm tham số `last_structure_break: int` (BULLISH/BEARISH)
  - Trả về giá trị đó trực tiếp; chỉ fallback sang HH/HL pattern nếu chưa có structure break
  - Cập nhật `SMCAnalyzer.run()` để pass structure break từ task 4 vào
  - Verify: Trend = BULLISH sau BOS bullish, BEARISH sau BOS bearish

- [ ] **Task 6: Sửa `order_block.py` — gắn OB với structure break event**
  - Xóa logic tạo OB từ cặp swing liên tiếp
  - Khi `displayStructure` detect BOS/CHoCH bullish: tìm bar có `parsedLow = min` trong range `[pivot.bar_time → current_bar]` → tạo Bullish OB
  - Khi bearish: tìm bar có `parsedHigh = max` trong range → tạo Bearish OB
  - Thêm **volatility filter**: `parsedHigh = low if (high-low) >= 2*atr else high`, `parsedLow = high if (high-low) >= 2*atr else low`
  - Sửa mitigation: Bullish OB bị xóa khi `low < ob.barLow`; Bearish OB khi `high > ob.barHigh`
  - Verify: OB số lượng ~ bằng số BOS/CHoCH events

- [ ] **Task 7: Cập nhật `analyzer.py` — kết nối lại pipeline**
  - Cập nhật `SMCAnalyzer.run()` để dùng API mới từ tasks 3-6
  - Đảm bảo `export()` vẫn trả về đủ 15 elements
  - Verify: `SMCAnalyzer().run(df)` không raise exception

- [ ] **Task 8: Chạy tests**
  - Chạy `pytest tests/ -k smc` (hoặc toàn bộ test suite nếu không có test SMC riêng)
  - Verify: Không có regression

---

## Done When
- [x] `external_order` default = 50
- [x] BOS chỉ trigger khi `close` thực sự cross qua level
- [x] CHoCH xác định đúng dựa trên trend state tại thời điểm cross
- [ ] Order Block được tạo cùng lúc với BOS/CHoCH event (Task 6)
- [x] `SMCAnalyzer().run(df)` chạy thành công end-to-end

## Notes
- Tasks 3 + 4 + 5 phụ thuộc nhau — nên làm cùng lúc trong một lần refactor
- Task 6 phụ thuộc Task 3 (cần structure break event mới)
- FVG / Premium-Discount / MTF Levels **không nằm trong scope** của plan này
