Kiêm# Gann Fan Lines Refactor

## Goal
Thay thế horizontal bands (Fibonacci-style) bằng Gann Fan Lines chéo radiating từ pivot point.

## Tasks

- [x] Task 1: Redesign `GannZone` dataclass — thêm `pivot_index`, `pivot_price`, `slope`; thêm method `price_at(candle_index)`; xóa `upper_price`/`lower_price` cố định
  → Verify: `GannZone(pivot_index=0, pivot_price=100.0, slope=-1.0).price_at(5) == 95.0`

- [x] Task 2: Rewrite `GannCalculator` — tính `price_per_candle = price_range / (t_low - t_high)`; tạo 4 fan lines với slopes `[1/4, 2/4, 3/4, 4/4] * price_per_candle`; update `_find_zone()` → dùng `zone.price_at(current_index)` thay vì bounds cố định
  → Verify: `GannSquareResult` có `current_zone` đúng khi cho OHLCV sample

- [x] Task 3: Rewrite `GannChartGenerator` — thay `axhspan`/`axhline` bằng `ax.plot(timestamps, fan_prices)` + `ax.fill_between(timestamps, upper_fan, lower_fan)`; đặt zone label giữa vùng tại candle cuối
  → Verify: chart smoke test tạo file `.png` > 0 bytes, không exception

- [x] Task 4: Update `prompts/gann_analysis.txt` — đổi "4 horizontal Gann zones" → "4 diagonal Gann Fan zones radiating from the pivot point"; cập nhật Zone Breakdown mô tả boundaries là dynamic
  → Verify: file không còn chứa chuỗi "horizontal"

- [x] Task 5: Rewrite `tests/test_gann_calculator.py` — thay assertions `upper_price`/`lower_price` float cố định bằng `zone.price_at(pivot_index)`; thay `zone.contains(price)` bằng `zone.contains_at(price, current_index)`
  → Verify: `pytest modules/gemini_gann_square/tests/test_gann_calculator.py -q` passes

- [x] Task 6: Update `tests/test_gann_chart_generator.py` — thêm test kiểm tra chart có fan lines (không còn `axhspan`)
  → Verify: `pytest modules/gemini_gann_square/tests/ -q` — all pass

## Done When
- [x] Tất cả tests pass (`pytest modules/gemini_gann_square/tests/ -q`)
- [x] Chart visual có đường chéo fan lines từ pivot, không phải dải ngang
- [x] Không còn `upper_price`/`lower_price` cố định trong `GannZone`
