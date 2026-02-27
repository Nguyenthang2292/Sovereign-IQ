# SMC v3.0 Refactor — Sub-Module Implementation

## Goal

Tách `SMC_v3_0.py` (1430 dòng, global state) thành package con có cấu trúc
`models/` → `core/` → `analyzer.py` → `charts/` → `cli.py`.
Chi tiết thiết kế: `modules/smart_money_concept/docs/2026-02-28-smc-refactor-design.md`

---

## Tasks

- [ ] **Task 1: Tạo `models/`** ❌ CHƯA ĐẠT
  - Tạo `models/__init__.py`
  - Move `pivot.py` → `models/pivot.py` (update import paths)
  - Tạo `models/order_block.py` từ `data_class.class_order_block` (port toàn bộ `OrderBlock` dataclass)
  - Verify: `from modules.smart_money_concept.models import Pivot, OrderBlock` không lỗi
  - **Lý do chưa đạt cụ thể:**
    - `modules/smart_money_concept/pivot.py` (file cũ) vẫn còn tồn tại, chưa hoàn tất bước move/cleanup.
    - `tests/smart_money_concept/test_pivot.py` vẫn import từ đường dẫn cũ `modules.smart_money_concept.pivot`, chưa update import path sang `models`.
    - Không có đủ bằng chứng trong repo để xác minh việc `OrderBlock` đã được port **toàn bộ** từ `data_class.class_order_block` (không tìm thấy file nguồn để đối chiếu).

- [x] **Task 2: Tạo `core/trend.py` + `core/swing.py`** ✅ 2026-02-28
  - `trend.py`: `detect_trend(highs, lows) -> int`, `compute_atr(...)` — không global
  - `swing.py`: dataclass `SwingResult`, hàm `detect_swings(df, ...) -> SwingResult`, `classify_swing_types(highs, lows)`
  - Verify: gọi trực tiếp với list `Pivot` giả → trả về đúng kiểu

- [x] **Task 3: Tạo `core/bos.py` + `core/choch.py`** ✅ 2026-02-28
  - `bos.py`: dataclass `BOSResult(high_bos, low_bos)`, hàm `identify_bos(df, highs, lows) -> BOSResult`
  - `choch.py`: dataclass `ChochResult(bullish, bearish)`, hàm `identify_choch(bos, highs, lows) -> ChochResult`
  - Verify: input mock `SwingResult` → output `BOSResult` / `ChochResult` có đúng fields

- [ ] **Task 4: Tạo `core/equal_hl.py` + `core/order_block.py`**
  - `equal_hl.py`: dataclass `EqualHLResult`, hàm `identify_equal_hl(internal_highs, internal_lows, highs_arr, lows_arr, closes_arr, ...)`
  - `order_block.py`: port toàn bộ `build_internal_order_blocks`, `build_swing_order_blocks`, `process_swings`, `filter_order_blocks`, `update_order_blocks` → wrap vào `identify_order_blocks(df, highs, lows, trend) -> list[OrderBlock]`
  - Verify: `core/` hoàn toàn không có `import plotly`

- [ ] **Task 5: Tạo `analyzer.py`**
  - Dataclass `SMCState` với đủ 9 fields (swings, trend, bos×2, choch×2, equal_hl, ob×2, ohlcv)
  - Class `SMCAnalyzer` với `run(df) -> SMCState`
  - Method `export(df) -> tuple` trả về đúng 15 values như `export_data()` cũ
  - Verify: `SMCAnalyzer().export(df)` cho ra cùng kiểu dữ liệu như `export_data()` trong `SMC_v3_0.py`

- [ ] **Task 6: Tạo `charts/`**
  - `charts/__init__.py`, `charts/renderer.py` (class `SMCChartRenderer` với `render(state, ticker) -> go.Figure`)
  - `swing_chart.py`, `bos_chart.py`, `choch_chart.py`, `equal_hl_chart.py`, `order_block_chart.py`
  - Port toàn bộ `draw_*` functions từ `SMC_v3_0.py` vào các file tương ứng, nhận `SMCState` thay vì đọc global
  - Verify: `SMCChartRenderer().render(state)` tạo ra figure hiển thị được

- [ ] **Task 7: Tạo `cli.py`**
  - Viết lại `main()` từ `SMC_v3_0.py` dùng `SMCAnalyzer` + `SMCChartRenderer`
  - `python -m modules.smart_money_concept` hoặc `python cli.py` chạy được
  - Verify: nhập `AAPL` → chart hiện lên đúng như trước refactor

- [ ] **Task 8: Cập nhật `__init__.py` public API**
  - Re-export: `SMCAnalyzer`, `SMCState`, `Pivot`, `OrderBlock`
  - Verify: `from modules.smart_money_concept import SMCAnalyzer` không lỗi

- [ ] **Task 9: Viết unit tests cho `core/`**
  - `tests/smart_money_concept/test_trend.py` — 3 case: BULLISH / BEARISH / NEUTRAL
  - `tests/smart_money_concept/test_swing.py` — detect_swings với df giả
  - `tests/smart_money_concept/test_bos.py` — identify_bos với df + list Pivot mock
  - `tests/smart_money_concept/test_analyzer.py` — integration: `SMCAnalyzer().run(df)` trả về `SMCState` đủ fields
  - Verify: `pytest tests/smart_money_concept/ -v` toàn bộ PASS

- [ ] **Task 10: Kiểm tra backward compat + dọn dẹp**
  - Chạy bất kỳ code nào đang import từ `SMC_v3_0.py` (ví dụ `auto_trade`), đổi sang dùng `SMCAnalyzer.export()`
  - Kiểm tra `SMC_v3_0.py` có thể giữ lại như file legacy (deprecated) hoặc xoá nếu không còn dependent
  - Verify: `pytest tests/ -v` toàn project không có regression mới

---

## Done When

- [ ] `from modules.smart_money_concept import SMCAnalyzer` hoạt động
- [ ] `core/` không có bất kỳ `import plotly` hay `global` nào
- [ ] `pytest tests/smart_money_concept/ -v` — tất cả PASS
- [ ] `cli.py` chạy standalone cho ra chart đúng

## Notes

- Bắt đầu từ Task 1 → Task 5 trước khi chạm vào charts/
- Task 2–4 có thể làm song song nếu cần (không phụ thuộc nhau trong core/)
- Giữ `SMC_v3_0.py` nguyên vẹn cho đến khi Task 10 hoàn tất (safety net)
