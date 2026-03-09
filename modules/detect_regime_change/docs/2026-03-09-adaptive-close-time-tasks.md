# Adaptive Close Time — Implementation Tasks

## Goal

Triển khai hệ thống Adaptive Close Time dựa trên Regime Change Detection (PELT + HMM) để thay thế `max_duration_hours` cố định bằng giá trị tự động tính theo regime của từng symbol.

> **Design doc:** [2026-03-09-adaptive-close-time-design.md](./2026-03-09-adaptive-close-time-design.md)

---

## Phase 1 — Python Core

### 1.1 Data Models

- [x] Tạo `modules/detect_regime_change/models.py` — dataclass `ChangePoint`, `RegimeSegment`, `RegimeDurationResult` → Verify: `python -c "from modules.detect_regime_change.models import RegimeDurationResult; r = RegimeDurationResult(symbol='TEST', timeframe='15m'); print(r.is_valid)"` trả `False`

### 1.2 PELT Detector

- [x] Thêm `ruptures` vào `requirements.txt` → Verify: `pip install ruptures && python -c "import ruptures"`
- [x] Tạo `modules/detect_regime_change/pelt_detector.py` — hàm `detect_change_points_pelt()` và `calculate_pelt_avg_duration()` → Verify: pytest unit test với fake returns array cho ra `List[ChangePoint]` và `(avg_hours, median_hours)` hợp lệ

### 1.3 HMM Bridge

- [x] Tạo `modules/detect_regime_change/hmm_regime_bridge.py` — hàm `estimate_hmm_regime_duration()` bridge sang `modules.hmm.hmm_swings` → Verify: pytest mock `hmm_swings` return known values, assert output `(duration_hours, state, probability)` đúng đơn vị

### 1.4 Main Engine

- [x] Tạo `modules/detect_regime_change/regime_duration_analyzer.py` — class `RegimeDurationAnalyzer` với method `analyze()` và `_combine_results()` → Verify: pytest test với mock PELT + HMM kết quả, assert `recommended_duration_hours` đúng weighted average formula

### 1.5 Module Init

- [x] Tạo `modules/detect_regime_change/__init__.py` — export `RegimeDurationAnalyzer`, `RegimeDurationResult` → Verify: `python -c "from modules.detect_regime_change import RegimeDurationAnalyzer, RegimeDurationResult"`

### 1.6 Adaptive Close Calculator

- [x] Tạo `modules/auto_trade/execution/adaptive_close_calculator.py` — class `AdaptiveCloseCalculator` với `compute_adaptive_deadline()` và `_fetch_ohlcv()` → Verify: pytest test mock `RegimeDurationAnalyzer` + `settings_manager`, assert deadline = `opened_at + clamped_hours`

### 1.7 Settings Update

- [x] Cập nhật `settings.yaml` — thêm section `auto_close.adaptive` (`enabled`, `min_duration_hours`, `max_duration_hours`, `lookback_days`, `timeframe`) → Verify: YAML load thành công, giá trị mặc định `enabled: false`

### 1.8 Settings Manager Parse

- [x] Cập nhật `settings_manager.py` — parse `auto_close.adaptive.*` config → Verify: `settings_manager.get("auto_close.adaptive.enabled")` trả `False`

### 1.9 Order Flow Integration

- [x] Tích hợp `AdaptiveCloseCalculator` vào flow mở order — sau khi place thành công, gọi `compute_adaptive_deadline()` rồi set `auto_close_deadline_utc` lên order record trước khi save DB → Verify: `pytest tests/auto_trade/execution/test_adaptive_close_integration.py -v` pass, payload DB có `auto_close_deadline_utc`

### 1.10 Tests

- [x] Viết pytest cho `models.py`: `RegimeDurationResult.is_valid` trả đúng → Verify: `pytest tests/detect_regime_change/test_models.py -v`
- [x] Viết pytest cho `pelt_detector.py`: detect trên synthetic data có known breakpoints → Verify: `pytest tests/detect_regime_change/test_pelt_detector.py -v`
- [x] Viết pytest cho `hmm_regime_bridge.py`: mock `hmm_swings`, verify unit conversion → Verify: `pytest tests/detect_regime_change/test_hmm_bridge.py -v`
- [x] Viết pytest cho `regime_duration_analyzer.py`: mock PELT + HMM, test combine logic (high conf/low conf/only PELT/only HMM/both fail) → Verify: `pytest tests/detect_regime_change/test_analyzer.py -v`
- [x] Viết pytest cho `adaptive_close_calculator.py`: mock analyzer, test clamp + fallback + disabled → Verify: `pytest tests/auto_trade/test_adaptive_close.py -v`

---

## Phase 2 — Rust Optimization (sau Phase 1)

- [x] Setup `modules/detect_regime_change/rust_extensions/Cargo.toml` với `pyo3` dependency → Verify: `cargo check` pass
- [x] Implement PELT core trong Rust (`rust_extensions/src/lib.rs`) → Verify: `cargo test` pass
- [x] Tạo PyO3 bindings export hàm `detect_change_points_pelt_rs()` → Verify: `python -c "from detect_regime_change.rust_extensions import detect_change_points_pelt_rs"`
- [x] Thêm fallback logic trong `pelt_detector.py`: thử Rust trước, nếu import fail → dùng `ruptures` Python → Verify: pytest pass cả khi Rust module chưa build
- [x] Benchmark Python vs Rust trên 5000-candle data → Verify: Rust nhanh hơn ≥ 3x
- [x] Fix fallback order + model routing: không import `ruptures` sớm; route Rust theo `RUST_SUPPORTED_MODELS` → Verify: `pytest tests/detect_regime_change/test_pelt_detector.py -q`
- [x] Mở rộng Rust backend hỗ trợ model `normal` (ngoài `l2`) qua tham số `model` trong binding `detect_change_points_pelt_rs(..., model="l2")` → Verify: `cargo test` pass (bao gồm test `normal`)
- [x] Bỏ chế độ tương thích legacy Rust ABI (3 args), chuẩn hóa detector dùng duy nhất ABI model-aware 4 args → Verify: build lại extension và chạy `pytest tests/detect_regime_change/test_pelt_detector.py -q`
- [x] Bổ sung regression tests cho các case thiếu dependency (`rust missing`, `ruptures missing`) và model route (`l2`/`normal`) theo chế độ Rust thuần model-aware → Verify: `pytest tests/detect_regime_change/test_pelt_detector.py -q` (11 passed, 0 skipped)

---

## Phase 3 — GUI & Polish (sau Phase 2)

- [ ] Thêm section "Adaptive Close" trong GUI settings (toggle enable, min/max/lookback inputs) → Verify: mở GUI, thấy section mới, toggle hoạt động
- [ ] Hiển thị adaptive deadline trên Scheduled Exits panel (cột "Deadline Source: adaptive/static") → Verify: mở order với adaptive on, panel hiện "adaptive"
- [ ] Log chi tiết regime analysis trong live log panel → Verify: log hiện PELT/HMM/combined results khi order mở

---

## Done When

- [ ] Phase 1 hoàn thành: tất cả pytest pass, adaptive close **tắt** mặc định, bật lên tính đúng deadline
- [ ] Flow cũ (`max_duration_hours` cố định) vẫn hoạt động bình thường khi `adaptive.enabled: false`
- [ ] Không thay đổi `auto_close_timer.py` hay `auto_close_timer_job.py`

## Notes

- Dependency direction: `auto_trade → detect_regime_change → modules/hmm` (một chiều)
- `detect_regime_change` **không biết** về orders, auto-close, hay trading logic
- Safety: 4 layers — clamp boundary, fallback tĩnh, data sufficiency check, exception handling
- `adaptive.enabled: false` mặc định → zero-risk deploy
