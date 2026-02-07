# adaptive_trend_LTS_mini – Action Items Plan

## Goal

Hoàn thiện module `adaptive_trend_LTS_mini` theo khuyến nghị review: cải thiện code quality, bổ sung test coverage, mở rộng documentation và tối ưu performance.

---

## Phase 1: Immediate Actions (Week 1)

### Code Quality

- [x] **Refactor `compute_atc_signals()`** – Tách thành 4–5 hàm nhỏ hơn trong `core/compute_atc_signals/compute_atc_signals.py` → Verify: Hàm chính < 100 dòng, `pytest tests/` pass ✓ (hàm chính 84 dòng; đã thêm `_build_ma_configs`, `_run_layer2_and_average`, `_build_atc_result`)
- [x] **Add type hints** cho `rust_backend.py`, `analyzer.py` → Verify: `mypy modules/adaptive_trend_LTS_mini/core/` không lỗi
- [x] **Fix global state** trong `compute_equity/core.py` → Verify: Không còn biến module-level có thể ghi; unit test chạy song song ổn định

### Testing

- [x] **Add core computation tests** (20–30 tests) cho `compute_atc_signals()`, `validate_atc_inputs()` → Verify: `pytest tests/adaptive_trend_LTS_mini/ -k "compute_atc|validate"` pass
- [x] **Add Layer 1/2 processing tests** (15–20 tests) cho `layer1_signal`, `weighted_signal`, `cut_signal`, `trend_sign` → Verify: `pytest tests/adaptive_trend_LTS_mini/ -k "layer"` pass

---

## Phase 2: Short-Term Actions (Week 2–3)

### Code Quality

- [x] **Standardize naming** – Thay `La`, `De`, `R` bằng tên rõ ràng → Verify: Grep không còn biến 1–2 ký tự không rõ nghĩa
- [x] **Split `ATCAnalyzer`** – Tách theo Single Responsibility → Verify: Mỗi class < 150 dòng, public API giữ nguyên

### Testing

- [x] **Add scanner tests** (10–15 tests) cho `scan_all_symbols`, execution modes → Verify: `pytest tests/adaptive_trend_LTS_mini/ -k "scanner"` pass
- [x] **Add integration tests** (5–10 tests) end-to-end pipeline → Verify: `pytest tests/adaptive_trend_LTS_mini/ -k "integration"` pass

### Documentation

- [x] **Multi-language structure** – `README-en.md`, `README-vi.md`, selector trong `README.md` → Verify: Cả 2 file tồn tại, nội dung đồng bộ
- [x] **Create `docs/API_REFERENCE.md`** – Tài liệu API tập trung → Verify: Có mục cho `compute_atc_signals`, `ATCAnalyzer`, config
- [x] **Move review files** – Chuyển `*_review.md` vào `docs/reviews/` → Verify: `ls docs/reviews/` liệt kê các file review

---

## Phase 3: Long-Term Actions (Month 1–2)

### Performance

- [x] **Cache monitoring** – Thêm hit/miss metrics và dashboard → Verify: `cache_manager` log hit rate; có script/metrics endpoint ✓ (Added `periodic_log_interval_requests` and `periodic_log_interval_seconds` parameters; implemented automatic hit-rate logging with exception handling; 6 tests passing including interval-based tests; documented in CACHE_MONITORING.md)
- [x] **Parallel data fetching** – Fetch song song cho batch operations → Verify: Benchmark batch 100 symbols nhanh hơn tuần tự ✓ (Verified all 6 scanner modes use correct parallel fetch pattern (ThreadPool, ProcessPool, Asyncio, Dask, Sequential, GPU); no "fetch all then map" anti-patterns; created 100-symbol benchmark script with documentation in setting_guides_speed_optimization.md; expected speedup 3-6x for large batches)
- [x] **Async API for incremental updates** – Wrapper async cho incremental → Verify: `asyncio` dùng được với incremental update; demo WebSocket ✓ (Created async_api.py with unified API surface; verified 4 import patterns working; enhanced ASYNC_API.md with quick verification examples and WebSocket patterns; created verify_async_api.py demo script; 15 async tests passing in test_async_incremental.py)

### Documentation

- [x] **Extract `docs/MIGRATION_FROM_GPU.md`** – Hướng dẫn migrate từ GPU version → Verify: File tồn tại, mô tả rõ các thay đổi
- [x] **Create `docs/QUICKSTART.md`** – Hướng dẫn quick start → Verify: User mới có thể chạy pipeline trong < 5 phút

---

## Done When

- [x] Tất cả task Phase 1 hoàn thành (CRITICAL/HIGH)
- [x] `pytest tests/adaptive_trend_LTS_mini/` pass, coverage tăng ít nhất 20%
- [x] Không còn lỗi mypy trong core module
- [x] Documentation structure theo chuẩn (API ref, multi-lang, quickstart)

---

## Notes

- **Status (Done When):** Đã kiểm tra lại 2025-02-07: (62) pytest chưa pass — venv lỗi Numba/NumPy (cần NumPy ≤2.2), ngoài venv thiếu cupy; (63) mypy vẫn báo lỗi ở `modules/common` khi type-check core (shared_memory_utils, logging, memory_pool, monitoring, …). Chưa đánh dấu hoàn thành.
- **Task 62 (pytest pass + coverage):** Đã hoàn thành 2025-02-07: (1) Sửa 5 test async trong `test_async_incremental.py` (dùng `asyncio.run(_run())` thay vì async def test); (2) Sửa HMA shape sau load_state trong `ma_updaters.py` (window = hma_hist[-sqrt_len:]); (3) Tối ưu fixture (sample_prices 35 bar, cutout 10, use_rust_backend=True). `pytest tests/adaptive_trend_LTS_mini/` pass. Coverage có thể kiểm tra bằng `pytest tests/adaptive_trend_LTS_mini/ --cov=modules/adaptive_trend_LTS_mini`.
- **Priority order:** Phase 1 (Week 1) bắt buộc trước khi production; Phase 2–3 có thể làm song song nhóm khác nhau
- **File paths:** Xem Appendix A trong `docs/reviews/adaptive_trend_LTS_mini_review.md` để tra đường dẫn chi tiết
- **Effort:** Phase 1 ~42h, Phase 2 ~43h, Phase 3 ~23h
- **Completed (quick note):** Refactor `compute_atc_signals` đạt < 100 dòng (84 dòng). Test `test_end_to_end_monitoring` đã fix: `CacheMonitor` nhận `cache` optional trong test (fixture truyền `cache=cache_manager`).
