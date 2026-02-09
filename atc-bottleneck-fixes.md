# ATC Module Bottleneck Fixes

## Goal
Giảm thời gian scan 300+ symbols bằng cách sửa 5 bottleneck đã xác định trong ATC (ưu tiên I/O trước compute).

## Tasks

- [x] **1. Throttled call: rate limiter thay global lock** — `public.py`: lock chỉ cho bookkeeping (`_last_request_ts`), `func()` gọi ngoài lock, sleep ngoài lock. ✅ DONE
- [x] **2. TTL-based OHLCV cache** — `ohlcv.py`: cache có `_ohlcv_cache_timestamps`, TTL = timeframe × `cache_ttl_multiplier`; `check_freshness=True` vẫn đọc cache nếu trong TTL. ✅ DONE
- [x] **3. Parallel exchange fallback** — `ohlcv.py`: `_try_exchanges_parallel` probe 3 exchange song song (`ThreadPoolExecutor`), `as_completed` lấy kết quả đầu tiên thành công, còn lại fallback sequential. ✅ DONE
- [x] **4. Reuse thread pool cho MA** — `compute_atc_signals.py`: 1 `shared_executor` truyền vào `set_of_moving_averages(executor=shared_executor)` cho cả 6 MA types. ✅ DONE
- [x] **5. Single scanner pool + giảm GC** — `threadpool.py`: 1 `ThreadPoolExecutor` cho toàn bộ scan; `gc.collect()` chỉ gọi 1 lần sau scan. ✅ DONE
- [x] **6. Verification** — Chạy ATC scan 50–100 symbols, đo end-to-end, xác nhận không regression. ✅ DONE

## Done When

- [x] `throttled_call` không serialize toàn bộ requests (lock chỉ cho bookkeeping).
- [x] Cache OHLCV có TTL khi `check_freshness=True`, giảm redundant fetches.
- [x] Fallback exchange chạy song song thay vì tuần tự.
- [x] MA compute và scanner dùng pool reuse; không gọi `gc.collect()` mỗi batch.
- [ ] Tests pass; thời gian scan 50–100 symbols giảm so với baseline.

## Test Coverage (đã bổ sung)

- **TTL cache** (`tests/common/test_data_fetcher.py`): `test_fetch_ohlcv_check_freshness_uses_cache_within_ttl`, `test_fetch_ohlcv_check_freshness_refetches_when_cache_expired`, `test_fetch_ohlcv_cache_ttl_multiplier_extends_cache_window`, `test_fetch_ohlcv_ttl_boundary_just_under_ttl_hits_cache`, `test_fetch_ohlcv_ttl_boundary_just_over_ttl_refetches`.
- **Parallel fallback** (`tests/common/test_data_fetcher.py`): `test_fetch_ohlcv_parallel_probe_first_success_wins`, `test_fetch_ohlcv_parallel_probe_first_fails_second_succeeds`, `test_fetch_ohlcv_parallel_probe_all_fail_then_sequential_fallback`, `test_fetch_ohlcv_parallel_probe_all_exchanges_fail_returns_none`.
- **Scanner threadpool** (`tests/adaptive_trend_LTS_mini/test_scanner_threadpool.py`): `test_threadpool_uses_single_executor_for_entire_scan`, `test_threadpool_worker_exception_continues_other_symbols`, `test_threadpool_gc_called_once_after_scan`, `test_threadpool_keyboard_interrupt_handled_cleanly`.

## Notes

- REF: `ATC_MODULE_BOTTLENECK_ANALYSIS.md` (impact order: #1 → #3 → #2 → #4 → #5).
- ccxt `enableRateLimit=True` đã có; tránh double rate-limit khi refactor #1.
- Có thể thêm `pytest-benchmark` hoặc script `scripts/bench_atc_scan.py` để đo trước/sau.
