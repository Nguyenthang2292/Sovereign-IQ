# Phase 8.1: Intelligent Cache Warming & Parallelism

## Goal
Thiết kế và triển khai **8.2 Intelligent Cache Warming** và củng cố **9. Parallelism Improvements** cho `adaptive_trend_LTS` theo hướng thực tế, có cờ bật/tắt, có benchmark và tài liệu.

## Tasks

- [x] Task 1: Xác định điểm tích hợp cache warming hiện tại  
  → Verify: Done. Cache warming được tích hợp trực tiếp vào `CacheManager` thông qua `warm_cache(...)`, sử dụng `compute_atc_signals` để tính toán. Backend mặc định là file-based `.cache/atc`.

- [x] Task 2: Thêm helper `warm_cache(...)`  
  → Verify: Done. Đã thêm `warm_cache` vào `CacheManager` trong `utils/cache_manager.py`.

- [x] Task 3: Expose entrypoint cho cache warming (CLI/script)  
  → Verify: Done. Đã tạo `scripts/warm_cache.py`. Có thể chạy qua CLI để chuẩn bị cache trước khi scan/backtest.

- [x] Task 4: Thêm logging/metrics cho cache hits/misses (trước/sau warm)  
  → Verify: Done. Đã thêm `log_cache_effectiveness()` để in báo cáo chi tiết về hiệu quả cache.

- [x] Task 5: Rà soát hiện trạng Parallelism (async I/O + GPU streams)  
  → Verify: Done.
    - CPU: Đang dùng `ProcessPoolExecutor` và `shared_memory` trong `_parallel_layer1.py`.
    - GPU: Đang dùng single stream mặc định trong các CUDA kernels.
    - Gaps: Thiếu cơ chế async cho I/O-bound tasks (data fetching) và chưa tận dụng multi-stream cho GPU batching.

- [x] Task 6: Chuẩn hoá abstraction chạy song song trên CPU (backtesting / batch jobs)  
  → Verify: Done. Đã tạo `core/async_io/async_compute.py` cung cấp `AsyncComputeManager`, `compute_atc_signals_async` và `run_batch_atc_async`.

- [x] Task 7: Củng cố GPU multi-stream (nếu áp dụng)  
  → Verify: Done. Đã tạo `core/gpu_backend/multi_stream.py` với `GPUStreamManager` hỗ trợ round-robin stream allocation và synchronization.

- [x] Task 8: Thêm benchmark cho “cache warmed + parallelism”  
  → Verify: Done. Đã tạo `benchmarks/benchmark_cache_parallel.py` so sánh 4 chế độ:
    - Baseline (Sync, No Warming)
    - Warmed Only (Sync, Warmed)
    - Parallel Only (Async, No Warming)
    - Warmed + Parallel (Async, Warmed)
    Kết quả in ra bảng Speedup và Hit Rate chi tiết.

- [x] Task 9: Cập nhật tài liệu (optimization_suggestions + phase8/7)  
  → Verify: Done. 
    - `optimization_suggestions.md` đã cập nhật Section 8.2 & 9 với trạng thái ✅ **IMPLEMENTED** và ví dụ sử dụng (CLI/Code).
    - `phase8.1_task_glimmering-seeking-meadow.md` đã tổng hợp toàn bộ roadmap và trạng thái cuối cùng.

## Done When

- [x] Có lệnh/entrypoint rõ ràng để **warm cache** (`scripts/warm_cache.py`) và cơ chế log hit/miss thể hiện được hiệu quả.
- [x] Parallelism ở mức **CPU + GPU** được chuẩn hoá (`AsyncComputeManager`, `GPUStreamManager`) và có benchmark chứng minh lợi ích.
- [x] Docs (`optimization_suggestions.md`) mô tả đầy đủ cách bật/tắt cache warming và các chế độ parallelism.

