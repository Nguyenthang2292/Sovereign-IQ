# Phase 8.2: Code Generation & JIT Specialization

## Goal
Khai thác **code generation / JIT specialization** cho các cấu hình ATC phổ biến nhằm giảm overhead cấu hình và đạt thêm ~10–20% tốc độ cho các config được lặp lại nhiều lần, mà vẫn giữ code base rõ ràng, có thể tắt/bật.

## Tasks

- [ ] Task 1: Xác định các cấu hình ATC “hot path” cần chuyên biệt hóa  
  → Verify: Có danh sách ngắn (3–5 cấu hình) trong doc hoặc comment (vd. EMA-only, KAMA-only, combo phổ biến) kèm thống kê sơ bộ (từ logs/usage) cho thấy đây là config được gọi thường xuyên.

- [ ] Task 2: Thiết kế API specialization (wrapper hoặc factory)  
  → Verify: Có một interface rõ ràng (vd. `get_specialized_compute_fn(config)` hoặc `compute_atc_specialized(prices, config)`) được định nghĩa trong một module riêng (vd. `core/codegen/specialization.py`), chưa cần tối ưu nặng nhưng type/signature ổn định.

- [ ] Task 3: Implement JIT specialization tối thiểu cho 1–2 case (EMA-first)  
  → Verify: Với một cấu hình EMA đơn giản, đường gọi chuyên biệt (Numba `generated_jit` hoặc pattern tương đương) chạy được, trả kết quả giống hệt đường chuẩn (`compute_atc_signals`) trên cùng bộ test small dataset.

- [ ] Task 4: Thêm fallback an toàn & cờ cấu hình  
  → Verify: Có flag (trong config hoặc param) cho phép bật/tắt specialization (vd. `use_codegen_specialization: bool`), và khi tắt thì toàn bộ pipeline quay về code path chuẩn mà không thay đổi kết quả; unit test so sánh 2 mode cho cùng input.

- [ ] Task 5: Benchmark micro cho specialized vs non-specialized  
  → Verify: Có benchmark nhỏ (script hoặc test benchmark) chạy lặp lại 1–2 cấu hình phổ biến, log ra thời gian cho: (a) đường chuẩn, (b) đường specialized, và cho thấy xu hướng >= 10% cải thiện trên repeated calls (sau warm-up JIT).

- [ ] Task 6: Quyết định scope mở rộng (có tiếp tục hay giữ ở mức experimental)  
  → Verify: Trong doc (phase8.2 hoặc optimization_suggestions), có ghi rõ: case nào được chuyên biệt hóa chính thức, case nào vẫn đi đường generic, và lý do (complexity vs lợi ích), để dev khác nắm được chiến lược dài hạn.

- [ ] Task 7: Cập nhật tài liệu & ví dụ sử dụng  
  → Verify: `optimization_suggestions.md` (mục 10) và `phase8_task.md`/`phase8.2_task.md` mô tả cách bật `use_codegen_specialization`, luồng fallback, và cung cấp ít nhất một snippet code mẫu cho cách gọi specialized path.

## Done When

- [ ] Có ít nhất một đường chạy ATC phổ biến được JIT-specialize thành công với kết quả y hệt đường chuẩn.
- [ ] Có flag/bật–tắt rõ ràng và fallback an toàn về đường generic khi cần.
- [ ] Benchmark micro cho thấy lợi ích thực tế (≥ 10% trên repeated calls) hoặc có kết luận rõ ràng trong docs nếu lợi ích không đủ để mở rộng phạm vi.

