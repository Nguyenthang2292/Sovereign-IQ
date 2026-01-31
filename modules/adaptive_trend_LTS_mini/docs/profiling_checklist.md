# Minimal Profiling Checklist

Checklist tóm tắt để chạy profiling khi cần điều tra hiệu năng trong adaptive_trend_LTS.

---

## Checklist Steps

### Step 1: Xác nhận vấn đề (Identify the Problem)

- [ ] **Chạy benchmark bình thường** để xác nhận vấn đề:
  ```bash
  python -m modules.adaptive_trend_LTS.benchmarks.benchmark_comparison.main --symbols 20 --bars 500
  ```

- [ ] **Ghi chú performance:**
  - Thời gian execute total.
  - RAM usage (nếu có thể quan sát).
  - Module nào chạy chậm nhất.

- [ ] **Xác nhận hot path:** Module/function nào cần điều tra (Ví dụ: benchmark comparison, specific module runner).

---

### Step 2: Chạy Profiling (Run Profiling)

#### Option A: Dùng Profiling Helper Script (Recommended)

- [ ] **Chạy script helper với flag --both:**
  ```bash
  python -m modules.adaptive_trend_LTS.scripts.profile_benchmarks --both --symbols 20 --bars 500
  ```

- [ ] **Hoặc chạy --cprofile hoặc --pyspy nếu chỉ cần một công cụ:**
  - `--cprofile`: Chỉ chạy cProfile (để lấy stats file).
  - `--pyspy`: Chỉ chạy py-spy (để lấy flamegraph SVG).

- [ ] **Xác nhận profiling output được tạo:**
  - File `.stats` trong `modules/adaptive_trend_LTS/profiles/` (cProfile output).
  - File `.svg` trong `modules/adaptive_trend_LTS/profiles/` (py-spy flamegraph).
  - **Note**: Script helper tự động tạo thư mục `profiles/` nếu chưa tồn tại.

#### Option B: Chạy Profiling Directly (Alternative)

- [ ] **Chạy cProfile trực tiếp:**
  ```bash
  # Tạo thư mục profiles/ trước (nếu chưa có)
  mkdir -p modules/adaptive_trend_LTS/profiles
  
  python -m cProfile -o modules/adaptive_trend_LTS/profiles/benchmark_comparison.stats -m modules.adaptive_trend_LTS.benchmarks.benchmark_comparison.main --symbols 20 --bars 500
  ```

- [ ] **Chạy py-spy trực tiếp:**
  ```bash
  # Tạo thư mục profiles/ trước (nếu chưa có)
  mkdir -p modules/adaptive_trend_LTS/profiles
  
  py-spy record -o modules/adaptive_trend_LTS/profiles/benchmark_comparison_flame.svg -- python -m modules.adaptive_trend_LTS.benchmarks.benchmark_comparison.main --symbols 20 --bars 500
  ```

---

### Step 3: Phân tích kết quả (Analyze Results)

#### Phân tích cProfile Stats

- [ ] **Mở cProfile stats trong interactive pstats:**
  ```bash
  python -m pstats modules/adaptive_trend_LTS/profiles/benchmark_comparison.stats
  ```

- [ ] **Xem top functions by cumulative time:**
  - Trong pstats prompt, gõ: `stats 20`
  - Ghi lại top 10-20 functions tốn nhiều thời gian nhất.

- [ ] **Xem top functions by self-time:**
  - Trong pstats prompt, gõ: `stats 20` sau đó `sort time`
  - Ghi lại functions tốn nhiều time tự chúng (không bao gồm subcalls).

- [ ] **Explore callees của hàm top:**
  - Trong pstats prompt, gõ: `callees <function_name>`
  - Xem hàm nào được gọi bởi hàm top.

#### Phân tích Flamegraph

- [ ] **Mở flamegraph SVG trong browser:**
  ```bash
  # Windows
  start modules/adaptive_trend_LTS/profiles/benchmark_comparison_flame.svg
  # macOS
  open modules/adaptive_trend_LTS/profiles/benchmark_comparison_flame.svg
  # Linux
  xdg-open modules/adaptive_trend_LTS/profiles/benchmark_comparison_flame.svg
  ```

- [ ] **Xác nhận hot paths (wide bars):**
  - Nhìn vào flamegraph để tìm những stack bars rộng nhất.
  - Những bars này đại diện cho functions tốn nhiều thời gian.

- [ ] **Ghi lại functions trong hot paths:**
  - Ghi lại tên hàm, path trong stack.
  - Xem có patterns nào lặp lại (ví dụ: nhiều calls đến cùng hàm optimization).

---

### Step 4: Tìm Bottleneck (Find Bottlenecks)

- [ ] **Xác định bottleneck trong cProfile stats:**
  - Function có cumulative time cao nhất.
  - Function có call count cao nhất.
  - Function có average time per call cao nhất.

- [ ] **Xác định bottleneck trong flamegraph:**
  - Stack bar rộng nhất (chiếm nhiều screen real estate).
  - Những bars lặp lại nhiều lần trên flamegraph (frequent hot paths).

- [ ] **Cross-verify cProfile và flamegraph:**
  - Hãy xem functions top trong cProfile có tương ứng với hot paths trong flamegraph?
  - Nếu có, đó là true bottleneck cần ưu tiên tối ưu hóa.

---

### Step 5: Tối ưu hóa (Optimize)

- [ ] **Xem code của bottleneck function:**
  - Mở file source code chứa function đó.
  - Xem có loops, repeated calculations, hoặc heavy computations?

- [ ] **Xem có thể refactors để giảm complexity:**
  - Có thể vectorize bằng NumPy?
  - Có thể cache kết quả intermediate?
  - Có thể move đến Rust/C++ extension?

- [ ] **Triển khai tối ưu hóa:**
  - Refactor code theo hướng tìm được.
  - Thêm caching nếu phù hợp.
  - Thêm parallelization nếu phù hợp.

- [ ] **Chạy profiling lại sau tối ưu hóa:**
  - Lặp lại Step 1-4 để verify improvement.
  - So sánh kết quả profiling cũ vs mới để đo lường giảm % thời gian.

---

## Notes & Tips

### Profiling Overhead

- **cProfile overhead**: ~20-30% slowdown.
- **py-spy overhead**: ~5-15% slowdown.
- **Lưu ý**: Profiling results bao gồm overhead, nên thời gian execute sẽ lâu hơn benchmark bình thường.

### Profiling Frequency

- **Chạy profiling 2-3 lần**: Để đảm bảo kết quả ổn định, hãy chạy profiling nhiều lần và lấy trung bình.
- **Tắt profiling cho final benchmark**: Khi đo lường final performance improvement, hãy tắt profiling (chạy benchmark bình thường).

### Profiling Artifacts

- **Git ignore**: Thư mục `profiles/` (cả ở root và trong `modules/adaptive_trend_LTS/`) đã được ignore trong `.gitignore`, nên profiling artifacts không bị commit.
- **Output location**: Script helper tạo output trong `modules/adaptive_trend_LTS/profiles/` (tự động tạo thư mục nếu chưa có).
- **Cleanup**: Có thể xóa files trong `profiles/` sau khi đã hoàn thành analysis (giữ disk space).

---

## Example Output Template

Khi hoàn thành profiling session, có thể điền template sau để ghi chú findings:

```
=== Profiling Session ===
Date: YYYY-MM-DD HH:MM:SS
Benchmark: --symbols 20 --bars 500
Issue: [Mô tả vấn đề cần điều tra]

=== Profiling Results ===
Top 5 Functions (by cumulative time):
1. function_name() - X.XXs (XX% total)
2. function_name() - X.XXs (XX% total)
...

Hot Paths (from flamegraph):
1. function_name() - [Stack depth: X]
2. function_name() - [Stack depth: Y]
...

=== Bottleneck Identified ===
Primary bottleneck: function_name()
Root cause: [Mô tả nguyên nhân - e.g., nested loop, missing cache]

=== Optimization Plan ===
Optimization: [Mô tả tối ưu hóa đã áp dụng]
Expected improvement: [Mô tả kỳ vọng - e.g., 10-20% faster]

=== Verification ===
Pre-optimize time: X.XXs
Post-optimize time: X.XXs
Improvement: XX% faster
```

---

**End of Minimal Profiling Checklist**
