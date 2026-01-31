# Profiling Guide: adaptive_trend_LTS

Hướng dẫn sử dụng công cụ profiling (cProfile + py-spy) để phân tích hiệu năng và tìm bottleneck trong module adaptive_trend_LTS.

## Overview

- **cProfile**: Python built-in profiler để đo chi tiết thời gian thực thi theo hàm.
- **py-spy**: Công cụ visualization tạo flame graph từ profiling data.
- **Target**: Benchmark comparison pipeline tại `modules/adaptive_trend_LTS/benchmarks/benchmark_comparison/main.py`.

---

## 1. Installation

### 1.1 Install Required Tools

```bash
# cProfile (built-in, không cần cài)
# py-spy (cho flamegraph visualization)
pip install py-spy

# Optional: snakeviz cho pstats visualization
pip install snakeviz

# Optional: gprof2dot cho call graph visualization
pip install gprof2dot graphviz
```

### 1.2 Verify Installation

```bash
# Verify py-spy
py-spy --version
# Expected output: py-spy vX.X.X (or similar)

# Verify cProfile (built-in)
python -c "import cProfile; print('cProfile available')"
```

---

## 2. Using Profiling Helper Script

Để đơn giản hóa quy trình, chúng tôi đã cung cấp script helper:

### 2.1 Run Profiling Helper

Script `scripts/profile_benchmarks.py` cung cấp giao diện thống nhất để chạy cả cProfile và py-spy.

```bash
# Chạy cả cProfile và py-spy (mặc định)
python -m modules.adaptive_trend_LTS.scripts.profile_benchmarks --both

# Chỉ chạy cProfile
python -m modules.adaptive_trend_LTS.scripts.profile_benchmarks --cprofile --symbols 20 --bars 500

# Chỉ chạy py-spy
python -m modules.adaptive_trend_LTS.scripts.profile_benchmarks --pyspy --symbols 20 --bars 500
```

### 2.2 Profiling Helper Options

| Option | Description |
|--------|-------------|
| `--cprofile` | Chạy với cProfile chỉ |
| `--pyspy` | Chạy với py-spy chỉ |
| `--both` | Chạy cả hai (mặc định) |
| `--symbols N` | Số symbols benchmark (passed to benchmark) |
| `--bars N` | Số bars per symbol (passed to benchmark) |
| `--timeframe T` | Timeframe (passed to benchmark) |
| `--clear-cache` | Xóa cache trước khi benchmark |

---

## 3. Using cProfile Directly

Nếu script helper không phù hợp với nhu cầu, bạn có thể chạy cProfile trực tiếp.

### 3.1 Run Benchmark with cProfile

```bash
# Lệnh cơ bản
python -m cProfile -o profiles/benchmark_comparison.stats -m modules.adaptive_trend_LTS.benchmarks.benchmark_comparison.main --symbols 20 --bars 500

# Với các tham số đầy đủ
python -m cProfile -o profiles/benchmark_comparison_stats.stats -m modules.adaptive_trend_LTS.benchmarks.benchmark_comparison.main --symbols 20 --bars 500 --timeframe 1h --clear-cache
```

### 3.2 Analyze cProfile Output

#### Option A: Interactive pstats

```bash
# Mở interactive prompt
python -m pstats profiles/benchmark_comparison.stats

# Các lệnh cơ bản trong pstats prompt:
stats 10              # Show top 10 functions by cumulative time
callees <func_name>  # Show functions called by <func_name>
callers <func_name>   # Show functions calling <func_name>
sort cumulative      # Sort by cumulative time (default)
sort time             # Sort by time spent in function (excluding subcalls)
```

#### Option B: Snakeviz Visualization

```bash
# Tạo visualization từ pstats
snakeviz profiles/benchmark_comparison.stats

# Mở visualization trong browser (trên port 8080 mặc định)
# Tự động mở tab mới với visualization
```

#### Option C: gprof2dot Call Graph

```bash
# Tạo call graph
gprof2dot -f pstats profiles/benchmark_comparison.stats | dot -Tpng -o profile.png

# Mở file PNG
```

---

## 4. Using py-spy Flamegraph

py-spy tạo flame graph visualization để dễ dàng nhìn thấy call stack bottleneck.

### 4.1 Run Benchmark with py-spy

```bash
# Lệnh cơ bản
py-spy record -o profiles/benchmark_comparison_flame.svg -- python -m modules.adaptive_trend_LTS.benchmarks.benchmark_comparison.main --symbols 20 --bars 500

# Với các tham số đầy đủ
py-spy record -o profiles/benchmark_comparison_flame.svg -- python -m modules.adaptive_trend_LTS.benchmarks.benchmark_comparison.main --symbols 20 --bars 500 --timeframe 1h --clear-cache
```

### 4.2 View Flamegraph

```bash
# Mở SVG file trong browser
start profiles/benchmark_comparison_flame.svg  # Windows
open profiles/benchmark_comparison_flame.svg     # macOS
xdg-open profiles/benchmark_comparison_flame.svg # Linux

# Hoặc mở trực tiếp từ URL file://
# file:///D:/path/to/profiles/benchmark_comparison_flame.svg
```

### 4.3 Reading Flamegraph

- **Width of bar**: Thời gian tổng của hàm đó.
- **Height of stack**: Depth trong call stack (nested calls).
- **Color**: Random (để phân biệt các hàm khác nhau).
- **Hot paths**: Những stack rộng màu sắc nổi bật, chỉ ra nơi hầu hết thời gian được tiêu tốn.

---

## 5. Profiling Artifacts

### 5.1 Output Locations

Tất cả profiling output được lưu vào thư mục `profiles/`:

```
modules/adaptive_trend_LTS/
└── profiles/
    ├── benchmark_comparison.stats        # cProfile binary stats
    ├── benchmark_comparison_flame.svg # py-spy flamegraph
    └── ...                            # Other profiling files
```

### 5.2 Git Ignore

Thư mục `profiles/` đã được ignore trong `.gitignore` để tránh commit profiling artifacts.

```bash
# Kiểm tra .gitignore (đã cấu hình)
cat .gitignore | grep profiles
# Output: profiles/
```

---

## 6. Minimal Profiling Checklist

Dưới đây là checklist tóm tắt để chạy profiling khi cần điều tra hiệu năng:

### Step 1: Xác nhận vấn đề
- [ ] Chạy benchmark bình thường để xác nhận vấn đề.
- [ ] Ghi chú thời gian execute và RAM usage.

### Step 2: Chạy Profiling
- [ ] Sử dụng script helper: `python -m modules.adaptive_trend_LTS.scripts.profile_benchmarks --both`
- [ ] Hoặc chạy trực tiếp: `python -m cProfile -o profiles/profile.stats -m ...`
- [ ] Hoặc chạy py-spy: `py-spy record -o profiles/flame.svg -- ...`

### Step 3: Phân tích kết quả
- [ ] Mở cProfile stats: `python -m pstats profiles/profile.stats` và dùng `stats 20` để xem top hàm.
- [ ] Xem flamegraph: Mở file `.svg` trong browser để tìm hot paths.
- [ ] Ghi lại top 5-10 hàm tốn nhiều thời gian.

### Step 4: Tìm Bottleneck
- [ ] Xác định hàm nào đang tiêu tốn nhiều thời gian (trong flamegraph hoặc stats).
- [ ] Xác định hàm nào được gọi nhiều nhất nhất (frequency).
- [ ] Xác định hàm nào có call stack sâu nhất (depth).

### Step 5: Tối ưu hóa
- [ ] Xem code của hàm bottleneck.
- [ ] Xem có thể refactors để giảm complexity?
- [ ] Xem có thể dùng vectorization, caching, hoặc Rust extension?
- [ ] Chạy benchmark lại sau tối ưu hóa để đo lường improvement.

---

## 7. Common Issues & Troubleshooting

### Issue 1: py-spy Permission Error (Linux/macOS)

**Error:** `OSError: [Errno 1] Operation not permitted`

**Solution:**
```bash
# Chạy py-spy với sudo (Linux/macOS)
sudo py-spy record -o profiles/flame.svg -- python -m modules.adaptive_trend_LTS.benchmarks.benchmark_comparison.main
```

**Note:** Trên Windows không cần sudo.

### Issue 2: Profiling Overhead

**Symptom:** Profiling làm benchmark chạy chậm hơn 2-3x so với bình thường.

**Cause:** cProfile và py-spy có overhead đo lường.

**Solution:**
- Chấp nhận overhead là điều bình thường.
- Chỉ dùng profiling khi cần điều tra, không phải cho mọi benchmark run.
- Tắt profiling khi đo lường performance improvement cuối cùng (chạy benchmark bình thường).

### Issue 3: File Not Found (profiles/)

**Error:** `No such file or directory: 'profiles/'`

**Solution:**
- Tạo thư mục profiles trước khi chạy:
  ```bash
  mkdir profiles
  ```
- Hoặc dùng script helper (tự tạo thư mục).

---

## 8. Integration with CI/CD

Profiling không nên chạy trong CI/CD pipeline vì:
1.  Overhead làm tests chậm hơn.
2.  Profiling artifacts không cần lưu trong CI.
3.  Profiling nên chỉ chạy khi developer cần điều tra hiệu năng.

Nếu cần profiling trong CI, thêm conditional flag:
```bash
# Chỉ chạy profiling khi biến environment được bật
if [ "$RUN_PROFILING" = "true" ]; then
    python -m modules.adaptive_trend_LTS.scripts.profile_benchmarks --both
else
    python -m modules.adaptive_trend_LTS.benchmarks.benchmark_comparison.main
fi
```

---

## 9. Best Practices

1. **Chạy profiling nhiều lần**: Để đảm bảo kết quả ổn định, hãy chạy profiling 2-3 lần.
2. **Tắt profiling khi đo lường improvement**: Đo performance cuối cùng bằng benchmark bình thường.
3. **Ghi chú**: Sau mỗi lần profiling, ghi chú lại findings vào docs (để dễ tra cứu sau).
4. **Profile hot paths**: Ưu tiên profiling benchmark với nhiều symbols/bars (reproduces realistic workload).
5. **Sử dụng flamegraph cho quick view**: Flamegraph dễ đọc hơn pstats stats để tìm bottleneck nhanh.
6. **Sử dụng pstats cho detailed view**: Khi cần deep dive vào call graph, dùng pstats hoặc snakeviz.

---

## 10. References

- **cProfile documentation**: https://docs.python.org/3/library/profile.html
- **py-spy GitHub**: https://github.com/benfred/py-spy
- **Snakeviz**: https://jiffyclub.github.io/snakeviz/
- **gprof2dot**: https://code.google.com/archive/p/jrfonse/downloads

---

**End of Profiling Guide**

