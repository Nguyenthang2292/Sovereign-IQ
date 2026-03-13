# Phase 8: Profiling-Guided Optimizations

> **Scope**: Establish profiling workflows (cProfile + py-spy) to guide targeted optimizations for `adaptive_trend_LTS`, starting from benchmark comparison pipeline.
> **Expected Gain**: 5–10% improvement in hot paths + faster diagnosis of regressions
> **Timeline**: 1–2 weeks
> **Status**: ✅ **COMPLETED**

---

## 1. Mục tiêu

Thiết lập quy trình profiling chuẩn cho module `adaptive_trend_LTS`:

- **cProfile**: Đo chi tiết thời gian thực thi theo hàm cho các entrypoint chính (đặc biệt là benchmark).
- **py-spy flamegraph**: Trực quan hoá bottleneck trên call stack để ưu tiên tối ưu hoá.
- **Workflow lặp lại được**: Dễ dàng chạy lại khi cần điều tra chậm hoặc regression.

---

## 2. Implementation Tasks

### 2.1 Task 1 – Add Profiling Entrypoints

**Status**: ✅ **COMPLETED**

- ✅ Tạo script helper tại `modules/adaptive_trend_LTS/scripts/profile_benchmarks.py`
  - Hỗ trợ chạy cProfile với flag `--cprofile`
  - Hỗ trợ chạy py-spy với flag `--pyspy`
  - Hỗ trợ chạy cả hai với flag `--both` (mặc định)
  - Tự động tạo thư mục `profiles/` nếu chưa tồn tại
  - Truyền các tham số benchmark (`--symbols`, `--bars`, `--timeframe`, `--clear-cache`) đến benchmark pipeline

- ✅ Tạo `__init__.py` cho module `scripts` để có thể import

**Verify**:
- ✅ Chạy lệnh profiling tạo được file `profiles/benchmark_comparison.stats` (hoặc tương đương, không rỗng).
- ✅ Script helper có thể chạy với cProfile và py-spy.

---

### 2.2 Task 2 – Document cProfile Usage

**Status**: ✅ **COMPLETED**

- ✅ Tạo tài liệu đầy đủ tại `modules/adaptive_trend_LTS/docs/profiling_guide.md`
  - Hướng dẫn installation cProfile, py-spy, snakeviz, gprof2dot
  - Hướng dẫn chạy profiling trực tiếp (cProfile, py-spy)
  - Hướng dẫn chạy qua script helper
  - Hướng dẫn phân tích kết quả (pstats interactive, snakeviz, gprof2dot)
  - Hướng dẫn đọc flamegraph py-spy
  - Hướng dẫn troubleshooting (permission errors, overhead)
  - Best practices cho profiling

**Verify**:
- ✅ Docs phản ánh đúng entrypoint và đường dẫn hiện tại (`modules/adaptive_trend_LTS/benchmarks/benchmark_comparison/main.py`)
- ✅ Docs bao gồm cả cProfile và py-spy usage.

---

### 2.3 Task 3 – Integrate py-spy Flamegraph Workflow

**Status**: ✅ **COMPLETED**

- ✅ Tài liệu về py-spy flamegraph workflow trong `docs/profiling_guide.md`:
  - Lệnh chạy py-spy để tạo SVG flamegraph
  - Hướng dẫn mở SVG trong browser
  - Hướng dẫn đọc flamegraph (width=total time, height=stack depth, hot paths)
  - Tích hợp với script helper (flag `--pyspy` và `--both`)

**Verify**:
- ✅ File `profiles/benchmark_comparison_flame.svg` được tạo và mở được trong browser.
- ✅ Workflow được ghi chú rõ ràng trong docs.

---

### 2.4 Task 4 – Minimal Profiling Checklist

**Status**: ✅ **COMPLETED**

- ✅ Định nghĩa checklist đầy đủ (5 bước) trong `modules/adaptive_trend_LTS/docs/profiling_checklist.md`:
  1. Xác nhận vấn đề (run benchmark bình thường)
  2. Chạy profiling (helper script hoặc direct commands)
  3. Phân tích kết quả (cProfile stats + flamegraph)
  4. Tìm bottleneck (top functions, hot paths)
  5. Tối ưu hóa và verify

- ✅ Bổ sung các template ghi chú findings
- ✅ Bổ sung tips về profiling overhead và frequency

**Verify**:
- ✅ Checklist nằm trong docs và trỏ tới đúng lệnh/entrypoint đã thiết lập.
- ✅ Checklist có thể dùng làm guide cho profiling session.

---

### 2.5 Task 5 – One-Command Profiling Helper

**Status**: ✅ **COMPLETED**

- ✅ Tạo script `scripts/profile_benchmarks.py` (đã triển khai trong Task 2.1):
  - Tự tạo thư mục `profiles/` (nếu chưa có)
  - Chạy cProfile với flag `--cprofile`
  - Chạy py-spy với flag `--pyspy`
  - Chạy cả hai với flag `--both` (mặc định)
  - Log đường dẫn file output
  - Tự động verify installation py-spy (với thông báo lỗi nếu thiếu)
  - Support truyền tham số benchmark (`--symbols`, `--bars`, `--timeframe`, `--clear-cache`)

**Verify**:
- ✅ Một lệnh (hoặc script) duy nhất chạy profiling end-to-end và sinh đủ file `.stats` + `.svg` vào `profiles/`.
- ✅ Script helper hoạt động đúng như mô tả trong task.

---

## 3. Validation & Integration

### 3.1 Validation

**Status**: ✅ **COMPLETED**

- ✅ Đảm bảo:
  - Profiling không được bật mặc định trong production (script helper cần flag cụ thể).
  - Profiling không phá vỡ logging/CI (script helper chạy subprocess, không can thiệp với benchmark).
  - `profiles/` được gitignore (đã verify ở task 3.2).

---

### 3.2 Gitignore & Artifacts

**Status**: ✅ **COMPLETED** (Verified: `profiles/` already exists in project root `.gitignore`)

- ✅ Cập nhật `.gitignore` (đã tồn tại) để ignore:
  - `profiles/`
  - Any `*.stats`, `*.svg` tạo bởi profiling.

**Verify**:
- ✅ `git status` không hiển thị file trong `profiles/` sau khi chạy profiling.
- ✅ Dòng `profiles/` đã có trong file `.gitignore` tại project root (line 78).

---

## 4. Done When

- [x] Có quy trình rõ ràng để sinh `profile.stats` và flamegraph (`*.svg`) cho benchmark chính.
- [x] Docs mô tả cách chạy và đọc kết quả profiling.
- [x] Artifacts profiling nằm trong `profiles/` và không bị git track.
- [x] Dev có thể dùng plan này để điều tra và tối ưu các hot paths trong tương lai.

---

## 5. Notes

- ✅ Profiling chỉ nên dùng khi cần điều tra hiệu năng, không bật mặc định.
- ✅ Ưu tiên các công cụ nhẹ, cross‑platform: `cProfile`, `py-spy`, `snakeviz`.
- ✅ Workflow đã được chuẩn hóa: Use script helper `--both` để lấy cả cProfile stats và flamegraph trong một lần chạy.

---

## 6. Summary of Deliverables

| Task | Status | Deliverable | Location |
|------|--------|-------------|------------|
| 2.1 Add Profiling Entrypoints | ✅ Completed | Script helper (`scripts/profile_benchmarks.py`), `__init__.py` |
| 2.2 Document cProfile Usage | ✅ Completed | Full profiling guide (`docs/profiling_guide.md`) |
| 2.3 Integrate py-spy Workflow | ✅ Completed | py-spy flamegraph guide (in `docs/profiling_guide.md`) |
| 2.4 Minimal Profiling Checklist | ✅ Completed | Checklist template (`docs/profiling_checklist.md`) |
| 2.5 One-Command Profiling Helper | ✅ Completed | Integrated in Task 2.1 script |
| 3.1 Validation | ✅ Completed | Verified requirements |
| 3.2 Gitignore & Artifacts | ✅ Completed | `profiles/` ignored in project `.gitignore` |

---

## 7. Usage Examples

### Example 1: Quick Profiling Session

```bash
# Chạy cả cProfile và py-spy với default parameters
python -m modules.adaptive_trend_LTS.scripts.profile_benchmarks --both

# Output sẽ tạo:
# - profiles/benchmark_comparison.stats (cProfile)
# - profiles/benchmark_comparison_flame.svg (flamegraph)
```

### Example 2: Profiling Custom Benchmark

```bash
# Chạy profiling với custom symbols/bars
python -m modules.adaptive_trend_LTS.scripts.profile_benchmarks --both --symbols 50 --bars 1000 --timeframe 15m

# Xem kết quả
python -m pstats profiles/benchmark_comparison.stats
start profiles/benchmark_comparison_flame.svg
```

### Example 3: Using Checklist

1. Chạy benchmark bình thường để xác nhận vấn đề.
2. Chạy profiling với script helper.
3. Mở checklist (`docs/profiling_checklist.md`) để theo dõi progress.
4. Phân tích kết quả (cProfile stats + flamegraph).
5. Tìm bottleneck và tối ưu hóa.
6. Chạy profiling lại để verify improvement.

---

**End of Phase 8 Task List**
