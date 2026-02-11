# Task 7: Benchmark against Python Implementation

## 📋 Mô tả
Benchmark so sánh hiệu năng giữa `modules/adaptive_trend_LTS_serverless` (Rust) và `modules/adaptive_trend_LTS_mini` (Python) trên 3 cặp tiền điện tử (BTC, ETH, XMR) với 3 timeframe (15m, 1h, 4h).

## 🎯 Mục tiêu
1. **So sánh tốc độ chạy**: Đo thờigian xử lý của cả hai implementation
2. **So sánh tín hiệu**: Kiểm tra tính đồng nhất của kết quả LONG/SHORT/NEUTRAL

## 📁 Files đã tạo

### 1. Rust Benchmark Binary
- **File**: `modules/adaptive_trend_LTS_serverless/src/bin/benchmark.rs`
- **Mô tả**: Binary Rust để benchmark, nhận JSON input và trả về kết quả với timing
- **Build**: `cargo build --release --bin atc_benchmark`

### 2. Python Benchmark Script
- **File**: `benchmark_atc_comparison.py` (trong thư mục gốc project)
- **Mô tả**: Script Python so sánh cả hai implementation
- **Chức năng**:
  - Generate dữ liệu OHLCV mẫu cho 3 symbols × 3 timeframes
  - Benchmark Python module với 5 runs averaging
  - Benchmark Rust module qua subprocess với 5 runs averaging
  - So sánh tốc độ và tính đồng nhất tín hiệu
  - In báo cáo tổng hợp

## 🚀 Cách chạy benchmark

```bash
# 1. Đảm bảo đang ở thư mục gốc project
cd /path/to/crypto-probability

# 2. Chạy benchmark script
python benchmark_atc_comparison.py
```

## 📊 Kết quả mong đợi

### Tốc độ
- **Rust serverless**: ~10-20x nhanh hơn Python mini
- **Thờigian xử lý trung bình**:
  - Python: 50-200ms per symbol/tf
  - Rust: 5-20ms per symbol/tf

### Tính đồng nhất tín hiệu
- **Mục tiêu**: >90% tín hiệu giống nhau (LONG/SHORT/NEUTRAL)
- **Sai khác giá trị**: <0.01 (do khác biệt floating-point precision)

## 🔧 Cấu hình

### Symbols test
- BTCUSDT
- ETHUSDT  
- XMRUSDT

### Timeframes
- 15m
- 1h
- 4h

### Parameters
```python
ATC_CONFIG = {
    "ema_len": 28,
    "hma_len": 28,
    "wma_len": 28,
    "dema_len": 28,
    "lsma_len": 28,
    "kama_len": 28,
    "lambda_param": 0.02,
    "decay": 0.03,
    "long_threshold": 0.1,
    "short_threshold": -0.1,
    "robustness": "Narrow",
    "cutout": 0,
}
```

## 📈 Output mẫu

```
================================================================================
ATC BENCHMARK: Python (mini) vs Rust (serverless)
================================================================================

Symbols: BTCUSDT, ETHUSDT, XMRUSDT
Timeframes: 15m, 1h, 4h
Bars per timeframe: 500

Building Rust benchmark binary...
✓ Rust binary built successfully

Benchmarking BTCUSDT @ 15m...
  Running Python module...
    Time: 125.34 ± 5.21 ms
    Signal: LONG (0.4523)
  Running Rust module...
    Time: 12.45 ± 0.89 ms
    Signal: LONG (0.4519)
...

================================================================================
BENCHMARK SUMMARY
================================================================================

1. SPEED COMPARISON
--------------------------------------------------------------------------------
Symbol       TF     Python (ms)     Rust (ms)       Speedup
--------------------------------------------------------------------------------
BTCUSDT      15m      125.34 ± 5.21     12.45 ± 0.89     10.1x
BTCUSDT      1h       118.92 ± 4.56     11.23 ± 0.76     10.6x
...
TOTAL              1125.50            112.50            10.0x

2. SIGNAL CONSISTENCY
--------------------------------------------------------------------------------
Symbol       TF     Python     Rust       Match
--------------------------------------------------------------------------------
BTCUSDT      15m    LONG       LONG       YES
...
Consistency Rate: 9/9 (100.0%)

3. SIGNAL VALUE DIFFERENCE
--------------------------------------------------------------------------------
Symbol       TF     Python       Rust         Diff
--------------------------------------------------------------------------------
BTCUSDT      15m      0.452345     0.451987     0.000358
...
Maximum Difference: 0.000512

4. CONCLUSION
--------------------------------------------------------------------------------
Rust implementation is 10.0x faster on average
Signal consistency: 100.0% (9/9)
Maximum signal difference: 0.000512
Excellent speedup achieved (>5x)
Perfect signal consistency

================================================================================
```

## ✅ Status
- [x] Tạo Rust benchmark binary
- [x] Tạo Python benchmark script
- [x] Cấu hình test cho 3 symbols × 3 timeframes
- [x] Implement so sánh tốc độ
- [x] Implement so sánh tín hiệu
- [x] Chạy benchmark và ghi nhận kết quả thực tế

## 📊 Kết quả Benchmark Thực tế (Feb 11, 2026)

### Tốc độ
- **Rust serverless**: **61.9x nhanh hơn** Python mini
- **Thờigian xử lý trung bình**:
  - Python: ~109-111ms per symbol/tf
  - Rust: ~1.5-1.9ms per symbol/tf

### Chi tiết theo Symbol/Timeframe
| Symbol | TF | Python (ms) | Rust (ms) | Speedup |
|--------|-----|-------------|-----------|---------|
| BTCUSDT | 15m | 109.69 ± 0.98 | 1.47 ± 0.20 | **74.8x** |
| BTCUSDT | 1h | 111.58 ± 1.26 | 1.84 ± 0.08 | **60.8x** |
| BTCUSDT | 4h | 109.36 ± 1.35 | 1.83 ± 0.05 | **59.7x** |
| ETHUSDT | 15m | 110.63 ± 0.85 | 1.86 ± 0.11 | **59.4x** |
| ETHUSDT | 1h | 108.68 ± 2.64 | 1.79 ± 0.08 | **60.8x** |
| ETHUSDT | 4h | 107.53 ± 3.33 | 1.77 ± 0.02 | **60.7x** |
| XMRUSDT | 15m | 110.86 ± 2.36 | 1.82 ± 0.04 | **60.9x** |
| XMRUSDT | 1h | 110.39 ± 0.78 | 1.76 ± 0.01 | **62.6x** |
| XMRUSDT | 4h | 109.62 ± 1.91 | 1.83 ± 0.05 | **59.8x** |

### Tính đồng nhất tín hiệu
- **Consistency Rate**: **66.7%** (6/9 tests)
- **Signal mismatch** ở:
  - BTCUSDT @ 15m: Python SHORT vs Rust LONG
  - XMRUSDT @ 15m: Python SHORT vs Rust NEUTRAL
  - XMRUSDT @ 4h: Python SHORT vs Rust LONG
- **Maximum difference**: 0.408675

### Phân tích
⚠️ **Cần điều tra**: Có sự khác biệt đáng kể về tín hiệu (33.3% mismatch) giữa Python và Rust implementation. Nguyên nhân có thể:
1. Khác biệt trong MA calculation algorithms
2. Khác biệt trong Layer 1 signal detection logic
3. Khác biệt floating-point precision
4. Khác biệt trong equity calculation

✅ **Performance**: Rust implementation vượt trội với speedup 61.9x, phù hợp cho production serverless deployment.

## 📝 Notes
- Benchmark sử dụng synthetic data (seeded random) để đảm bảo reproducibility
- Mỗi test case chạy 5 lần và lấy average để giảm variance
- Rust binary được build với release profile optimizations
- Python sử dụng fast_mode=True để tối ưu hiệu năng
