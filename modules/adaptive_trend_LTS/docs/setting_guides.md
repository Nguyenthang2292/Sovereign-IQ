# 📋 TỔNG KẾT TẤT CẢ SETTINGS - MODULE ADAPTIVE_TREND_LTS

## 🎯 Overview

Module **Adaptive Trend Classification LTS** là phiên bản ổn định với Rust backend, GPU acceleration và automatic memory management.

---

## ⚙️ CÁC PARAMETERS CHÍNH

### 1. **Moving Average Lengths** (Độ dài các MA)

| Parameter | Type | Default | Mô tả |
|-----------|------|---------|-------|
| `ema_len` | int | 28 | Độ dài EMA (Exponential Moving Average) |
| `hull_len` | int | 28 | Độ dài HMA (Hull Moving Average) |
| `wma_len` | int | 28 | Độ dài WMA (Weighted Moving Average) |
| `dema_len` | int | 28 | Độ dài DEMA (Double Exponential MA) |
| `lsma_len` | int | 28 | Độ dài LSMA (Least Squares MA) |
| `kama_len` | int | 28 | Độ dài KAMA (Kaufman Adaptive MA) |

**Lưu ý**:

- Giá trị thấp (10-20): Nhạy hơn, phù hợp timeframe ngắn
- Giá trị cao (30-50): Ổn định hơn, phù hợp timeframe dài

---

### 2. **MA Weights** (Trọng số ban đầu)

| Parameter | Type | Default | Mô tả |
|-----------|------|---------|-------|
| `ema_w` | float | 1.0 | Trọng số ban đầu cho EMA |
| `hma_w` | float | 1.0 | Trọng số ban đầu cho HMA |
| `wma_w` | float | 1.0 | Trọng số ban đầu cho WMA |
| `dema_w` | float | 1.0 | Trọng số ban đầu cho DEMA |
| `lsma_w` | float | 1.0 | Trọng số ban đầu cho LSMA |
| `kama_w` | float | 1.0 | Trọng số ban đầu cho KAMA |

**Lưu ý**: Trọng số sẽ tự động điều chỉnh dựa trên equity curves

---

### 3. **ATC Core Parameters**

| Parameter | Type | Default | Range | Mô tả |
|-----------|------|---------|-------|-------|
| `robustness` | str | "Medium" | "Narrow", "Medium", "Wide" | Độ nhạy của signal |
| `La` | float | 0.02 | 0.01-0.05 | Lambda - Growth rate (equity tăng) |
| `De` | float | 0.03 | 0.01-0.10 | Decay - Tỷ lệ giảm equity |
| `cutout` | int | 0 | 0-100 | Số bars bỏ qua ở đầu |

**Robustness Modes**:

- **"Narrow"**:
  - Offset nhỏ (length ± 1-3 steps)
  - Nhạy cảm hơn với price changes
  - Phù hợp: Trending markets
  
- **"Medium"** ✅ **RECOMMENDED**:
  - Offset trung bình (length ± 4 steps)
  - Cân bằng giữa sensitivity và stability
  - Phù hợp: Most market conditions
  
- **"Wide"**:
  - Offset lớn (length ± 9 steps)
  - Ổn định, ít nhiễu
  - Phù hợp: Volatile/choppy markets

**Lambda & Decay**:

- **La cao** (0.03-0.05): Equity tăng nhanh → trọng số thay đổi nhanh
- **La thấp** (0.01-0.02): Equity tăng chậm → trọng số ổn định
- **De cao** (0.05-0.10): Equity giảm nhanh khi sai → nhanh loại bỏ bad MAs
- **De thấp** (0.01-0.03): Equity giảm chậm → cho phép recovery

---

### 4. **Signal Thresholds**

| Parameter | Type | Default | Mô tả |
|-----------|------|---------|-------|
| `long_threshold` | float | 0.1 | Ngưỡng để classify LONG signal |
| `short_threshold` | float | -0.1 | Ngưỡng để classify SHORT signal |

**Signal Classification**:

- Signal > `long_threshold` → **LONG** (1.0)
- Signal < `short_threshold` → **SHORT** (-1.0)
- Otherwise → **NEUTRAL** (0.0)

---

### 5. **Data & Processing Parameters**

| Parameter | Type | Default | Mô tả |
|-----------|------|---------|-------|
| `prices` | pd.Series | **Required** | Price data (close prices) |
| `src` | pd.Series | None | Custom source (optional, defaults to prices) |
| `limit` | int | 1500 | Số bars để fetch |
| `timeframe` | str | "15m" | Timeframe (1m, 5m, 15m, 1h, 4h, 1d...) |

---

### 6. **Performance & Optimization**

| Parameter | Type | Default | Mô tả |
|-----------|------|---------|-------|
| `use_cuda` | bool | False | Sử dụng CUDA batch processing |
| `batch_processing` | bool | True | Sử dụng Rayon multi-threaded CPU batch |
| `parallel_l1` | bool | None | Parallel processing Layer 1 (auto-detect) |
| `parallel_l2` | bool | True | Parallel processing Layer 2 |
| `prefer_gpu` | bool | True | Ưu tiên GPU nếu có |
| `use_cache` | bool | True | Cache MA results |
| `fast_mode` | bool | True | Optimization mode |
| `precision` | str | "float64" | "float32" hoặc "float64" |

**Backend Priority**:

1. **Rust (Rayon Batch)** ⭐ **EXTREME SPEED** - Max CPU utilization
2. **Rust (Sequential)** - standard per-symbol execution
3. **CUDA (True Batch)** - GPU acceleration for hundreds of symbols
4. **Numba JIT** (fallback)
5. **Pure Python** (slowest)

---

### 7. **Strategy Mode**

| Parameter | Type | Default | Mô tả |
|-----------|------|---------|-------|
| `strategy_mode` | bool | False | Shift signal 1 bar (for backtesting) |

**Lưu ý**: Set `True` nếu dùng cho backtesting để tránh look-ahead bias

---

## 📊 KẾT QUẢ OUTPUT

`compute_atc_signals()` trả về **dictionary** chứa:

### Layer 1 Signals (cho từng MA type)

- `EMA_Signal`, `HMA_Signal`, `WMA_Signal`
- `DEMA_Signal`, `LSMA_Signal`, `KAMA_Signal`

### Layer 2 Metrics

- `EMA_S`, `HMA_S`, `WMA_S` (Layer 2 equities)
- `DEMA_S`, `LSMA_S`, `KAMA_S`

### Final Output

- **`Average_Signal`** ⭐ **MAIN RESULT** - Combined weighted signal

**Signal Range**: -1.0 (Strong Short) → 0.0 (Neutral) → +1.0 (Strong Long)

---

## 🎛️ RECOMMENDED PRESETS

### 1. **Scalping** (Timeframe: 1m - 5m)

```python
config = {
    'ema_len': 14, 'hull_len': 14, 'wma_len': 14,
    'dema_len': 14, 'lsma_len': 14, 'kama_len': 14,
    'robustness': 'Narrow',
    'La': 0.03, 'De': 0.05,
    'cutout': 20,
    'timeframe': '1m',
    'limit': 500
}
```

### 2. **Intraday Trading** (Timeframe: 15m - 1h) ✅ **DEFAULT**

```python
config = {
    'ema_len': 28, 'hull_len': 28, 'wma_len': 28,
    'dema_len': 28, 'lsma_len': 28, 'kama_len': 28,
    'robustness': 'Medium',
    'La': 0.02, 'De': 0.03,
    'cutout': 0,
    'timeframe': '15m',
    'limit': 1500
}
```

### 3. **Swing Trading** (Timeframe: 4h - 1d)

```python
config = {
    'ema_len': 50, 'hull_len': 50, 'wma_len': 50,
    'dema_len': 50, 'lsma_len': 50, 'kama_len': 50,
    'robustness': 'Wide',
    'La': 0.015, 'De': 0.02,
    'cutout': 0,
    'timeframe': '4h',
    'limit': 2000
}
```

### 4. **High-Performance** (Rust + Multi-symbol)

```python
config = {
    # ... standard params ...
    'batch_processing': True, # Use Rayon Multi-threading
    'use_cuda': False,        # Rust Rayon often faster for < 500 symbols
    'parallel_l1': True,
    'parallel_l2': True,
    'use_cache': True,
    'fast_mode': True,
    'use_dask': True,         # Enable Dask for 1000+ symbols (Out-of-Core)
    'npartitions': 20,        # Number of parallel partitions
}
```

### 5. **Out-of-Core Processing** (Dask Integration) ⭐ **NEW**

Khi xử lý danh sách symbol cực lớn (>1000 symbols) vượt quá dung lượng RAM, hoặc khi muốn tận dụng tối đa CPU core, hãy sử dụng Dask:

- **`use_dask`**: Bật chế độ xử lý song song và phân đoạn bộ nhớ (partitioning).
- **`npartitions`**: Số lượng mảnh dữ liệu xử lý cùng lúc. Mặc định hệ thống tự tính toán dựa trên số lượng symbol.

```python
from modules.adaptive_trend_LTS.core.scanner.scan_all_symbols import scan_all_symbols

longs, shorts = scan_all_symbols(
    data_fetcher,
    atc_config,
    execution_mode="dask",  # Chế độ tối ưu cho dữ liệu lớn
    npartitions=10
)
```

### 6. **True Batch Processing** (Best for 100+ symbols)

Nếu bạn có danh sách nhiều symbols (ví dụ: Binance Futures), hãy dùng hàm batch thay vì loop:

```python
from modules.adaptive_trend_LTS.core.compute_atc_signals.batch_processor import process_symbols_batch_rust

# symbols_data = {'BTCUSDT': prices_series, 'ETHUSDT': series, ...}
results = process_symbols_batch_rust(symbols_data, config)
```

---

## 🚀 PERFORMANCE COMPARISON

**Benchmark** (100 symbols × 1500 bars):

| Implementation | Time | Speedup | Memory | Accuracy |
|----------------|------|---------|--------|----------|
| Original Python | 52.58s | 1.00x | 140.5 MB | 100% |
| Enhanced Python | 22.99s | 2.29x | 125.0 MB | 100% |
| Rust (Seq) | 13.94s | 3.77x | 25.7 MB | 100% |
| Rust (Rayon) | 8.12s | 6.47x | 18.2 MB | 100% |
| **Rust + Dask Hybrid** ⭐ | **9.45s** | **5.56x** | **12.5 MB** | **100%** |
| CUDA Batch | 15.04s | 3.49x | 71.3 MB | 100% |

**Note**: Rust + Dask Hybrid có tốc độ gần bằng Rayon nhưng **không giới hạn kích thước dataset** và tiêu tốn ít RAM hơn nhờ cơ chế chunking.

**Recommendation**:
- < 1000 symbols: **USE RUST (RAYON)**
- \> 1000 symbols: **USE RUST + DASK HYBRID**

---

## 🔧 SETUP & BUILD

### Rust Backend (Recommended)

```bash
cd modules/adaptive_trend_LTS/rust_extensions
maturin develop --release
```

### CUDA Backend (Optional)

```bash
cd modules/adaptive_trend_LTS/rust_extensions
powershell -ExecutionPolicy Bypass -File build_cuda.ps1
```

---

## 📝 EXAMPLE USAGE

```python
from modules.adaptive_trend_LTS.core.compute_atc_signals import compute_atc_signals
import pandas as pd

# Prepare data
prices = pd.Series([100, 101, 102, ...])

# Compute signals
results = compute_atc_signals(
    prices=prices,
    ema_len=28,
    hull_len=28,
    wma_len=28,
    dema_len=28,
    lsma_len=28,
    kama_len=28,
    robustness='Medium',
    La=0.02,
    De=0.03,
)

# Get final signal
final_signal = results['Average_Signal']

# Interpret
current_signal = final_signal.iloc[-1]
if current_signal > 0.1:
    print("LONG signal")
elif current_signal < -0.1:
    print("SHORT signal")
else:
    print("NEUTRAL")
```

---

## 📞 TROUBLESHOOTING

**Common Issues**:

1. **Rust not found**: Install from <https://rustup.rs/>
2. **Maturin error**: `pip install maturin`
3. **Import error**: Run `maturin develop --release` in `rust_extensions/`
4. **CUDA error**: Check CUDA Toolkit 12.8 installed
5. **Memory issue**: Reduce `limit` or enable `fast_mode`

---

## ✅ BEST PRACTICES

1. **Start with defaults** (Medium, 28 lengths)
2. **Adjust for your timeframe**:
   - Shorter TF → Lower lengths (14-21)
   - Longer TF → Higher lengths (50+)
3. **Use Rust backend** for production (3.77x faster)
4. **Enable caching** (`use_cache=True`)
5. **Test parameters** on historical data first
6. **Monitor memory** with large datasets

---

**Last Updated**: 2026-01-25
**Version**: LTS (Long-Term Support)
**Backend**: Rust v2 + CUDA (optional)
