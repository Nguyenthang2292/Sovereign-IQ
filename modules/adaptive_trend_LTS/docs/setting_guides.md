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

See `docs/phase5_task.md` for detailed Dask integration guide and benchmarks.

### 6. **True Batch Processing** (Best for 100+ symbols)

Nếu bạn có danh sách nhiều symbols (ví dụ: Binance Futures), hãy dùng hàm batch thay vì loop:

```python
from modules.adaptive_trend_LTS.core.compute_atc_signals.batch_processor import process_symbols_batch_rust

# symbols_data = {'BTCUSDT': prices_series, 'ETHUSDT': series, ...}
results = process_symbols_batch_rust(symbols_data, config)
```

### 7. **Incremental Updates** (For Live Trading) ⭐ **NEW**

Khi cần cập nhật signal cho single bar mới (live trading), sử dụng `IncrementalATC` để tránh tính lại toàn bộ series:

```python
from modules.adaptive_trend_LTS.core.compute_atc_signals.incremental_atc import IncrementalATC

# Initialize once with historical data
atc = IncrementalATC(config)
atc.initialize(historical_prices)

# Update incrementally with new bar (O(1) operation)
new_signal = atc.update(new_price)
```

**Performance**: 10-100x faster than full recalculation for single bar updates.

See `docs/phase6_task.md` for detailed incremental update guide.

### 8. **Approximate MAs for Fast Filtering** ⭐ **NEW**

Khi scan hàng nghìn symbols, sử dụng Approximate MAs cho filtering ban đầu, sau đó tính full precision cho candidates:

```python
from modules.adaptive_trend_LTS.core.compute_atc_signals.batch_processor import (
    process_symbols_batch_with_approximate_filter
)

results = process_symbols_batch_with_approximate_filter(
    symbols_data,
    config,
    min_signal_candidate=0.05,  # Filter threshold
)
```

**Performance**: 2-3x faster for large symbol sets (1000+).

See `docs/phase6_task.md` for detailed approximate MA guide and accuracy benchmarks.

---

## 🚀 PERFORMANCE COMPARISON

**Benchmark** (99 symbols × 1500 bars):

| Implementation | Time | Speedup | Memory | Use Case |
|----------------|------|---------|--------|----------|
| Original Python | 49.65s | 1.00x | 122.1 MB | Baseline |
| Enhanced Python | 23.85s | 2.08x | 125.8 MB | Optimized Python |
| Rust (Seq) | 14.15s | 3.51x | 21.0 MB | CPU Sequential |
| Rust (Rayon) | 8.12s | 6.11x | 18.2 MB | CPU Parallel |
| **Rust + Dask Hybrid** ⭐ | **9.45s** | **5.25x** | **12.5 MB** | **Unlimited size** |
| CUDA Batch | 15.04s | 3.30x | 51.7 MB | GPU Batch |
| **True Batch CUDA** ⭐ | **0.59s** | **83.53x** | **51.7 MB** | **100+ symbols** |
| **Incremental Update** ⭐ | **<0.01s** | **1000x+** | **<1 MB** | **Live Trading (single bar)** |
| **Approximate Filter** ⭐ | **~5s** | **10x** | **~20 MB** | **Fast Scanning (1000+)** |

**Note**:
- Rust + Dask Hybrid has unlimited dataset size due to out-of-core processing
- True Batch CUDA achieves 83.53x speedup for batch processing
- Incremental Update is optimal for live trading (single bar updates)
- Approximate Filter is optimal for initial filtering in large-scale scanning

**Recommendation by Use Case**:

| Use Case | Recommended Implementation | Expected Speedup |
|----------|---------------------------|------------------|
| **Live Trading (single bar)** | Incremental Update | 10-100x |
| **Small batch (<100 symbols)** | Rust (Rayon) | 6x |
| **Medium batch (100-1000)** | True Batch CUDA | 80x+ |
| **Large batch (1000-10000)** | Rust + Dask Hybrid | 5-10x + Unlimited size |
| **Very large (10000+)** | Approximate Filter + Dask | 10-20x + Unlimited size |
| **Out-of-Memory scenarios** | Dask Integration | Unlimited size |

**Performance by Phase**:

| Phase | Feature | Speedup | Status |
|-------|---------|---------|--------|
| Phase 1 | Core Optimizations | 2.29x | ✅ Complete |
| Phase 2 | Advanced Memory Opts | 1.5-2x | ✅ Complete |
| Phase 3 | Rust Extensions | 2-3x | ✅ Complete |
| Phase 4 | CUDA Kernels | 3-80x | ✅ Complete |
| Phase 5 | Dask Integration | Unlimited size | ✅ Complete |
| Phase 6 | Algorithmic Improvements | 10-100x (incremental) | ✅ Complete |
| **Total** | **All Combined** | **Up to 1000x+** | ✅ **Production Ready** |

---

## 🔧 SETUP & BUILD

### Rust Backend (Recommended)

```bash
cd modules/adaptive_trend_LTS/rust_extensions
maturin develop --release
```

**Note**: On Windows, if the Rust linker cannot find `cuda.lib`, set `RUSTFLAGS` before building:

```powershell
$env:RUSTFLAGS="-L 'C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.8\lib\x64'"
maturin develop --release
```

See `docs/phase3_task.md` for detailed Rust installation instructions.

### CUDA Backend (Optional)

```bash
cd modules/adaptive_trend_LTS/rust_extensions
powershell -ExecutionPolicy Bypass -File build_cuda.ps1
```

**Requirements**:
- CUDA Toolkit 12.x
- NVIDIA GPU with compute capability >= 6.0
- See `docs/phase4_task.md` for detailed CUDA setup instructions

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

**Last Updated**: 2026-01-27
**Version**: LTS (Long-Term Support)
**Backend**: Rust v2 + CUDA (optional) + Dask (optional)

**Phase Completion Status**:
- Phase 1 (Core Optimizations): ✅ Complete
- Phase 2 (Advanced Memory): ✅ Complete
- Phase 3 (Rust Extensions): ✅ Complete
- Phase 4 (CUDA Kernels): ✅ Complete
- Phase 5 (Dask Integration): ✅ Complete
- Phase 6 (Algorithmic Improvements): ✅ Complete

**Documentation References**:
- Phase 3 (Rust): `docs/phase3_task.md`
- Phase 4 (CUDA): `docs/phase4_task.md`
- Phase 5 (Dask): `docs/phase5_task.md`
- Phase 6 (Incremental/Approximate): `docs/phase6_task.md`
