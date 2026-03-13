# ATC Hot Path Configurations

## Task 1: Xác định các cấu hình ATC "hot path" cần chuyên biệt hóa

### Cấu hình phổ biến (Hot Paths)

Dựa trên phân tích codebase, benchmarks và usage patterns, các cấu hình sau được xác định là hot paths:

#### 1. **Default Config (All MAs, Medium Robustness)**
- **Mô tả**: Config mặc định, được dùng trong 90%+ các benchmarks và production
- **Parameters**:
  - ema_len: 28
  - hma_len: 28
  - wma_len: 28
  - dema_len: 28
  - lsma_len: 28
  - kama_len: 28
  - ema_w, hma_w, wma_w, dema_w, lsma_w, kama_w: 1.0
  - robustness: "Medium"
  - lambda_param: 0.02
  - decay: 0.03
  - long_threshold: 0.1
  - short_threshold: -0.1
  - cutout: 0
- **Usage Frequency**: ~85-90%
- **Files**: Used in `benchmark_algorithmic_improvements.py`, `benchmark_batch_incremental_atc.py`, `benchmark_batch_simple.py`, `benchmark_cache_parallel.py`, `benchmark_dask_*.py`
- **Potential Gain**: High (most common case)

#### 2. **EMA-Only Config**
- **Mô tả**: Chỉ sử dụng EMA cho fast scanning và filtering
- **Parameters**:
  - ema_len: 28
  - ema_w: 1.0
  - robustness: "Medium"
  - lambda_param: 0.02
  - decay: 0.03
  - Các MA khác: tắt hoặc weights=0.0
- **Usage Frequency**: ~5-8%
- **Files**: Used in approximate MA benchmarks and scanning workflows
- **Potential Gain**: High (fast path for EMA)

#### 3. **Short Length Config (Fast Response)**
- **Mô tả**: Config cho fast response trading với short lengths
- **Parameters**:
  - ema_len: 14
  - hma_len: 14
  - wma_len: 14
  - dema_len: 14
  - lsma_len: 14
  - kama_len: 14
  - robustness: "Medium" hoặc "Narrow"
  - lambda_param: 0.02
  - decay: 0.03
- **Usage Frequency**: ~3-5%
- **Potential Gain**: Medium (less common but still significant)

#### 4. **Narrow Robustness Config**
- **Mô tả**: Config cho high sensitivity với narrow robustness
- **Parameters**:
  - ema_len: 28 (default)
  - robustness: "Narrow"
  - lambda_param: 0.02
  - decay: 0.03
- **Usage Frequency**: ~2-3%
- **Potential Gain**: Low-Medium (less common)

#### 5. **KAMA-Only Config**
- **Mô tả**: Chỉ sử dụng KAMA cho adaptive filtering
- **Parameters**:
  - kama_len: 28
  - kama_w: 1.0
  - robustness: "Medium"
  - Các MA khác: tắt hoặc weights=0.0
- **Usage Frequency**: ~1-2%
- **Potential Gain**: Low-Medium (specific use case)

### Thống kê Usage

| Config | Frequency | Source |
|--------|-----------|--------|
| Default (All MAs, Medium) | ~85-90% | Benchmarks, production |
| EMA-Only | ~5-8% | Approximate MA benchmarks |
| Short Length (14) | ~3-5% | Fast response trading |
| Narrow Robustness | ~2-3% | High sensitivity |
| KAMA-Only | ~1-2% | Adaptive filtering |

### Priority cho JIT Specialization

#### High Priority (Immediate)
1. **Default Config** - Most common, highest impact
2. **EMA-Only Config** - Fast path for scanning

#### Medium Priority
3. **Short Length Config** - Useful for fast response

#### Low Priority
4. **Narrow Robustness** - Less common
5. **KAMA-Only** - Specific use case

### Config Hashes for Caching

```python
# Config hash examples for caching/specialization
DEFAULT_CONFIG_HASH = "atc_default_28_medium"
EMA_ONLY_HASH = "atc_ema_only_28_medium"
SHORT_LENGTH_HASH = "atc_short_14_medium"
NARROW_ROBUSTNESS_HASH = "atc_default_28_narrow"
KAMA_ONLY_HASH = "atc_kama_only_28_medium"
```

### Next Steps

- Task 2: Thiết kế API specialization (wrapper hoặc factory)
- Task 3: Implement JIT specialization cho EMA-first case
