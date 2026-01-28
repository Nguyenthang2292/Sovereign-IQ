# ⚡ RECOMMENDED SETTINGS FOR MAXIMUM PROCESSING SPEED

**Purpose**: Bộ cấu hình tối ưu hóa tốc độ xử lý cho `adaptive_trend_LTS` module  
**Use Case**: Batch scanning, multi-symbol processing, production deployment  
**Last Updated**: 2026-01-29

---

## 🎯 Quick Summary

Để đạt **tốc độ xử lý tối đa**, sử dụng các setting sau:

### 1. **Backend Selection** (Quan trọng nhất!)

| Scenario | Recommended Backend | Expected Speedup | Config |
|----------|-------------------|------------------|--------|
| **Live Trading (single bar)** | Incremental Update | 10-100x | `use_incremental=True` |
| **Small batch (<100 symbols)** | Rust (Rayon Parallel) | 6x | `batch_processing=True` |
| **Medium batch (100-1000)** | CUDA Batch | 80x+ | `use_cuda=True` |
| **Large batch (1000-10000)** | Rust + Dask Hybrid | 5-10x + Unlimited size | `use_dask=True` |
| **Very large (10000+)** | Approximate Filter + Dask | 10-20x + Unlimited size | `use_approximate=True, use_dask=True` |

### 2. **Core Performance Settings**

```yaml
# Performance & Optimization
batch_processing: true          # Enable Rust Rayon multi-threading
use_cuda: false                 # Set true for 100+ symbols (requires CUDA Toolkit)
parallel_l1: true               # Parallel Layer 1 processing
parallel_l2: true               # Parallel Layer 2 processing
prefer_gpu: true                # Auto-select GPU if available
use_cache: true                 # Enable MA caching
fast_mode: true                 # Enable all optimizations
precision: "float32"            # Use float32 for speed (vs float64 for accuracy)

# Dask Integration (for large datasets)
use_dask: false                 # Set true for 1000+ symbols
npartitions: 20                 # Number of parallel partitions (auto-calculated if null)

# Incremental Updates (for live trading)
use_incremental: false          # Set true for live trading (single bar updates)

# Approximate MAs (for fast filtering)
use_approximate: false          # Set true for initial filtering (2-3x faster)
use_adaptive_approximate: false # Set true for volatility-aware filtering
```

### 3. **Memory Optimization Settings**

```yaml
# Memory Optimizations (Phase 7)
use_memory_mapped: false        # Set true for very large backtesting datasets
use_compression: false          # Set true to reduce cache storage (5-10x reduction)
compression_level: 5            # Compression level (1-9, higher = smaller but slower)
```

### 4. **Advanced Settings** (Optional)

```yaml
# Cache Warming (Phase 8.1)
warm_cache: false               # Pre-warm cache for repeated patterns

# JIT Specialization (Phase 8.2)
use_codegen_specialization: false  # Enable JIT for hot path configs (EMA-only)
```

---

## 📋 Recommended Presets by Use Case

### Preset 1: **Live Trading (Maximum Speed for Single Bar)**

```yaml
# Use Incremental ATC for O(1) updates
use_incremental: true
batch_processing: false         # Not needed for single symbol
use_cuda: false                 # Not needed for single symbol
parallel_l1: false
parallel_l2: false
use_cache: true
fast_mode: true
precision: "float32"
```

**Expected Performance**: <0.01s per update (1000x+ faster than full recalculation)

---

### Preset 2: **Small Batch Scanning (<100 symbols)**

```yaml
# Use Rust Rayon Parallel
batch_processing: true
use_cuda: false                 # Rust Rayon often faster for small batches
parallel_l1: true
parallel_l2: true
use_cache: true
fast_mode: true
precision: "float32"
use_dask: false
```

**Expected Performance**: 6x speedup vs sequential, ~8-12s for 99 symbols × 1500 bars

---

### Preset 3: **Medium Batch Scanning (100-1000 symbols)**

```yaml
# Use CUDA Batch Processing
batch_processing: false         # CUDA handles batching
use_cuda: true                  # Requires CUDA Toolkit 12.x
prefer_gpu: true
parallel_l1: false              # CUDA handles parallelism
parallel_l2: false
use_cache: true
fast_mode: true
precision: "float32"
use_dask: false
```

**Expected Performance**: 80x+ speedup, ~0.6s for 99 symbols × 1500 bars

**Requirements**:

- NVIDIA GPU with compute capability >= 6.0
- CUDA Toolkit 12.x installed
- Build with: `powershell -ExecutionPolicy Bypass -File build_cuda.ps1`

---

### Preset 4: **Large Batch Scanning (1000-10000 symbols)**

```yaml
# Use Rust + Dask Hybrid
batch_processing: true
use_cuda: false
parallel_l1: true
parallel_l2: true
use_cache: true
fast_mode: true
precision: "float32"
use_dask: true                  # Enable out-of-core processing
npartitions: 20                 # Adjust based on CPU cores (typically 2x cores)
```

**Expected Performance**: 5-10x speedup + unlimited dataset size (out-of-core)

**Memory Usage**: 10-20% of in-memory approach

---

### Preset 5: **Very Large Batch with Filtering (10000+ symbols)**

```yaml
# Use Approximate Filter + Dask
batch_processing: true
use_cuda: false
parallel_l1: true
parallel_l2: true
use_cache: true
fast_mode: true
precision: "float32"
use_dask: true
npartitions: 30
use_approximate: true           # Fast filtering (2-3x faster)
use_adaptive_approximate: false # Or use this for volatility-aware
```

**Workflow**:

1. **Stage 1**: Fast approximate scan to filter candidates (use_approximate=True)
2. **Stage 2**: Full precision calculation for filtered candidates (use_approximate=False)

**Expected Performance**: 10-20x speedup for large-scale scanning

---

## 🔧 Integration Examples

### Example 1: Python Script Integration

```python
from modules.adaptive_trend_LTS.core.compute_atc_signals import compute_atc_signals
from modules.adaptive_trend_LTS.core.compute_atc_signals.incremental_atc import IncrementalATC
import pandas as pd

# For live trading (single bar updates)
atc = IncrementalATC(config={
    'ema_len': 28, 'hull_len': 28, 'wma_len': 28,
    'dema_len': 28, 'lsma_len': 28, 'kama_len': 28,
    'robustness': 'Medium',
    'La': 0.02, 'De': 0.03,
})
atc.initialize(historical_prices)
new_signal = atc.update(new_price)  # O(1) operation

# For batch scanning (100+ symbols)
from modules.adaptive_trend_LTS.core.compute_atc_signals.batch_processor import process_symbols_batch_rust

symbols_data = {'BTCUSDT': prices_series, 'ETHUSDT': prices_series, ...}
config = {
    'ema_len': 28, 'hull_len': 28, 'wma_len': 28,
    'dema_len': 28, 'lsma_len': 28, 'kama_len': 28,
    'robustness': 'Medium',
    'La': 0.02, 'De': 0.03,
    'batch_processing': True,
    'use_cache': True,
    'fast_mode': True,
    'precision': 'float32',
}
results = process_symbols_batch_rust(symbols_data, config)
```

### Example 2: YAML Config Integration (`standard_batch_scan_config.yaml`)

```yaml
# Adaptive Trend Classification Settings
adaptive_trend_lts:
  # Core Parameters
  ema_len: 28
  hull_len: 28
  wma_len: 28
  dema_len: 28
  lsma_len: 28
  kama_len: 28
  
  # ATC Core
  robustness: "Medium"
  La: 0.02
  De: 0.03
  cutout: 0
  
  # Signal Thresholds
  long_threshold: 0.1
  short_threshold: -0.1
  
  # Performance Settings (ADJUST BASED ON USE CASE)
  batch_processing: true      # Rust Rayon for <100 symbols
  use_cuda: false             # Set true for 100+ symbols
  parallel_l1: true
  parallel_l2: true
  prefer_gpu: true
  use_cache: true
  fast_mode: true
  precision: "float32"
  
  # Dask Integration (for 1000+ symbols)
  use_dask: false             # Set true for large batches
  npartitions: 20
  
  # Incremental Updates (for live trading)
  use_incremental: false      # Set true for live trading
  
  # Approximate MAs (for fast filtering)
  use_approximate: false      # Set true for initial filtering
  use_adaptive_approximate: false
  
  # Memory Optimizations
  use_memory_mapped: false
  use_compression: false
  compression_level: 5
  
  # Advanced
  warm_cache: false
  use_codegen_specialization: false
```

---

## ⚙️ Configuration Decision Tree

```
START
  │
  ├─ Live Trading (single symbol, real-time updates)?
  │   └─ YES → use_incremental=True
  │   └─ NO → Continue
  │
  ├─ How many symbols?
  │   ├─ <100 → batch_processing=True, use_cuda=False
  │   ├─ 100-1000 → use_cuda=True (if GPU available)
  │   ├─ 1000-10000 → use_dask=True, batch_processing=True
  │   └─ >10000 → use_dask=True, use_approximate=True
  │
  ├─ Memory constraints?
  │   └─ YES → use_dask=True, use_memory_mapped=True
  │   └─ NO → Continue
  │
  ├─ Need initial filtering?
  │   └─ YES → use_approximate=True (Stage 1), then full precision (Stage 2)
  │   └─ NO → Continue
  │
  └─ Always enable:
      - use_cache=True
      - fast_mode=True
      - precision="float32" (unless need high precision)
```

---

## 📊 Performance Comparison

| Configuration | Symbols | Time | Speedup | Memory | Notes |
|--------------|---------|------|---------|--------|-------|
| **Baseline (Python)** | 99 | 49.65s | 1.00x | 122.1 MB | Original implementation |
| **Rust Rayon** | 99 | 8.12s | 6.11x | 18.2 MB | ⭐ **Best for <100 symbols** |
| **CUDA Batch** | 99 | 0.59s | 83.53x | 51.7 MB | ⭐ **Best for 100-1000 symbols** |
| **Rust + Dask** | 1000+ | ~9.45s | 5.25x | 12.5 MB | ⭐ **Best for 1000+ symbols** |
| **Incremental** | 1 | <0.01s | 1000x+ | <1 MB | ⭐ **Best for live trading** |
| **Approximate Filter** | 1000+ | ~5s | 10x | ~20 MB | ⭐ **Best for initial filtering** |

---

## ✅ Best Practices

1. **Start with Rust Rayon** (batch_processing=True) for most use cases
2. **Enable CUDA** only if you have 100+ symbols and NVIDIA GPU
3. **Use Dask** for 1000+ symbols or out-of-memory scenarios
4. **Use Incremental ATC** for live trading (single bar updates)
5. **Use Approximate MAs** for initial filtering, then full precision for final decisions
6. **Always enable caching** (use_cache=True)
7. **Use float32** for speed unless you need high precision
8. **Monitor memory** with large datasets
9. **Profile your workload** to identify bottlenecks

---

## 🔍 Troubleshooting

### Issue: Slow performance with Rust backend

**Check**:

- Is `batch_processing=True`?
- Is `parallel_l1=True` and `parallel_l2=True`?
- Is `use_cache=True`?

**Solution**: Enable all parallelism and caching flags

---

### Issue: CUDA not working

**Check**:

- CUDA Toolkit 12.x installed?
- NVIDIA GPU with compute capability >= 6.0?
- Rust extensions built with CUDA support?

**Solution**: Run `powershell -ExecutionPolicy Bypass -File build_cuda.ps1` in `rust_extensions/`

---

### Issue: Out of memory with large datasets

**Check**:

- Dataset size > available RAM?

**Solution**: Enable Dask (`use_dask=True`) and optionally memory-mapped arrays (`use_memory_mapped=True`)

---

### Issue: Signals differ from expected

**Check**:

- Using `use_approximate=True`?

**Expected Behavior**: Approximate MAs have ~5% tolerance. Use full precision for final trading decisions.

---

## 📄 References

- **Full Settings Guide**: `modules/adaptive_trend_LTS/docs/setting_guides.md`
- **Features Summary**: `modules/adaptive_trend_LTS/docs/features_summary.md`
- **Phase Documentation**:
  - Phase 3 (Rust): `docs/phase3_task.md`
  - Phase 4 (CUDA): `docs/phase4_task.md`
  - Phase 5 (Dask): `docs/phase5_task.md`
  - Phase 6 (Incremental/Approximate): `docs/phase6_task.md`
  - Phase 7 (Memory): `docs/phase7_task.md`
  - Phase 8 (Profiling): `docs/phase8_task.md`
  - Phase 8.1 (Cache & Parallelism): `docs/phase8.1_task.md`
  - Phase 8.2 (JIT Specialization): `docs/phase8.2_task.md`

---

**Last Updated**: 2026-01-29  
**Version**: LTS (Long-Term Support)  
**Status**: ✅ Production Ready
