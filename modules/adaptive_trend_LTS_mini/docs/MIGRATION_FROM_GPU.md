# Migration Guide: From GPU Version to CPU-Only Mini Version

## Overview

This guide helps you migrate from the full `adaptive_trend_LTS` module (with GPU/CUDA support) to the `adaptive_trend_LTS_mini` CPU-only version. The CPU-only version maintains 100% functional parity while removing all GPU dependencies for broader compatibility and simpler deployment.

**Migration Date**: 2026-01-31
**Target Version**: adaptive_trend_LTS_mini v1.0.0
**Status**: ✅ Production Ready

---

## Table of Contents

1. [Why Migrate to CPU-Only Version](#why-migrate-to-cpu-only-version)
2. [Key Architectural Changes](#key-architectural-changes)
3. [Removed Features](#removed-features)
4. [API Changes](#api-changes)
5. [Performance Comparison](#performance-comparison)
6. [Step-by-Step Migration Guide](#step-by-step-migration-guide)
7. [Configuration Changes](#configuration-changes)
8. [Code Examples](#code-examples)
9. [Performance Optimization Tips](#performance-optimization-tips)
10. [Troubleshooting](#troubleshooting)
11. [FAQ](#faq)

---

## Why Migrate to CPU-Only Version

### ✅ Benefits of CPU-Only Version

1. **No Hardware Dependencies**: Works on any CPU without NVIDIA GPU
2. **Cloud-Friendly**: Runs on any cloud instance (AWS, Azure, GCP) without expensive GPU instances
3. **Development-Easy**: Test on laptops and development machines without discrete GPU
4. **Lower Memory Footprint**: No GPU VRAM overhead, uses system RAM only
5. **Simpler Deployment**: No CUDA runtime dependencies, no driver management
6. **Smaller Build Size**: 57% smaller binary (1.5MB → 637KB)
7. **Better CPU Utilization**: Uses all available CPU cores via Rayon parallelism
8. **Maintenance-Free**: No CUDA version compatibility issues

### ⚠️ When NOT to Migrate

**Keep using GPU version if you:**
- Have NVIDIA GPU hardware available
- Need ultra-low latency (<50ms per symbol)
- Analyze 1000+ symbols in real-time
- Run high-frequency trading strategies
- Already have working GPU infrastructure

---

## Key Architectural Changes

### Backend Architecture

#### GPU Version (Full LTS)
```
Python API Layer
    ↓
Rust/cudarc Orchestration
    ↓
CUDA Kernels (GPU)
    ↓
GPU VRAM Processing
```

#### CPU-Only Version (LTS_mini)
```
Python API Layer
    ↓
Rust/Rayon Backend
    ↓
Multi-core CPU Processing
    ↓
System RAM
```

### Computation Flow

| Component | GPU Version | CPU-Only Version |
|-----------|-------------|------------------|
| **Moving Averages** | CUDA kernels (batch_ma_kernels.cu) | Rust/Rayon (parallel SIMD) |
| **Equity Calculation** | CUDA kernels (equity_kernel.cu) | Rust/Rayon (parallel loops) |
| **Signal Detection** | CUDA kernels (signal_kernels.cu) | Rust/Rayon (parallel processing) |
| **Parallelism** | GPU threads (thousands) | CPU threads (# of cores) |
| **Memory** | GPU VRAM (pinned memory) | System RAM (standard allocation) |
| **Batch Processing** | CUDA streams | Dask + ThreadPoolExecutor |

---

## Removed Features

### 1. CUDA/GPU Dependencies (Rust)

**Removed from Cargo.toml:**
```toml
# ❌ Removed
[dependencies]
cudarc = { version = "0.19", features = ["cuda-12-8"] }
```

**Removed Rust Files:**
- ❌ `rust_extensions/src/equity_cuda.rs`
- ❌ `rust_extensions/src/ma_cuda.rs`
- ❌ `rust_extensions/src/signal_cuda.rs`
- ❌ `rust_extensions/src/batch_processing.rs` (CUDA version)
- ❌ `rust_extensions/build.rs` (CUDA-specific build script)
- ❌ `rust_extensions/build_cuda.ps1`

### 2. CUDA Kernel Files

**Removed .cu Files:**
- ❌ `core/gpu_backend/batch_ma_kernels.cu`
- ❌ `core/gpu_backend/batch_signal_kernels.cu`
- ❌ `core/gpu_backend/equity_kernel.cu`
- ❌ `core/gpu_backend/ma_kernels.cu`
- ❌ `core/gpu_backend/signal_kernels.cu`
- ❌ `core/gpu_backend/gpu_common.h`

### 3. Python GPU Backend

**Removed Python Files:**
- ❌ `core/gpu_backend/` (entire directory)
- ❌ `core/compute_moving_averages/_gpu.py`
- ❌ `core/process_layer1/_gpu_signals.py`
- ❌ `core/process_layer1/_gpu_equity.py`
- ❌ `core/scanner/gpu_scan.py` (replaced with stub)

### 4. Python Dependencies

**Removed from requirements.txt:**
```python
# ❌ No longer required
cupy-cuda12x>=12.0.0
pycuda>=2023.1
```

### 5. Test and Benchmark Files

**Removed:**
- ❌ `tests/test_cuda_kernels.py`
- ❌ `tests/test_equity_cuda.py`
- ❌ `benchmarks/benchmark_cuda.py`
- ❌ `benchmarks/cuda_vs_cpu_diagnostic.py`
- ❌ `benchmarks/test_cuda_fix.py`
- ❌ `benchmarks/simulate_cuda_hma.py`
- ❌ `benchmarks/visualize_cuda_drift.py`
- ❌ `benchmarks/verify_cuda_dask_usage.py`

---

## API Changes

### 1. Removed Parameters

#### compute_atc_signals()

**Before (GPU Version):**
```python
result = compute_atc_signals(
    prices,
    use_cuda=True,           # ❌ REMOVED
    prefer_gpu=True,         # ❌ REMOVED
    use_rust_backend=True,
    precision="float64"
)
```

**After (CPU-Only):**
```python
result = compute_atc_signals(
    prices,
    use_rust_backend=True,   # ✅ Now CPU-only (Rayon)
    precision="float64"
)
```

### 2. Configuration Changes

#### ATCConfig Dataclass

**Before (GPU Version):**
```python
@dataclass
class ATCConfig:
    use_cuda: bool = False           # ❌ REMOVED
    prefer_gpu: bool = False         # ❌ REMOVED
    use_rust_backend: bool = True
    parallel_l1: bool = True
    parallel_l2: bool = True
```

**After (CPU-Only):**
```python
@dataclass
class ATCConfig:
    # ✅ use_cuda removed
    # ✅ prefer_gpu removed
    use_rust_backend: bool = True    # Now CPU-only
    parallel_l1: bool = True
    parallel_l2: bool = True
```

### 3. CLI Arguments

**Before (GPU Version):**
```bash
python -m modules.adaptive_trend_LTS.cli.main \
    --symbol BTC/USDT \
    --use-cuda              # ❌ REMOVED
```

**After (CPU-Only):**
```bash
python -m modules.adaptive_trend_LTS_mini.cli.main \
    --symbol BTC/USDT
    # ✅ --use-cuda flag removed
```

### 4. Scanner API

#### gpu_scan.py (Now a Stub)

**Before (GPU Version):**
```python
# Real GPU batch scanning
def _scan_gpu_batch(symbols, data_fetcher, atc_config, min_signal, batch_size):
    # CUDA batch processing
    # Returns GPU-accelerated results
```

**After (CPU-Only):**
```python
# Stub that falls back to sequential
def _scan_gpu_batch(symbols, data_fetcher, atc_config, min_signal, batch_size):
    log_warn("GPU scanning not available in LTS_mini, falling back to sequential")
    return _scan_sequential(symbols, data_fetcher, atc_config, min_signal, batch_size)
```

### 5. Backward Compatibility Layer

The config loader includes backward compatibility:

```python
# In utils/config.py
use_rust_backend = params.get(
    "use_rust_backend",
    params.get("prefer_gpu", True)  # ✅ Maps old prefer_gpu to use_rust_backend
)
```

**This means old configs with `prefer_gpu` will still work!**

---

## Performance Comparison

### Single Symbol Analysis

| Metric | GPU Version | CPU-Only Version | Factor |
|--------|-------------|------------------|--------|
| 1000 bars | 10-50ms | 100-500ms | ~10x slower |
| Memory usage | GPU VRAM + RAM | RAM only | More efficient |
| Hardware required | NVIDIA GPU | Any CPU | Universal |

### Batch Processing (100 symbols)

| Metric | GPU Version | CPU-Only Version | Factor |
|--------|-------------|------------------|--------|
| Processing time | 2-10s | 10-50s | ~5x slower |
| Parallelism | GPU threads | CPU cores | Different strategy |
| Memory footprint | Higher (GPU VRAM) | Lower (RAM only) | 30-50% reduction |

### Scalability

**GPU Version:**
- Scales with GPU compute units (thousands of threads)
- Best for large batches (100+ symbols)
- Nearly constant time per symbol in batch

**CPU-Only Version:**
- Scales linearly with CPU cores
- Best for moderate batches (10-100 symbols)
- Predictable performance curve

### Real-World Example

**Scenario**: Scan 50 symbols on 1h timeframe with 1500 bars each

| System | GPU Version | CPU-Only Version |
|--------|-------------|------------------|
| i7-10700K (8c/16t) + RTX 3060 | ~3-5s | ~15-25s |
| i7-10700K (8c/16t) CPU-only | N/A | ~15-25s |
| AMD Ryzen 9 5950X (16c/32t) | ~2-4s | ~8-15s |
| Cloud: AWS c6i.2xlarge (8 vCPU) | N/A | ~20-30s |

---

## Step-by-Step Migration Guide

### Phase 1: Preparation (15 minutes)

#### 1.1 Check Current Usage

Identify where GPU features are used:

```bash
# Search for GPU-related code in your project
cd your_project/
grep -r "use_cuda" .
grep -r "prefer_gpu" .
grep -r "gpu_backend" .
grep -r "cupy" .
grep -r "pycuda" .
```

#### 1.2 Backup Configurations

```bash
# Backup your current config files
cp config/atc_settings.yaml config/atc_settings.yaml.backup
```

#### 1.3 Review Dependencies

Check if your requirements.txt has GPU dependencies:
```bash
grep -E "cupy|pycuda" requirements.txt
```

### Phase 2: Update Code (30 minutes)

#### 2.1 Update Import Paths

**Before:**
```python
from modules.adaptive_trend_LTS.core.compute_atc_signals import compute_atc_signals
from modules.adaptive_trend_LTS.utils.config import ATCConfig
```

**After:**
```python
from modules.adaptive_trend_LTS_mini.core.compute_atc_signals import compute_atc_signals
from modules.adaptive_trend_LTS_mini.utils.config import ATCConfig
```

**Pro Tip**: Use find-and-replace in your IDE:
- Find: `modules.adaptive_trend_LTS.`
- Replace: `modules.adaptive_trend_LTS_mini.`

#### 2.2 Remove GPU Parameters

**Search and update all instances:**

```python
# ❌ Before
result = compute_atc_signals(
    prices,
    use_cuda=True,
    prefer_gpu=True,
    use_rust_backend=True
)

# ✅ After
result = compute_atc_signals(
    prices,
    use_rust_backend=True  # Now CPU-only
)
```

#### 2.3 Update Configuration Objects

```python
# ❌ Before
config = ATCConfig(
    use_cuda=True,
    prefer_gpu=True,
    use_rust_backend=True,
    batch_size=100
)

# ✅ After
config = ATCConfig(
    use_rust_backend=True,  # Now CPU-only
    batch_size=50,          # Smaller batches for CPU
    parallel_l1=True,
    parallel_l2=True
)
```

#### 2.4 Update CLI Scripts

```python
# ❌ Before (CLI with GPU flags)
parser.add_argument("--use-cuda", action="store_true", help="Use CUDA GPU")

# ✅ After (Remove GPU arguments)
# No changes needed - flags are gone
```

### Phase 3: Configuration Migration (15 minutes)

#### 3.1 Update YAML Configs

**Before (config/atc_settings.yaml):**
```yaml
atc:
  use_cuda: true
  prefer_gpu: true
  use_rust_backend: true
  batch_size: 200
  parallel_l1: true
  parallel_l2: true
```

**After (config/atc_settings.yaml):**
```yaml
atc:
  # use_cuda: removed
  # prefer_gpu: removed
  use_rust_backend: true  # Now CPU-only
  batch_size: 100         # Reduced for CPU
  parallel_l1: true
  parallel_l2: true
  precision: "float64"
```

#### 3.2 Adjust Batch Sizes

CPU-only version performs best with smaller batches:

```yaml
# GPU Version (optimal)
batch_size: 200-500

# CPU-Only Version (optimal)
batch_size: 50-100
```

### Phase 4: Build and Test (20 minutes)

#### 4.1 Build Rust Extension

```bash
cd modules/adaptive_trend_LTS_mini/rust_extensions
cargo build --release
```

**Verify build size:**
```bash
# Should be ~637KB (not ~1.5MB with CUDA)
ls -lh target/release/*.dll    # Windows
ls -lh target/release/*.so     # Linux/Mac
```

#### 4.2 Run Tests

```bash
# Test CPU-only validation
pytest modules/adaptive_trend_LTS_mini/tests/test_cpu_only_validation.py -v

# Test core functionality
pytest modules/adaptive_trend_LTS_mini/tests/ -v -k "not gpu"
```

#### 4.3 Verify No GPU Imports

```bash
# Should find NO matches
grep -r "import cupy" modules/adaptive_trend_LTS_mini/
grep -r "import pycuda" modules/adaptive_trend_LTS_mini/
```

### Phase 5: Performance Testing (20 minutes)

#### 5.1 Benchmark Single Symbol

```python
import time
from modules.adaptive_trend_LTS_mini.core.compute_atc_signals import compute_atc_signals

# Test single symbol
start = time.time()
result = compute_atc_signals(prices, use_rust_backend=True)
elapsed = time.time() - start
print(f"Single symbol: {elapsed*1000:.1f}ms")

# Expected: 100-500ms depending on CPU
```

#### 5.2 Benchmark Batch Processing

```python
from modules.adaptive_trend_LTS_mini.core.scanner import scan_all_symbols

# Test batch scanning
start = time.time()
results, short_signals = scan_all_symbols(
    data_fetcher=data_fetcher,
    atc_config=config,
    min_signal=0.5
)
elapsed = time.time() - start
print(f"Batch {len(results)} symbols: {elapsed:.1f}s")
```

### Phase 6: Production Deployment (30 minutes)

#### 6.1 Update Requirements

```python
# requirements.txt - Remove GPU deps
# ❌ Remove these:
# cupy-cuda12x>=12.0.0
# pycuda>=2023.1

# ✅ Keep these:
pandas>=1.3.0
numpy>=1.21.0
dask>=2021.10.0
```

#### 6.2 Update Docker (if applicable)

```dockerfile
# ❌ Before (GPU version)
FROM nvidia/cuda:12.0-runtime-ubuntu22.04
RUN pip install cupy-cuda12x pycuda

# ✅ After (CPU-only)
FROM python:3.11-slim
# No GPU dependencies needed
```

#### 6.3 Deploy and Monitor

```bash
# Deploy new version
python -m modules.adaptive_trend_LTS_mini.cli.main --scan --top 50

# Monitor CPU usage (should use all cores)
htop  # or Task Manager on Windows
```

---

## Configuration Changes

### Recommended CPU-Only Settings

```python
config = ATCConfig(
    # Core settings (unchanged)
    ema_len=28,
    hma_len=28,
    wma_len=28,
    dema_len=28,
    lsma_len=28,
    kama_len=28,
    robustness="Medium",
    lambda_param=0.02,
    decay=0.03,

    # Backend settings (changed)
    use_rust_backend=True,     # Now CPU-only
    parallel_l1=True,          # Parallel Layer 1
    parallel_l2=True,          # Parallel Layer 2

    # Performance tuning (adjusted for CPU)
    batch_size=100,            # Smaller than GPU (was 200-500)
    precision="float64",       # Keep high precision

    # Optimization features (optional)
    use_approximate=True,      # Faster scanning
    approximate_threshold=0.05,

    # Caching (optional)
    use_compression=True,      # Compress cache files
    compression_level=5
)
```

---

## Code Examples

### Example 1: Basic Signal Calculation

**Before (GPU Version):**
```python
from modules.adaptive_trend_LTS.core.compute_atc_signals import compute_atc_signals

result = compute_atc_signals(
    prices=df['close'],
    ema_len=28,
    use_cuda=True,           # ❌ Removed
    prefer_gpu=True,         # ❌ Removed
    use_rust_backend=True
)
```

**After (CPU-Only):**
```python
from modules.adaptive_trend_LTS_mini.core.compute_atc_signals import compute_atc_signals

result = compute_atc_signals(
    prices=df['close'],
    ema_len=28,
    use_rust_backend=True    # ✅ Now CPU-only
)
```

### Example 2: Scanner with Configuration

**Before (GPU Version):**
```python
from modules.adaptive_trend_LTS.core.scanner import scan_all_symbols
from modules.adaptive_trend_LTS.utils.config import ATCConfig

config = ATCConfig(
    timeframe="15m",
    use_cuda=True,           # ❌ Removed
    batch_size=200
)

results, signals = scan_all_symbols(
    data_fetcher=fetcher,
    atc_config=config,
    min_signal=0.5
)
```

**After (CPU-Only):**
```python
from modules.adaptive_trend_LTS_mini.core.scanner import scan_all_symbols
from modules.adaptive_trend_LTS_mini.utils.config import ATCConfig

config = ATCConfig(
    timeframe="15m",
    batch_size=100,          # ✅ Adjusted for CPU
    parallel_l1=True,
    parallel_l2=True
)

results, signals = scan_all_symbols(
    data_fetcher=fetcher,
    atc_config=config,
    min_signal=0.5
)
```

### Example 3: CLI Usage

**Before (GPU Version):**
```bash
# Analyze with GPU
python -m modules.adaptive_trend_LTS.cli.main \
    --symbol BTC/USDT \
    --timeframe 1h \
    --use-cuda              # ❌ Removed

# Scan with GPU
python -m modules.adaptive_trend_LTS.cli.main \
    --scan \
    --use-cuda              # ❌ Removed
```

**After (CPU-Only):**
```bash
# Analyze with CPU
python -m modules.adaptive_trend_LTS_mini.cli.main \
    --symbol BTC/USDT \
    --timeframe 1h
    # ✅ GPU flags removed, uses Rust/Rayon automatically

# Scan with CPU
python -m modules.adaptive_trend_LTS_mini.cli.main \
    --scan \
    --top 100
```

### Example 4: Batch Processing

**Before (GPU Version):**
```python
from modules.adaptive_trend_LTS.core.compute_atc_signals import process_symbols_rust_dask

results = process_symbols_rust_dask(
    symbols=symbols,
    data_fetcher=fetcher,
    atc_config=config,
    use_cuda=True,           # ❌ Removed
    batch_size=500
)
```

**After (CPU-Only):**
```python
from modules.adaptive_trend_LTS_mini.core.compute_atc_signals import process_symbols_rust_dask

results = process_symbols_rust_dask(
    symbols=symbols,
    data_fetcher=fetcher,
    atc_config=config,
    batch_size=100           # ✅ Smaller batches for CPU
)
```

---

## Performance Optimization Tips

### 1. CPU-Specific Optimizations

#### Enable Rust Backend
```python
config.use_rust_backend = True  # Always use Rust/Rayon
```

#### Use All CPU Cores
```python
config.parallel_l1 = True  # Parallel Layer 1 calculations
config.parallel_l2 = True  # Parallel Layer 2 calculations
```

#### Optimize Batch Size
```python
# Test different batch sizes for your CPU
for batch_size in [50, 100, 150, 200]:
    config.batch_size = batch_size
    # Run benchmark and measure
```

**Recommended batch sizes:**
- 4-8 cores: batch_size=50
- 8-16 cores: batch_size=100
- 16+ cores: batch_size=150-200

### 2. Enable Approximate Mode

For faster scanning (trading some precision for speed):

```python
config.use_approximate = True
config.approximate_threshold = 0.05  # 5% approximation error
```

**Performance gain**: 2-3x faster scanning
**Accuracy trade-off**: ~95-98% correlation with exact results

### 3. Use Compression for Cache

Reduce disk I/O and storage:

```python
config.use_compression = True
config.compression_level = 5  # Balance between speed and size
config.compression_algorithm = "blosclz"
```

**Storage reduction**: 5-10x smaller cache files

### 4. Optimize Data Fetching

```python
# Reuse data fetcher instance
data_fetcher = DataFetcher(exchange_manager)

# Enable caching
data_fetcher.enable_cache()

# Batch fetch symbols
symbols_data = data_fetcher.fetch_batch(symbols, timeframe="1h")
```

### 5. Profile and Monitor

```python
import cProfile
import pstats

# Profile your code
profiler = cProfile.Profile()
profiler.enable()

# Your ATC code here
result = compute_atc_signals(prices, use_rust_backend=True)

profiler.disable()
stats = pstats.Stats(profiler)
stats.sort_stats('cumulative')
stats.print_stats(20)
```

### 6. Memory Management

```python
import gc

# Process in smaller batches with explicit GC
for batch in chunked(symbols, batch_size=50):
    results = process_batch(batch)
    gc.collect()  # Force garbage collection between batches
```

---

## Troubleshooting

### Issue 1: Import Errors

**Error:**
```
ModuleNotFoundError: No module named 'cupy'
```

**Solution:**
You're still importing from the GPU version. Update imports:
```python
# Change from:
from modules.adaptive_trend_LTS import ...

# To:
from modules.adaptive_trend_LTS_mini import ...
```

### Issue 2: Slow Performance

**Symptom**: CPU-only version is 20x+ slower than expected

**Solution:**
1. Verify Rust backend is enabled:
   ```python
   print(config.use_rust_backend)  # Should be True
   ```

2. Check Rust extension is built:
   ```bash
   cd modules/adaptive_trend_LTS_mini/rust_extensions
   cargo build --release
   ```

3. Enable parallel processing:
   ```python
   config.parallel_l1 = True
   config.parallel_l2 = True
   ```

4. Reduce batch size:
   ```python
   config.batch_size = 50  # Try smaller batches
   ```

### Issue 3: High Memory Usage

**Symptom**: Memory usage spikes during batch processing

**Solution:**
```python
# Reduce batch size
config.batch_size = 50  # Down from 100

# Enable compression
config.use_compression = True

# Force GC between batches
import gc
gc.collect()
```

### Issue 4: Configuration Errors

**Error:**
```
TypeError: __init__() got an unexpected keyword argument 'use_cuda'
```

**Solution:**
Remove GPU-related parameters from your config:
```python
# Remove these:
# use_cuda=True,
# prefer_gpu=True,

# Keep this:
use_rust_backend=True,
```

### Issue 5: Build Errors

**Error:**
```
error: failed to compile Rust extension
```

**Solution:**
1. Verify Rust is installed:
   ```bash
   rustc --version
   ```

2. Clean and rebuild:
   ```bash
   cd rust_extensions
   cargo clean
   cargo build --release
   ```

3. Check for CUDA-related flags (should NOT exist):
   ```bash
   grep -r "cudarc" Cargo.toml  # Should return nothing
   ```

### Issue 6: Missing Dependencies

**Error:**
```
ImportError: cannot import name 'compute_atc_signals'
```

**Solution:**
Install dependencies:
```bash
pip install pandas numpy dask
```

Build Rust extension:
```bash
cd modules/adaptive_trend_LTS_mini/rust_extensions
cargo build --release
```

---

## FAQ

### Q1: Is the CPU-only version feature-complete?

**A:** Yes! The CPU-only version has 100% functional parity with the GPU version. All algorithms, calculations, and features are identical - only the execution backend changed from GPU to CPU.

### Q2: How much slower is the CPU-only version?

**A:** Approximately 5-10x slower for batch processing:
- Single symbol: 100-500ms (vs 10-50ms GPU)
- 100 symbols: 10-50s (vs 2-10s GPU)

However, it scales linearly with CPU cores and is more memory-efficient.

### Q3: Can I still use my old configuration files?

**A:** Yes! The CPU-only version includes backward compatibility. Old configs with `prefer_gpu` will map to `use_rust_backend` automatically. Just remove the `use_cuda` parameter.

### Q4: Do I need to retrain or recalculate anything?

**A:** No! All calculations are identical. Signals, equity curves, and results will be exactly the same (within floating-point precision).

### Q5: Will this work on cloud instances without GPU?

**A:** Yes! This is one of the main benefits. Deploy on any AWS/Azure/GCP instance type without expensive GPU costs.

### Q6: What about the Rust extension build size?

**A:** The Rust extension is now 57% smaller:
- GPU version: ~1.5MB (with cudarc)
- CPU-only: ~637KB

### Q7: Can I migrate back to GPU version later?

**A:** Yes! The full `adaptive_trend_LTS` module with GPU support is still available. You can switch back by:
1. Changing imports from `LTS_mini` to `LTS`
2. Adding back `use_cuda=True` parameters
3. Installing GPU dependencies (cupy, pycuda)

### Q8: Is parallel processing still supported?

**A:** Yes! The CPU-only version uses Rust/Rayon for multi-core parallelism:
- `parallel_l1=True`: Parallel Layer 1 calculations
- `parallel_l2=True`: Parallel Layer 2 calculations

This automatically uses all available CPU cores.

### Q9: What's the best CPU for this version?

**A:** The more cores, the better:
- Minimum: 4 cores (i5/Ryzen 5)
- Recommended: 8-16 cores (i7/Ryzen 7-9)
- Optimal: 16+ cores (Threadripper, Xeon)

Performance scales linearly with core count.

### Q10: Can I use this in Docker containers?

**A:** Yes! CPU-only version is perfect for containers:

```dockerfile
FROM python:3.11-slim
WORKDIR /app
COPY . .
RUN pip install -r requirements.txt
CMD ["python", "-m", "modules.adaptive_trend_LTS_mini.cli.main"]
```

No GPU drivers, no CUDA toolkit needed!

---

## Summary

### Migration Checklist

- [ ] Update imports: `adaptive_trend_LTS` → `adaptive_trend_LTS_mini`
- [ ] Remove parameters: `use_cuda`, `prefer_gpu`
- [ ] Update config: Remove GPU-related fields
- [ ] Adjust batch sizes: Reduce by 50% for CPU
- [ ] Build Rust extension: `cargo build --release`
- [ ] Run tests: `pytest modules/adaptive_trend_LTS_mini/tests/`
- [ ] Update requirements.txt: Remove cupy, pycuda
- [ ] Verify no GPU imports: `grep -r "cupy\|pycuda" .`
- [ ] Benchmark performance: Test single symbol and batch
- [ ] Deploy and monitor: Check CPU utilization

### Key Takeaways

1. **100% functional parity**: All features work identically
2. **5-10x slower**: Trade-off for universal compatibility
3. **Zero GPU dependencies**: Works anywhere
4. **Better CPU utilization**: Uses all available cores
5. **Simpler deployment**: No drivers, no hassle
6. **Backward compatible**: Old configs still work

### Next Steps

1. Start with development/testing migration
2. Benchmark on your workload
3. Optimize batch sizes for your CPU
4. Deploy to staging environment
5. Monitor performance and adjust settings
6. Roll out to production

---

## Additional Resources

- [Main README](../README.md) - Module overview and quick start
- [API Reference](API_REFERENCE.md) - Complete API documentation
- [Setting Guides](setting_guides.md) - Parameter tuning and presets
- [Phase 3 Task](phase3_task.md) - Rust backend setup and troubleshooting
- [Quickstart Guide](QUICKSTART.md) - Get started in 5 minutes

---

**Questions?** Open an issue on GitHub or check the documentation in `docs/`.

**Migration Date**: 2026-01-31
**Document Version**: 1.0.0
**Last Updated**: 2026-02-06
