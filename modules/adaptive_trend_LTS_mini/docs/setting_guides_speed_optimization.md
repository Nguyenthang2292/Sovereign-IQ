# ⚡ RECOMMENDED SETTINGS FOR MAXIMUM PROCESSING SPEED

**Purpose**: Bộ cấu hình tối ưu hóa tốc độ xử lý cho `adaptive_trend_LTS_mini` module  
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

## 🔬 Benchmarking Parallel vs Sequential Execution

To verify the performance benefits of parallel execution on your system, use the included benchmark script:

### Quick Benchmark

```bash
# Benchmark 100 symbols (sequential vs parallel with 10 workers)
python benchmarks/benchmark_parallel_scan.py

# Custom symbol count
python benchmarks/benchmark_parallel_scan.py --symbols 50

# Custom worker count
python benchmarks/benchmark_parallel_scan.py --symbols 100 --workers 20
```

### Expected Results

| Execution Mode | Symbols | Expected Time | Notes |
|----------------|---------|---------------|-------|
| Sequential | 100 | ~60-120s | Single-threaded processing |
| Parallel (10 workers) | 100 | ~10-20s | 5-10x speedup typical |

**Note**: Actual speedup depends on:
- CPU core count and performance
- Network latency (data fetching)
- Exchange rate limits
- System load

### Interpreting Results

The benchmark will display:
- **Time per symbol**: Total time divided by symbol count
- **Speedup**: Ratio of sequential time to parallel time
- **Per-symbol metrics**: Average processing time per symbol

Use these results to:
1. **Verify parallel execution is faster** than sequential
2. **Tune worker count** for your system (typically 5-20 workers optimal)
3. **Identify bottlenecks** (if speedup is low, check network/rate limits)

---

## 📋 Recommended Presets by Use Case

### Preset 1: **Live Trading (Maximum Speed for Single Bar)**

**Use Case**: Real-time trading bot, WebSocket updates, live signal monitoring

**Configuration**:

```yaml
# Use Incremental ATC for O(1) updates
use_incremental: true
use_o1_mas: true                # Enable O(1) MA algorithms
use_rust_incremental: true      # Use Rust backend
batch_processing: false         # Not needed for single symbol
use_cuda: false                 # Not needed for single symbol
parallel_l1: false
parallel_l2: false
use_cache: true
fast_mode: true
precision: "float32"
```

**Python Implementation**:

```python
from modules.adaptive_trend_LTS_mini.core.compute_atc_signals import IncrementalATC
import pandas as pd
from pathlib import Path

# Setup configuration
config = {
    "ema_len": 28, "hma_len": 28, "wma_len": 28,
    "dema_len": 28, "lsma_len": 28, "kama_len": 28,
    "robustness": "Medium",
    "La": 0.02, "De": 0.03,
    "use_o1_mas": True,
    "use_rust_incremental": True,
}

# Initialize once with historical data
atc = IncrementalATC(config)
historical_prices = pd.Series([...])  # Your historical data
atc.initialize(historical_prices)

# Optional: Save state for quick restart
state_path = Path("states/BTCUSDT_1h.msgpack")
state_path.parent.mkdir(parents=True, exist_ok=True)
atc.save_state(state_path)

# In your trading loop:
def on_new_candle(new_price: float):
    signal = atc.update(new_price)  # O(1) operation, <0.01s
    
    if signal > 0.1:
        print(f"LONG signal: {signal:.4f}")
    elif signal < -0.1:
        print(f"SHORT signal: {signal:.4f}")
    else:
        print(f"NEUTRAL: {signal:.4f}")
    
    # Auto-save state periodically
    if should_save_state():
        atc.save_state(state_path)
    
    return signal

# Restart from saved state (no warmup needed)
# atc = IncrementalATC.load_state(state_path)
```

**Expected Performance**: <0.01s per update (1000x+ faster than full recalculation)

---

### Preset 2: **Small Batch Scanning (<100 symbols)**

**Use Case**: Portfolio scanner, daily watchlist, strategy testing

**Configuration**:

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

**Python Implementation**:

```python
from modules.adaptive_trend_LTS_mini.core.compute_atc_signals.batch_processor import process_symbols_batch_rust
from modules.common.core.data_fetcher import DataFetcher
import pandas as pd

# Setup
data_fetcher = DataFetcher()
symbols = ["BTCUSDT", "ETHUSDT", "BNBUSDT", "ADAUSDT", "SOLUSDT"]  # Your symbol list (<100)

config = {
    "ema_len": 28, "hma_len": 28, "wma_len": 28,
    "dema_len": 28, "lsma_len": 28, "kama_len": 28,
    "robustness": "Medium",
    "La": 0.02, "De": 0.03,
    "batch_processing": True,
    "parallel_l1": True,
    "parallel_l2": True,
    "use_cache": True,
    "fast_mode": True,
    "precision": "float32",
}

# Fetch data for all symbols
symbols_data = {}
for symbol in symbols:
    try:
        df, _ = data_fetcher.fetch_ohlcv_with_fallback_exchange(
            symbol, timeframe="15m", limit=1500
        )
        symbols_data[symbol] = df["close"]
    except Exception as e:
        print(f"Error fetching {symbol}: {e}")

# Batch processing with Rust
results = process_symbols_batch_rust(symbols_data, config)

# Extract signals
signals = {}
for symbol, result in results.items():
    final_signal = result["Average_Signal"].iloc[-1]
    signals[symbol] = final_signal

# Filter for trading opportunities
longs = {s: sig for s, sig in signals.items() if sig > 0.1}
shorts = {s: sig for s, sig in signals.items() if sig < -0.1}

print(f"\nLONG candidates ({len(longs)}):")
for symbol, signal in sorted(longs.items(), key=lambda x: x[1], reverse=True):
    print(f"  {symbol}: {signal:.4f}")

print(f"\nSHORT candidates ({len(shorts)}):")
for symbol, signal in sorted(shorts.items(), key=lambda x: x[1]):
    print(f"  {symbol}: {signal:.4f}")
```

**Expected Performance**: 6x speedup vs sequential, ~8-12s for 99 symbols × 1500 bars

---

### Preset 3: **Medium Batch Scanning (100-1000 symbols)**

**Use Case**: Exchange-wide scanning, market analysis, multi-strategy backtesting

**Configuration**:

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

**Python Implementation**:

```python
from modules.adaptive_trend_LTS_mini.core.compute_atc_signals import compute_atc_signals
from modules.common.core.data_fetcher import DataFetcher
import pandas as pd

# Setup
data_fetcher = DataFetcher()
symbols = data_fetcher.get_all_futures_symbols()  # 100-1000 symbols

config = {
    "ema_len": 28, "hma_len": 28, "wma_len": 28,
    "dema_len": 28, "lsma_len": 28, "kama_len": 28,
    "robustness": "Medium",
    "La": 0.02, "De": 0.03,
    "use_cuda": True,
    "prefer_gpu": True,
    "use_cache": True,
    "fast_mode": True,
    "precision": "float32",
}

# Process with CUDA
signals = {}
for symbol in symbols:
    try:
        df, _ = data_fetcher.fetch_ohlcv_with_fallback_exchange(
            symbol, timeframe="15m", limit=1500
        )
        
        results = compute_atc_signals(
            prices=df["close"],
            **config
        )
        
        signals[symbol] = results["Average_Signal"].iloc[-1]
    except Exception as e:
        print(f"Error processing {symbol}: {e}")

# Create results DataFrame
results_df = pd.DataFrame([
    {"symbol": s, "signal": sig}
    for s, sig in signals.items()
]).sort_values("signal", ascending=False)

# Export top candidates
top_longs = results_df[results_df["signal"] > 0.1].head(20)
top_shorts = results_df[results_df["signal"] < -0.1].tail(20)

print(f"\nTop 20 LONG signals:\n{top_longs}")
print(f"\nTop 20 SHORT signals:\n{top_shorts}")

# Save to CSV
results_df.to_csv("atc_signals.csv", index=False)
```

**Expected Performance**: 80x+ speedup, ~0.6s for 99 symbols × 1500 bars

**Requirements**:
- NVIDIA GPU with compute capability >= 6.0
- CUDA Toolkit 12.x installed
- Build with: `powershell -ExecutionPolicy Bypass -File build_cuda.ps1`

---

### Preset 4: **Large Batch Scanning (1000-10000 symbols)**

**Use Case**: Crypto market-wide scanning, multi-exchange analysis, research

**Configuration**:

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

**Python Implementation**:

```python
from modules.adaptive_trend_LTS_mini.core.scanner.scan_all_symbols import scan_all_symbols
from legacy.adaptive_trend_enhance.utils.config import ATCConfig
from modules.common.core.data_fetcher import DataFetcher
import pandas as pd

# Setup
data_fetcher = DataFetcher()

# ATCConfig with Dask enabled
atc_config = ATCConfig(
    timeframe="15m",
    limit=1500,
    ema_len=28, hma_len=28, wma_len=28,
    dema_len=28, lsma_len=28, kama_len=28,
    robustness="Medium",
    La=0.02, De=0.03,
    batch_processing=True,
    parallel_l1=True,
    parallel_l2=True,
    use_cache=True,
    fast_mode=True,
    precision="float32",
)

# Scan with Dask (out-of-core processing)
long_df, short_df = scan_all_symbols(
    data_fetcher=data_fetcher,
    atc_config=atc_config,
    min_signal=0.1,              # Signal threshold
    max_symbols=None,            # Scan all available symbols
    execution_mode="dask",       # Use Dask for large datasets
    npartitions=20,              # Parallel partitions
)

# Results
print(f"\nFound {len(long_df)} LONG candidates")
print(f"Found {len(short_df)} SHORT candidates")

# Top signals
print(f"\nTop 30 LONG signals:\n{long_df.head(30)}")
print(f"\nTop 30 SHORT signals:\n{short_df.head(30)}")

# Save results
long_df.to_csv("long_signals.csv", index=False)
short_df.to_csv("short_signals.csv", index=False)

# Export for further analysis
combined_df = pd.concat([
    long_df.assign(direction="LONG"),
    short_df.assign(direction="SHORT")
])
combined_df.to_parquet("all_signals.parquet")
```

**Expected Performance**: 5-10x speedup + unlimited dataset size (out-of-core)

**Memory Usage**: 10-20% of in-memory approach

---

### Preset 5: **Very Large Batch with Filtering (10000+ symbols)**

**Use Case**: Global market screening, cross-exchange arbitrage, ML feature generation

**Configuration**:

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

**Python Implementation**:

```python
from legacy.adaptive_trend_enhance.utils.config import ATCConfig
from modules.adaptive_trend_LTS_mini.core.scanner.scan_all_symbols import scan_all_symbols
from modules.adaptive_trend_LTS_mini.core.compute_atc_signals import compute_atc_signals
from modules.common.core.data_fetcher import DataFetcher
import pandas as pd

data_fetcher = DataFetcher()

# ===== STAGE 1: Fast Approximate Filtering =====
print("Stage 1: Fast approximate scan...")

fast_config = ATCConfig(
    timeframe="15m",
    limit=1500,
    ema_len=28, hma_len=28, wma_len=28,
    dema_len=28, lsma_len=28, kama_len=28,
    robustness="Medium",
    La=0.02, De=0.03,
    use_approximate=True,        # 2-3x faster
    use_dask=True,
    npartitions=30,
    batch_processing=True,
    use_cache=True,
    fast_mode=True,
    precision="float32",
)

long_df, short_df = scan_all_symbols(
    data_fetcher=data_fetcher,
    atc_config=fast_config,
    min_signal=0.05,             # Lower threshold for filtering
    max_symbols=None,
    execution_mode="dask",
)

print(f"Stage 1 complete: {len(long_df)} longs, {len(short_df)} shorts")

# ===== STAGE 2: Full Precision on Candidates =====
print("\nStage 2: Full precision on top candidates...")

full_config = ATCConfig(
    timeframe="15m",
    limit=1500,
    ema_len=28, hma_len=28, wma_len=28,
    dema_len=28, lsma_len=28, kama_len=28,
    robustness="Medium",
    La=0.02, De=0.03,
    use_approximate=False,       # Full precision
    batch_processing=True,
    use_cache=True,
    fast_mode=True,
    precision="float32",
)

# Process top 100 candidates with full precision
top_candidates = pd.concat([
    long_df.head(50),
    short_df.head(50)
])["symbol"].tolist()

final_signals = {}
for symbol in top_candidates:
    try:
        df, _ = data_fetcher.fetch_ohlcv_with_fallback_exchange(
            symbol, timeframe="15m", limit=1500
        )
        
        results = compute_atc_signals(
            prices=df["close"],
            **full_config.__dict__
        )
        
        final_signals[symbol] = results["Average_Signal"].iloc[-1]
    except Exception as e:
        print(f"Error processing {symbol}: {e}")

# Create final results
final_df = pd.DataFrame([
    {"symbol": s, "signal": sig, "direction": "LONG" if sig > 0 else "SHORT"}
    for s, sig in final_signals.items()
]).sort_values("signal", ascending=False)

# Filter by final threshold
final_longs = final_df[final_df["signal"] > 0.1]
final_shorts = final_df[final_df["signal"] < -0.1]

print(f"\nFinal results:")
print(f"  Confirmed LONGs: {len(final_longs)}")
print(f"  Confirmed SHORTs: {len(final_shorts)}")

print(f"\nTop 20 final LONG signals:\n{final_longs.head(20)}")
print(f"\nTop 20 final SHORT signals:\n{final_shorts.head(20)}")

# Save final results
final_longs.to_csv("final_longs.csv", index=False)
final_shorts.to_csv("final_shorts.csv", index=False)
```

**Workflow**:

1. **Stage 1**: Fast approximate scan to filter candidates (use_approximate=True)
2. **Stage 2**: Full precision calculation for filtered candidates (use_approximate=False)

**Expected Performance**: 10-20x speedup for large-scale scanning

---

## 🔧 Integration Examples

### Example 1: Python Script Integration

```python
from modules.adaptive_trend_LTS_mini.core.compute_atc_signals import compute_atc_signals
from modules.adaptive_trend_LTS_mini.core.compute_atc_signals.incremental_atc import IncrementalATC
import pandas as pd

# For live trading (single bar updates)
atc = IncrementalATC(config={
    'ema_len': 28, 'hma_len': 28, 'wma_len': 28,
    'dema_len': 28, 'lsma_len': 28, 'kama_len': 28,
    'robustness': 'Medium',
    'La': 0.02, 'De': 0.03,
})
atc.initialize(historical_prices)
new_signal = atc.update(new_price)  # O(1) operation

# For batch scanning (100+ symbols)
from modules.adaptive_trend_LTS_mini.core.compute_atc_signals.batch_processor import process_symbols_batch_rust

symbols_data = {'BTCUSDT': prices_series, 'ETHUSDT': prices_series, ...}
config = {
    'ema_len': 28, 'hma_len': 28, 'wma_len': 28,
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
  hma_len: 28
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

## 💡 Advanced Usage Examples

Detailed examples for O(1) MA, Rust incremental backend, multi-timeframe (MTF), batch update, and state serialization.

### Example 1: Incremental ATC with O(1) MA and Rust (default)

For **live trading** with real-time single-bar updates:

```python
from modules.adaptive_trend_LTS_mini.core.compute_atc_signals import IncrementalATC
import pandas as pd

config = {
    "ema_len": 28,
    "hma_len": 28,
    "wma_len": 28,
    "dema_len": 28,
    "lsma_len": 28,
    "kama_len": 28,
    "use_o1_mas": True,   # O(1) WMA/HMA/LSMA/KAMA (default True)
    "use_rust_incremental": True,  # Rust backend when available (default True)
}

# Initialize once with historical data
atc = IncrementalATC(config)
prices = pd.Series([100.0, 101.0, 102.0, 103.0, 104.0] + [104.0 + i * 0.5 for i in range(100)])
atc.initialize(prices)

# Update incrementally with new bar (O(1) operation)
signal = atc.update(110.0)
print(f"New signal: {signal}")
```

**Performance**: <0.01s per update (1000x+ faster than full recalculation)

---

### Example 2: Legacy mode (disable O(1) and Rust)

For **compatibility testing** or **debugging**:

```python
config_legacy = {
    **config,
    "use_o1_mas": False,
    "use_rust_incremental": False,
}
atc_legacy = IncrementalATC(config_legacy)
atc_legacy.initialize(prices)
signal_legacy = atc_legacy.update(110.0)
```

---

### Example 3: Batch update (multiple bars at once)

For **catching up** after reconnection or **processing historical gaps**:

```python
atc = IncrementalATC(config)
atc.initialize(prices)

# Process multiple new bars at once
new_prices = [110.0, 111.0, 112.0, 113.0, 114.0]
signals = atc.batch_update(new_prices)  # list[float], one signal per price
assert len(signals) == len(new_prices)

for i, signal in enumerate(signals):
    print(f"Bar {i+1}: {signal}")
```

**Performance**: Still significantly faster than full recalculation per bar

---

### Example 4: Multi-Timeframe (MTF)

For **simultaneous analysis** across multiple timeframes:

```python
from modules.adaptive_trend_LTS_mini.core.compute_atc_signals import MultiTimeframeIncrementalATC

mtf = MultiTimeframeIncrementalATC(config, timeframes=["1m", "5m", "15m"])

# Initialize: dict per TF or single series for base TF
historical_1m = prices  # pd.Series
mtf.initialize({"1m": historical_1m})  # or provide 5m, 15m if available

# Call update on each 1m bar; 5m/15m advance when their bar completes
signals = mtf.update(110.0, timeframe="1m")  # dict {"1m": float, "5m": float, "15m": float}

print(f"1m signal: {signals['1m']}")
print(f"5m signal: {signals['5m']}")
print(f"15m signal: {signals['15m']}")
```

**Use Case**: Multi-timeframe confirmation strategies, alignment indicators

---

### Example 5: State serialization (zero-warmup restart)

For **production systems** requiring fast restart without re-initialization:

```python
from pathlib import Path

# Initialize and run for some time
atc = IncrementalATC(config)
atc.initialize(prices)
atc.update(108.0)
atc.update(109.0)

# Save state to disk
path = Path("states/BTCUSDT_1h.msgpack")
path.parent.mkdir(parents=True, exist_ok=True)
atc.save_state(path)

# --- System restart ---

# Load and continue from saved state (no initialization needed)
atc2 = IncrementalATC.load_state(path)
next_signal = atc2.update(110.0)  # no need to initialize again
print(f"Signal after restart: {next_signal}")
```

**Benefits**:
- **Zero warmup time**: No need to process historical data
- **Instant recovery**: Resume from exact state before shutdown
- **Efficient storage**: Compact binary format (msgpack)

---

### Example 6: Two-Stage Scanning with Approximate Filter

For **large-scale scanning** (1000+ symbols) with filtering:

```python
from legacy.adaptive_trend_enhance.utils.config import ATCConfig
from modules.adaptive_trend_LTS_mini.core.scanner.scan_all_symbols import scan_all_symbols
from modules.common.core.data_fetcher import DataFetcher

data_fetcher = DataFetcher()

# Stage 1: Fast approximate scan (1000+ symbols)
fast_config = ATCConfig(
    timeframe="15m",
    use_approximate=True,  # 2-3x faster
)

long_df, short_df = scan_all_symbols(
    data_fetcher=data_fetcher,
    atc_config=fast_config,
    min_signal=0.05,  # Lower threshold for initial filter
    max_symbols=None,  # Scan all symbols
    execution_mode="threadpool",
)

# Stage 2: Full precision on filtered candidates
full_config = ATCConfig(
    timeframe="15m",
    use_approximate=False,  # Full precision
)

# Re-analyze top 50 candidates with full precision
top_candidates = long_df.head(50)["symbol"].tolist()

for symbol in top_candidates:
    df, _ = data_fetcher.fetch_ohlcv_with_fallback_exchange(
        symbol, timeframe="15m", limit=1500
    )
    prices = df["close"]
    
    from modules.adaptive_trend_LTS_mini.core.compute_atc_signals import compute_atc_signals
    
    results = compute_atc_signals(
        prices=prices,
        use_approximate=False,  # Full precision
    )
    
    signal = results["Average_Signal"].iloc[-1]
    print(f"{symbol}: {signal:.4f}")
```

**Performance**: 5-10x faster than full precision for all symbols

---

### Example 7: Running tests and benchmarks

To **verify** your setup and **measure** performance:

```bash
# Incremental & advanced-feature tests
pytest modules/adaptive_trend_LTS_mini/tests/test_incremental_atc_o1.py -v
pytest modules/adaptive_trend_LTS_mini/tests/test_incremental_rust.py -v
pytest modules/adaptive_trend_LTS_mini/tests/test_incremental_mtf.py -v
pytest modules/adaptive_trend_LTS_mini/tests/test_incremental_batch.py -v
pytest modules/adaptive_trend_LTS_mini/tests/test_incremental_serialization.py -v

# Benchmarks
python -m modules.adaptive_trend_LTS_mini.benchmarks.benchmark_incremental_o1 --iterations 1000
python -m modules.adaptive_trend_LTS_mini.benchmarks.benchmark_incremental_rust
python -m modules.adaptive_trend_LTS_mini.benchmarks.benchmark_incremental_batch
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

- **Full Settings Guide**: `modules/adaptive_trend_LTS_mini/docs/setting_guides.md`
- **Features Summary**: `modules/adaptive_trend_LTS_mini/docs/features_summary.md`
- **Phase Documentation**:
  - Phase 3 (Rust): `docs/phase3_task.md`
  - Phase 4 (CUDA): `docs/phase4_task.md`
  - Phase 5 (Dask): `docs/phase5_task.md`
  - Phase 6 (Incremental/Approximate): `docs/phase6_task.md`
  - Phase 7 (Memory): `docs/phase7_task.md`
  - Phase 8 (Profiling): `docs/phase8_task.md`
  - Phase 8.1 (Cache & Parallelism): `docs/phase8.1_task.md`
  - Phase 8.2 (JIT Specialization): `docs/phase8.2_task.md`
  - Phase 9 (Advanced Usage - O(1) MA, Rust, MTF, batch, serialization): `docs/phase9_task.md`

---

**Last Updated**: 2026-01-29  
**Version**: LTS (Long-Term Support)  
**Status**: ✅ Production Ready
