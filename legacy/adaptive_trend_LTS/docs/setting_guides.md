# 📋 Settings Reference — Adaptive Trend LTS Module

**Version**: LTS (Long-Term Support)
**Last Updated**: 2026-01-29
**Status**: ✅ All features complete
**Backend**: Rust v2 + CUDA (optional) + Dask (optional)

## 🎯 Overview

The **Adaptive Trend Classification LTS** module is a stable build with Rust backend, GPU acceleration, and automatic memory management.

## 📑 Quick Navigation

- [Core Parameters](#-core-parameters)
  - [Moving Average Lengths](#1-moving-average-lengths)
  - [MA Weights](#2-ma-weights)
  - [ATC Core Parameters](#3-atc-core-parameters)
  - [Signal Thresholds](#4-signal-thresholds)
  - [Data & Processing](#5-data--processing-parameters)
  - [Performance & Optimization](#6-performance--optimization)
  - [Strategy Mode](#7-strategy-mode)
- [Output Results](#-output-results)
- [Recommended Presets](#-recommended-presets)
  - [Scalping (1m-5m)](#1-scalping-timeframe-1m---5m)
  - [Intraday Trading (15m-1h)](#2-intraday-trading-timeframe-15m---1h--default)
  - [Swing Trading (4h-1d)](#3-swing-trading-timeframe-4h---1d)
  - [High-Performance](#4-high-performance-rust--multi-symbol)
  - [Out-of-Core Processing (Dask)](#5-out-of-core-processing-dask-integration--new)
  - [True Batch Processing](#6-true-batch-processing-best-for-100-symbols)
  - [Incremental Updates](#7-incremental-updates-for-live-trading--new)
  - [Approximate MAs](#8-approximate-mas-for-fast-filtering--new)
  - [Advanced Usage Examples](#-advanced-usage-examples) *(O(1) MA, Rust, MTF, batch, save/load)*
- [Performance Comparison](#-performance-comparison)
- [Setup & Build](#-setup--build)
- [Example Usage](#-example-usage)
- [Troubleshooting](#-troubleshooting)
- [Best Practices](#-best-practices)

---

## ⚙️ Core Parameters

### 1. **Moving Average Lengths**

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `ema_len` | int | 28 | EMA period (Exponential Moving Average) |
| `hma_len` | int | 28 | HMA period (Hull Moving Average) |
| `wma_len` | int | 28 | WMA period (Weighted Moving Average) |
| `dema_len` | int | 28 | DEMA period (Double Exponential MA) |
| `lsma_len` | int | 28 | LSMA period (Least Squares MA) |
| `kama_len` | int | 28 | KAMA period (Kaufman Adaptive MA) |

**Note**:

- Lower values (10–20): More responsive; suited to shorter timeframes
- Higher values (30–50): Smoother; suited to longer timeframes

---

### 2. **MA Weights**

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `ema_w` | float | 1.0 | Initial weight for EMA |
| `hma_w` | float | 1.0 | Initial weight for HMA |
| `wma_w` | float | 1.0 | Initial weight for WMA |
| `dema_w` | float | 1.0 | Initial weight for DEMA |
| `lsma_w` | float | 1.0 | Initial weight for LSMA |
| `kama_w` | float | 1.0 | Initial weight for KAMA |

**Note**: Weights are adjusted automatically from equity curves.

---

### 3. **ATC Core Parameters**

| Parameter | Type | Default | Range | Description |
|-----------|------|---------|-------|-------------|
| `robustness` | str | "Medium" | "Narrow", "Medium", "Wide" | Signal sensitivity |
| `La` | float | 0.02 | 0.01-0.05 | Lambda — equity growth rate |
| `De` | float | 0.03 | 0.01-0.10 | Decay — equity decay rate |
| `cutout` | int | 0 | 0-100 | Number of bars to skip at start |

**Robustness Modes**:

- **"Narrow"**:
  - Small offset (length ± 1–3 steps)
  - More sensitive to price changes
  - Suited to: Trending markets
  
- **"Medium"** ✅ **RECOMMENDED**:
  - Medium offset (length ± 4 steps)
  - Balance of sensitivity and stability
  - Suited to: Most market conditions
  
- **"Wide"**:
  - Large offset (length ± 9 steps)
  - Stable, less noise
  - Suited to: Volatile/choppy markets

**Lambda & Decay**:

- **Higher La** (0.03–0.05): Equity rises faster → weights change faster
- **Lower La** (0.01–0.02): Equity rises slowly → more stable weights
- **Higher De** (0.05–0.10): Equity drops faster when wrong → quicker removal of bad MAs
- **Lower De** (0.01–0.03): Equity drops slowly → allows recovery

---

### 4. **Signal Thresholds**

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `long_threshold` | float | 0.1 | Threshold for LONG signal classification |
| `short_threshold` | float | -0.1 | Threshold for SHORT signal classification |

**Signal Classification**:

- Signal > `long_threshold` → **LONG** (1.0)
- Signal < `short_threshold` → **SHORT** (-1.0)
- Otherwise → **NEUTRAL** (0.0)

---

### 5. **Data & Processing Parameters**

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `prices` | pd.Series | **Required** | Price data (close prices) |
| `src` | pd.Series | None | Custom source (optional, defaults to prices) |
| `limit` | int | 1500 | Number of bars to fetch |
| `timeframe` | str | "15m" | Timeframe (1m, 5m, 15m, 1h, 4h, 1d...) |

---

### 6. **Performance & Optimization**

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `use_cuda` | bool | False | Use CUDA batch processing |
| `batch_processing` | bool | True | Use Rayon multi-threaded CPU batch |
| `parallel_l1` | bool | None | Parallel processing Layer 1 (auto-detect) |
| `parallel_l2` | bool | True | Parallel processing Layer 2 |
| `prefer_gpu` | bool | True | Prefer GPU when available |
| `use_cache` | bool | True | Cache MA results |
| `fast_mode` | bool | True | Optimization mode |
| `precision` | str | "float64" | "float32" or "float64" |

**Backend Priority**:

1. **Rust (Rayon Batch)** ⭐ **EXTREME SPEED** - Max CPU utilization
2. **Rust (Sequential)** - standard per-symbol execution
3. **CUDA (True Batch)** - GPU acceleration for hundreds of symbols
4. **Numba JIT** (fallback)
5. **Pure Python** (slowest)

---

### 7. **Strategy Mode**

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `strategy_mode` | bool | False | Shift signal 1 bar (for backtesting) |

**Note**: Set to `True` for backtesting to avoid look-ahead bias.

---

## 📊 Output Results

`compute_atc_signals()` returns a **dictionary** containing:

### Layer 1 Signals (per MA type)

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
    'ema_len': 14, 'hma_len': 14, 'wma_len': 14,
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
    'ema_len': 28, 'hma_len': 28, 'wma_len': 28,
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
    'ema_len': 50, 'hma_len': 50, 'wma_len': 50,
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

For very large symbol lists (>1000 symbols) that exceed RAM, or to maximize CPU utilization, use Dask:

- **`use_dask`**: Enable parallel, partitioned (out-of-core) processing.
- **`npartitions`**: Number of data partitions processed in parallel. Default is derived from symbol count.

```python
from modules.adaptive_trend_LTS.core.scanner.scan_all_symbols import scan_all_symbols

longs, shorts = scan_all_symbols(
    data_fetcher,
    atc_config,
    execution_mode="dask",  # Optimized for large datasets
    npartitions=10
)
```

See `docs/phase5_task.md` for detailed Dask integration guide and benchmarks.

### 6. **True Batch Processing** (Best for 100+ symbols)

For many symbols (e.g. Binance Futures), use the batch API instead of a loop:

```python
from modules.adaptive_trend_LTS.core.compute_atc_signals.batch_processor import process_symbols_batch_rust

# symbols_data = {'BTCUSDT': prices_series, 'ETHUSDT': series, ...}
results = process_symbols_batch_rust(symbols_data, config)
```

### 7. **Incremental Updates** (For Live Trading) ⭐ **NEW**

For single-bar updates (live trading), use `IncrementalATC` to avoid full series recalculation:

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

For scanning thousands of symbols, use Approximate MAs for initial filtering, then full precision for candidates.

#### 8.1 When to Use Approximate MAs

**✅ Good Use Cases:**

- **Large-scale scanning** (1000+ symbols): Quickly filter candidates
- **Initial screening**: Eliminate obvious non-candidates before detailed analysis
- **Resource-constrained environments**: Lower CPU usage for scanning
- **Real-time monitoring**: Faster updates when monitoring many symbols

**❌ When NOT to Use:**

- **Final trading decisions**: Use full precision for actual entry/exit signals
- **Backtesting**: Need exact values for accurate historical analysis
- **High-frequency trading**: Precision matters more than speed
- **Single symbol analysis**: Speed difference is negligible

#### 8.2 Configuration Options

**Basic Approximate MAs (Fast Mode)**

Uses simplified calculations (e.g., SMA for EMA approximation) for ~5% tolerance:

```python
from legacy.adaptive_trend_enhance.utils.config import ATCConfig

# Basic approximate MAs - 2-3x faster
config = ATCConfig(
    timeframe="15m",
    limit=1500,
    use_approximate=True,  # Enable basic approximate MAs
)
```

**Adaptive Approximate MAs (Volatility-Aware)**

Dynamically adjusts tolerance based on market volatility:

```python
# Adaptive approximate MAs - volatility-aware tolerance
config = ATCConfig(
    timeframe="15m",
    limit=1500,
    use_adaptive_approximate=True,  # Enable adaptive approximate MAs
    approximate_volatility_window=20,  # Window for volatility calculation
    approximate_volatility_factor=1.0,  # Multiplier for volatility effect
)
```

**How it works:**

- **Low volatility**: Tighter tolerance for better accuracy
- **High volatility**: Looser tolerance for faster computation
- Automatically adapts to market conditions

**Full Precision (Default)**

Standard behavior with exact MA calculations:

```python
# Full precision - default mode
config = ATCConfig(
    timeframe="15m",
    limit=1500,
    # use_approximate=False (default)
    # use_adaptive_approximate=False (default)
)
```

#### 8.3 Usage Examples

**Example 1: Two-Stage Scanning Workflow**

Combine approximate MAs for initial filtering with full precision for final analysis:

```python
from legacy.adaptive_trend_enhance.utils.config import ATCConfig
from modules.adaptive_trend_LTS.core.scanner.scan_all_symbols import scan_all_symbols
from modules.common.core.data_fetcher import DataFetcher

# Initialize data fetcher
data_fetcher = DataFetcher()

# Stage 1: Fast approximate scan (1000+ symbols)
fast_config = ATCConfig(
    timeframe="15m",
    use_approximate=True,  # 2-3x faster
)

long_df, short_df = scan_all_symbols(
    data_fetcher=data_fetcher,
    atc_config=fast_config,
    min_signal=0.05,  # Lower threshold for initial filter (passed to scan_all_symbols)
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
    # Fetch data and run full precision ATC
    df, _ = data_fetcher.fetch_ohlcv_with_fallback_exchange(
        symbol, timeframe="15m", limit=1500
    )
    prices = df["close"]

    from modules.adaptive_trend_LTS.core.compute_atc_signals import compute_atc_signals

    results = compute_atc_signals(
        prices=prices,
        use_approximate=False,  # Full precision
    )

    # Make trading decision based on full precision signals
    signal = results["Average_Signal"].iloc[-1]
    print(f"{symbol}: {signal:.4f}")
```

**Example 2: Using Batch Processor with Approximate Filter**

Two-stage batch processing with approximate filtering, then full precision for candidates:

```python
from modules.adaptive_trend_LTS.core.compute_atc_signals.batch_processor import (
    process_symbols_batch_with_approximate_filter
)

# symbols_data = {'BTCUSDT': prices_series, 'ETHUSDT': prices_series, ...}
# config = dict of parameters (ema_len, hma_len, etc.)

results = process_symbols_batch_with_approximate_filter(
    symbols_data,  # Dict[str, pd.Series]
    config,  # Configuration dict
    approximate_threshold=0.1,  # Not used in current implementation
    min_signal_candidate=0.05,  # Minimum signal to pass filtering stage
)

# Returns: Dict[str, Dict[str, pd.Series]] - Full precision results for candidates only
```

**Note**: This function performs two-stage processing:

1. **Stage 1**: Fast approximate MAs to filter candidates (symbols with signal >= min_signal_candidate)
2. **Stage 2**: Full precision calculation only for filtered candidates

**Performance**: 5-10x faster than full precision for all symbols when filtering reduces candidates significantly.

**Example 3: Adaptive Approximate for Mixed Volatility Markets**

```python
# Adaptive approximate scan - adjusts tolerance based on volatility
adaptive_config = ATCConfig(
    timeframe="15m",
    use_adaptive_approximate=True,
    approximate_volatility_window=20,  # 20-bar volatility window
    approximate_volatility_factor=1.5,  # Increase tolerance in volatile markets
)

long_df, short_df = scan_all_symbols(
    data_fetcher=data_fetcher,
    atc_config=adaptive_config,
    execution_mode="threadpool",
)
```

#### 8.4 Performance Comparison

| Mode | Speed | Accuracy | Use Case |
|------|-------|----------|----------|
| Full Precision | 1x (baseline) | 100% | Final trading decisions, backtesting |
| Basic Approximate | 2-3x faster | ~95% | Initial screening, large-scale scanning |
| Adaptive Approximate | 2-3x faster | ~95% (adaptive) | Mixed volatility markets, smart filtering |

**Performance**: 2-3x faster for large symbol sets (1000+).

#### 8.5 Parameters Reference

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `use_approximate` | bool | False | Enable basic approximate MAs (2-3x faster) |
| `use_adaptive_approximate` | bool | False | Enable adaptive approximate MAs (volatility-aware) |
| `approximate_volatility_window` | int | 20 | Window size for volatility calculation |
| `approximate_volatility_factor` | float | 1.0 | Multiplier for volatility effect on tolerance |

**Note**: `use_approximate` and `use_adaptive_approximate` are mutually exclusive. If both are True, `use_adaptive_approximate` takes precedence.

#### 8.6 Best Practices

1. **Two-Stage Workflow**: Use approximate MAs for initial filtering, then full precision for final decisions
2. **Threshold Tuning**: Lower the `min_signal` threshold slightly when using approximate MAs to avoid missing candidates
3. **Volatility Factor**: Increase `approximate_volatility_factor` (e.g., 1.5-2.0) for crypto markets with high volatility
4. **Testing**: Always backtest your strategy with full precision MAs before live trading
5. **Monitoring**: Compare approximate vs full precision results periodically to ensure acceptable accuracy

#### 8.7 Troubleshooting

**Issue: Signals differ from full precision**

**Expected Behavior**: Approximate MAs have ~5% tolerance, so signals will differ slightly.

**Solution**: This is by design. Use approximate MAs only for initial filtering, not final trading decisions.

**Issue: Adaptive approximate not adapting**

**Check**: Verify that `approximate_volatility_window` is appropriate for your timeframe.

**Solution**: Increase window size for longer timeframes (e.g., 30-50 for 1h+).

**Issue: Performance improvement less than expected**

**Check**: Approximate MAs provide 2-3x speedup for MA calculation, but total speedup depends on other factors (data fetching, network, etc.).

**Solution**: Use with large symbol sets (1000+) for maximum benefit.

#### 8.8 Technical Details

**Basic Approximate MAs** (implemented in `approximate_mas.py`):

- **EMA**: Uses SMA approximation (~5% tolerance)
- **HMA**: Simplified WMA calculations
- **WMA**: Simplified linear weights
- **DEMA**: Double EMA approximation
- **LSMA**: Simplified linear regression (endpoints)
- **KAMA**: Fixed smoothing constant

**Adaptive Approximate MAs** (implemented in `adaptive_approximate_mas.py`):

- Calculates rolling volatility (std dev)
- Adjusts tolerance: `tolerance = base_tolerance * (1 + normalized_volatility * volatility_factor)`
- **Low volatility**: Tighter tolerance, better accuracy
- **High volatility**: Looser tolerance, faster computation

**Backward Compatibility**: The approximate MA feature is fully backward compatible:

- Defaults to full precision when both flags are False
- Existing code continues to work without modification
- Optional feature enabled only when explicitly requested
- No changes to output format or API

See `docs/phase6_task.md` for detailed approximate MA guide and accuracy benchmarks.

---

## 💡 Advanced Usage Examples

Examples for O(1) MA, Rust incremental backend, multi-timeframe (MTF), batch update, and state serialization.

*Section 8 (Approximate MAs) uses 8.1–8.8; this section uses 9.1–9.6 — structure differs by purpose.*

### 9.1 Incremental ATC with O(1) MA and Rust (default)

```python
from modules.adaptive_trend_LTS.core.compute_atc_signals import IncrementalATC
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
atc = IncrementalATC(config)
prices = pd.Series([100.0, 101.0, 102.0, 103.0, 104.0] + [104.0 + i * 0.5 for i in range(100)])
atc.initialize(prices)
signal = atc.update(110.0)
```

### 9.2 Legacy mode (disable O(1) and Rust)

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

### 9.3 Batch update (multiple bars at once)

```python
atc = IncrementalATC(config)
atc.initialize(prices)
new_prices = [110.0, 111.0, 112.0, 113.0, 114.0]
signals = atc.batch_update(new_prices)  # list[float], one signal per price
assert len(signals) == len(new_prices)
```

### 9.4 Multi-Timeframe (MTF)

```python
from modules.adaptive_trend_LTS.core.compute_atc_signals import MultiTimeframeIncrementalATC

mtf = MultiTimeframeIncrementalATC(config, timeframes=["1m", "5m", "15m"])
# Initialize: dict per TF or single series for base TF
historical_1m = prices  # pd.Series
mtf.initialize({"1m": historical_1m})  # or provide 5m, 15m if available

# Call update on each 1m bar; 5m/15m advance when their bar completes
signals = mtf.update(110.0, timeframe="1m")  # dict {"1m": float, "5m": float, "15m": float}
```

### 9.5 State serialization (zero-warmup restart)

```python
from pathlib import Path

atc = IncrementalATC(config)
atc.initialize(prices)
atc.update(108.0)
atc.update(109.0)

path = Path("states/BTCUSDT_1h.msgpack")
path.parent.mkdir(parents=True, exist_ok=True)
atc.save_state(path)

# Restart: load and continue
atc2 = IncrementalATC.load_state(path)
next_signal = atc2.update(110.0)  # no need to initialize again
```

### 9.6 Running tests and benchmarks

```bash
# Incremental & advanced-feature tests
pytest modules/adaptive_trend_LTS/tests/test_incremental_atc_o1.py -v
pytest modules/adaptive_trend_LTS/tests/test_incremental_rust.py -v
pytest modules/adaptive_trend_LTS/tests/test_incremental_mtf.py -v
pytest modules/adaptive_trend_LTS/tests/test_incremental_batch.py -v
pytest modules/adaptive_trend_LTS/tests/test_incremental_serialization.py -v

# Benchmarks
python -m modules.adaptive_trend_LTS.benchmarks.benchmark_incremental_o1 --iterations 1000
python -m modules.adaptive_trend_LTS.benchmarks.benchmark_incremental_rust
python -m modules.adaptive_trend_LTS.benchmarks.benchmark_incremental_batch
```

---

## 🚀 PERFORMANCE COMPARISON

**Benchmark** (99 symbols × 1500 bars):

| Implementation | Time | Speedup | Memory | Use Case |
|----------------|------|---------|--------|----------|
| Original Python | 49.65s | 1.00x | 122.1 MB | Baseline |
| Enhanced Python | 23.85s | 2.08x | 125.8 MB | Optimized Python |
| Rust (Sequential) | 14.15s | 3.51x | 21.0 MB | CPU Sequential |
| Rust (Rayon Parallel) | 8.12s | 6.11x | 18.2 MB | CPU Parallel |
| **Rust + Dask Hybrid** ⭐ | **9.45s** | **5.25x** | **12.5 MB** | **Unlimited size** |
| **CUDA Batch** ⭐ | **0.59s** | **83.53x** | **51.7 MB** | **100+ symbols** |
| **Incremental Update** ⭐ | **<0.01s** | **1000x+** | **<1 MB** | **Live Trading (single bar)** |
| **Approximate Filter** ⭐ | **~5s** | **10x** | **~20 MB** | **Fast Scanning (1000+)** |

**Note**:

- Rust + Dask Hybrid has unlimited dataset size due to out-of-core processing
- CUDA Batch achieves 83.53x speedup for batch processing (100+ symbols)
- Incremental Update is optimal for live trading (single bar updates)
- Approximate Filter is optimal for initial filtering in large-scale scanning

**Recommendation by Use Case**:

| Use Case | Recommended Implementation | Expected Speedup |
|----------|---------------------------|------------------|
| **Live Trading (single bar)** | Incremental Update | 10-100x |
| **Small batch (<100 symbols)** | Rust (Rayon Parallel) | 6x |
| **Medium batch (100-1000)** | CUDA Batch | 80x+ |
| **Large batch (1000-10000)** | Rust + Dask Hybrid | 5-10x + Unlimited size |
| **Very large (10000+)** | Approximate Filter + Dask | 10-20x + Unlimited size |
| **Out-of-Memory scenarios** | Dask Integration | Unlimited size |

**Performance by feature**:

| Feature | Speedup | Status |
|---------|---------|--------|
| Core optimizations | 2.29x | ✅ Complete |
| Advanced memory | 1.5–2x | ✅ Complete |
| Rust extensions | 2–3x | ✅ Complete |
| CUDA kernels | 3–80x | ✅ Complete |
| Dask integration | Unlimited size | ✅ Complete |
| Algorithmic (incremental) | 10–100x | ✅ Complete |
| **All combined** | **Up to 1000x+** | ✅ **Production ready** |

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
    hma_len=28,
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

## 📄 Document Information

**Last Updated**: 2026-01-29
**Version**: LTS (Long-Term Support)
**Backend**: Rust v2 + CUDA (optional) + Dask (optional)

**Feature completion**:

- Core optimizations: ✅ Complete
- Advanced memory: ✅ Complete
- Rust extensions: ✅ Complete
- CUDA kernels: ✅ Complete
- Dask integration: ✅ Complete
- Algorithmic improvements (incremental): ✅ Complete
- Memory optimizations: ✅ Complete
- Profiling-guided optimizations: ✅ Complete
- Cache warming & parallelism: ✅ Complete
- Code generation & JIT specialization: ✅ Complete
- Advanced incremental (O(1) MA, Rust, MTF, batch, save/load): ✅ Complete

**Documentation references**:

- Core & advanced: `docs/phase2_task.md`
- Rust: `docs/phase3_task.md`
- CUDA: `docs/phase4_task.md`
- Dask: `docs/phase5_task.md`
- Incremental / approximate: `docs/phase6_task.md`
- Memory optimizations: `docs/phase7_task.md`
- Profiling: `docs/phase8_task.md`
- Cache & parallelism: `docs/phase8.1_task.md`
- JIT specialization: `docs/phase8.2_task.md`
- Advanced usage (O(1) MA, Rust, MTF, batch, serialization): `docs/phase9_task.md`, `docs/phase9_usage_examples.md`
- Features summary: `docs/features_summary.md`
