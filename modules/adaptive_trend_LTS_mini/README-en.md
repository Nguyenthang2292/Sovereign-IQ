# Adaptive Trend Classification LTS (ATC LTS)

**Long-term support version with Rust-accelerated backend, GPU/CPU optimization, and automatic memory management**

> **Language / Ngôn ngữ**: [English](README-en.md) | [Tiếng Việt](README-vi.md)

Adaptive Trend Classification LTS is the stable version of ATC with:

- **Rust backend**: Equity, KAMA, MAs (EMA/WMA/DEMA/LSMA/HMA), signal persistence run on Rust when built; fallback to Numba if not built.
- **Parallel computing**: Multi-processing + multi-threading with auto-detection CPU/RAM
- **GPU acceleration**: Automatic detection and use of GPU (CUDA/OpenCL) if available
- **Memory management**: Automatic cleanup, monitoring and prevention of memory leaks
- **Numba JIT**: Fallback for MA calculations when Rust is not available
- **Caching**: Intelligent caching for MA results
- **Memory Optimizations**: Memory-mapped arrays for backtesting and blosc compression for cache
- **NumPy optimization**: Pre-allocated arrays and NumPy operations instead of Pandas

The module provides an adaptive trend analysis system using multiple Moving Averages with adaptive weighting based on equity curves.

## Table of Contents

- [Overview](#overview)
- [Module Structure](#module-structure)
- [CPU-Only Mini Version](#cpu-only-mini-version)
- [How It Works](#how-it-works)
- [Installation](#installation)
- [Usage](#usage)
- [Configuration](#configuration)
- [Results](#results)
- [Signal Interpretation](#signal-interpretation)
- [Utilities](#utilities)
- [Performance](#performance)
- [Memory Optimizations](#memory-optimizations)
- [Important Notes](#important-notes)
- [CLI Commands](#cli-commands)
- [Advanced Examples](#advanced-examples)
- [Troubleshooting](#troubleshooting)
- [References](#references)
- [Changelog](#changelog)

## Overview

ATC is an adaptive trend classification system using:

- **6 types of Moving Averages**: EMA, HMA, WMA, DEMA, LSMA, KAMA
- **2-layer architecture**:
  - Layer 1: Calculate signals for each MA type based on equity curves
  - Layer 2: Calculate weights and combine all to create Average_Signal
- **Adaptive weighting**: Uses equity curves to automatically adjust the weight of each MA
- **Robustness modes**: "Narrow", "Medium", "Wide" to adjust sensitivity

## Module Structure

```text
adaptive_trend_LTS_mini/
├── __init__.py              # Module exports
├── README.md                # Language selector
├── README-en.md             # This documentation (English)
├── README-vi.md             # Vietnamese documentation
├── core/
│   ├── rust_backend.py      # Rust extension wrapper (equity, KAMA, MAs, persistence)
│   ├── compute_atc_signals/ # ATC signals (Rust-accelerated when built)
│   ├── compute_moving_averages/  # MA with Rust or Numba fallback
│   ├── compute_equity/      # Equity curves
│   ├── process_layer1/      # Layer 1 processing
│   ├── signal_detection/    # Signal detection
│   ├── scanner/             # Multi-symbol scanning
│   └── ...
├── rust_extensions/         # Rust crate (PyO3); see rust_extensions/README.md
├── cli/                     # CLI (argument_parser, display, main, ...)
├── docs/                    # Detailed documentation (setting_guides, phase tasks, ...)
└── utils/                   # config, cache_manager, rate_of_change, ...
```

**Documentation:** See full parameters, presets and troubleshooting: [docs/setting_guides.md](docs/setting_guides.md).

## CPU-Only Mini Version

This is a **CPU-only mini version** of the Adaptive Trend LTS module with all CUDA/GPU code removed. It maintains full functionality using the Rust/Rayon CPU backend for excellent multi-core performance.

### ✅ Available Features (CPU-Only)
- All ATC signal calculations (Layer 1 & Layer 2)
- 6 MA types: EMA, HMA, WMA, DEMA, LSMA, KAMA
- Multi-threaded CPU processing via Rust/Rayon
- Dask batch processing support
- Approximate scanning for faster exploration
- Full CLI interface
- Incremental ATC updates

### ❌ Not Available (Removed)
- GPU/CUDA acceleration
- CuPy/PyCUDA dependencies
- CUDA kernel compilation

### Performance

| Operation | CPU Version (Rayon) |
|-----------|---------------------|
| Single symbol (1000 bars) | ~100-500ms |
| 10 symbols batch | ~1-5s |
| 100 symbols batch | ~10-50s |
| 1000 symbols batch | ~100-500s |

**Scalability**: Linear scaling with CPU cores via Rayon parallelism

### When to Use CPU-Only Version

✅ **Use when:**
- No NVIDIA GPU available
- Cloud environments without GPU instances
- Development/testing on laptops
- Lower memory footprint required
- Production deployment on CPU-only servers

❌ **Don't use when:**
- Real-time analysis of 1000+ symbols required
- Ultra-low latency needs (<50ms)
- High-frequency trading requirements

## How It Works

### Layer 1: Individual MA Signals

For each MA type (EMA, HMA, WMA, DEMA, LSMA, KAMA):

1. Calculate 9 MAs with different lengths (base length ± offsets based on robustness)
2. Calculate signals for each MA based on price/MA crossovers
3. Calculate equity curves for each signal using exponential growth
4. Weighted average of 9 signals based on equity curves → Layer 1 signal for that MA type

### Layer 2: Combined Signal

1. Calculate weights for each MA type based on Layer 1 signals
2. Weighted average of all Layer 1 signals → **Average_Signal** (final output)

### Equity Curves

Equity curves simulate the performance of trading strategies:

- Uses exponential growth factor (La) and decay rate (De)
- Higher equity → higher weight → that MA has greater influence
- Adaptive: Automatically adjusts weights based on performance

## Installation

### Prerequisites
- Python 3.9+
- Rust toolchain (for building extensions)
- No CUDA/GPU required

### Build Rust Extension

```bash
cd modules/adaptive_trend_LTS_mini/rust_extensions
cargo build --release
```

Or from project root: `.\build_rust.bat` (Windows) / `.\build_rust.ps1`.

**Requirements:** [Rust](https://rustup.rs/), [maturin](https://www.maturin.rs/) (`pip install maturin`). Details and troubleshooting: [docs/phase3_task.md#prerequisites--setup](docs/phase3_task.md#prerequisites--setup).

### Install Python Dependencies

```bash
pip install pandas numpy dask
```

## Usage

### Basic Analysis

```bash
python -m modules.adaptive_trend_LTS_mini.cli.main --symbol BTC/USDT --timeframe 1h
```

### Batch Scanning

```bash
python -m modules.adaptive_trend_LTS_mini.cli.main --scan --top 100
```

### Python API

Examples below use `legacy.adaptive_trend_enhance`; can be replaced with `modules.adaptive_trend_LTS_mini` (same API, uses Rust backend when built).

```python
import pandas as pd
from legacy.adaptive_trend_enhance import compute_atc_signals, ATCConfig

# Prepare data
prices = pd.Series([...])  # Close prices

# Configuration
config = ATCConfig(
    ema_len=28,
    hma_len=28,
    wma_len=28,
    dema_len=28,
    lsma_len=28,
    kama_len=28,
    robustness="Medium",  # "Narrow", "Medium", "Wide"
    lambda_param=0.02,     # Growth rate
    decay=0.03,            # Decay rate
    cutout=0,              # Bars to skip
)

# Calculate ATC signals
results = compute_atc_signals(
    prices=prices,
    ema_len=config.ema_len,
    hma_len=config.hma_len,
    wma_len=config.wma_len,
    dema_len=config.dema_len,
    lsma_len=config.lsma_len,
    kama_len=config.kama_len,
    robustness=config.robustness,
    La=config.lambda_param,
    De=config.decay,
    cutout=config.cutout,
)

# Results
average_signal = results["Average_Signal"]  # Final combined signal
ema_signal = results["EMA_Signal"]         # Layer 1: EMA signal
hma_signal = results["HMA_Signal"]         # Layer 1: HMA signal
# ... other signals
```

### Analyze Single Symbol

```python
from legacy.adaptive_trend_enhance import analyze_symbol, ATCConfig
from modules.common.core.data_fetcher import DataFetcher
from modules.common.core.exchange_manager import ExchangeManager

# Initialize
exchange_manager = ExchangeManager()
data_fetcher = DataFetcher(exchange_manager)

# Configuration
config = ATCConfig(
    timeframe="15m",
    limit=1500,
    ema_len=28,
    # ... other parameters
)

# Analyze
result = analyze_symbol(
    symbol="BTC/USDT",
    data_fetcher=data_fetcher,
    config=config,
)

if result:
    print(f"Symbol: {result['symbol']}")
    print(f"Current Price: {result['current_price']}")
    print(f"ATC Results: {result['atc_results']}")
```

### Scan Multiple Symbols

```python
from legacy.adaptive_trend_enhance import scan_all_symbols, ATCConfig
from modules.common.core.data_fetcher import DataFetcher
from modules.common.core.exchange_manager import ExchangeManager

# Initialize
exchange_manager = ExchangeManager()
data_fetcher = DataFetcher(exchange_manager)

# Configuration
config = ATCConfig(
    timeframe="15m",
    limit=1500,
    # ... other parameters
)

# Scan
results, short_signals = scan_all_symbols(
    data_fetcher=data_fetcher,
    atc_config=config,
    min_signal=0.5,  # Minimum signal strength
)

# Results
for _, result in results.iterrows():
    print(f"{result['symbol']}: Signal = {result['signal']}")
```

### Using CLI

```bash
# Analyze single symbol
python -m legacy.adaptive_trend_enhance.cli.main BTC/USDT

# Scan all futures symbols
python -m legacy.adaptive_trend_enhance.cli.main --auto

# Interactive mode
python -m legacy.adaptive_trend_enhance.cli.main

# Custom timeframe
python -m legacy.adaptive_trend_enhance.cli.main BTC/USDT --timeframe 1h
```

## Configuration

### ATCConfig

```python
@dataclass
class ATCConfig:
    # Moving Average lengths
    ema_len: int = 28
    hma_len: int = 28
    wma_len: int = 28
    dema_len: int = 28
    lsma_len: int = 28
    kama_len: int = 28

    # ATC parameters
    robustness: str = "Medium"  # "Narrow", "Medium", or "Wide"
    lambda_param: float = 0.02  # Growth rate for equity
    decay: float = 0.03         # Decay rate for equity
    cutout: int = 0            # Bars to skip at beginning

    # Data parameters
    limit: int = 1500          # Number of candles to fetch
    timeframe: str = "15m"     # Timeframe
```

### Robustness Modes

- **Narrow**: Small offsets → less variation in MA lengths → more sensitive
- **Medium**: Medium offsets → balanced
- **Wide**: Large offsets → more variation → more stable, less sensitive

## Results

`compute_atc_signals` returns a dictionary containing:

- **Average_Signal**: Final signal (combination of all MAs)
- **EMA_Signal**, **HMA_Signal**, **WMA_Signal**, **DEMA_Signal**, **LSMA_Signal**, **KAMA_Signal**: Layer 1 signals for each MA type
- **EMA_Weight**, **HMA_Weight**, **WMA_Weight**, **DEMA_Weight**, **LSMA_Weight**, **KAMA_Weight**: Weights for each MA type
- **EMA_Equity**, **HMA_Equity**, ...: Equity curves for each MA type

All are `pd.Series` with the same index as input prices.

## Signal Interpretation

- **Positive values (> 0)**: Bullish signal, price above MA
- **Negative values (< 0)**: Bearish signal, price below MA
- **Zero (0)**: Neutral, no clear signal
- **Magnitude**: Signal strength (higher = stronger)

## Utilities

### rate_of_change

Calculate rate of change of a series:

```python
from legacy.adaptive_trend_enhance.utils import rate_of_change

roc = rate_of_change(prices, period=1)
```

### diflen

Calculate difference length based on robustness mode:

```python
from legacy.adaptive_trend_enhance.utils import diflen

offset = diflen(robustness="Medium")  # Returns offset value
```

### exp_growth

Calculate exponential growth factor:

```python
from legacy.adaptive_trend_enhance.utils import exp_growth

growth = exp_growth(La=0.02, period=1)
```

## Performance

### Rust Backend

**Rust backend** is used by default when built (see [Installation](#installation) above). Equity, KAMA, MAs (EMA/WMA/DEMA/LSMA/HMA), signal persistence functions run on Rust; fallback to Numba if not built.

**Benchmarks (10k bars, `cargo bench` in `rust_extensions/`):**

| Component         | Time (µs) | Note        |
|-------------------|-----------|-------------|
| Equity            | ~32       | 2–3x+ vs Numba |
| KAMA              | ~164      | 2–3x+ vs Numba |
| Signal persistence| ~8.5      | ~5x vs Numba   |
| EMA / DEMA        | ~14 / ~31 | MA Rust        |
| WMA / LSMA / HMA  | ~131 / ~194 / ~232 | MA Rust   |

- **Numba JIT**: Fallback when Rust is not available; equity and MA compile with Numba.
- **Vectorized operations**: NumPy for final calculations.
- **Caching**: Rate of change is cached.
- **Parallel scanning**: Scanner supports parallel processing for multiple symbols.

### CPU-Only Configuration

All configuration is CPU-only:

```python
from modules.adaptive_trend_LTS_mini.utils.config import ATCConfig

config = ATCConfig(
    use_rust_backend=True,  # Use Rust/Rayon CPU backend
    use_approximate=True,   # Enable approximate for faster scanning
    parallel_l1=True,       # Parallel Layer 1 calculations
    parallel_l2=True,       # Parallel Layer 2 calculations
)
```

### CPU-Only Performance Tips

1. **Use Rust backend**: Always set `use_rust_backend=True`
2. **Enable approximate mode**: Set `use_approximate=True` for scanning
3. **Optimize batch size**: 50-200 symbols per batch (tune for your CPU)
4. **Use all CPU cores**: Rayon automatically uses available cores
5. **Cache data**: Reuse fetched OHLCV data where possible

## Memory Optimizations

The module supports memory optimizations for large datasets:

1.  **Memory-Mapped Arrays**:
    - Handle large datasets without loading everything into RAM
    - Reduce 90%+ RAM usage for backtesting
    - Enable via `use_memory_mapped=True` in `ATCConfig`

2.  **Data Compression**:
    - Compress cache files using `blosc`
    - Reduce 5-10x storage footprint
    - Enable via `use_compression=True` in `ATCConfig`

See details: [docs/memory_optimizations_usage_guide.md](docs/memory_optimizations_usage_guide.md)

## Important Notes

1. **Data quality**: ATC requires high-quality OHLCV data. Ensure data has no large gaps.

2. **Timeframe**: ATC works well across multiple timeframes, but parameters may need adjustment:
   - Short timeframe (1m, 5m): May need to reduce lengths
   - Long timeframe (4h, 1d): May need to increase lengths

3. **Robustness**:
   - "Narrow" for strong trending markets
   - "Medium" for balanced markets
   - "Wide" for volatile markets

4. **Lambda and Decay**:
   - High Lambda → equity increases quickly → weights change quickly
   - High Decay → equity decreases quickly → weights decrease quickly

5. **Cutout**: Skip some initial bars to avoid initialization artifacts.

## CLI Commands

The module provides CLI interface via `legacy/adaptive_trend_enhance/cli/main.py`:

```bash
# Basic usage
python -m legacy.adaptive_trend_enhance.cli.main <SYMBOL>

# Options
--timeframe TIMEFRAME    # Set timeframe (default: 15m)
--auto                   # Auto mode (scan all futures symbols)
--min-signal FLOAT       # Minimum signal strength for scan
--no-menu                # Skip interactive menu
--batch-size INT         # Batch size for memory optimization
```

## Advanced Examples

### Custom configuration from dictionary

```python
from legacy.adaptive_trend_enhance.utils.config import create_atc_config_from_dict

params = {
    "ema_len": 21,
    "hma_len": 21,
    "wma_len": 21,
    "dema_len": 21,
    "lsma_len": 21,
    "kama_len": 21,
    "robustness": "Narrow",
    "lambda_param": 0.03,
    "decay": 0.02,
    "limit": 2000,
}

config = create_atc_config_from_dict(params, timeframe="1h")
```

### Combining with other indicators

```python
from legacy.adaptive_trend_enhance import compute_atc_signals
from modules.common.core.indicator_engine import IndicatorEngine

# Calculate ATC signals
atc_results = compute_atc_signals(prices=df['close'], ...)

# Calculate other indicators
engine = IndicatorEngine()
df_with_indicators, metadata = engine.compute(df)

# Combine signals
combined_signal = (
    atc_results['Average_Signal'] * 0.6 +
    (df_with_indicators['RSI_14'] - 50) / 50 * 0.4
)
```

## Troubleshooting

### Issue: Rust not detected / `rustc` not in PATH

**Solution**: Add `%USERPROFILE%\.cargo\bin` to PATH, or run `.\build_rust.bat` / `.\build_rust.ps1` (auto-adds PATH). Details: [docs/phase3_task.md#troubleshooting](docs/phase3_task.md#troubleshooting).

### Issue: Maturin build errors

**Solution**: Check `rustc --version`, `python --version`; activate venv before building.

### Issue: `atc_rust` import errors

**Solution**: Run `maturin develop --release` in `rust_extensions/`; verify with `pip show atc-rust`.

### Issue: Numba cache after module rename

**Solution**: Delete `__pycache__` containing `*.nbc` / `*.nbi` in `core/signal_detection/` if encountering `ModuleNotFoundError` with old module path.

### Issue: Slow Performance

**Solution:**
1. Verify Rust backend is enabled: `use_rust_backend=True`
2. Build in release mode: `cargo build --release`
3. Enable parallel processing: `parallel_l1=True, parallel_l2=True`
4. Check CPU core utilization during processing

### Issue: Build Errors

**Solution:**
```bash
cd rust_extensions
cargo clean
cargo build --release
```

### Issue: Import Errors

**Solution:**
Ensure the Rust extension is built:
```bash
ls rust_extensions/target/release/*.dll  # Windows
ls rust_extensions/target/release/*.so   # Linux/Mac
```

## References

- Ported from Pine Script indicator "Adaptive Trend Classification"
- Uses multiple Moving Averages with adaptive weighting
- Equity-based weighting to automatically adjust weights

---

## Changelog - CPU-Only Mini Version

### Version 1.0.0 (2026-01-31)

#### Overview
Complete migration from GPU/CUDA dependencies to CPU-only implementation using Rust/Rayon backend.

#### Removed

##### CUDA/GPU Dependencies
- All CUDA-related Rust dependencies
- CuPy and PyCUDA Python packages
- All `.cu` CUDA kernel files
- GPU backend Python modules
- GPU test and benchmark files

#### Modified

##### API Changes
- Removed `use_cuda` parameter from all functions
- Removed `prefer_gpu` parameter from `compute_atc_signals()`
- Removed `--use-cuda` CLI flag
- Removed `use_cuda` field from `ATCConfig` dataclass

##### Build Changes
- Rust build size reduced: ~1.5MB → ~637KB (57% smaller)
- Simplified routing logic (CPU-only)

#### Kept

##### Core Functionality (100% preserved)
- All ATC signal calculation algorithms
- All 6 MA types (EMA, HMA, WMA, DEMA, LSMA, KAMA)
- Layer 1 & Layer 2 calculation logic
- Rust/Rayon CPU backend (multi-threaded)
- Dask batch processing support
- Approximate scanning mode
- Full CLI interface
- Incremental ATC updates
- Signal persistence calculations

#### Performance Comparison

| Metric | GPU Version | CPU-Only Version | Factor |
|--------|-------------|------------------|--------|
| Single symbol | 10-50ms | 100-500ms | ~10x slower |
| Batch 100 symbols | 2-10s | 10-50s | ~5x slower |
| Memory usage | GPU VRAM + RAM | RAM only | More efficient |
| CPU cores used | 1-2 | All available | Better utilization |

#### Benefits of CPU-Only Version

1. **No Hardware Dependencies**: Works on any CPU, no NVIDIA GPU required
2. **Lower Memory Footprint**: No GPU memory overhead
3. **Cloud-Friendly**: Runs on any cloud instance without GPU
4. **Development-Easy**: Test on laptops without discrete GPU
5. **Smaller Build**: 57% smaller binary size
6. **Simpler Deployment**: No CUDA runtime dependencies

#### Validation Results
- All Rust extension builds successfully without CUDA
- All imports work without GPU dependencies
- No cupy/pycuda imports in codebase
- All non-GPU tests pass
- CPU-only validation tests pass

**Migration Date**: 2026-01-31
**Version**: 1.0.0
**Status**: Production Ready
