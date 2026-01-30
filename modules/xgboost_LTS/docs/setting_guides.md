# 📋 Settings Reference — XGBoost LTS Module

**Version**: LTS (Long-Term Support)
**Last Updated**: 2026-01-30
**Status**: ✅ Core features complete (Phases 1-4, 6)
**Backend**: Rust + Numba + XGBoost GPU (optional) + Dask (planned)

## 🎯 Overview

The **XGBoost LTS** module is a stable build with Rust backend for feature engineering and labeling, GPU acceleration for XGBoost training, and automatic memory management.

## 📑 Quick Navigation

- [Core Parameters](#-core-parameters)
  - [Model Parameters](#1-xgboost-model-parameters)
  - [Labeling Parameters](#2-labeling-parameters)
  - [Feature Engineering](#3-feature-engineering-parameters)
  - [Cross-Validation](#4-cross-validation-parameters)
  - [Data & Processing](#5-data--processing-parameters)
  - [Performance & Optimization](#6-performance--optimization)
- [Output Results](#-output-results)
- [Recommended Presets](#-recommended-presets)
  - [Scalping (1m-5m)](#1-scalping-timeframe-1m---5m)
  - [Intraday Trading (15m-1h)](#2-intraday-trading-timeframe-15m---1h--default)
  - [Swing Trading (4h-1d)](#3-swing-trading-timeframe-4h---1d)
  - [High-Performance](#4-high-performance-rust--multi-symbol)
  - [GPU Acceleration](#5-gpu-acceleration-for-training)
- [Performance Comparison](#-performance-comparison)
- [Setup & Build](#-setup--build)
- [Example Usage](#-example-usage)
- [Troubleshooting](#-troubleshooting)
- [Best Practices](#-best-practices)

---

## ⚙️ Core Parameters

### 1. **XGBoost Model Parameters**

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `max_depth` | int | 6 | Maximum tree depth |
| `learning_rate` | float | 0.01 | Step size shrinkage |
| `n_estimators` | int | 200 | Number of boosting rounds |
| `min_child_weight` | int | 3 | Minimum sum of instance weight |
| `gamma` | float | 0.1 | Minimum loss reduction |
| `subsample` | float | 0.8 | Training data subsample ratio |
| `colsample_bytree` | float | 0.8 | Feature subsample ratio per tree |
| `reg_alpha` | float | 0.5 | L1 regularization |
| `reg_lambda` | float | 2.0 | L2 regularization |
| `tree_method` | str | "hist" | Tree construction algorithm |
| `device` | str | "cuda" (if GPU) | Training device ("cpu" or "cuda") |

**Note**:
- GPU training (`device="cuda"`) requires NVIDIA GPU and CUDA Toolkit
- `tree_method="hist"` is optimized for both CPU and GPU

---

### 2. **Labeling Parameters**

| Parameter | Type | Default | Range | Description |
|-----------|------|---------|-------|-------------|
| `TARGET_HORIZON` | int | 12 | 1-100 | Bars ahead to look for price movement |
| `TARGET_BASE_THRESHOLD` | float | 0.0008 | 0.0001-0.01 | Base threshold for price change (%) |
| `XGBOOST_VOLATILITY_ROLLING_WINDOW` | int | 100 | 20-200 | Window for volatility calculation |

**Labeling Logic**:

The module uses **dynamic threshold labeling** based on market volatility:

1. Calculate volatility multiplier using ATR or rolling returns
2. Adjust threshold dynamically based on volatility regime
3. Classify into:
   - **UP** (1): Future price > current + dynamic threshold
   - **DOWN** (0): Future price < current - dynamic threshold
   - **NEUTRAL** (2): Otherwise

**Volatility Regimes**:

- **Low volatility**: Tighter thresholds (more sensitive)
- **Medium volatility**: Standard thresholds
- **High volatility**: Wider thresholds (less noise)

---

### 3. **Feature Engineering Parameters**

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `MODEL_FEATURES` | list | See config | List of features used in model |
| `use_advanced_features` | bool | True | Include advanced technical indicators |
| `use_price_derived` | bool | True | Include price-derived features |

**Feature Categories**:

1. **Price-Derived Features** (via Rust if available):
   - Returns (log returns, simple returns)
   - High-low range
   - Volume-based features

2. **Advanced Features**:
   - Moving averages (EMA, SMA)
   - Momentum indicators (RSI, MACD)
   - Volatility indicators (ATR, Bollinger Bands)
   - Volume indicators

---

### 4. **Cross-Validation Parameters**

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `n_splits` | int | 5 | Number of time-series CV folds |
| `test_size` | int | 500 | Number of bars in test set per fold |
| `gap` | int | 12 | Gap between train and test (avoid leakage) |
| `XGBOOST_USE_PARALLEL_CV` | bool | True | Parallel fold execution |

**Cross-Validation Strategy**:

Uses **TimeSeriesSplit** with gap to respect temporal ordering and avoid look-ahead bias.

---

### 5. **Data & Processing Parameters**

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `timeframe` | str | "15m" | Timeframe (1m, 5m, 15m, 1h, 4h, 1d...) |
| `limit` | int | 5000 | Number of bars to fetch |
| `min_samples_per_class` | int | 30 | Minimum samples per class for CV fold |

---

### 6. **Performance & Optimization**

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `XGBOOST_USE_FLOAT32` | bool | True | Use float32 precision (2x memory reduction) |
| `XGBOOST_USE_PARALLEL_CV` | bool | True | Parallel cross-validation folds |
| `OPTUNA_PARALLEL_TRIALS` | int | 4 | Parallel Optuna trials |
| `use_cache` | bool | True | Cache models and labels |
| `use_rust` | bool | auto-detect | Use Rust for feature engineering/labeling |

**Backend Priority**:

1. **Rust Extensions** ⭐ **3-5x FASTER** - Labeling and feature engineering
2. **Numba JIT** - Fallback for labeling
3. **Pure Python** - Slowest fallback

**GPU Training**:
- Automatically detected via `nvidia-smi`
- Cached detection result for performance
- 2-5x faster training on compatible GPUs

**Parallel Processing**:
- Parallel CV folds: 2-4x faster cross-validation
- Parallel Optuna trials: 2-8x faster hyperparameter tuning
- Batch symbol processing: Linear scaling with CPU cores

---

## 📊 Output Results

`train_and_predict()` returns a **dictionary** containing:

### Training Metrics

- **`best_params`**: Best hyperparameters from cross-validation
- **`cv_scores`**: Cross-validation accuracy scores per fold
- **`mean_cv_accuracy`**: Average CV accuracy
- **`feature_importance`**: Feature importance scores

### Predictions

- **`predictions`**: Predicted classes for test set
- **`probabilities`**: Prediction probabilities per class
- **`test_accuracy`**: Accuracy on final test set

### Model

- **`model`**: Trained XGBoost classifier

---

## 🎛️ RECOMMENDED PRESETS

### 1. **Scalping** (Timeframe: 1m - 5m)

```python
config = {
    # Labeling
    'TARGET_HORIZON': 5,
    'TARGET_BASE_THRESHOLD': 0.001,
    'XGBOOST_VOLATILITY_ROLLING_WINDOW': 50,
    
    # Model
    'max_depth': 5,
    'learning_rate': 0.02,
    'n_estimators': 150,
    
    # Data
    'timeframe': '1m',
    'limit': 2000,
    
    # Performance
    'XGBOOST_USE_FLOAT32': True,
    'XGBOOST_USE_PARALLEL_CV': True,
}
```

### 2. **Intraday Trading** (Timeframe: 15m - 1h) ✅ **DEFAULT**

```python
config = {
    # Labeling
    'TARGET_HORIZON': 12,
    'TARGET_BASE_THRESHOLD': 0.0008,
    'XGBOOST_VOLATILITY_ROLLING_WINDOW': 100,
    
    # Model
    'max_depth': 6,
    'learning_rate': 0.01,
    'n_estimators': 200,
    'min_child_weight': 3,
    'gamma': 0.1,
    'subsample': 0.8,
    'colsample_bytree': 0.8,
    'reg_alpha': 0.5,
    'reg_lambda': 2.0,
    
    # Data
    'timeframe': '15m',
    'limit': 5000,
    
    # CV
    'n_splits': 5,
    'test_size': 500,
    'gap': 12,
    
    # Performance
    'XGBOOST_USE_FLOAT32': True,
    'XGBOOST_USE_PARALLEL_CV': True,
}
```

### 3. **Swing Trading** (Timeframe: 4h - 1d)

```python
config = {
    # Labeling
    'TARGET_HORIZON': 24,
    'TARGET_BASE_THRESHOLD': 0.002,
    'XGBOOST_VOLATILITY_ROLLING_WINDOW': 150,
    
    # Model
    'max_depth': 7,
    'learning_rate': 0.005,
    'n_estimators': 300,
    
    # Data
    'timeframe': '4h',
    'limit': 7000,
    
    # Performance
    'XGBOOST_USE_FLOAT32': True,
    'XGBOOST_USE_PARALLEL_CV': True,
}
```

### 4. **High-Performance** (Rust + Multi-symbol)

```python
config = {
    # ... standard params ...
    
    # Performance optimization
    'XGBOOST_USE_FLOAT32': True,        # 2x memory reduction
    'XGBOOST_USE_PARALLEL_CV': True,     # 2-4x CV speedup
    'OPTUNA_PARALLEL_TRIALS': 8,         # 2-8x optimization speedup
    'use_rust': True,                    # 3-5x feature/labeling speedup
    'use_cache': True,                   # 50-100x for repeated runs
}
```

**Python Implementation**:

```python
from modules.xgboost_LTS.utils.batch_symbols import batch_train_symbols
from modules.common.core.data_fetcher import DataFetcher

# Setup
data_fetcher = DataFetcher()
symbols = ["BTCUSDT", "ETHUSDT", "BNBUSDT", "ADAUSDT", "SOLUSDT"]

# Fetch data for all symbols
symbols_data = {}
for symbol in symbols:
    df, _ = data_fetcher.fetch_ohlcv_with_fallback_exchange(
        symbol, timeframe="15m", limit=5000
    )
    symbols_data[symbol] = df

# Batch training with parallel processing
results = batch_train_symbols(
    symbols_data=symbols_data,
    max_workers=4,  # Parallel processes
)

# Extract results
for symbol, result in results.items():
    print(f"{symbol}: CV Accuracy = {result['mean_cv_accuracy']:.4f}")
```

**Expected Performance**: Linear scaling with CPU cores (4-8x on 8-core)

### 5. **GPU Acceleration** (for Training)

```python
config = {
    # ... standard params ...
    
    # GPU settings (auto-configured if GPU available)
    'tree_method': 'hist',
    'device': 'cuda',  # Automatically set if GPU detected
    
    # Performance
    'XGBOOST_USE_FLOAT32': True,  # Important for GPU (1.2-1.5x speedup)
}
```

**Requirements**:
- NVIDIA GPU with compute capability >= 3.5
- CUDA Toolkit 11.0+
- GPU detection is cached for performance

**Expected Performance**: 2-5x faster training on compatible GPUs

---

## 🚀 PERFORMANCE COMPARISON

**Benchmark** (Single symbol, 5000 bars):

| Component | Without Optimization | With Optimization | Speedup |
|-----------|---------------------|-------------------|---------|
| Feature Engineering | ~2.0s | ~0.4s (Rust) | 5x |
| Labeling | ~1.0s | ~0.2s (Rust) | 5x |
| Training (CPU) | ~15.0s | ~6.0s (Parallel CV) | 2.5x |
| Training (GPU) | ~15.0s | ~3.0s (GPU) | 5x |
| **Total (CPU)** | **~18s** | **~7s** | **2.6x** |
| **Total (GPU + Rust)** | **~18s** | **~4s** | **4.5x** |

**Cache Performance**:

| Component | First Run | Cached Run | Speedup |
|-----------|-----------|------------|---------|
| Labels | 1.0s | <0.01s | 100x |
| Model | 15.0s | <0.1s | 150x |
| **Total** | **18s** | **<1s** | **18x+** |

---

## 🔧 SETUP & BUILD

### Rust Backend (Recommended)

```bash
cd modules/xgboost_LTS/rust_extensions
maturin develop --release
```

**Note**: On Windows, if the Rust linker cannot find dependencies:

```powershell
$env:RUSTFLAGS="-L 'C:\path\to\libs'"
maturin develop --release
```

See `docs/phase4_task.md` for detailed Rust installation instructions.

---

## 📝 EXAMPLE USAGE

```python
from modules.xgboost_LTS.core.model import train_and_predict
from modules.xgboost_LTS.core.labeling import apply_directional_labels
from modules.common.core.indicator_engine import IndicatorEngine, IndicatorConfig, IndicatorProfile
from modules.common.core.data_fetcher import DataFetcher
import pandas as pd

# Fetch data
data_fetcher = DataFetcher()
df, _ = data_fetcher.fetch_ohlcv_with_fallback_exchange(
    "BTCUSDT", timeframe="15m", limit=5000
)

# Calculate indicators
indicator_engine = IndicatorEngine(
    IndicatorConfig.for_profile(IndicatorProfile.XGBOOST)
)
df = indicator_engine.compute_features(df)

# Add advanced features
from modules.xgboost_LTS.utils.features import add_advanced_features
df = add_advanced_features(df)

# Apply labels
df = apply_directional_labels(df)

# Train and predict
results = train_and_predict(df, use_cache=True)

# Get prediction for latest bar
latest_prediction = results['predictions'][-1]
latest_proba = results['probabilities'][-1]

from config import ID_TO_LABEL
predicted_label = ID_TO_LABEL[latest_prediction]

print(f"Prediction: {predicted_label}")
print(f"Confidence: UP={latest_proba[1]:.2%}, DOWN={latest_proba[0]:.2%}, NEUTRAL={latest_proba[2]:.2%}")
print(f"CV Accuracy: {results['mean_cv_accuracy']:.2%}")
```

---

## 📞 TROUBLESHOOTING

**Common Issues**:

1. **Rust not found**: Install from <https://rustup.rs/>
2. **Maturin error**: `pip install maturin`
3. **Import error**: Run `maturin develop --release` in `rust_extensions/`
4. **GPU not detected**: Ensure CUDA Toolkit installed and `nvidia-smi` works
5. **Class imbalance**: Reduce `min_samples_perclass` or increase `limit`
6. **Memory issue**: Enable `XGBOOST_USE_FLOAT32` or reduce `limit`

---

## ✅ BEST PRACTICES

1. **Start with defaults** (Intraday preset)
2. **Adjust for your timeframe**:
   - Shorter TF → Lower `TARGET_HORIZON` (5-10)
   - Longer TF → Higher `TARGET_HORIZON` (20-30)
3. **Enable Rust backend** for production (3-5x faster)
4. **Enable caching** (`use_cache=True`) for development
5. **Use GPU** if available (2-5x faster training)
6. **Parallel CV** for faster model selection
7. **Monitor class distribution** to avoid imbalanced data
8. **Test parameters** on historical data first
9. **Use float32** for memory-constrained environments

---

## 📄 Document Information

**Generated**: 2026-01-30
**Module**: xgboost_LTS
**Purpose**: Settings reference for XGBoost LTS module
**Related Docs**: 
- `optimization_suggestions.md` - Optimization phases overview
- `phase4_task.md` - Rust extensions implementation
- `phase6_task.md` - Profiling and monitoring
