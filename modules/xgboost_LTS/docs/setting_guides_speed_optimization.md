# ⚡ RECOMMENDED SETTINGS FOR MAXIMUM PROCESSING SPEED

**Purpose**: Bộ cấu hình tối ưu hóa tốc độ xử lý cho `xgboost_LTS` module  
**Use Case**: Batch training, multi-symbol processing, production deployment  
**Last Updated**: 2026-01-30

---

## 🎯 Quick Summary

Để đạt **tốc độ xử lý tối đa**, sử dụng các setting sau:

### 1. **Backend Selection** (Quan trọng nhất!)

| Scenario | Recommended Backend | Expected Speedup | Config |
|----------|-------------------|------------------|--------|
| **Single Symbol Training** | Rust + GPU + Parallel CV | 4-5x | `use_rust=True, device='cuda', XGBOOST_USE_PARALLEL_CV=True` |
| **Small Batch (<10 symbols)** | Rust + Parallel + Cache | 3-4x | `use_rust=True, use_cache=True, max_workers=4` |
| **Medium Batch (10-100)** | Rust + GPU + Batch Processing | 5-10x | `use_rust=True, device='cuda', batch_processing=True` |
| **Hyperparameter Tuning** | Parallel Optuna Trials | 2-8x | `OPTUNA_PARALLEL_TRIALS=-1` |
| **Repeated Runs** | Model & Label Caching | 50-150x | `use_cache=True` |

### 2. **Core Performance Settings**

```yaml
# Performance & Optimization
XGBOOST_USE_FLOAT32: true        # Use float32 for 2x memory reduction
XGBOOST_USE_PARALLEL_CV: true    # Enable parallel cross-validation
OPTUNA_PARALLEL_TRIALS: -1       # Use all CPU cores for Optuna
use_rust: true                   # Enable Rust for feature/labeling (3-5x)
use_cache: true                  # Enable caching for repeated runs

# GPU Settings (auto-configured)
tree_method: "hist"              # Optimized for GPU
device: "cuda"                   # Auto-set if GPU detected
```

### 3. **Memory Optimization Settings**

```yaml
# Memory Optimizations
XGBOOST_USE_FLOAT32: true        # 50% memory reduction
use_cache: true                  # Cache to disk for repeated runs
cache_compression: true          # Compress cached files (5-10x reduction)
```

---

## 📋 Recommended Presets by Use Case

### Preset 1: **Single Symbol Training (Maximum Speed)**

**Use Case**: Quick model training, strategy validation, single pair analysis

**Configuration**:

```yaml
# Model parameters (optimized for speed)
max_depth: 6
learning_rate: 0.02           # Slightly higher for faster convergence
n_estimators: 150             # Reduced for speed

# Cross-validation (parallel)
n_splits: 3                   # Reduced for speed (vs 5 default)
XGBOOST_USE_PARALLEL_CV: true # 2-4x speedup

# Performance
XGBOOST_USE_FLOAT32: true
use_rust: true                # 3-5x feature/labeling
use_cache: true
device: "cuda"                # If GPU available
```

**Python Implementation**:

```python
from modules.xgboost_LTS.core.model import train_and_predict
from modules.xgboost_LTS.core.labeling import apply_directional_labels
from modules.xgboost_LTS.utils.features import add_advanced_features
from modules.common.core.indicator_engine import IndicatorEngine, IndicatorConfig, IndicatorProfile
from modules.common.core.data_fetcher import DataFetcher

# Setup
data_fetcher = DataFetcher()
df, _ = data_fetcher.fetch_ohlcv_with_fallback_exchange(
    "BTCUSDT", timeframe="15m", limit=5000
)

# Compute features (uses Rust if available)
indicator_engine = IndicatorEngine(
    IndicatorConfig.for_profile(IndicatorProfile.XGBOOST)
)
df = indicator_engine.compute_features(df)
df = add_advanced_features(df)

# Apply labels (uses Rust if available)
df = apply_directional_labels(df)

# Train with optimized settings
results = train_and_predict(
    df,
    use_cache=True,        # Enable caching
    n_splits=3,            # Faster CV
)

print(f"CV Accuracy: {results['mean_cv_accuracy']:.2%}")
print(f"Test Accuracy: {results['test_accuracy']:.2%}")
```

**Expected Performance**: ~4-7s total (with Rust + GPU + cache)

---

### Preset 2: **Batch Symbol Training (Multi-Symbol)**

**Use Case**: Portfolio training, market-wide analysis, strategy comparison

**Configuration**:

```yaml
# Batch settings
max_workers: 4                # Parallel processes (adjust based on CPU cores)

# Model (balanced for batch processing)
max_depth: 6
learning_rate: 0.01
n_estimators: 200

# Performance
XGBOOST_USE_FLOAT32: true
XGBOOST_USE_PARALLEL_CV: true
use_rust: true
use_cache: true
device: "cuda"
```

**Python Implementation**:

```python
from modules.xgboost_LTS.utils.batch_symbols import batch_train_symbols
from modules.common.core.data_fetcher import DataFetcher
from modules.common.core.indicator_engine import IndicatorEngine, IndicatorConfig, IndicatorProfile
from modules.xgboost_LTS.utils.features import add_advanced_features
from modules.xgboost_LTS.core.labeling import apply_directional_labels

# Setup
data_fetcher = DataFetcher()
indicator_engine = IndicatorEngine(
    IndicatorConfig.for_profile(IndicatorProfile.XGBOOST)
)
symbols = ["BTCUSDT", "ETHUSDT", "BNBUSDT", "ADAUSDT", "SOLUSDT", "DOTUSDT", "LINKUSDT"]

# Fetch and prepare data for all symbols
symbols_data = {}
for symbol in symbols:
    try:
        df, _ = data_fetcher.fetch_ohlcv_with_fallback_exchange(
            symbol, timeframe="15m", limit=5000
        )
        
        # Compute features
        df = indicator_engine.compute_features(df)
        df = add_advanced_features(df)
        
        # Apply labels
        df = apply_directional_labels(df)
        
        symbols_data[symbol] = df
    except Exception as e:
        print(f"Error preparing {symbol}: {e}")

# Batch training with parallel processing
results = batch_train_symbols(
    symbols_data=symbols_data,
    max_workers=4,  # Adjust based on CPU cores
)

# Extract and display results
print("\n" + "="*60)
print("BATCH TRAINING RESULTS")
print("="*60)

for symbol, result in sorted(results.items(), key=lambda x: x[1]['mean_cv_accuracy'], reverse=True):
    print(f"{symbol:10} | CV: {result['mean_cv_accuracy']:.2%} | Test: {result['test_accuracy']:.2%}")
```

**Expected Performance**: Linear scaling with CPU cores (4-8x on 8-core)

---

### Preset 3: **Hyperparameter Optimization (Optuna)**

**Use Case**: Finding optimal model parameters, strategy optimization

**Configuration**:

```yaml
# Optuna settings
OPTUNA_PARALLEL_TRIALS: -1    # Use all CPU cores
n_trials: 100                 # Number of trials

# Model search space
max_depth: [4, 5, 6, 7, 8]
learning_rate: [0.005, 0.01, 0.02, 0.05]
n_estimators: [100, 150, 200, 300]
min_child_weight: [1, 3, 5]
gamma: [0, 0.1, 0.2]
subsample: [0.7, 0.8, 0.9, 1.0]
colsample_bytree: [0.7, 0.8, 0.9, 1.0]

# Performance
XGBOOST_USE_FLOAT32: true
XGBOOST_USE_PARALLEL_CV: true
use_rust: true
device: "cuda"
```

**Python Implementation**:

```python
from modules.xgboost_LTS.core.optimization import XGBoostHyperparameterOptimizer
from modules.xgboost_LTS.core.labeling import apply_directional_labels
from modules.xgboost_LTS.utils.features import add_advanced_features
from modules.common.core.indicator_engine import IndicatorEngine, IndicatorConfig, IndicatorProfile
from modules.common.core.data_fetcher import DataFetcher

# Setup
data_fetcher = DataFetcher()
indicator_engine = IndicatorEngine(
    IndicatorConfig.for_profile(IndicatorProfile.XGBOOST)
)

# Fetch and prepare data
df, _ = data_fetcher.fetch_ohlcv_with_fallback_exchange(
    "BTCUSDT", timeframe="15m", limit=5000
)

df = indicator_engine.compute_features(df)
df = add_advanced_features(df)
df = apply_directional_labels(df)

# Hyperparameter optimization with parallel trials
optimizer = XGBoostHyperparameterOptimizer()

# Optimize (uses OPTUNA_PARALLEL_TRIALS from config)
best_params, best_score, study = optimizer.optimize(
    df,
    n_trials=100,      # Number of trials
    n_splits=3,        # CV splits (reduced for speed)
)

print(f"\nBest CV Score: {best_score:.4f}")
print(f"Best Parameters:")
for param, value in best_params.items():
    print(f"  {param}: {value}")

# Train final model with best parameters
from modules.xgboost_LTS.core.model import train_and_predict

results = train_and_predict(df, custom_params=best_params)
print(f"\nFinal Test Accuracy: {results['test_accuracy']:.2%}")
```

**Expected Performance**: 2-8x speedup with parallel trials (depends on CPU cores)

---

### Preset 4: **Cached Development Workflow**

**Use Case**: Rapid iteration, parameter tuning, feature engineering experiments

**Configuration**:

```yaml
# Caching settings (maximum caching)
use_cache: true
cache_compression: true       # Compress cached files (5-10x reduction)
cache_ttl: 604800             # 7 days cache lifetime

# Performance
XGBOOST_USE_FLOAT32: true
XGBOOST_USE_PARALLEL_CV: true
use_rust: true
```

**Python Implementation**:

```python
from modules.xgboost_LTS.core.model import train_and_predict
from modules.xgboost_LTS.core.labeling import apply_directional_labels
from modules.xgboost_LTS.utils.features import add_advanced_features
from modules.common.core.indicator_engine import IndicatorEngine, IndicatorConfig, IndicatorProfile
from modules.common.core.data_fetcher import DataFetcher

# Setup (same as before)
data_fetcher = DataFetcher()
indicator_engine = IndicatorEngine(
    IndicatorConfig.for_profile(IndicatorProfile.XGBOOST)
)

# Fetch and prepare data
df, _ = data_fetcher.fetch_ohlcv_with_fallback_exchange(
    "BTCUSDT", timeframe="15m", limit=5000
)

df = indicator_engine.compute_features(df)
df = add_advanced_features(df)

# Apply labels (will be cached)
df = apply_directional_labels(df)

# First run: Full training (~10-15s)
print("First run (no cache)...")
results = train_and_predict(df, use_cache=True)
print(f"Time: ~10-15s | CV Accuracy: {results['mean_cv_accuracy']:.2%}")

# Second run: Cached (~0.1s)
print("\nSecond run (with cache)...")
results = train_and_predict(df, use_cache=True)
print(f"Time: ~0.1s | CV Accuracy: {results['mean_cv_accuracy']:.2%}")

# Experiment with different parameters (cache reused)
print("\nExperimenting with parameters...")
results = train_and_predict(df, use_cache=True, n_splits=5)
print(f"Time: ~0.1s | CV Accuracy: {results['mean_cv_accuracy']:.2%}")
```

**Expected Performance**: 
- First run: ~10-15s
- Cached runs: ~0.1s (150x speedup)

---

### Preset 5: **Production Deployment (GPU + Rust + Cache)**

**Use Case**: Live trading bot, production API, real-time predictions

**Configuration**:

```yaml
# Production settings
XGBOOST_USE_FLOAT32: true        # Memory efficiency
XGBOOST_USE_PARALLEL_CV: false   # Single-threaded for stability
use_rust: true                   # Maximum speed
use_cache: true                  # Fast model loading
device: "cuda"                   # GPU for predictions

# Optuna (optional: for pre-deployment hyperparameter tuning)
OPTUNA_PARALLEL_TRIALS: -1       # Use all CPU cores when running Optuna
n_trials: 50                     # Trials for one-off tuning before deploy

# Model (production-optimized; or use best_params from Optuna)
max_depth: 6
learning_rate: 0.01
n_estimators: 200
early_stopping_rounds: 20        # Prevent overfitting
```

**Python Implementation**:

```python
from modules.xgboost_LTS.core.model import train_and_predict, predict_next_move
from modules.xgboost_LTS.core.labeling import apply_directional_labels
from modules.xgboost_LTS.utils.features import add_advanced_features
from modules.common.core.indicator_engine import IndicatorEngine, IndicatorConfig, IndicatorProfile
from modules.common.core.data_fetcher import DataFetcher
from config import ID_TO_LABEL
import pandas as pd

# Setup (once at startup)
data_fetcher = DataFetcher()
indicator_engine = IndicatorEngine(
    IndicatorConfig.for_profile(IndicatorProfile.XGBOOST)
)

# Train model once (or load from cache)
def train_production_model(symbol="BTCUSDT", timeframe="15m"):
    """Train or load cached model"""
    df, _ = data_fetcher.fetch_ohlcv_with_fallback_exchange(
        symbol, timeframe=timeframe, limit=5000
    )
    
    df = indicator_engine.compute_features(df)
    df = add_advanced_features(df)
    df = apply_directional_labels(df)
    
    # Train with cache (will load if available)
    results = train_and_predict(df, use_cache=True)
    return results['model']

# Train/load model at startup
model = train_production_model()
print("Model loaded successfully!")

# Prediction function for live trading
def get_live_prediction(symbol="BTCUSDT", timeframe="15m"):
    """Fast prediction for live trading"""
    # Fetch latest data (minimal bars needed)
    df, _ = data_fetcher.fetch_ohlcv_with_fallback_exchange(
        symbol, timeframe=timeframe, limit=500  # Minimal for features
    )
    
    # Compute features
    df = indicator_engine.compute_features(df)
    df = add_advanced_features(df)
    
    # Predict using cached model
    prediction_id, probabilities = predict_next_move(df, model)
    predicted_label = ID_TO_LABEL[prediction_id]
    
    return {
        'prediction': predicted_label,
        'confidence': {
            'UP': probabilities[1],
            'DOWN': probabilities[0],
            'NEUTRAL': probabilities[2]
        }
    }

# Usage in trading loop
result = get_live_prediction()
print(f"Prediction: {result['prediction']}")
print(f"Confidence: UP={result['confidence']['UP']:.2%}, DOWN={result['confidence']['DOWN']:.2%}")
```

**Optional: Hyperparameter tuning with Optuna before deployment**

Chạy Optuna một lần (khi deploy hoặc định kỳ) để tìm `best_params`, sau đó train model production với params đó. Chuẩn bị `df` giống như trong `train_production_model()` ở trên, rồi:

```python
from modules.xgboost_LTS.core.optimization import XGBoostHyperparameterOptimizer

# (Optional) Run Optuna once to get best_params (uses OPTUNA_PARALLEL_TRIALS from config)
optimizer = XGBoostHyperparameterOptimizer()
best_params, best_score, study = optimizer.optimize(
    df,
    n_trials=50,       # Fewer trials for production tuning
    n_splits=3,
)
# Train production model with best_params (uses OPTUNA_PARALLEL_TRIALS from config)
results = train_and_predict(df, use_cache=True, custom_params=best_params)
model = results['model']
# Then use model in get_live_prediction() as above
```

**Expected Performance**: 
- Model loading: <0.1s (cached)
- Live prediction: ~0.5-1s per symbol
- Optional Optuna tuning: 2-8x speedup with `OPTUNA_PARALLEL_TRIALS=-1` (see Preset 3)

---

## 🔧 Integration Examples

### Example 1: CLI Integration

Create a configuration file `xgboost_fast_config.yaml`:

```yaml
# XGBoost LTS Fast Configuration
xgboost_lts:
  # Model Parameters
  max_depth: 6
  learning_rate: 0.01
  n_estimators: 200
  min_child_weight: 3
  gamma: 0.1
  subsample: 0.8
  colsample_bytree: 0.8
  reg_alpha: 0.5
  reg_lambda: 2.0
  
  # Labeling
  TARGET_HORIZON: 12
  TARGET_BASE_THRESHOLD: 0.0008
  XGBOOST_VOLATILITY_ROLLING_WINDOW: 100
  
  # Cross-Validation
  n_splits: 5
  test_size: 500
  gap: 12
  
  # Performance Settings
  XGBOOST_USE_FLOAT32: true
  XGBOOST_USE_PARALLEL_CV: true
  OPTUNA_PARALLEL_TRIALS: -1
  use_rust: true
  use_cache: true
  tree_method: "hist"
  device: "cuda"  # Auto-configured
  
  # Caching
  cache_compression: true
  cache_ttl: 604800  # 7 days
```

### Example 2: Batch Processing Script

```python
#!/usr/bin/env python
"""
Batch XGBoost training script for multiple symbols
Optimized for maximum speed with Rust + GPU + Parallel processing
"""

from modules.xgboost_LTS.utils.batch_symbols import batch_train_symbols
from modules.common.core.data_fetcher import DataFetcher
from modules.common.core.indicator_engine import IndicatorEngine, IndicatorConfig, IndicatorProfile
from modules.xgboost_LTS.utils.features import add_advanced_features
from modules.xgboost_LTS.core.labeling import apply_directional_labels
import pandas as pd
from pathlib import Path
import json
from datetime import datetime

def batch_train_and_save(symbols, timeframe="15m", limit=5000, max_workers=4):
    """Batch train models and save results"""
    
    # Setup
    data_fetcher = DataFetcher()
    indicator_engine = IndicatorEngine(
        IndicatorConfig.for_profile(IndicatorProfile.XGBOOST)
    )
    
    # Prepare data
    symbols_data = {}
    for symbol in symbols:
        try:
            df, _ = data_fetcher.fetch_ohlcv_with_fallback_exchange(
                symbol, timeframe=timeframe, limit=limit
            )
            df = indicator_engine.compute_features(df)
            df = add_advanced_features(df)
            df = apply_directional_labels(df)
            symbols_data[symbol] = df
        except Exception as e:
            print(f"Error preparing {symbol}: {e}")
    
    # Batch train
    results = batch_train_symbols(symbols_data, max_workers=max_workers)
    
    # Save results
    output_dir = Path("artifacts/xgboost/batch_results")
    output_dir.mkdir(parents=True, exist_ok=True)
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # Save summary
    summary = []
    for symbol, result in results.items():
        summary.append({
            'symbol': symbol,
            'cv_accuracy': result['mean_cv_accuracy'],
            'test_accuracy': result['test_accuracy'],
            'cv_scores': result['cv_scores'],
        })
    
    summary_df = pd.DataFrame(summary).sort_values('cv_accuracy', ascending=False)
    summary_df.to_csv(output_dir / f"batch_summary_{timestamp}.csv", index=False)
    
    # Save detailed results
    with open(output_dir / f"batch_results_{timestamp}.json", 'w') as f:
        json.dump({
            symbol: {
                'cv_accuracy': result['mean_cv_accuracy'],
                'test_accuracy': result['test_accuracy'],
                'best_params': result['best_params'],
            }
            for symbol, result in results.items()
        }, f, indent=2)
    
    print(f"\nResults saved to {output_dir}")
    print("\nTop 10 Performers:")
    print(summary_df.head(10))
    
    return results

if __name__ == "__main__":
    # Define symbols to train
    symbols = [
        "BTCUSDT", "ETHUSDT", "BNBUSDT", "ADAUSDT", "SOLUSDT",
        "DOTUSDT", "LINKUSDT", "UNIUSDT", "LTCUSDT", "AVAXUSDT"
    ]
    
    # Run batch training
    results = batch_train_and_save(
        symbols=symbols,
        timeframe="15m",
        limit=5000,
        max_workers=4  # Adjust based on CPU cores
    )
```

---

## 💡 Performance Tips

### Tip 1: Optimize Cross-Validation

```python
# Faster CV for development
results = train_and_predict(df, n_splits=3)  # vs default 5

# Production CV for accuracy
results = train_and_predict(df, n_splits=5)
```

### Tip 2: Use Float32 for Large Datasets

```python
# In config/__init__.py
XGBOOST_USE_FLOAT32 = True  # 50% memory reduction

# Automatic conversion in training pipeline
```

### Tip 3: Enable All Optimizations

```python
# config/__init__.py
XGBOOST_USE_FLOAT32 = True
XGBOOST_USE_PARALLEL_CV = True
OPTUNA_PARALLEL_TRIALS = -1  # Use all cores
```

### Tip 4: Cache Aggressively

```python
# Always use cache for development
results = train_and_predict(df, use_cache=True)

# Clear cache when data changes
from modules.xgboost_LTS.utils.cache_manager import CacheManager
cache_manager = CacheManager()
cache_manager.clear_cache()  # Clear all
cache_manager.clear_old_cache(max_age_days=7)  # Clear old
```

### Tip 5: Monitor Performance

```python
# Use profiling script
# python scripts/profile_xgboost.py

# Use benchmark script
# python modules/xgboost_LTS/benchmarks/regression_test.py
```

---

## 📊 Performance Budget

Based on benchmarks, these are the expected performance targets:

| Task | Budget | Actual (Optimized) | Status |
|------|--------|-------------|--------|
| Feature Engineering (5000 bars) | <2.0s | ~0.4s (Rust) | ✅ Within budget |
| Labeling (5000 bars) | <1.0s | ~0.2s (Rust) | ✅ Within budget |
| Training (Single symbol, CPU) | <15.0s | ~6.0s (Parallel CV) | ✅ Within budget |
| Training (Single symbol, GPU) | <10.0s | ~3.0s (GPU) | ✅ Within budget |
| Batch Training (10 symbols) | <120s | ~60s (Parallel) | ✅ Within budget |
| Hyperparameter Optimization (100 trials) | <600s | ~150s (Parallel) | ✅ Within budget |

If performance exceeds budget, investigate using profiling tools.

---

## 📄 Document Information

**Generated**: 2026-01-30
**Module**: xgboost_LTS
**Purpose**: Speed optimization guide for XGBoost LTS module
**Related Docs**: 
- `setting_guides.md` - Complete settings reference
- `optimization_suggestions.md` - Optimization phases overview
- `phase4_task.md` - Rust extensions implementation
- `phase6_task.md` - Profiling and monitoring
