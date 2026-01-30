# 🚀 Optimization Suggestions for XGBoost Module

**Date**: 2026-01-30  
**Based on**: `adaptive_trend_LTS` optimization patterns (Phases 2-9)  
**Target Speedup**: 5-50x (depending on use case)

---

## 📊 Current State Analysis

### Module Structure

```
modules/xgboost/
├── core/
│   ├── labeling.py      # Dynamic threshold labeling (~286 lines)
│   ├── model.py         # XGBoost training & prediction (~391 lines)
│   └── optimization.py  # Optuna hyperparameter tuning (~491 lines)
├── utils/
│   ├── display.py       # Classification report formatting
│   └── utils.py         # Utility functions
└── cli/
    ├── main.py          # CLI entry point
    └── argument_parser.py
```

### Identified Bottlenecks (Based on Code Analysis)

| Component | Current State | Bottleneck Type | Priority |
|-----------|--------------|-----------------|----------|
| **Labeling** (`apply_directional_labels`) | Pure pandas/numpy | CPU-bound, vectorized but sequential | Medium |
| **Model Training** (`train_and_predict`) | XGBoost with CV | CPU-bound, sequential folds | High |
| **Hyperparameter Tuning** (`optimize`) | Optuna with TSCV | CPU-bound, sequential trials/folds | High |
| **Prediction** (`predict_next_move`) | Single prediction | I/O-bound, already fast | Low |

---

## 🔴 Phase 1: Core Optimizations (HIGH PRIORITY)

### 1.1 GPU Acceleration for XGBoost Training ⭐

**Current**: CPU-based training with optional GPU (`tree_method="hist", device="cuda"`)  
**Issue**: GPU detection via `nvidia-smi` subprocess call is slow; GPU not fully utilized  
**Improvement**: Pre-detect GPU at module import, cache result, ensure proper GPU utilization

```python
# BEFORE (model.py lines 153-169)
import subprocess
result = subprocess.run(["nvidia-smi"], capture_output=True, timeout=5)
if result.returncode == 0:
    params["tree_method"] = "hist"
    params["device"] = "cuda"

# AFTER (suggested)
import functools

@functools.lru_cache(maxsize=1)
def _detect_gpu_available() -> bool:
    """Detect GPU availability once at startup."""
    try:
        import subprocess
        result = subprocess.run(
            ["nvidia-smi", "--query-gpu=name", "--format=csv,noheader"],
            capture_output=True, timeout=5
        )
        return result.returncode == 0
    except Exception:
        return False

# In build_model():
if USE_GPU and _detect_gpu_available():
    params["tree_method"] = "hist"
    params["device"] = "cuda"
```

**Expected Gain**: 2-5x faster training on GPU  
**Effort**: Low  

### 1.2 Parallel Cross-Validation Folds ⭐

**Current**: Sequential CV folds in `train_and_predict` (lines 278-356)  
**Issue**: Each fold trains independently but runs sequentially  
**Improvement**: Use `joblib` or `concurrent.futures` for parallel fold execution

```python
# BEFORE (model.py lines 284-340)
for fold, (train_idx, test_idx) in enumerate(tscv.split(X), start=1):
    # ... training and evaluation ...

# AFTER (suggested implementation)
from concurrent.futures import ProcessPoolExecutor, as_completed

def _train_fold(fold_data):
    """Train single CV fold."""
    fold, train_idx, test_idx, X, y, params = fold_data
    # Apply gap, validate classes, train, evaluate
    # Return fold accuracy
    return fold, accuracy, y_true, y_pred

# Parallel execution
with ProcessPoolExecutor(max_workers=os.cpu_count() // 2) as executor:
    futures = {
        executor.submit(_train_fold, fold_data): fold
        for fold_data in fold_data_list
    }
    for future in as_completed(futures):
        fold, acc, y_true, y_pred = future.result()
        cv_scores.append(acc)
```

**Expected Gain**: 2-4x faster CV (depends on n_splits and CPU cores)  
**Effort**: Medium  

### 1.3 Parallel Optuna Trials ⭐

**Current**: Sequential Optuna trials (optimization.py lines 433-437)  
**Issue**: `study.optimize()` runs trials sequentially by default  
**Improvement**: Use Optuna's built-in parallel execution with `n_jobs`

```python
# BEFORE (optimization.py line 433-437)
study.optimize(
    lambda trial: self._objective(trial, X, y, n_splits=n_splits),
    n_trials=n_trials,
    show_progress_bar=True,
)

# AFTER (suggested)
study.optimize(
    lambda trial: self._objective(trial, X, y, n_splits=n_splits),
    n_trials=n_trials,
    n_jobs=-1,  # Use all available CPU cores
    show_progress_bar=True,
    gc_after_trial=True,  # Prevent memory leaks
)
```

**Expected Gain**: 2-8x faster optimization (depends on CPU cores)  
**Effort**: Very Low  

---

## 🟡 Phase 2: Memory & Vectorization (MEDIUM PRIORITY)

### 2.1 Numba JIT for Labeling Functions

**Current**: Pure pandas/numpy in `apply_directional_labels`  
**Issue**: Multiple rolling operations are slow for large datasets  
**Improvement**: Use Numba JIT for hot path calculations

```python
# BEFORE (labeling.py lines 167-243)
volatility_multiplier = _calculate_volatility_multiplier(df)
vol_low_rolling = volatility_multiplier.rolling(...).quantile(0.33)
# ... more rolling operations

# AFTER (suggested with Numba)
from numba import njit
import numpy as np

@njit(cache=True, parallel=True)
def _calculate_rolling_quantile_numba(arr: np.ndarray, window: int, q: float) -> np.ndarray:
    """Numba-optimized rolling quantile calculation."""
    n = len(arr)
    result = np.empty(n)
    for i in range(n):
        start = max(0, i - window + 1)
        result[i] = np.quantile(arr[start:i+1], q)
    return result

# Use in apply_directional_labels:
vol_low_rolling = pd.Series(
    _calculate_rolling_quantile_numba(volatility_multiplier.values, rolling_window, 0.33),
    index=volatility_multiplier.index
)
```

**Expected Gain**: 3-5x faster labeling for large datasets (>10,000 rows)  
**Effort**: Medium  

### 2.2 Memory-Efficient DataFrame Operations

**Current**: Multiple intermediate DataFrames created  
**Issue**: Memory usage spikes with large datasets  
**Improvement**: Use inplace operations and explicit memory cleanup

```python
# BEFORE (labeling.py)
df["DynamicThreshold"] = threshold_series
df["TargetLabel"] = np.where(...)
df["Target"] = df["TargetLabel"].map(LABEL_TO_ID)

# AFTER (memory-efficient)
import gc

# Use inplace operations where possible
df.loc[:, "DynamicThreshold"] = threshold_series
df.loc[:, "TargetLabel"] = np.where(...)
df.loc[:, "Target"] = df["TargetLabel"].map(LABEL_TO_ID)

# Explicit cleanup of large intermediates
del threshold_series, volatility_multiplier
gc.collect()
```

**Expected Gain**: 30-50% memory reduction for large datasets  
**Effort**: Low  

### 2.3 Float32 Precision Option

**Current**: Default float64 for all calculations  
**Issue**: Double memory usage, slower GPU operations  
**Improvement**: Optional float32 mode for memory/speed tradeoff

```python
# Add to config
XGBOOST_USE_FLOAT32 = True  # Set False for full precision

# In model.py
if XGBOOST_USE_FLOAT32:
    X = X.astype(np.float32)
    # XGBoost natively supports float32
```

**Expected Gain**: 2x memory reduction, 1.2-1.5x faster GPU training  
**Effort**: Low  

---

## 🟢 Phase 3: Caching & Persistence (MEDIUM PRIORITY)

### 3.1 Model Caching

**Current**: Model trained from scratch each run  
**Issue**: Repeated training for same data wastes resources  
**Improvement**: Cache trained models with configuration hash

```python
# Suggested implementation
import hashlib
import joblib
from pathlib import Path

class ModelCache:
    def __init__(self, cache_dir: str = "artifacts/xgboost/models"):
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
    
    def _get_cache_key(self, df_hash: str, config: dict) -> str:
        config_str = json.dumps(config, sort_keys=True)
        combined = f"{df_hash}:{config_str}"
        return hashlib.sha256(combined.encode()).hexdigest()[:16]
    
    def get_model(self, df: pd.DataFrame, config: dict) -> Optional[Any]:
        """Load cached model if available."""
        df_hash = hashlib.sha256(
            pd.util.hash_pandas_object(df).values.tobytes()
        ).hexdigest()[:16]
        cache_key = self._get_cache_key(df_hash, config)
        cache_path = self.cache_dir / f"model_{cache_key}.joblib"
        if cache_path.exists():
            return joblib.load(cache_path)
        return None
    
    def save_model(self, model, df: pd.DataFrame, config: dict):
        """Cache trained model."""
        df_hash = hashlib.sha256(
            pd.util.hash_pandas_object(df).values.tobytes()
        ).hexdigest()[:16]
        cache_key = self._get_cache_key(df_hash, config)
        cache_path = self.cache_dir / f"model_{cache_key}.joblib"
        joblib.dump(model, cache_path)
```

**Expected Gain**: Instant model loading for repeated runs (100x+)  
**Effort**: Medium  

### 3.2 Label Caching

**Current**: Labels recalculated each run  
**Issue**: Expensive calculation for unchanged data  
**Improvement**: Cache labeled DataFrames

```python
# Similar caching pattern for labels
def apply_directional_labels_cached(df: pd.DataFrame, cache_dir: str = "artifacts/xgboost/labels") -> pd.DataFrame:
    cache_path = Path(cache_dir)
    cache_path.mkdir(parents=True, exist_ok=True)
    
    # Calculate hash of input data
    df_hash = hashlib.sha256(
        pd.util.hash_pandas_object(df[["close"]]).values.tobytes()
    ).hexdigest()[:16]
    
    cached_file = cache_path / f"labels_{df_hash}.parquet"
    
    if cached_file.exists():
        return pd.read_parquet(cached_file)
    
    # Apply labels
    result = apply_directional_labels(df.copy())
    
    # Cache result
    result.to_parquet(cached_file)
    return result
```

**Expected Gain**: Near-instant for repeated labeling (50x+)  
**Effort**: Low  

---

## 🔵 Phase 4: Rust Extensions (LOW PRIORITY - After Core Done)

### 4.1 Rust Backend for Labeling

**Current**: Python/NumPy for all labeling  
**Opportunity**: Critical path optimization with Rust (similar to `adaptive_trend_LTS` Phase 3)

```rust
// rust_extensions/src/labeling.rs (suggested structure)
use pyo3::prelude::*;
use numpy::{PyArray1, PyReadonlyArray1};

#[pyfunction]
fn calculate_volatility_multiplier_rust(
    close: PyReadonlyArray1<f64>,
    atr_14: Option<PyReadonlyArray1<f64>>,
) -> PyResult<Py<PyArray1<f64>>> {
    // Rust implementation with SIMD optimization
}

#[pyfunction]
fn apply_directional_labels_rust(
    close: PyReadonlyArray1<f64>,
    atr_14: Option<PyReadonlyArray1<f64>>,
    target_horizon: usize,
    base_threshold: f64,
) -> PyResult<(Py<PyArray1<i32>>, Py<PyArray1<f64>>)> {
    // Returns (labels, thresholds)
}
```

**Expected Gain**: 2-5x faster labeling vs NumPy  
**Effort**: High  

### 4.2 Rust Backend for Feature Engineering

**Current**: pandas-ta or custom calculations  
**Opportunity**: Pre-calculate features in Rust

```rust
// rust_extensions/src/features.rs
#[pyfunction]
fn batch_calculate_features_rust(
    prices: PyReadonlyArray2<f64>,  // [open, high, low, close, volume]
) -> PyResult<PyObject> {
    // Calculate all MODEL_FEATURES in one pass
    // Return HashMap<String, Vec<f64>>
}
```

**Expected Gain**: 3-5x faster feature calculation  
**Effort**: High  

---

## 🟣 Phase 5: Batch & Distributed Processing (OPTIONAL)

### 5.1 Dask Integration for Large Datasets

**Current**: In-memory processing only  
**Opportunity**: Out-of-core processing for large historical datasets (similar to `adaptive_trend_LTS` Phase 5)

```python
import dask.dataframe as dd

def train_and_predict_dask(df: dd.DataFrame) -> Any:
    """Train XGBoost with Dask integration for large datasets."""
    import dask_ml.xgboost as dxgb
    
    X = df[MODEL_FEATURES]
    y = df["Target"]
    
    # DaskXGBClassifier handles distributed training
    model = dxgb.XGBClassifier(**XGBOOST_PARAMS)
    model.fit(X, y)
    
    return model
```

**Expected Gain**: Unlimited dataset size, 90% memory reduction  
**Effort**: Medium  

### 5.2 Batch Symbol Processing

**Current**: One symbol at a time  
**Opportunity**: Process multiple symbols in parallel

```python
from concurrent.futures import ProcessPoolExecutor

def batch_train_symbols(
    symbols_data: Dict[str, pd.DataFrame],
    max_workers: int = None
) -> Dict[str, Any]:
    """Train models for multiple symbols in parallel."""
    results = {}
    
    with ProcessPoolExecutor(max_workers=max_workers) as executor:
        futures = {
            executor.submit(train_and_predict, df): symbol
            for symbol, df in symbols_data.items()
        }
        
        for future in as_completed(futures):
            symbol = futures[future]
            try:
                results[symbol] = future.result()
            except Exception as e:
                log_error(f"Failed to train {symbol}: {e}")
    
    return results
```

**Expected Gain**: Linear scaling with CPU cores (4-8x on 8-core)  
**Effort**: Low  

---

## 🟤 Phase 6: Profiling & Monitoring (FOUNDATION)

### 6.1 Profiling Infrastructure

**Similar to `adaptive_trend_LTS` Phase 8, establish profiling workflow:**

```python
# scripts/profile_xgboost.py
import cProfile
import pstats
from pathlib import Path

def profile_training(symbol: str, timeframe: str, output_dir: str = "profiles/xgboost"):
    """Profile XGBoost training pipeline."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    profiler = cProfile.Profile()
    profiler.enable()
    
    # Run training
    from modules.xgboost import train_and_predict
    # ... training code ...
    
    profiler.disable()
    
    # Save stats
    stats_file = output_dir / f"training_{symbol}_{timeframe}.stats"
    profiler.dump_stats(str(stats_file))
    
    # Print summary
    stats = pstats.Stats(profiler)
    stats.sort_stats("cumulative")
    stats.print_stats(20)
```

### 6.2 Benchmark Script

```python
# benchmarks/benchmark_xgboost.py
import time
import pandas as pd
from modules.xgboost import train_and_predict, apply_directional_labels

def benchmark_xgboost(df: pd.DataFrame, n_runs: int = 5):
    """Benchmark XGBoost pipeline."""
    results = {}
    
    # Benchmark labeling
    times = []
    for _ in range(n_runs):
        start = time.perf_counter()
        apply_directional_labels(df.copy())
        times.append(time.perf_counter() - start)
    results["labeling"] = {"mean": sum(times)/len(times), "times": times}
    
    # Benchmark training
    labeled_df = apply_directional_labels(df.copy())
    times = []
    for _ in range(n_runs):
        start = time.perf_counter()
        train_and_predict(labeled_df.copy())
        times.append(time.perf_counter() - start)
    results["training"] = {"mean": sum(times)/len(times), "times": times}
    
    return results
```

---

## 📋 Implementation Roadmap

### Phase 1: Core Optimizations (Week 1-2) - **HIGH ROI**

| Task | Effort | Expected Gain | Dependencies |
|------|--------|---------------|--------------|
| 1.1 GPU Detection Caching | Low | 2-5x training | None |
| 1.2 Parallel CV Folds | Medium | 2-4x CV | None |
| 1.3 Parallel Optuna Trials | Very Low | 2-8x optimization | None |

### Phase 2: Memory & Vectorization (Week 3-4) - **MEDIUM ROI**

| Task | Effort | Expected Gain | Dependencies |
|------|--------|---------------|--------------|
| 2.1 Numba JIT Labeling | Medium | 3-5x labeling | None |
| 2.2 Memory-Efficient Ops | Low | 30-50% memory | None |
| 2.3 Float32 Option | Low | 2x memory, 1.2x speed | None |

### Phase 3: Caching & Persistence (Week 5-6) - **HIGH ROI**

| Task | Effort | Expected Gain | Dependencies |
|------|--------|---------------|--------------|
| 3.1 Model Caching | Medium | 100x+ repeated runs | None |
| 3.2 Label Caching | Low | 50x+ repeated runs | None |

### Phase 4: Rust Extensions (Future - Optional) - **MEDIUM ROI**

| Task | Effort | Expected Gain | Dependencies |
|------|--------|---------------|--------------|
| 4.1 Rust Labeling | High | 2-5x labeling | Phase 2 complete |
| 4.2 Rust Features | High | 3-5x features | Phase 4.1 |

### Phase 5: Batch & Distributed (Future - Optional) - **VARIABLE ROI**

| Task | Effort | Expected Gain | Dependencies |
|------|--------|---------------|--------------|
| 5.1 Dask Integration | Medium | Unlimited size | None |
| 5.2 Batch Symbols | Low | Linear scaling | None |

---

## 🎯 Recommended Starting Points

Based on ROI and effort analysis:

### 🔥 Quick Wins (Start Here)

1. **1.3 Parallel Optuna Trials** - Add `n_jobs=-1` (1 line change, 2-8x gain)
2. **1.1 GPU Detection Caching** - Cache nvidia-smi result (10 lines, avoids repeated subprocess calls)
3. **2.3 Float32 Option** - Optional precision reduction (5 lines, 2x memory)

### 📊 High-Impact Medium Effort

1. **3.1 Model Caching** - Cache trained models (50 lines, 100x+ for repeated runs)
2. **1.2 Parallel CV Folds** - Parallelize cross-validation (30 lines, 2-4x gain)

### 🔬 Advanced Optimizations (After Core)

1. **2.1 Numba JIT Labeling** - JIT compile rolling calculations
2. **4.1 Rust Extensions** - Native performance for labeling

---

## 📄 Related Documentation

- **`adaptive_trend_LTS` Optimization**: `modules/adaptive_trend_LTS/docs/optimization_suggestions.md`
- **`adaptive_trend_LTS` Features Summary**: `modules/adaptive_trend_LTS/docs/features_summary_20260130.md`
- **Config**: `config/__init__.py` (XGBOOST_PARAMS, MODEL_FEATURES)

---

**Last Updated**: 2026-01-30  
**Status**: 📋 Proposals - Ready for Implementation  
**Target Speedup**: 5-50x (depending on use case and phases implemented)
