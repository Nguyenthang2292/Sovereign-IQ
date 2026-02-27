# Phase 1 Task: Core Optimizations for XGBoost Module

**Target**: Achieve 2-8x speedup with minimal code changes  
**Effort**: Low-Medium  
**Priority**: 🔴 HIGH

---

## 🎯 Objectives

1. **GPU Detection Caching** - Eliminate repeated `nvidia-smi` subprocess calls
2. **Parallel Cross-Validation** - Run CV folds in parallel using multiprocessing
3. **Parallel Optuna Trials** - Enable Optuna's built-in parallelization

---

## Task 1.1: GPU Detection Caching ✅ DONE

### Current Code (model.py lines 152-169)

```python
if USE_GPU:
    try:
        import subprocess
        result = subprocess.run(["nvidia-smi"], capture_output=True, timeout=5)
        if result.returncode == 0:
            params["tree_method"] = "hist"
            params["device"] = "cuda"
    except (subprocess.TimeoutExpired, FileNotFoundError, Exception):
        pass
```

### Issue

- `nvidia-smi` is called every time `build_model()` is invoked
- During cross-validation, this happens 5+ times
- Each subprocess call adds ~100-500ms overhead

### Solution

Create a cached GPU detection function that only runs once per module import.

### Implementation Steps

#### Step 1: Create GPU utility module

Create `modules/xgboost/utils/gpu_utils.py`:

```python
"""
GPU utilities for XGBoost module.

Provides cached GPU detection to avoid repeated subprocess calls.
"""

import functools
import subprocess
from typing import Optional


@functools.lru_cache(maxsize=1)
def detect_cuda_available() -> bool:
    """
    Detect if CUDA GPU is available for XGBoost.
    
    Uses nvidia-smi to check for GPU availability.
    Result is cached after first call.
    
    Returns:
        True if CUDA GPU is available, False otherwise.
    """
    try:
        result = subprocess.run(
            ["nvidia-smi", "--query-gpu=name", "--format=csv,noheader"],
            capture_output=True,
            timeout=5,
        )
        return result.returncode == 0
    except (subprocess.TimeoutExpired, FileNotFoundError, OSError, Exception):
        return False


@functools.lru_cache(maxsize=1)
def get_gpu_info() -> Optional[str]:
    """
    Get GPU name if available.
    
    Returns:
        GPU name string or None if not available.
    """
    try:
        result = subprocess.run(
            ["nvidia-smi", "--query-gpu=name", "--format=csv,noheader"],
            capture_output=True,
            text=True,
            timeout=5,
        )
        if result.returncode == 0:
            return result.stdout.strip()
    except Exception:
        pass
    return None
```

#### Step 2: Update model.py

Replace the inline GPU detection with cached function:

```python
# Add import at top of model.py
from modules.xgboost.utils.gpu_utils import detect_cuda_available

# In build_model() function, replace lines 152-169 with:
if USE_GPU and detect_cuda_available():
    params["tree_method"] = "hist"
    params["device"] = "cuda"
    if "n_jobs" in params:
        del params["n_jobs"]
```

#### Step 3: Update **init**.py

Add new utility to exports:

```python
# In modules/xgboost/utils/__init__.py
from modules.xgboost.utils.gpu_utils import detect_cuda_available, get_gpu_info
```

### Expected Result

- Eliminate 5+ redundant subprocess calls per training session
- **Speedup**: ~0.5-2 seconds per CV session

### Test

```python
# Test GPU detection caching
from modules.xgboost.utils.gpu_utils import detect_cuda_available

# First call - may take 100-500ms
result1 = detect_cuda_available()

# Second call - instant (cached)
result2 = detect_cuda_available()

assert result1 == result2
print(f"GPU Available: {result1}")
```

---

## Task 1.2: Parallel Cross-Validation ✅ DONE

### Current Code (model.py lines 278-356)

```python
for fold, (train_idx, test_idx) in enumerate(tscv.split(X), start=1):
    # ... gap prevention, validation, training, evaluation ...
    cv_model = build_model()
    cv_model.fit(X.iloc[train_idx_filtered], y.iloc[train_idx_filtered])
    preds = cv_model.predict(X.iloc[test_idx_filtered])
    acc = accuracy_score(y_test_fold, preds)
    cv_scores.append(acc)
```

### Issue

- CV folds run sequentially
- Each fold is independent and can run in parallel
- 5 folds = 5x potential speedup opportunity

### Solution

Use `concurrent.futures.ProcessPoolExecutor` to parallelize CV fold execution.

### Implementation Steps

#### Step 1: Create CV utility module

Create `modules/xgboost/utils/cv_parallel.py`:

```python
"""
Parallel cross-validation utilities for XGBoost module.
"""

import multiprocessing as mp
from concurrent.futures import ProcessPoolExecutor, as_completed
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score

from config import TARGET_HORIZON, TARGET_LABELS, ID_TO_LABEL


def _train_cv_fold(
    fold_num: int,
    train_idx: np.ndarray,
    test_idx: np.ndarray,
    X_values: np.ndarray,
    y_values: np.ndarray,
    feature_names: List[str],
    params: Dict[str, Any],
) -> Tuple[int, float, Optional[List[int]], Optional[List[int]], str]:
    """
    Train single CV fold (designed to run in separate process).
    
    Args:
        fold_num: Fold number for logging
        train_idx: Training indices
        test_idx: Test indices
        X_values: Feature values as numpy array (to avoid pickle issues)
        y_values: Target values as numpy array
        feature_names: List of feature names
        params: XGBoost parameters
    
    Returns:
        Tuple of (fold_num, accuracy, y_true_list, y_pred_list, message)
    """
    import xgboost as xgb
    
    # Apply gap to prevent data leakage
    train_idx_array = np.array(train_idx)
    if len(train_idx_array) <= TARGET_HORIZON:
        return (fold_num, 0.0, None, None, "Skipped (insufficient train data for gap)")
    
    train_idx_filtered = train_idx_array[:-TARGET_HORIZON]
    
    # Ensure test set doesn't overlap with gap
    test_idx_array = np.array(test_idx)
    if len(train_idx_filtered) == 0 or len(test_idx_array) == 0:
        return (fold_num, 0.0, None, None, "Skipped (no valid data)")
    
    min_test_start = train_idx_filtered[-1] + TARGET_HORIZON + 1
    if test_idx_array[0] < min_test_start:
        test_idx_filtered = test_idx_array[test_idx_array >= min_test_start]
        if len(test_idx_filtered) == 0:
            return (fold_num, 0.0, None, None, "Skipped (no valid test data after gap)")
    else:
        test_idx_filtered = test_idx_array
    
    # Class diversity validation
    y_train_fold = y_values[train_idx_filtered]
    unique_classes = sorted(np.unique(y_train_fold))
    
    if len(unique_classes) < 2:
        return (fold_num, 0.0, None, None, f"Skipped (insufficient class diversity: {unique_classes})")
    
    if len(unique_classes) < len(TARGET_LABELS):
        class_list = [ID_TO_LABEL[c] for c in unique_classes]
        return (fold_num, 0.0, None, None, f"Skipped (missing classes: expected {TARGET_LABELS}, got {class_list})")
    
    # Train model
    try:
        model = xgb.XGBClassifier(**params)
        
        # Use DataFrame for training to preserve feature names
        X_train = pd.DataFrame(X_values[train_idx_filtered], columns=feature_names)
        y_train = y_train_fold
        
        model.fit(X_train, y_train)
        
        # Evaluate
        X_test = pd.DataFrame(X_values[test_idx_filtered], columns=feature_names)
        y_test_fold = y_values[test_idx_filtered]
        
        preds = model.predict(X_test)
        acc = accuracy_score(y_test_fold, preds)
        
        message = f"Accuracy: {acc:.4f} (train: {len(train_idx_filtered)}, gap: {TARGET_HORIZON}, test: {len(test_idx_filtered)})"
        
        return (fold_num, acc, y_test_fold.tolist(), preds.tolist(), message)
        
    except Exception as e:
        return (fold_num, 0.0, None, None, f"Error: {str(e)}")


def run_parallel_cv(
    X: pd.DataFrame,
    y: pd.Series,
    tscv,
    params: Dict[str, Any],
    max_workers: Optional[int] = None,
) -> Tuple[List[float], List[int], List[int]]:
    """
    Run cross-validation folds in parallel.
    
    Args:
        X: Feature DataFrame
        y: Target Series
        tscv: TimeSeriesSplit object
        params: XGBoost parameters (will be filtered for pickle safety)
        max_workers: Maximum parallel workers (default: CPU count // 2)
    
    Returns:
        Tuple of (cv_scores, all_y_true, all_y_pred)
    """
    from modules.common.utils import log_model, log_warn
    
    # Prepare pickle-safe data
    X_values = X.values
    y_values = y.values
    feature_names = list(X.columns)
    
    # Filter params for pickle safety (remove non-serializable items)
    params_filtered = {k: v for k, v in params.items() if isinstance(v, (int, float, str, bool, type(None)))}
    
    # Prepare fold data
    fold_data = []
    for fold_num, (train_idx, test_idx) in enumerate(tscv.split(X), start=1):
        fold_data.append((fold_num, np.array(train_idx), np.array(test_idx)))
    
    # Determine workers
    if max_workers is None:
        max_workers = max(1, mp.cpu_count() // 2)
    
    # Run parallel CV
    cv_scores = []
    all_y_true = []
    all_y_pred = []
    
    # Use ProcessPoolExecutor for true parallelism
    with ProcessPoolExecutor(max_workers=max_workers) as executor:
        futures = {
            executor.submit(
                _train_cv_fold,
                fold_num,
                train_idx,
                test_idx,
                X_values,
                y_values,
                feature_names,
                params_filtered,
            ): fold_num
            for fold_num, train_idx, test_idx in fold_data
        }
        
        for future in as_completed(futures):
            fold_num, acc, y_true, y_pred, message = future.result()
            
            if acc > 0 and y_true is not None:
                cv_scores.append(acc)
                all_y_true.extend(y_true)
                all_y_pred.extend(y_pred)
                log_model(f"CV Fold {fold_num} {message}")
            else:
                log_warn(f"CV Fold {fold_num}: {message}")
    
    return cv_scores, all_y_true, all_y_pred
```

#### Step 2: Update model.py

Replace the sequential CV loop with parallel execution:

```python
# Add import at top
from modules.xgboost.utils.cv_parallel import run_parallel_cv

# Add config flag
XGBOOST_USE_PARALLEL_CV = True  # Set False to use sequential CV

# Replace lines 278-354 with:
max_splits = min(5, len(df) - 1)
if max_splits >= 2:
    tscv = TimeSeriesSplit(n_splits=max_splits)
    
    if XGBOOST_USE_PARALLEL_CV:
        # Parallel CV execution
        cv_scores, all_y_true, all_y_pred = run_parallel_cv(
            X, y, tscv, XGBOOST_PARAMS
        )
    else:
        # Sequential CV (original implementation)
        cv_scores = []
        all_y_true = []
        all_y_pred = []
        for fold, (train_idx, test_idx) in enumerate(tscv.split(X), start=1):
            # ... existing sequential code ...
    
    if len(cv_scores) > 0:
        mean_cv = sum(cv_scores) / len(cv_scores)
        log_success(f"CV Mean Accuracy ({len(cv_scores)} folds): {mean_cv:.4f}")
        
        if len(all_y_true) > 0 and len(all_y_pred) > 0:
            print_classification_report(
                np.array(all_y_true),
                np.array(all_y_pred),
                "Cross-Validation Aggregated Report (All Folds)",
            )
    else:
        log_warn("CV: No valid folds after applying gap.")
```

### Expected Result

- CV folds run in parallel on multiple CPU cores
- **Speedup**: 2-4x for 5-fold CV on 4+ core CPUs

### Test

```python
# Test parallel CV
import pandas as pd
from sklearn.model_selection import TimeSeriesSplit

# Create test data
df = pd.DataFrame({
    'feature1': range(1000),
    'feature2': range(1000),
    'Target': [i % 3 for i in range(1000)]
})

X = df[['feature1', 'feature2']]
y = df['Target']
tscv = TimeSeriesSplit(n_splits=5)

# Run parallel CV
from modules.xgboost.utils.cv_parallel import run_parallel_cv
cv_scores, y_true, y_pred = run_parallel_cv(X, y, tscv, {'n_estimators': 10})

print(f"CV Scores: {cv_scores}")
print(f"Mean: {sum(cv_scores)/len(cv_scores):.4f}")
```

---

## Task 1.3: Parallel Optuna Trials ✅ DONE

### Current Code (optimization.py lines 433-437)

```python
study.optimize(
    lambda trial: self._objective(trial, X, y, n_splits=n_splits),
    n_trials=n_trials,
    show_progress_bar=True,
)
```

### Issue

- Optuna trials run sequentially by default
- Each trial is independent
- Easily parallelizable with `n_jobs` parameter

### Solution

Add `n_jobs=-1` to use all available CPU cores for trials.

### Implementation Steps

#### Step 1: Add config flag

Add to `config/__init__.py` or `modules/xgboost/core/optimization.py`:

```python
# Configuration for parallel trials
OPTUNA_PARALLEL_TRIALS = True
OPTUNA_N_JOBS = -1  # -1 = use all CPU cores
```

#### Step 2: Update optimization.py

Modify the `study.optimize()` call:

```python
# Add at top of file
OPTUNA_PARALLEL_TRIALS = True
OPTUNA_N_JOBS = -1

# Replace lines 433-437 with:
optimize_kwargs = {
    "n_trials": n_trials,
    "show_progress_bar": True,
    "gc_after_trial": True,  # Prevent memory leaks in parallel execution
}

if OPTUNA_PARALLEL_TRIALS:
    optimize_kwargs["n_jobs"] = OPTUNA_N_JOBS

study.optimize(
    lambda trial: self._objective(trial, X, y, n_splits=n_splits),
    **optimize_kwargs
)
```

#### Step 3: Handle thread-safety in _objective

Update `_objective` method to be thread-safe:

```python
def _objective(
    self,
    trial: optuna.Trial,
    X: pd.DataFrame,
    y: pd.Series,
    n_splits: int = 5,
) -> float:
    # ... existing code ...
    
    # Ensure XGBClassifier uses single thread per trial when parallelizing trials
    params["n_jobs"] = 1  # Single thread per model when running parallel trials
    
    # ... rest of method ...
```

### Expected Result

- Optuna trials run in parallel across CPU cores
- **Speedup**: 2-8x for 100 trials on 4-8 core CPUs

### Test

```python
# Test parallel Optuna optimization
import pandas as pd
from modules.xgboost.core.optimization import HyperparameterTuner

# Create test data
df = pd.DataFrame({
    'feature1': range(500),
    'feature2': range(500),
    'Target': [i % 3 for i in range(500)]
})
df.columns = ['feature1', 'feature2', 'Target']

# Run optimization with timing
import time
tuner = HyperparameterTuner(symbol="TEST", timeframe="1h")

start = time.perf_counter()
best_params = tuner.optimize(df, n_trials=20, n_splits=3)
duration = time.perf_counter() - start

print(f"Optimization completed in {duration:.2f}s")
print(f"Best params: {best_params}")
```

---

## 📊 Verification Benchmarks

### Before Optimization Baseline

Run this before implementing changes to establish baseline:

```python
# benchmarks/benchmark_xgboost_baseline.py
import time
import pandas as pd
import numpy as np
from config import MODEL_FEATURES

# Generate test data
np.random.seed(42)
n_samples = 2000

df = pd.DataFrame({
    **{f: np.random.randn(n_samples) for f in MODEL_FEATURES},
    'close': np.cumsum(np.random.randn(n_samples)) + 100,
    'Target': np.random.randint(0, 3, n_samples),
})

# Benchmark labeling
from modules.xgboost.core.labeling import apply_directional_labels

times = []
for _ in range(5):
    test_df = df.copy()
    start = time.perf_counter()
    apply_directional_labels(test_df)
    times.append(time.perf_counter() - start)
print(f"Labeling: {sum(times)/len(times):.3f}s (mean of 5)")

# Benchmark training
from modules.xgboost.core.model import train_and_predict

labeled_df = apply_directional_labels(df.copy())
labeled_df = labeled_df.dropna(subset=['Target'])

times = []
for _ in range(3):
    test_df = labeled_df.copy()
    start = time.perf_counter()
    train_and_predict(test_df)
    times.append(time.perf_counter() - start)
print(f"Training: {sum(times)/len(times):.3f}s (mean of 3)")
```

### After Optimization Verification

Run after implementing changes:

```python
# benchmarks/benchmark_xgboost_optimized.py
# Same benchmark code as above
# Compare timing results
```

---

## 📋 Checklist

- [x] **Task 1.1**: GPU Detection Caching
  - [x] Create `modules/xgboost/utils/gpu_utils.py`
  - [x] Update `modules/xgboost/core/model.py` to use cached function
  - [x] Update `modules/xgboost/utils/__init__.py`
  - [x] Test GPU detection caching

- [x] **Task 1.2**: Parallel Cross-Validation
  - [x] Create `modules/xgboost/utils/cv_parallel.py`
  - [x] Update `modules/xgboost/core/model.py` with parallel CV option
  - [x] Add `XGBOOST_USE_PARALLEL_CV` config flag
  - [x] Test parallel CV execution

- [x] **Task 1.3**: Parallel Optuna Trials
  - [x] Add `OPTUNA_PARALLEL_TRIALS` and `OPTUNA_N_JOBS` config
  - [x] Update `modules/xgboost/core/optimization.py` with `n_jobs` parameter
  - [x] Update `_objective` to be thread-safe
  - [x] Test parallel optimization

- [x] **Verification**
  - [x] Run baseline benchmark before changes
  - [x] Run optimized benchmark after changes
  - [x] Document speedup achieved

---

**Status**: ✅ Completed
**Estimated Implementation Time**: 4-8 hours  
**Expected Speedup**: 2-8x (depending on CPU cores and GPU availability)
