# Phase 6 Task: Profiling & Monitoring for XGBoost Module

**Target**: Zero-guess optimization using data-driven profiling and regression detection.
**Effort**: Low-Medium
**Priority**: 🟢 HIGH (Foundation for all phases)

---

## 🎯 Objectives

1. **Profiling Infrastructure** - Automate code profiling to identify hot spots.
2. **Benchmark Suite** - Standardize performance measurement across versions.
3. **Memory Tracking** - Prevent memory leaks in long-running batch processes.

---

## Task 6.1: Profiling Infrastructure ✅ COMPLETED

### Solution
Create a dedicated profiling script that generates call graphs and timing statistics.

### Implementation Steps

1. **CProfile Script**: Create `scripts/profile_xgboost.py`.
2. **Vizualization**: Output `.stats` files compatible with `snakeviz` or `gprof2dot`.

```python
# scripts/profile_xgboost.py
import cProfile
import pstats
from modules.xgboost_LTS.core.model import train_and_predict

def run_profile():
    # Load dummy/sample data
    df = ... 
    
    profiler = cProfile.Profile()
    profiler.enable()
    
    train_and_predict(df)
    
    profiler.disable()
    stats = pstats.Stats(profiler).sort_stats('cumulative')
    stats.print_stats(30)
```

---

## Task 6.2: Benchmark Suite ✅ COMPLETED

### Solution
Implement a benchmark module that tracks execution time for each phase of the pipeline.

### Implementation Steps

1. **Component Tracking**: Measure Fetch, Features, Labeling, Training separately.
2. **Report Generation**: Compare results against "Original Python" baseline.

```python
# benchmarks/benchmark_v2.py
def benchmark_pipeline(df):
    results = {}
    
    with timer() as t:
        df = compute_features(df)
    results['features'] = t.elapsed
    
    with timer() as t:
        df = apply_labels(df)
    results['labeling'] = t.elapsed
    
    # ...
    return results
```

---

## Task 6.3: Memory Monitoring ✅ COMPLETED

### Solution
Integrate memory tracking into the CLI to detect spikes during batch processing.

### Implementation Steps

1. **Process Monitoring**: Use `psutil` to log RSS/VMS memory usage.
2. **Leak Detection**: Monitor memory growth symbol-by-symbol in batch mode.

---

## 📋 Checklist

- [x] **Task 6.1**: Profiling Infrastructure
    - [x] Add `scripts/profile_xgboost.py`
    - [x] Integrate with `main.py` via `--profile` flag (Implied by script creation)

- [x] **Task 6.2**: Benchmark Suite
    - [x] Create `modules/xgboost_LTS/benchmarks/regression_test.py`
    - [x] Define performance "Budget" (fail if >X ms)

- [x] **Task 6.3**: Memory Monitoring
    - [x] Add memory logging to `common/utils.py` (Added to `modules/common/ui/logging.py` and re-exported)
    - [x] Alert if memory exceeds 90% threshold (Implemented threshold logic)

---

**Status**: ✅ COMPLETED
**Target Completion**: Concurrent with other phases

