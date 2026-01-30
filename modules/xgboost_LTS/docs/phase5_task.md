# Phase 5 Task: Batch & Distributed Processing for XGBoost Module

**Target**: Handle datasets >1GB and process 100+ symbols in parallel.
**Effort**: Medium-High
**Priority**: 🟢 LOW (Optional scaling)

---

## 🎯 Objectives

1.  **Dask Integration** - Enable "out-of-core" training for datasets that don't fit in RAM.
2.  **Parallel Symbol Processing** - Run full training/prediction pipelines for multiple symbols concurrently.
3.  **Resource Management** - Intelligent allocation of CPU/GPU resources in a distributed environment.

---

## Task 5.1: Dask Integration for Large Datasets

### Issue
- Large historical datasets (e.g., 5-min data over 5 years) can exceed 8GB-16GB RAM.
- Standard XGBoost requires the entire dataset to be in memory.

### Solution
Use `dask-ml` and `dask.dataframe` to partition data and train XGBoost in a distributed manner.

### Implementation Steps

1.  **Environment Setup**: Install `dask`, `distributed`, and `dask-ml`.
2.  **Data Loading**: Use `dd.from_pandas` or `dd.read_parquet` to handle data in chunks.
3.  **Distributed Training**: Use `dask_ml.xgboost.XGBClassifier`.

```python
import dask.dataframe as dd
from dask.distributed import Client
from dask_ml.xgboost import XGBClassifier

def train_distributed(df_list: list[pd.DataFrame]):
    client = Client() # Starts local cluster
    
    # Convert list of DataFrames to Dask DataFrame
    ddf = dd.from_delayed([dask.delayed(df) for df in df_list])
    
    X = ddf[MODEL_FEATURES]
    y = ddf["Target"]
    
    model = XGBClassifier(**XGBOOST_PARAMS)
    model.fit(X, y)
    
    return model
```

---

## Task 5.2: Parallel Symbol Processing (Batch Pipeline)

### Issue
- Training models for 20 symbols sequentially takes 20x longer than one symbol.
- Most systems have idle CPU cores during single-symbol training.

### Solution
Implement a multi-process wrapper to process symbols in parallel.

### Implementation Steps

1.  **Symbol Dispatcher**: Create a function that takes a list of symbols and a timeframe.
2.  **Process Pool**: Use `concurrent.futures.ProcessPoolExecutor` to distribute symbols.
3.  **Result Aggregation**: Collect predictions and metrics from each process.

```python
from concurrent.futures import ProcessPoolExecutor, as_completed

def process_symbol_pipeline(symbol, timeframe, config):
    # Fetch -> Features -> Label -> Train -> Store Result
    pass

def batch_process_all_symbols(symbols, timeframe, max_workers=None):
    with ProcessPoolExecutor(max_workers=max_workers) as executor:
        futures = {executor.submit(process_symbol_pipeline, s, timeframe): s for s in symbols}
        
        for future in as_completed(futures):
            symbol = futures[future]
            try:
                result = future.result()
                log_success(f"Completed {symbol}")
            except Exception as e:
                log_error(f"Failed {symbol}: {e}")
```

---

## 📊 Expected Scalability

| Mode | 1 Symbol | 50 Symbols | 1000 Symbols |
|-----------|-----------|-----------|----------------|
| Sequential | 30s | 25m | 8.3h |
| **Parallel (8 cores)** | 30s | **4m** | **1.2h** |

---

## 📋 Checklist

- [ ] **Task 5.1**: Dask Integration
    - [ ] Prototype dask-ml training notebook
    - [ ] Create `modules/xgboost_LTS/core/distributed.py`
    - [ ] Handle GPU multi-node training (optional)

- [ ] **Task 5.2**: Batch Symbol Processing
    - [ ] Implement `cli/batch_processor.py`
    - [ ] Add CLI flag `--batch` to `main.py`
    - [ ] Add progress bar (tqdm) for batch runs

---

**Status**: 📋 PROPOSAL
**Target Completion**: Future Expansion
