# Phase 5 Completion Plan

## 1. Fix Unit Tests (Task 4)

### 1.1 Fix `tests/adaptive_trend_LTS/test_dask_scanner.py`

**Issue:**
1.  `test_process_single_symbol_dask_success` fails because `result["trend"]` is a numpy type (`np.int8`), but the test asserts `isinstance(..., (int, float))`.
2.  `test_progress_callback` and `test_progress_callback_logs` fail because `callback._posttask` is called with 4 arguments, but the implementation expects 5 (including `id`).

**Fix:**
Update `test_dask_scanner.py`:

```python
# In test_process_single_symbol_dask_success
assert isinstance(result["signal"], (int, float, np.number))
assert isinstance(result["trend"], (int, float, np.number))

# In test_progress_callback and test_progress_callback_logs
# Pass dummy id argument
callback._posttask("task_key", None, {}, {}, "dummy_id")
```

### 1.2 Fix `tests/adaptive_trend_LTS/test_dask_batch_processor.py`

**Issue:**
`test_process_partition_python_basic` fails because `compute_atc_signals` receives `hma_len` from `sample_config` fixture, but expects `hull_len`.

**Fix:**
Update `sample_config` fixture in `test_dask_batch_processor.py`:

```python
@pytest.fixture
def sample_config():
    return {
        "ema_len": 20,
        "hull_len": 20,  # Changed from hma_len
        "wma_len": 20,
        # ...
    }
```

## 2. Create Documentation (Task 5)

### 2.1 Create `modules/adaptive_trend_LTS/docs/dask_usage_guide.md`

Create a new file with the following content:

```markdown
# Dask Integration Guide for Adaptive Trend Classification (ATC)

This guide explains how to use the Dask-based features in the ATC system for out-of-core processing, large-scale batch scanning, and high-performance backtesting.

## Overview

The Dask integration allows the ATC system to:
1.  **Process unlimited datasets**: Handle datasets larger than RAM by streaming data in chunks.
2.  **Scale horizontally**: Utilize all CPU cores efficiently.
3.  **Hybrid Processing**: Combine Rust's raw speed with Dask's memory management.

---

## 1. Dask Scanner (For Live Scanning)

The Dask scanner is designed to process thousands of symbols efficiently.

### Usage

Use the `scan_all_symbols` function with `execution_mode="dask"`.

... (Rest of the content prepared previously)
```

### 2.2 Update `README.md`

Add a section about Dask features:

```markdown
## Dask Integration & Scalability

The system now supports **Out-of-Core Processing** via Dask, allowing for:
- Processing unlimited datasets (larger than RAM)
- Hybrid Rust + Dask execution for maximum performance
- Multi-file historical backtesting

See [Dask Usage Guide](modules/adaptive_trend_LTS/docs/dask_usage_guide.md) for details.
```

## 3. Update Task Tracker

Update `modules/adaptive_trend_LTS/docs/phase5_task.md`:

- Mark Task 4 (Testing & Validation) as `[x]`.
- Mark Task 5 (Documentation) as `[x]`.
- Mark Phase 5 as Completed? (Or wait for Rust Hybrid item).
  - The plan asked to "check and complete missing parts in task 4 and 5".
