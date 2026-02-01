# Code Review: modules/adaptive_trend_LTS_mini/core/backtesting/dask_backtest.py

## Refactoring Status (Last Updated: 2026-02-01)

**Overall Score: 5.5/10** ⚠️ NOT PRODUCTION READY

- ❌ **3 CRITICAL issues to fix**
- ⚠️ **8 HIGH priority issues**
- 📝 **5 MEDIUM priority improvements needed**

**Status: REQUIRES SIGNIFICANT FIXES**

### Critical Issues (Must Fix)

1. ❌ **Module Import Path Inconsistency** - Imports from wrong module, will cause ImportError
2. ❌ **Aggressive gc.collect()** - Called after every symbol (2-5x performance penalty)
3. ❌ **Incorrect Meta Schema** - Line 69 creates wrong DataFrame structure

### High Priority Issues

4. ⚠️ **Silent Error Handling** - Empty DataFrames hide failures, impossible to debug
5. ⚠️ **90% Code Duplication** - Two main functions share nearly identical logic
6. ⚠️ **Hardcoded Parameter Filtering** - Brittle parameter list, breaks with signature changes
7. ⚠️ **Sequential File Processing** - Not leveraging Dask parallelism for multiple files
8. ⚠️ **No File Path Validation** - Security issue, unclear errors
9. ⚠️ **Memory-Mapped Data Copied** - Defeats purpose, no memory benefit
10. ⚠️ **No Progress Tracking** - Users can't monitor large backtests
11. ⚠️ **No Unit Tests** - Critical functionality untested

### Medium Priority Improvements

12. 📝 **Unused Parameters** - chunksize, partition_size parameters accepted but ignored
13. 📝 **Redundant Import** - Dask imported twice
14. 📝 **No Scheduler Configuration** - Can't use distributed scheduler
15. 📝 **No Result Aggregation** - Missing summary statistics
16. 📝 **Inconsistent Partition Logic** - Uses `/` then `//` inconsistently

---

## Overview

This module provides Dask-based distributed backtesting for the Adaptive Trend Classification (ATC) strategy on large historical datasets. It supports:

- CSV/Parquet file loading with chunking
- Memory-mapped data for low-memory environments
- Multi-file backtesting
- Groupby-apply parallelization per symbol

## Critical Issues

### 1. Module Import Path Inconsistency (Lines 93, 169)

```python
from modules.adaptive_trend_LTS.core.compute_atc_signals import compute_atc_signals
from modules.adaptive_trend_LTS.utils.memory_mapped_data import load_memory_mapped_from_csv
```

**Issue:** File is in `adaptive_trend_LTS_mini/` but imports from `adaptive_trend_LTS/`. This will cause ImportError at runtime.

**Fix:**

```python
from modules.adaptive_trend_LTS_mini.core.compute_atc_signals import compute_atc_signals
from modules.adaptive_trend_LTS_mini.utils.memory_mapped_data import load_memory_mapped_from_csv
```

### 2. Incorrect Meta Schema Definition (Lines 69, 194-199, 266-271)

```python
meta=pd.DataFrame({col: "float64" for col in descriptor.columns})  # Wrong!

meta = {
    "symbol": "string",
    "signal": "float64",
    "price": "float64",
    "timestamp": "datetime64[ns]",
}
```

**Issue:** Line 69 creates a DataFrame with column names as values, which is incorrect. The meta parameter expects either:

- A proper DataFrame with correct dtypes and column structure
- A dict mapping column names to dtypes

**Fix for line 69:**

```python
meta = {col: descriptor.dtypes.get(col, 'float64') for col in descriptor.columns}
ddf = dd.from_delayed(delayed_objs, meta=meta)
```

### 3. Inefficient Garbage Collection in Hot Path (Lines 201-204, 273-276)

```python
def process_with_gc(group_df: pd.DataFrame) -> pd.DataFrame:
    result = _process_symbol_group(group_df, symbol_column, price_column, atc_config)
    gc.collect()  # Forces full GC after every symbol!
    return result
```

**Issue:** Calling `gc.collect()` after processing each symbol is extremely inefficient:

- Full GC is expensive (can take 100s of milliseconds)
- Python's automatic GC is usually sufficient
- In Dask distributed mode, this happens in worker processes unnecessarily
- For 1000 symbols, this adds 100+ seconds of overhead

**Impact:** Massive performance degradation (potentially 2-5x slower).

**Fix:**

```python
# Remove the wrapper entirely or only use for specific memory-constrained scenarios
results_ddf = grouped.apply(
    _process_symbol_group,
    symbol_column=symbol_column,
    price_column=price_column,
    atc_config=atc_config,
    meta=meta,
    include_groups=False
)

# If memory is truly constrained, batch the GC calls:
def process_batch_with_gc(group_df: pd.DataFrame, counter=[0]) -> pd.DataFrame:
    result = _process_symbol_group(group_df, symbol_column, price_column, atc_config)
    counter[0] += 1
    if counter[0] % 100 == 0:  # GC every 100 symbols
        gc.collect()
    return result
```

### 4. Hardcoded Parameter Filtering (Lines 109-122)

```python
for param in [
    "limit",
    "batch_size",
    "use_memory_mapped",
    # ... 8 more hardcoded params
]:
    func_args.pop(param, None)
```

**Issues:**

- Brittle: breaks if `compute_atc_signals` signature changes
- Magic list: no explanation of which params belong where
- Not maintainable: adding new config params requires updates here

**Better Approach - Use function introspection:**

```python
import inspect

def _get_valid_function_params(func: Callable, config: dict) -> dict:
    """Extract only parameters that the function accepts."""
    sig = inspect.signature(func)
    valid_params = set(sig.parameters.keys())

    # Map config keys to function parameter names
    param_mapping = {
        "lambda_param": "La",
        "decay": "De",
    }

    func_args = {}
    for key, value in config.items():
        # Map renamed parameters
        param_name = param_mapping.get(key, key)
        # Only include if function accepts it
        if param_name in valid_params:
            func_args[param_name] = value

    return func_args

# Usage:
func_args = _get_valid_function_params(compute_atc_signals, atc_config)
result = compute_atc_signals(prices=prices, **func_args)
```

### 5. Silent Errors with Empty DataFrame Returns (Lines 99, 129, 139-141)

```python
if prices.empty or len(prices) < atc_config.get("ema_len", 28):
    return pd.DataFrame()  # Silent failure

if avg_signal.empty:
    return pd.DataFrame()  # Silent failure

except Exception as e:
    log_error(f"Error processing symbol group: {e}")
    return pd.DataFrame()  # Silent failure
```

**Issue:** Returning empty DataFrames makes it impossible to distinguish between:

- Symbols with no data
- Symbols with insufficient data
- Symbols that failed due to errors

**Impact:** Users can't debug which symbols failed or why.

**Better Approach:**

```python
def _process_symbol_group(
    group_df: pd.DataFrame,
    symbol_column: str,
    price_column: str,
    atc_config: dict,
) -> pd.DataFrame:
    """Process a single symbol's historical data.

    Returns:
        DataFrame with columns: symbol, signal, price, timestamp, status, error_msg
    """
    symbol = group_df[symbol_column].iloc[0] if not group_df.empty else "UNKNOWN"

    try:
        prices = group_df[price_column].sort_index()

        min_length = atc_config.get("ema_len", 28)

        if prices.empty:
            return _create_error_result(symbol, "NO_DATA", "Empty price series")

        if len(prices) < min_length:
            return _create_error_result(
                symbol,
                "INSUFFICIENT_DATA",
                f"Need {min_length} candles, got {len(prices)}"
            )

        # ... process normally

        return pd.DataFrame({
            "symbol": [symbol] * len(avg_signal),
            "signal": avg_signal.values,
            "price": prices.values,
            "timestamp": prices.index,
            "status": ["SUCCESS"] * len(avg_signal),
            "error_msg": [None] * len(avg_signal),
        })

    except Exception as e:
        log_error(f"Error processing {symbol}: {e}")
        import traceback
        log_error(traceback.format_exc())
        return _create_error_result(symbol, "COMPUTATION_ERROR", str(e))

def _create_error_result(symbol: str, status: str, error_msg: str) -> pd.DataFrame:
    """Create a single-row error result."""
    return pd.DataFrame({
        "symbol": [symbol],
        "signal": [None],
        "price": [None],
        "timestamp": [pd.NaT],
        "status": [status],
        "error_msg": [error_msg],
    })
```

## Code Quality Issues

### 6. Unused Parameter (Lines 32, 181)

```python
def _create_dask_from_memmap(
    # ...
    chunksize: str = "100MB",  # Never used!
) -> dd.DataFrame:
```

**Issue:** `chunksize` parameter is accepted but never used. The function uses hardcoded 100,000 rows per partition instead.

**Fix:** Either remove the parameter or use it:

```python
def _create_dask_from_memmap(
    mmap_array,
    descriptor,
    symbol_column: str,
    price_column: str,
    rows_per_partition: int = 100000,  # More accurate name
) -> dd.DataFrame:
    """Create a Dask DataFrame from memory-mapped data."""
    total_rows = descriptor.shape[0]
    num_partitions = max(1, total_rows // rows_per_partition)
    # ...
```

### 7. Redundant Dask Import (Line 260)

```python
import dask.dataframe as dd  # Already imported at top (line 9)
```

**Fix:** Remove redundant import.

### 8. Inconsistent Partition Calculation Logic (Lines 49-50)

```python
total_rows = descriptor.shape[0]
num_partitions = max(1, int(total_rows / 100000))
rows_per_partition = total_rows // num_partitions
```

**Issue:**

- Uses `/` (float division) for `num_partitions`, then `//` for `rows_per_partition`
- The `int()` cast is redundant if using `//`

**Improvement:**

```python
total_rows = descriptor.shape[0]
target_rows_per_partition = 100_000  # More readable
num_partitions = max(1, total_rows // target_rows_per_partition)
rows_per_partition = total_rows // num_partitions
```

### 9. Missing Validation in backtest_from_dataframe (Lines 224-293)

```python
def backtest_from_dataframe(
    df: pd.DataFrame,
    atc_config: dict,
    # ...
    partition_size: int = 10,  # Parameter never used!
```

**Issues:**

- `partition_size` parameter is documented but never used
- `npartitions` default is None, letting Dask decide, but `partition_size` suggests manual control

**Fix:**

```python
def backtest_from_dataframe(
    df: pd.DataFrame,
    atc_config: dict,
    symbol_column: str = "symbol",
    price_column: str = "close",
    npartitions: Optional[int] = None,
) -> pd.DataFrame:
    """Backtest ATC signals on an existing DataFrame using Dask.

    Args:
        npartitions: Number of Dask partitions. If None, auto-calculated
                     based on DataFrame size (1 partition per 100k rows)
    """
    if npartitions is None:
        # Auto-calculate based on size
        target_rows_per_partition = 100_000
        npartitions = max(1, len(df) // target_rows_per_partition)
        log_info(f"Auto-calculated {npartitions} partitions for {len(df)} rows")

    ddf = dd.from_pandas(df, npartitions=npartitions)
```

### 10. Duplicate Code in Two Functions (Lines 144-221 vs 224-293)

```python
def backtest_with_dask(...):
    # ... setup
    grouped = ddf.groupby(symbol_column)
    meta = { ... }
    def process_with_gc(...): ...
    results_ddf = grouped.apply(process_with_gc, meta=meta, include_groups=False)
    results_df = results_ddf.compute()
    # ... return

def backtest_from_dataframe(...):
    # ... setup
    grouped = ddf.groupby(symbol_column)
    meta = { ... }  # EXACT SAME
    def process_with_gc(...): ...  # EXACT SAME
    results_ddf = grouped.apply(process_with_gc, meta=meta, include_groups=False)  # EXACT SAME
    results_df = results_ddf.compute()  # EXACT SAME
    # ... return
```

**Issue:** 90% code duplication between the two functions.

**Refactor:**

```python
def _execute_dask_backtest(
    ddf: dd.DataFrame,
    atc_config: dict,
    symbol_column: str,
    price_column: str,
) -> pd.DataFrame:
    """Common execution logic for Dask backtesting."""
    grouped = ddf.groupby(symbol_column)

    meta = {
        "symbol": "string",
        "signal": "float64",
        "price": "float64",
        "timestamp": "datetime64[ns]",
    }

    try:
        results_ddf = grouped.apply(
            _process_symbol_group,
            symbol_column=symbol_column,
            price_column=price_column,
            atc_config=atc_config,
            meta=meta,
            include_groups=False
        )
        results_df = results_ddf.compute()
        log_info(f"Completed backtesting with {len(results_df)} records")
        return results_df
    except Exception as e:
        log_error(f"Error in Dask computation: {e}")
        return pd.DataFrame()

def backtest_with_dask(...) -> pd.DataFrame:
    # ... load data into ddf
    return _execute_dask_backtest(ddf, atc_config, symbol_column, price_column)

def backtest_from_dataframe(...) -> pd.DataFrame:
    # ... convert to ddf
    return _execute_dask_backtest(ddf, atc_config, symbol_column, price_column)
```

## Performance Issues

### 11. Sequential File Processing (Lines 296-343)

```python
def backtest_multiple_files_dask(file_paths: List[str], ...):
    results_list = []
    for file_path in file_paths:  # Sequential!
        result = backtest_with_dask(file_path, ...)
        if not result.empty:
            results_list.append(result)
```

**Issue:** Files are processed sequentially, not leveraging Dask's parallel capabilities.

**Better Approach:**

```python
def backtest_multiple_files_dask(
    file_paths: List[str],
    atc_config: dict,
    chunksize: str = "100MB",
    symbol_column: str = "symbol",
    price_column: str = "close",
) -> pd.DataFrame:
    """Backtest across multiple files in parallel."""

    if not file_paths:
        log_warn("No file paths provided")
        return pd.DataFrame()

    log_info(f"Backtesting {len(file_paths)} files in parallel")

    # Read all files into a single Dask DataFrame
    try:
        ddf = dd.read_csv(
            file_paths,  # Dask can read multiple files!
            blocksize=chunksize,
            dtype={symbol_column: "string", price_column: "float64"},
        )
    except Exception as e:
        log_error(f"Failed to read files: {e}")
        return pd.DataFrame()

    return _execute_dask_backtest(ddf, atc_config, symbol_column, price_column)
```

**Alternative - Use delayed for more control:**

```python
@dask.delayed
def process_single_file(file_path: str, atc_config: dict, **kwargs) -> pd.DataFrame:
    """Process a single file (delayed)."""
    return backtest_with_dask(file_path, atc_config, **kwargs)

# Create delayed tasks
delayed_results = [
    process_single_file(fp, atc_config, chunksize, symbol_column, price_column)
    for fp in file_paths
]

# Compute all in parallel
results_list = dask.compute(*delayed_results)
results_list = [r for r in results_list if not r.empty]

if results_list:
    return pd.concat(results_list, ignore_index=True)
return pd.DataFrame()
```

### 12. Memory-Mapped Data Not Properly Used (Lines 168-181)

```python
descriptor, mmap_array = load_memory_mapped_from_csv(historical_data_path, symbol_column, price_column)
# ...
ddf = _create_dask_from_memmap(mmap_array, descriptor, symbol_column, price_column, chunksize)
```

**Issue:** Memory-mapped arrays are great for reducing memory, but:

- The `_create_dask_from_memmap` function copies data with `np.array(partition_data[col])` (line 61), defeating the purpose
- No memory benefit vs. regular CSV loading

**Fix - Keep data memory-mapped:**

```python
def read_memmap_partition(partition_idx):
    start = partition_idx * rows_per_partition
    end = start + rows_per_partition if partition_idx < num_partitions - 1 else total_rows

    # Don't copy - keep as memory-mapped view
    partition_slice = slice(start, end)
    partition_data = mmap_array[partition_slice]

    # Create DataFrame directly from structured array view
    return pd.DataFrame(partition_data, index=range(start, end))
```

## Missing Features

### 13. No Progress Tracking

**Issue:** For large backtests (thousands of symbols, gigabytes of data), users have no visibility into progress.

**Suggestion:**

```python
from dask.diagnostics import ProgressBar

def backtest_with_dask(..., show_progress: bool = True) -> pd.DataFrame:
    # ...

    if show_progress:
        with ProgressBar():
            results_df = results_ddf.compute()
    else:
        results_df = results_ddf.compute()

    return results_df
```

### 14. No Dask Scheduler Configuration

**Issue:** Dask defaults to single-machine threaded scheduler. For large backtests, users might want distributed scheduler.

**Suggestion:**

```python
def backtest_with_dask(
    ...,
    scheduler: Optional[str] = None,  # 'threads', 'processes', or client address
) -> pd.DataFrame:
    """
    Args:
        scheduler: Dask scheduler to use. Options:
                   - None or 'threads': Threaded scheduler (default)
                   - 'processes': Process-based parallelism
                   - 'address': Connect to distributed scheduler
    """
    # ...

    if scheduler:
        results_df = results_ddf.compute(scheduler=scheduler)
    else:
        results_df = results_ddf.compute()
```

### 15. No Result Aggregation/Statistics

**Issue:** Function returns raw signal data but provides no summary statistics.

**Suggestion:**

```python
def backtest_with_dask(..., include_summary: bool = False) -> Union[pd.DataFrame, Tuple[pd.DataFrame, dict]]:
    """
    Returns:
        If include_summary=False: DataFrame with signals
        If include_summary=True: Tuple of (signals_df, summary_dict)
    """
    results_df = # ... compute

    if not include_summary:
        return results_df

    summary = {
        "total_symbols": results_df["symbol"].nunique(),
        "total_signals": len(results_df),
        "avg_signal_per_symbol": len(results_df) / results_df["symbol"].nunique(),
        "long_signals": (results_df["signal"] > 0).sum(),
        "short_signals": (results_df["signal"] < 0).sum(),
        "neutral_signals": (results_df["signal"] == 0).sum(),
    }

    return results_df, summary
```

## Security & Safety Issues

### 16. No File Path Validation

```python
def backtest_with_dask(historical_data_path: str, ...):
    log_info(f"Loading historical data from {historical_data_path}")
    ddf = dd.read_csv(historical_data_path, ...)
```

**Issue:** No validation of file paths. Could lead to:

- Path traversal vulnerabilities
- Attempting to read sensitive files
- Unclear error messages

**Suggestion:**

```python
from pathlib import Path

def backtest_with_dask(historical_data_path: str, ...) -> pd.DataFrame:
    """Backtest ATC signals on large historical data using Dask."""

    # Validate path
    try:
        path = Path(historical_data_path)
        if not path.exists():
            log_error(f"File does not exist: {historical_data_path}")
            return pd.DataFrame()

        if not path.is_file():
            log_error(f"Path is not a file: {historical_data_path}")
            return pd.DataFrame()

        # Check file extension
        valid_extensions = {'.csv', '.parquet', '.pq'}
        if path.suffix.lower() not in valid_extensions:
            log_warn(f"Unexpected file extension: {path.suffix}. Expected: {valid_extensions}")

    except Exception as e:
        log_error(f"Invalid file path: {e}")
        return pd.DataFrame()

    log_info(f"Loading historical data from {path.resolve()}")
    # ...
```

## Testing

**Missing Tests:**

- Unit tests for `_process_symbol_group` with various edge cases
- Integration tests with sample CSV files
- Performance benchmarks (Dask vs pandas)
- Memory usage tests (especially for memory-mapped path)
- Tests for error conditions (missing columns, corrupt data, etc.)

**Suggested Test Structure:**

```python
# tests/adaptive_trend_LTS_mini/test_dask_backtest.py

import pytest
import pandas as pd
from modules.adaptive_trend_LTS_mini.core.backtesting.dask_backtest import (
    backtest_from_dataframe,
    _process_symbol_group,
)

@pytest.fixture
def sample_historical_data():
    """Create sample historical data for testing."""
    return pd.DataFrame({
        "symbol": ["BTC/USDT"] * 100 + ["ETH/USDT"] * 100,
        "close": list(range(100)) + list(range(100, 200)),
        "timestamp": pd.date_range("2024-01-01", periods=200, freq="1h"),
    })

@pytest.fixture
def sample_atc_config():
    """Sample ATC configuration."""
    return {
        "ema_len": 28,
        "hma_len": 28,
        "robustness": 5,
        "lambda_param": 0.0004,
        "decay": 0.5,
    }

def test_backtest_from_dataframe_basic(sample_historical_data, sample_atc_config):
    """Test basic backtesting functionality."""
    result = backtest_from_dataframe(
        sample_historical_data,
        sample_atc_config,
        npartitions=2
    )

    assert not result.empty
    assert "symbol" in result.columns
    assert "signal" in result.columns

def test_backtest_empty_dataframe(sample_atc_config):
    """Test handling of empty DataFrame."""
    result = backtest_from_dataframe(pd.DataFrame(), sample_atc_config)
    assert result.empty

def test_backtest_missing_columns(sample_atc_config):
    """Test handling of missing required columns."""
    df = pd.DataFrame({"wrong_column": [1, 2, 3]})
    result = backtest_from_dataframe(df, sample_atc_config, symbol_column="symbol")
    assert result.empty

def test_process_symbol_group_insufficient_data(sample_atc_config):
    """Test handling of insufficient data."""
    small_df = pd.DataFrame({
        "symbol": ["BTC/USDT"] * 10,
        "close": range(10),
    })
    result = _process_symbol_group(small_df, "symbol", "close", sample_atc_config)
    # Should return empty or error result
    assert isinstance(result, pd.DataFrame)
```

## Project Convention Compliance

- ❌ Module import paths wrong (critical - will cause ImportError)
- ✅ Type hints present
- ✅ Docstrings present (but could be more detailed)
- ✅ Error logging with project utilities
- ⚠️ No comprehensive error handling
- ⚠️ Significant code duplication
- ❌ No unit tests visible

## Overall Assessment

### Score: 5.5/10

Significant issues, not production-ready.

**Strengths:**

- ✅ Good use of Dask for parallel processing
- ✅ Supports multiple data sources (CSV, memory-mapped, DataFrames)
- ✅ Proper use of `include_groups=False` to avoid FutureWarning
- ✅ Comprehensive docstrings

**Critical Issues:**

1. 🔴 Wrong module import paths - will cause immediate ImportError
2. 🔴 Aggressive `gc.collect()` - massive performance penalty (2-5x slower)
3. 🔴 Silent error handling - impossible to debug failures
4. 🟡 90% code duplication between two main functions
5. 🟡 Hardcoded parameter filtering - brittle and unmaintainable
6. 🟡 Incorrect meta schema (line 69)

**Missing:**

- ❌ No unit tests
- ❌ No progress tracking
- ❌ No result summaries/statistics
- ❌ No file path validation
- ❌ Sequential file processing (not leveraging Dask parallelism)

## Priority Fixes

1. **CRITICAL (Must fix immediately):**
   - Fix module import paths
   - Remove or dramatically reduce `gc.collect()` calls
   - Fix meta schema definition (line 69)
2. **HIGH (Required for production):**
   - Implement proper error tracking (don't return empty DataFrames silently)
   - Eliminate code duplication with `_execute_dask_backtest` helper
   - Add unit tests
   - Add file path validation
3. **MEDIUM (Improves usability):**
   - Use function introspection instead of hardcoded parameter lists
   - Add progress tracking
   - Parallelize multi-file backtesting
   - Add result summaries
4. **LOW (Polish):**
   - Fix memory-mapped data copying
   - Add scheduler configuration
   - Remove unused parameters

**Recommendation:** Do not deploy to production until critical issues are resolved. The performance penalty from `gc.collect()` alone makes this unusable for large-scale backtesting. Module import errors will prevent execution entirely.
