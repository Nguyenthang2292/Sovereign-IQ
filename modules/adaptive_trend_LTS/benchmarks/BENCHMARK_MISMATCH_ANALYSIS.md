# Benchmark Mismatch Analysis - 0% Match Rate Investigation

## Executive Summary

The benchmark results show **0% match rate** for certain implementation combinations:
- **Original vs Enhanced**: 0% match (Max diff: 0.514)
- **Original vs Rust+Dask**: 0% match (Max diff: 0.00)
- **Original vs CUDA+Dask**: 0% match (Max diff: 0.00)
- **Original vs All Three (Rust+CUDA+Dask)**: 0% match (Max diff: 0.00)

In contrast, individual implementations work well:
- **Original vs Rust**: 90% match (18/20 symbols)
- **Original vs CUDA**: 90% match (18/20 symbols)
- **Original vs Dask**: 90% match (18/20 symbols)

**Root Cause Identified**: Index loss in Dask bridge functions causing comparison failure.

---

## Problem Analysis

### 1. Enhanced Module Mismatch (0% match, diff: 0.514)

**Location**: `comparison.py:140-158`

```python
# Original vs Enhanced
if enh_s is not None and len(enh_s) > 0:
    min_len = min(len(orig_s), len(enh_s))
    if min_len > 0:
        orig_values = orig_s.values[:min_len]  # Get values by position
        enh_values = enh_s.values[:min_len]

        valid_mask = np.isfinite(orig_values) & np.isfinite(enh_values)
        if np.any(valid_mask):
            diff_oe = np.abs(orig_values[valid_mask] - enh_values[valid_mask]).max()
            if diff_oe < 1e-6:
                orig_enh_matching += 1
```

**Issue**: The comment at line 141 reveals the problem:
```python
# Note: Enhanced may reset index after cutout, Original keeps original index
# Compare by position (values) to handle index mismatch
```

**Explanation**:
- `adaptive_trend_enhance` module applies `cutout` parameter which removes the first N bars
- When removing first `cutout` bars, the index is reset to start from 0
- Original module preserves the original datetime index
- Even though values are compared positionally, **the index mismatch causes issues in the comparison logic**

**Evidence from Results**:
- Max difference: 0.514 (significant!)
- Avg difference: 0.354
- This is NOT just an index issue - the values themselves differ

**Likely Cause**: The `adaptive_trend_enhance` module has **different calculation logic** or parameter handling than `adaptive_trend`.

---

### 2. Rust+Dask / CUDA+Dask / All Three Mismatches (0% match, diff: 0.00)

**Location**: `rust_dask_bridge.py:77-79` and `rust_dask_bridge.py:131-133`

```python
# CPU Rust version
results = {}
for symbol, signal_array in batch_results.items():
    results[symbol] = {"Average_Signal": pd.Series(signal_array)}  # ❌ NO INDEX!
```

```python
# CUDA Rust version
results = {}
for symbol, signal_array in batch_results.items():
    results[symbol] = {"Average_Signal": pd.Series(signal_array)}  # ❌ NO INDEX!
```

**Critical Bug**: When converting numpy arrays back to pandas Series, **the original index is lost**.

**Comparison Code** (`comparison.py:216-224`):
```python
# Original vs Rust+Dask
if rust_dask_s is not None and len(rust_dask_s) > 0:
    common_idx = orig_s.index.intersection(rust_dask_s.index)  # ❌ NO COMMON INDEX!
    if len(common_idx) > 0:
        diff_ord = np.abs(orig_s.loc[common_idx] - rust_dask_s.loc[common_idx]).max()
        orig_rust_dask_diffs.append(diff_ord)
        if diff_ord < 1e-6:
            orig_rust_dask_matching += 1
```

**What Happens**:
1. Original series has datetime index: `DatetimeIndex(['2024-01-01', '2024-01-02', ...])`
2. Rust+Dask series has default integer index: `RangeIndex(0, 500, 1)`
3. `index.intersection()` returns **empty set** (no common indices)
4. `len(common_idx) == 0` → comparison is **skipped entirely**
5. No differences are recorded
6. Match count stays at 0

**Evidence**:
- Max difference: **0.00** (not 0.000001, literally 0.0)
- Avg difference: **0.00**
- Median difference: **0.00**
- This indicates **no comparison was performed** (not that values match perfectly)

---

## Why Individual Implementations Work

### Rust (90% match)

**Location**: `batch_processor.py:66-74`

```python
formatted_results = {}
for symbol, classified_array in batch_results.items():
    orig_series = symbols_data.get(symbol)
    if orig_series is not None and hasattr(orig_series, "index"):
        # ✅ PRESERVES INDEX from original input
        formatted_results[symbol] = {
            "Average_Signal": pd.Series(classified_array, index=orig_series.index)
        }
```

**Correct Behavior**: The batch processor retrieves the original Series and **reuses its index**.

### CUDA (90% match)

Uses the same `batch_processor.py` code path → same correct index preservation.

### Dask (90% match)

**Location**: `dask_batch_processor.py` (not shown, but similar pattern)

Likely preserves indices correctly through Dask operations.

---

## Library Conversion Error Context

User mentioned: **"84% of accepted parts are due to library conversion errors"**

This refers to:
1. **Pandas → NumPy conversion**: When passing data to Rust
2. **NumPy → Pandas conversion**: When returning results from Rust
3. **Index preservation issues**: During these conversions

The Rust bridge functions perform:
```python
# INPUT: pandas Series with datetime index
symbols_numpy[s] = v.values.astype(np.float64)  # ✅ Converts to numpy (loses index)

# RUST PROCESSING: Returns numpy array (no index info)

# OUTPUT: Convert back to Series
results[symbol] = {"Average_Signal": pd.Series(signal_array)}  # ❌ Creates new default index
```

**Missing Step**: Need to pass through and restore original index.

---

## Root Cause Summary

| Implementation | Match Rate | Root Cause |
|----------------|-----------|------------|
| **Enhanced** | 0% | Different calculation logic + cutout index reset |
| **Rust+Dask** | 0% | Index lost in `rust_dask_bridge.py:79` |
| **CUDA+Dask** | 0% | Index lost in `rust_dask_bridge.py:133` |
| **All Three** | 0% | Index lost in `rust_dask_bridge.py:133` (uses CUDA path) |

All Dask hybrid versions use `rust_dask_bridge.py` which fails to preserve indices.

---

## Detailed Fix Required

### Fix 1: Rust-Dask Bridge Index Preservation

**File**: `modules/adaptive_trend_LTS/core/compute_atc_signals/rust_dask_bridge.py`

**Lines to Fix**: 79, 133, 176

#### Current Code (BROKEN):
```python
def _process_partition_with_rust_cpu(
    partition_data: Dict[str, np.ndarray],
    config: dict,
) -> Dict[str, Dict[str, pd.Series]]:
    # ... Rust processing ...

    results = {}
    for symbol, signal_array in batch_results.items():
        results[symbol] = {"Average_Signal": pd.Series(signal_array)}  # ❌ NO INDEX

    return results
```

#### Fixed Code:
```python
def _process_partition_with_rust_cpu(
    partition_data: Dict[str, pd.Series],  # ✅ Accept Series, not ndarray
    config: dict,
) -> Dict[str, Dict[str, pd.Series]]:
    # ... Rust processing ...

    results = {}
    for symbol, signal_array in batch_results.items():
        # ✅ Restore original index from input data
        orig_series = partition_data.get(symbol)
        if orig_series is not None and hasattr(orig_series, "index"):
            results[symbol] = {
                "Average_Signal": pd.Series(signal_array, index=orig_series.index)
            }
        else:
            results[symbol] = {"Average_Signal": pd.Series(signal_array)}

    return results
```

**Apply same fix to:**
- `_process_partition_with_rust_cuda()` (line 133)
- `_process_partition_python()` (line 176)

### Fix 2: Update Function Signature

The function signature says `Dict[str, np.ndarray]` but needs to accept Series to preserve index:

```python
def _process_partition_with_rust_cpu(
    partition_data: Dict[str, pd.Series | np.ndarray],  # ✅ Union type
    config: dict,
) -> Dict[str, Dict[str, pd.Series]]:
```

### Fix 3: Enhanced Module Alignment

**File**: `modules/adaptive_trend_enhance/core/compute_atc_signals.py`

**Issue**: Different calculation logic or cutout handling causes 0.514 max difference.

**Investigation Needed**:
1. Compare cutout implementation between `adaptive_trend` and `adaptive_trend_enhance`
2. Check parameter scaling differences (La/De)
3. Verify MA calculation implementations match
4. Check if there are floating-point precision differences

---

## Testing Strategy

### Step 1: Verify Index Presence

Add debug logging to benchmark comparison:

```python
# In comparison.py, after line 132
if rust_dask_s is not None:
    print(f"DEBUG {symbol}:")
    print(f"  orig_s index type: {type(orig_s.index)}")
    print(f"  orig_s index sample: {orig_s.index[:5]}")
    print(f"  rust_dask_s index type: {type(rust_dask_s.index)}")
    print(f"  rust_dask_s index sample: {rust_dask_s.index[:5]}")
    print(f"  common_idx length: {len(common_idx)}")
```

**Expected Output (BEFORE FIX)**:
```
DEBUG BTC/USDT:
  orig_s index type: <class 'pandas.core.indexes.datetimes.DatetimeIndex'>
  orig_s index sample: DatetimeIndex(['2024-01-01', ...])
  rust_dask_s index type: <class 'pandas.core.indexes.range.RangeIndex'>
  rust_dask_s index sample: RangeIndex(start=0, stop=5, step=1)
  common_idx length: 0  ← NO COMMON INDICES!
```

**Expected Output (AFTER FIX)**:
```
DEBUG BTC/USDT:
  orig_s index type: <class 'pandas.core.indexes.datetimes.DatetimeIndex'>
  orig_s index sample: DatetimeIndex(['2024-01-01', ...])
  rust_dask_s index type: <class 'pandas.core.indexes.datetimes.DatetimeIndex'>
  rust_dask_s index sample: DatetimeIndex(['2024-01-01', ...])
  common_idx length: 500  ← INDICES MATCH!
```

### Step 2: Unit Test for Index Preservation

```python
def test_rust_dask_index_preservation():
    """Test that Rust-Dask bridge preserves pandas index."""
    import pandas as pd
    from modules.adaptive_trend_LTS.core.compute_atc_signals.rust_dask_bridge import (
        _process_partition_with_rust_cpu
    )

    # Create test data with datetime index
    dates = pd.date_range('2024-01-01', periods=100, freq='D')
    test_data = {
        'TEST/USDT': pd.Series(range(100, 200), index=dates)
    }

    config = {
        'ema_len': 28,
        'robustness': 'Medium',
        # ... other config ...
    }

    results = _process_partition_with_rust_cpu(test_data, config)

    # Verify index is preserved
    assert 'TEST/USDT' in results
    result_series = results['TEST/USDT']['Average_Signal']

    assert isinstance(result_series.index, pd.DatetimeIndex), \
        f"Expected DatetimeIndex, got {type(result_series.index)}"

    assert len(result_series.index) == len(dates), \
        f"Index length mismatch: {len(result_series.index)} vs {len(dates)}"

    assert (result_series.index == dates).all(), \
        "Index values don't match original"

    print("✅ Index preservation test PASSED")
```

### Step 3: Re-run Benchmark

After applying fixes:

```bash
cd modules/adaptive_trend_LTS/benchmarks/benchmark_comparison
python main.py --symbols 20 --bars 500 --timeframe 1h
```

**Expected Results (AFTER FIX)**:
```
Signal Comparison Table:
| Signal Comparison    | vs Enhanced | vs Rust | vs CUDA | vs Dask | vs Rust+Dask | vs CUDA+Dask | vs All Three |
|----------------------|-------------|---------|---------|---------|--------------|--------------|--------------|
| Match Rate           | 0.00%       | 90.00%  | 90.00%  | 90.00%  | 90.00% ✅    | 90.00% ✅    | 90.00% ✅    |
| Matching Symbols     | 0/20        | 18/20   | 18/20   | 18/20   | 18/20 ✅     | 18/20 ✅     | 18/20 ✅     |
| Max Difference       | 5.14e-01    | 3.29e-01| 3.29e-01| 3.29e-01| 3.29e-01 ✅  | 3.29e-01 ✅  | 3.29e-01 ✅  |
```

---

## Enhanced Module Investigation

The Enhanced module still needs investigation for the 0.514 max difference. Possible causes:

### 1. Cutout Parameter Handling

**Original** (`adaptive_trend`):
```python
# Preserves original index, just marks first cutout values as NaN
result["Average_Signal"][:cutout] = np.nan
```

**Enhanced** (`adaptive_trend_enhance`):
```python
# Removes first cutout bars entirely, resetting index
result["Average_Signal"] = result["Average_Signal"][cutout:].reset_index(drop=True)
```

### 2. Parameter Scaling Differences

Check if `La` and `De` parameters are scaled differently:

```python
# Original
la_scaled = la / 1000.0  # 0.02 → 0.00002

# Enhanced (possible bug)
la_scaled = la / 100.0   # 0.02 → 0.0002  ← 10x different!
```

### 3. Floating Point Precision

Check if calculations use different precision:
- Original: `np.float64`
- Enhanced: `np.float32` (lower precision)

---

## Conclusion

**Primary Issue**: Index loss in `rust_dask_bridge.py` causing 0% match for all Dask hybrid versions.

**Secondary Issue**: Different calculation logic in `adaptive_trend_enhance` causing significant value differences.

**Fix Priority**:
1. ✅ **HIGH**: Fix index preservation in `rust_dask_bridge.py` (lines 79, 133, 176)
2. ✅ **MEDIUM**: Investigate Enhanced module differences
3. ✅ **LOW**: Add comprehensive tests for index preservation

**Impact After Fix**:
- Rust+Dask match rate: **0% → 90%**
- CUDA+Dask match rate: **0% → 90%**
- All Three match rate: **0% → 90%**
- Enhanced match rate: **Requires further investigation**

---

## Additional Notes

### Why Median Difference is 0.00

For the 0% match implementations, even though Max Difference is 0.00, this doesn't mean values match. It means:

```python
if len(common_idx) > 0:  # ← This condition is FALSE
    diff_ord = np.abs(orig_s.loc[common_idx] - rust_dask_s.loc[common_idx]).max()
    orig_rust_dask_diffs.append(diff_ord)  # ← NEVER EXECUTED
```

Since no differences are ever appended to the list:
- `orig_rust_dask_diffs = []` (empty list)
- `max(orig_rust_dask_diffs)` = 0.0 (default for empty list)
- No actual comparison occurred!

This is a **silent failure** - the benchmark reports "0.00" difference but actually performed no comparison at all.

---

## Recommendations

1. **Immediate Action**: Apply index preservation fix to `rust_dask_bridge.py`
2. **Testing**: Run comprehensive benchmark suite after fix
3. **Investigation**: Deep dive into Enhanced module differences
4. **Documentation**: Update benchmark documentation to explain index requirements
5. **Monitoring**: Add assertions to detect index mismatches early

---

## References

- Benchmark Results: `modules/adaptive_trend_LTS/benchmarks/benchmark_results.txt`
- Comparison Code: `modules/adaptive_trend_LTS/benchmarks/benchmark_comparison/comparison.py`
- Rust-Dask Bridge: `modules/adaptive_trend_LTS/core/compute_atc_signals/rust_dask_bridge.py`
- Batch Processor: `modules/adaptive_trend_LTS/core/compute_atc_signals/batch_processor.py`
