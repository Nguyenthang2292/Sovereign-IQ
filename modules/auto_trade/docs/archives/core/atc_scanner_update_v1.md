# Phân Tích Tối Ưu Hóa Rust cho ATC Scanner

**Module**: `modules/auto_trade/core/atc_scanner.py`  
**Ngày phân tích**: 2026-02-01  
**Mục đích**: Xác định các thành phần có thể viết lại bằng Rust để tăng hiệu năng

---

## Tổng Quan

Module `ATCScanner` hiện đang được viết hoàn toàn bằng Python với các tính năng:

- Parallel scanning với ThreadPoolExecutor
- Weighted voting aggregation
- Caching với TTL
- Batch processing
- Signal strength calculation

---

## 🟢 Các Phần NÊN Viết Lại Bằng Rust

### 1. ⭐ **Signal Aggregation Logic** (Ưu tiên cao)

**Vị trí**: Lines 234-274 trong `scan_symbols()`

**Lý do**:

- ✅ **Tính toán số học nặng**: Nhiều phép tính với float, loop qua symbols và timeframes
- ✅ **Không có I/O**: Pure computation, không cần async
- ✅ **Hot path**: Chạy cho mọi symbol trong mọi lần scan
- ✅ **Predictable data structures**: Dict, Set, List - dễ map sang Rust

**Hiệu năng dự kiến**:

- ⚡ **10-50x nhanh hơn** cho 100-500 symbols
- 🧠 **Giảm 30-50% memory** nhờ zero-copy và efficient data structures

**Implementation approach**:

```rust
// Rust extension module
#[pyfunction]
fn aggregate_signals(
    symbols: Vec<String>,
    results_by_tf: HashMap<String, ScanResult>,
    weights: HashMap<String, f64>,
    threshold: f64,
    use_signal_strength: bool,
) -> Vec<SignalResult> {
    // Fast aggregation with SIMD potential
}
```

**Độ khó**: 🟡 Trung bình (cần PyO3 binding)

---

### 2. ⭐ **Cache Management** (Ưu tiên cao)

**Vị trí**: Lines 190-247 (`_get_cache_key`, `_get_cached_result`, `_set_cache`)

**Lý do**:

- ✅ **Dictionary operations**: Rust HashMap nhanh hơn Python dict 3-5x
- ✅ **TTL checking**: Tính toán thời gian liên tục
- ✅ **LRU eviction**: Sorting và deletion - Rust efficient hơn
- ✅ **Thread-safe**: Rust có Arc, RwLock tốt hơn Python dict

**Hiệu năng dự kiến**:

- ⚡ **3-5x nhanh hơn** cho cache operations
- 🧠 **20-30% ít memory hơn** với compact data structures
- 🔒 **Better concurrency** với lock-free structures

**Implementation approach**:

```rust
use std::collections::HashMap;
use std::sync::{Arc, RwLock};
use lru::LruCache;

pub struct ScanCache {
    cache: Arc<RwLock<LruCache<String, (CachedData, f64)>>>,
    ttl_seconds: u64,
}
```

**Độ khó**: 🟢 Dễ (straightforward data structure conversion)

---

### 3. ⭐⭐ **Weighted Score Calculation** (Ưu tiên rất cao)

**Vị trí**: Lines 131-161 (`_calculate_weighted_score`)

**Lý do**:

- ✅ **Pure math**: Không có side effects
- ✅ **Called frequently**: Mỗi symbol × mỗi timeframe
- ✅ **SIMD potential**: Vector operations có thể dùng SIMD
- ✅ **No Python overhead**: Tránh GIL và function call overhead

**Hiệu năng dự kiến**:

- ⚡ **50-100x nhanh hơn** với SIMD vectorization
- 🧠 **Minimal memory**: Stack-allocated, no heap

**Implementation approach**:

```rust
#[inline]
pub fn calculate_weighted_score(
    signal_type: &str,
    tf_weight: f64,
    strength: f64,
    use_signal_strength: bool,
) -> f64 {
    // Inline, zero-cost abstraction
}
```

**Độ khó**: 🟢 Rất dễ (simple function)

---

### 4. 🟡 **Batch Processing Logic** (Ưu tiên trung bình)

**Vị trí**: Lines 254-286 (`_scan_symbols_batched`)

**Lý do**:

- ✅ **List slicing**: Rust slices zero-copy
- ✅ **Iterator composition**: Rust iterators efficient
- ⚠️ **Nhưng**: Phần này chủ yếu là orchestration, không phải computation

**Hiệu năng dự kiến**:

- ⚡ **2-3x nhanh hơn** cho list operations
- 🧠 **Zero-copy slicing** giảm memory allocation

**Implementation approach**:

```rust
pub fn batch_process<F>(
    symbols: Vec<String>,
    batch_size: usize,
    process_fn: F,
) -> Vec<SignalResult>
where
    F: Fn(&[String]) -> Vec<SignalResult>,
{
    symbols.chunks(batch_size)
        .flat_map(process_fn)
        .collect()
}
```

**Độ khó**: 🟢 Dễ (iterator patterns)

---

## 🔴 Các Phần KHÔNG NÊN Viết Lại Bằng Rust

### 1. **ThreadPoolExecutor Orchestration**

**Vị trí**: Lines 288-340 (parallel scanning with futures)

**Lý do**:

- ❌ **I/O bound**: Chờ network/disk, không phải CPU
- ❌ **Python integration**: Gọi `scan_all_symbols` (Python function)
- ❌ **Complexity**: Async/await coordination phức tạp với PyO3
- ❌ **Marginal benefit**: ThreadPoolExecutor đã tốt cho I/O

**Kết luận**: Giữ nguyên Python

---

### 2. **Configuration và Initialization**

**Vị trí**: Lines 60-104 (`__init__`, `_validate_weights`)

**Lý do**:

- ❌ **Run once**: Chỉ chạy khi khởi tạo
- ❌ **Python integration**: Truy cập config dict, logging
- ❌ **Error handling**: Python exceptions dễ hơn

**Kết luận**: Giữ nguyên Python

---

### 3. **DataFrame Operations**

**Vị trí**: Lines 413-437 (DataFrame reconstruction trong `_run_single_scan`)

**Lý do**:

- ❌ **Pandas dependency**: Cần tạo pd.DataFrame
- ❌ **PyO3 complexity**: Convert giữa Rust và Pandas phức tạp
- ⚠️ **Có thể dùng Polars**: Nếu muốn toàn bộ Rust, cân nhắc Polars

**Kết luận**: Giữ nguyên Python (hoặc migrate toàn bộ sang Polars)

---

## 📊 Kiến Trúc Hybrid Đề Xuất

### Phương Án 1: **Incremental Optimization** (Khuyến nghị)

```
[Python ATCScanner]
  ├── Configuration & Validation (Python) ✅
  ├── ThreadPoolExecutor (Python) ✅
  │   └── _run_single_scan (Python) ✅
  │       └── scan_all_symbols (Python) ✅
  │
  └── Performance-critical components → Rust
      ├── aggregate_signals() → Rust ⚡
      ├── calculate_weighted_score() → Rust ⚡
      └── ScanCache → Rust ⚡
```

**Ưu điểm**:

- ✅ Giữ được API Python hiện tại
- ✅ Tối ưu chỉ các phần hot path
- ✅ Dễ test và maintain
- ✅ Không breaking changes

**Hiệu năng dự kiến tổng thể**: **5-20x nhanh hơn** cho aggregation logic

---

### Phương Án 2: **Full Rust Rewrite** (Tích cực nhưng rủi ro)

```
[Rust ATCScanner] (PyO3 wrapper)
  ├── All logic in Rust
  ├── Parallel scanning với Tokio/Rayon
  ├── Polars thay vì Pandas
  └── Expose Python API với PyO3
```

**Ưu điểm**:

- ✅ **10-100x nhanh hơn** toàn bộ pipeline
- ✅ Memory safety
- ✅ Concurrency tốt hơn

**Nhược điểm**:

- ❌ Công sức lớn
- ❌ Khó maintain hơn
- ❌ Breaking changes
- ❌ Cần rewrite tests

---

## 🎯 Khuyến Nghị Ưu Tiên

### Phase 1: **Quick Wins** (1-2 tuần)

1. ✅ Viết Rust extension cho `calculate_weighted_score()`
2. ✅ Benchmark và verify 50-100x improvement
3. ✅ Integrate với PyO3

**Expected impact**: 20-30% faster cho small-medium workloads

---

### Phase 2: **High-Impact Optimization** (2-3 tuần)

1. ✅ Viết `aggregate_signals()` trong Rust
2. ✅ Thread-safe `ScanCache` với Rust
3. ✅ Benchmark với 100-500 symbols

**Expected impact**: 5-10x faster cho large workloads

---

### Phase 3: **Complete Optimization** (1-2 tháng) - Optional

1. ⚠️ Evaluate Polars migration
2. ⚠️ Rewrite parallel scanning với Rayon
3. ⚠️ Full Rust core với Python wrapper

**Expected impact**: 10-50x faster toàn bộ pipeline

---

## 🔧 Implementation Roadmap

### Bước 1: Setup Rust Extension Module

```bash
# Tạo Rust extension
cd modules/auto_trade/core
cargo init --lib rust_scanner
cd rust_scanner
cargo add pyo3
```

### Bước 2: Implement First Function

```toml
# Cargo.toml
[lib]
name = "rust_scanner"
crate-type = ["cdylib"]

[dependencies]
pyo3 = { version = "0.20", features = ["extension-module"] }
```

```rust
// src/lib.rs
use pyo3::prelude::*;

#[pyfunction]
fn calculate_weighted_score(
    signal_type: &str,
    tf_weight: f64,
    strength: f64,
    use_signal_strength: bool,
) -> f64 {
    match signal_type {
        "LONG" => {
            if use_signal_strength {
                tf_weight * strength.abs()
            } else {
                tf_weight
            }
        }
        "SHORT" => {
            if use_signal_strength {
                tf_weight * strength
            } else {
                -tf_weight
            }
        }
        _ => 0.0,
    }
}

#[pymodule]
fn rust_scanner(_py: Python, m: &PyModule) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(calculate_weighted_score, m)?)?;
    Ok(())
}
```

### Bước 3: Build và Test

```bash
cd rust_scanner
maturin develop --release
```

```python
# Test trong Python
import rust_scanner

score = rust_scanner.calculate_weighted_score("LONG", 0.5, 0.8, True)
print(score)  # 0.4
```

### Bước 4: Integrate vào Module

```python
# atc_scanner.py
try:
    from .rust_scanner import calculate_weighted_score as rust_calc_score
    USE_RUST = True
except ImportError:
    USE_RUST = False

def _calculate_weighted_score(self, signal_type, tf_weight, strength):
    if USE_RUST:
        return rust_calc_score(signal_type, tf_weight, strength, self.use_signal_strength)
    else:
        # Fallback to Python implementation
        ...
```

---

## 📈 Hiệu Năng Dự Kiến

| Component | Current (Python) | With Rust | Speedup | Difficulty |
|-----------|-----------------|-----------|---------|------------|
| `calculate_weighted_score()` | ~1µs | ~10ns | **100x** | 🟢 Dễ |
| `aggregate_signals()` | ~5ms (100 symbols) | ~200µs | **25x** | 🟡 TB |
| `ScanCache` operations | ~100µs | ~20µs | **5x** | 🟢 Dễ |
| Batch processing | ~50ms (500 symbols) | ~20ms | **2.5x** | 🟢 Dễ |
| **Overall pipeline** | **100ms** | **20-30ms** | **3-5x** | - |

---

## ✅ Kết Luận

**Những phần NÊN viết lại bằng Rust**:

1. ⭐⭐ `_calculate_weighted_score()` - Pure math, 100x speedup
2. ⭐⭐ `aggregate_signals()` logic - Hot path, 25x speedup  
3. ⭐ Cache management - Thread-safe, 5x speedup
4. 🟡 Batch processing - Nice to have, 2.5x speedup

**Những phần KHÔNG NÊN viết lại**:

- ❌ ThreadPoolExecutor orchestration (I/O bound)
- ❌ Configuration/validation (run once)
- ❌ DataFrame operations (Pandas integration phức tạp)

**Chiến lược khuyến nghị**: **Incremental optimization** với Rust extensions cho các hàm hot path, giữ nguyên orchestration logic bằng Python.


---

# ATC Scanner Hybrid + Polars: Conflict Analysis & Improvements

## Executive Summary

The migration strategy to Polars + Hybrid Rust architecture has **6 critical conflicts** and **9 high-priority improvements** needed. The biggest issue is that the migration only touches `atc_scanner.py` but the Pandas DataFrames propagate through 5+ interconnected modules.

---

## Critical Conflicts Identified

### 1. **Downstream Module Dependencies** ⚠️ BLOCKER

**Issue**: The strategy only migrates `atc_scanner.py` but doesn't account for downstream consumers.

**Affected Files**:
- `signal_pipeline.py` (lines 29-42): Imports and uses `ATCScanner`, passes results to XGBoostFilter
- `xgboost_filter.py` (lines 36, 64-68): Expects `List[SignalResult]` but processes data internally with pandas
- `signal_selector.py` (lines 31, 99-141): Receives `List[SignalResult]` from XGBoostFilter
- `persistence.py` (lines 115-187): Saves signals but doesn't interact with DataFrames directly (OK)

**Current Data Flow**:
```
ATCScanner (pandas) → SignalResult → XGBoostFilter (pandas) → SignalResult →
SignalSelector → FinalSignal → SignalPersistence
```

**Proposed Flow (Task 5)**:
```
ATCScanner (polars) → SignalResult → XGBoostFilter (pandas?) → ...
```

**Problem**: Task #5 converts pandas→polars at boundary but doesn't specify:
- Where exactly is the boundary?
- Does `SignalResult` contain DataFrame references? (No, it's a NamedTuple)
- Will downstream modules handle the change? (They use `SignalResult`, not raw DataFrames)

**Resolution**: ✅ **Conflict is MINIMAL**
- `SignalResult` is a NamedTuple (atc_scanner.py:44-51) with no DataFrame fields
- The pandas→polars conversion is internal to `atc_scanner.py`
- Downstream modules receive `List[SignalResult]` which is DataFrame-agnostic

**Action**: Update Task #5 to clarify: "Convert internal DataFrames to Polars; `SignalResult` API remains unchanged"

---

### 2. **Upstream Dependency: `scan_all_symbols` Returns Pandas** ⚠️ MEDIUM

**Issue**: Task #5 notes that `scan_all_symbols` (from `adaptive_trend_LTS_mini`) returns `pd.DataFrame`, requiring conversion at boundary.

**Current Code** (atc_scanner.py:394-476):
```python
def _run_single_scan(self, symbols: List[str], timeframe: str) -> Tuple[pd.DataFrame, pd.DataFrame]:
    # ...
    long_signals, short_signals = scan_all_symbols(...)  # Returns pandas
    return long_signals, short_signals  # pandas DataFrames
```

**Task #5 Plan**:
```python
long_signals = pl.from_pandas(long_pd)
short_signals = pl.from_pandas(short_pd)
```

**Conflict**: This conversion happens in `_run_single_scan`, but:
1. The function signature still shows `Tuple[pd.DataFrame, pd.DataFrame]` (line 394)
2. Task #2 says to change type hint to `Tuple[pl.DataFrame, pl.DataFrame]`
3. Cache reconstruction (Task #4, lines 409-420) uses `pd.DataFrame(...)`

**Action**:
- Task #2 and #5 must be coordinated: change signature AND convert result
- Ensure cache paths (hit + miss) both return Polars DataFrames

---

### 3. **Caching Module Conflict** ⚠️ HIGH PRIORITY

**Issue**: `caching.py` has no DataFrame-specific logic BUT:

**atc_scanner.py cache usage** (lines 86, 206-244, 405-420):
```python
# Cache stores Dict[str, Dict[str, Any]] - OK, no DataFrames
self._cache: Dict[str, Tuple[Dict[str, Dict[str, Any]], float]] = {}

# Cache reconstruction returns pandas DataFrames (Task #4 conflict)
return pd.DataFrame(longs_data), pd.DataFrame(shorts_data)  # Line 420
```

**Related Issue**: `caching.py` itself has critical thread-safety bug (see separate review):
- Claims to be thread-safe (line 11) but lacks locking
- Used in `atc_scanner.py` line 86 and `signal_pipeline.py` line 77

**Actions**:
1. Task #4: Change cache reconstruction to use Polars: `pl.DataFrame(longs_data)`
2. **CRITICAL**: Fix thread-safety in `caching.py` BEFORE Polars migration (separate from this plan)
3. Consider replacing `caching.py` with Rust LRU cache (Task #11) for thread-safety

---

### 4. **Empty DataFrame Schema Mismatch** ⚠️ MEDIUM

**Issue**: Task #3 defines empty schema but doesn't specify ALL required columns.

**Current Code** (atc_scanner.py:476):
```python
return pd.DataFrame(), pd.DataFrame()  # Empty on error
```

**Task #3 Proposal**:
```python
_EMPTY_DF_SCHEMA = {"symbol": pl.Utf8, "signal": pl.Float64}
```

**Problem**: `_scan_symbols_internal` (lines 324-338) expects these columns:
- `symbol` ✅
- `signal` ✅
- But also checks `"signal" in longs.columns` (line 327) - this works

**Verification Needed**: Do empty DataFrames cause issues in:
- Line 325: `if not longs.empty` → Polars: `if not longs.is_empty()`
- Lines 327-330: Column access with fallback (works with empty DF)

**Action**: Task #3 schema is sufficient. Update Task #6 to ensure empty DataFrame creation uses schema.

---

### 5. **Test Suite Pandas Dependencies** ⚠️ MEDIUM-HIGH

**Issue**: Task #8 says "update tests" but doesn't specify extent of changes needed.

**Affected Test Files**:
```
tests/auto_trade/core/test_atc_scanner.py
tests/auto_trade/core/test_atc_scanner_enhancements.py
tests/auto_trade/core/test_signal_pipeline.py
tests/auto_trade/core/test_xgboost_filter.py (may create pandas fixtures)
```

**Potential Issues**:
1. Tests that mock `scan_all_symbols` to return pandas DataFrames (OK - still valid)
2. Tests that assert on DataFrame properties using pandas API (needs update)
3. Tests that inspect internal `_run_single_scan` return type (needs update)

**Action**: Expand Task #8:
```markdown
**8. Update Tests**
- Update `test_atc_scanner.py`:
  - Replace `assert df.empty` with `assert df.is_empty()`
  - Replace pandas column access patterns with Polars equivalents
  - Update type assertions if checking DataFrame types
- Verify `test_signal_pipeline.py` still passes (uses SignalResult, not DataFrames)
- Check `test_atc_scanner_enhancements.py` for pandas-specific logic
- Run full test suite: `pytest tests/auto_trade/core/test_atc_scanner*.py -v`
```

---

### 6. **Rust Integration Phase Conflicts** ⚠️ MEDIUM

**Issue**: Tasks #9-13 (Rust implementation) are marked optional/low-priority but the document title claims this is a Hybrid architecture migration.

**Architectural Ambiguity**:
- Is this migration **Phase 1** (Polars only, Rust later)?
- Or **Full Hybrid** (Polars + Rust together)?

**Current Task Priority**:
- Tasks 1-8: Polars migration (required)
- Tasks 9-13: Rust hot path (optional, Task #11 marked optional)

**Conflict with `atc_scanner_update_v1.md`**:
The referenced document (`atc_scanner_update_v1.md`) likely specifies a full Hybrid approach, but this checklist treats Rust as optional.

**Action**: Clarify in document intro:
```markdown
## Migration Phases

**Phase 1 (This Document)**: Polars Migration
- Tasks 1-8: Replace Pandas with Polars
- Deliverable: atc_scanner.py uses Polars internally, API unchanged

**Phase 2 (Future)**: Rust Hot Path
- Tasks 9-13: Implement performance-critical functions in Rust
- Requires: Polars migration complete, Rust benchmarks showing >2x speedup
```

---

## High-Priority Improvements

### 1. **Add Rollback Strategy** 🔄

**Missing**: No rollback plan if migration fails.

**Add Task #0**:
```markdown
**0. Backup and Branch Strategy**
- Create feature branch: `feature/atc-scanner-polars-migration`
- Copy `atc_scanner.py` to `atc_scanner.py.backup` ✅ (already in Task list)
- Tag current commit: `git tag pre-polars-migration`
- Document rollback: `git checkout pre-polars-migration && git cherry-pick <fixes>`
```

---

### 2. **Add Performance Benchmarks** 📊

**Missing**: No verification that Polars actually improves performance.

**Add Task #8.5**:
```markdown
**8.5. Performance Benchmarking**
- Create benchmark script: `benchmarks/atc_scanner_polars.py`
- Measure:
  - Scan time for 10, 50, 100 symbols
  - Memory usage (pandas vs polars)
  - Cache reconstruction speed
- Target: Polars should be ≥20% faster or use ≤30% less memory
- If not: Consider whether migration is worth the effort
```

---

### 3. **Specify Polars API Equivalents** 📖

**Missing**: Task #7 mentions API changes but doesn't list all conversions needed.

**Expand Task #7**:
```markdown
**7. Update Polars API Usage in _scan_symbols_internal**

**Pandas → Polars Conversions**:
| Pandas API               | Polars API                          | Line #      |
|--------------------------|-------------------------------------|-------------|
| `df.empty`               | `df.is_empty()`                     | 325, 332    |
| `df["col"]`              | `df["col"]` or `df.select("col")`   | 328, 335    |
| `"col" in df.columns`    | `"col" in df.columns`               | 327, 334    |
| `dict(zip(df["a"], df["b"]))` | `dict(zip(df["a"].to_list(), df["b"].to_list()))` | 328, 335 |
| `pd.DataFrame(data)`     | `pl.DataFrame(data)`                | 420         |

**Notes**:
- Polars column access is lazy; use `.to_list()` for eager evaluation
- Schema validation: `df.schema` returns `OrderedDict[str, DataType]`
```

---

### 4. **Add Schema Validation** ✅

**Missing**: Task #3 defines schema but doesn't validate it.

**Add to Task #5**:
```markdown
**5. Convert scan_all_symbols Result to Polars**
# After conversion
long_signals = pl.from_pandas(long_pd)
short_signals = pl.from_pandas(short_pd)

# Validate schema
expected_cols = {"symbol", "signal"}  # From Task #3
if not expected_cols.issubset(long_signals.columns):
    log_warn(f"Missing columns in long_signals: {expected_cols - set(long_signals.columns)}")
```

---

### 5. **Clarify Cache Key Generation** 🔑

**Issue**: Cache keys are based on symbols + timeframe + minute (line 200-204), but Polars DataFrames might have different hash behavior.

**Verify**: Does cache work correctly when:
1. Same symbols, different order? (Task #3 sorts symbols in cache key - OK)
2. DataFrame internal structure changes? (Cache stores dicts, not DataFrames - OK)

**Action**: No change needed; cache design is DataFrame-agnostic.

---

### 6. **Add Polars Lazy API Option** ⚡

**Enhancement**: Polars supports lazy evaluation for better performance.

**Add Optional Task #14**:
```markdown
**14. (Optional) Evaluate Polars Lazy API**
- Convert DataFrames to LazyFrames: `pl.scan_df(df)`
- Defer execution until `.collect()` called
- Potential 10-30% speedup for filtering operations
- Test with: `longs.lazy().filter(pl.col("signal") > threshold).collect()`
```

---

### 7. **Improve Error Handling in Task #6** 🚨

**Current Task #6**:
```markdown
Exception path return empty Polars DataFrame using schema from step 3
```

**Problem**: Silent failures hide errors.

**Improve**:
```python
except Exception as e:
    log_error(f"ATCScanner: Error scanning {timeframe}: {e}")
    # Return empty DataFrames with schema (not silent)
    empty_long = pl.DataFrame(schema=_EMPTY_DF_SCHEMA)
    empty_short = pl.DataFrame(schema=_EMPTY_DF_SCHEMA)
    return empty_long, empty_short
```

---

### 8. **Add Compatibility Layer** 🔌

**Risk**: If downstream code secretly depends on pandas, migration breaks silently.

**Add Task #7.5**:
```markdown
**7.5. Add Pandas Compatibility Assertion**
- In `_scan_symbols_internal` after line 320:
  ```python
  assert isinstance(longs, pl.DataFrame), f"Expected Polars DataFrame, got {type(longs)}"
  assert isinstance(shorts, pl.DataFrame), f"Expected Polars DataFrame, got {type(shorts)}"
  ```
- Remove after Phase Verification passes
```

---

### 9. **Document Rust Integration Points** 🦀

**Issue**: Tasks #9-13 mention Rust but don't specify integration API.

**Add to Task #9-10**:
```markdown
**9. Implement Rust: Weighted Score**
- Function signature:
  ```rust
  #[pyfunction]
  fn calculate_weighted_score(
      signal_type: &str,  // "LONG", "SHORT", "NEUTRAL"
      tf_weight: f64,
      strength: f64
  ) -> PyResult<f64>
  ```
- Python usage:
  ```python
  from sovereign_prime import calculate_weighted_score
  score = calculate_weighted_score("LONG", 0.5, 0.8)
  ```

**10. Implement Rust: Signal Aggregation**
- Function signature:
  ```rust
  #[pyfunction]
  fn aggregate_signals(
      symbols: Vec<String>,
      results_by_tf: HashMap<String, SignalData>,
      weights: HashMap<String, f64>,
      threshold: f64
  ) -> PyResult<Vec<SignalResult>>
  ```
- Returns list of dicts convertible to `SignalResult` in Python
```

---

## Revised Task Order

**Current order has dependencies out of sequence.** Recommended:

### Phase 1: Preparation (Tasks 0-3)
1. **Task 0** (NEW): Backup and branch strategy
2. **Task 1**: Add Polars dependency
3. **Task 2**: Update imports and type hints
4. **Task 3**: Define empty DataFrame schema

### Phase 2: Core Migration (Tasks 4-6)
5. **Task 4**: Cache reconstruction with Polars
6. **Task 5**: Convert scan_all_symbols result (with schema validation)
7. **Task 6**: Exception path with empty Polars DataFrames

### Phase 3: API Updates (Task 7)
8. **Task 7**: Update Polars API usage (with conversion table)
9. **Task 7.5** (NEW): Add compatibility assertions

### Phase 4: Verification (Tasks 8-8.5)
10. **Task 8**: Update tests
11. **Task 8.5** (NEW): Performance benchmarks

### Phase 5: Documentation (Task 13-14)
12. **Task 13**: Update docstrings
13. **Task 14**: Phase verification
14. **Task 15** (NEW): Document lessons learned

### Phase 6 (Future): Rust Hot Path (Tasks 9-12)
15. **Task 9**: Rust weighted score
16. **Task 10**: Rust signal aggregation
17. **Task 11**: Rust ScanCache (optional)
18. **Task 12**: Integrate Rust with switches

---

## Requirements.txt Impact

**Current** (line 2):
```
pandas>=2.1.4,<2.2.0
```

**After Task #1**:
```
pandas>=2.1.4,<2.2.0  # Still required by adaptive_trend_LTS_mini
polars>=0.20.0,<1.0.0  # New: ATC scanner internal DataFrames
```

**Notes**:
- Keep pandas because `scan_all_symbols` still uses it
- Polars 0.20+ has stable API; avoid 1.0 until tested
- Consider adding `polars[pyarrow]` for better pandas interop

---

## Dependency Graph

```
scan_all_symbols (pandas)
    ↓
atc_scanner._run_single_scan → pl.from_pandas() → pl.DataFrame
    ↓
atc_scanner._scan_symbols_internal → SignalResult (no DataFrame)
    ↓
SignalPipeline.run_pipeline → passes SignalResult list
    ↓
XGBoostFilter.filter_signals → SignalResult (no DataFrame)
    ↓
SignalSelector.select_best_signal → FinalSignal
    ↓
SignalPersistence.save_signal → JSON
```

**Conclusion**: Polars is isolated to `atc_scanner.py` internal logic. No propagation needed. ✅

---

## Risk Assessment

| Risk | Severity | Mitigation |
|------|----------|------------|
| Polars API incompatibility | Medium | Task #7 conversion table |
| Test suite failures | High | Task #8 comprehensive updates |
| Performance regression | Low | Task #8.5 benchmarks |
| Cache thread-safety bug | **CRITICAL** | Fix `caching.py` first (separate) |
| Rust integration complexity | Medium | Make Tasks 9-13 truly optional |
| Rollback difficulty | Medium | Task #0 branching strategy |

---

## Recommendations

### Must Do (Blocking)
1. ✅ Fix `caching.py` thread-safety BEFORE starting Polars migration
2. ✅ Add Task #0: Backup and branch strategy
3. ✅ Add Task #8.5: Performance benchmarks
4. ✅ Expand Task #7: Polars API conversion table
5. ✅ Clarify Phase 1 (Polars) vs Phase 2 (Rust) in document intro

### Should Do (High Value)
6. ✅ Add Task #7.5: Compatibility assertions
7. ✅ Document Rust function signatures (Tasks #9-10)
8. ✅ Reorder tasks by dependency

### Nice to Have
9. ⚠️ Evaluate Polars lazy API (Task #14)
10. ⚠️ Add schema validation to Task #5

---

## Conclusion

**Overall Assessment**: The migration strategy is **sound but incomplete**. The core insight—that `SignalResult` API isolates Polars changes—is correct. However, the checklist lacks:
- Rollback planning
- Performance validation
- Detailed API conversion guide
- Clear separation of Polars (Phase 1) vs Rust (Phase 2)

**Priority**: Fix `caching.py` thread-safety bug **before** starting this migration, as the bug will be harder to diagnose after introducing Polars.

**Timeline Estimate** (with improvements):
- Phase 1 (Polars): 2-3 days (Tasks 0-8.5)
- Phase 2 (Rust): 5-7 days (Tasks 9-12) - only if benchmarks justify

**Approval**: Ready to proceed **after** adding Tasks 0, 7.5, 8.5 and fixing `caching.py`.

---

**Document Version**: v1.0
**Review Date**: 2026-02-01
**Reviewer**: Claude Code
**Related Documents**:
- `caching_review_v1.md` (thread-safety issues)
- `atc_scanner_update_v1.md` (Hybrid architecture reference)
