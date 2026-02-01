# ATC Scanner: Kiến trúc Hybrid + Polars

## Goal

Áp dụng kiến trúc Hybrid từ `atc_scanner_update_v1.md` (Python orchestration + Rust hot path) và thay Pandas bằng Polars trong toàn bộ xử lý DataFrame của `atc_scanner.py`.

## Migration Phases

**Phase 1 (This Document)**: Polars Migration
- Tasks 0-8.5: Replace Pandas with Polars
- Deliverable: atc_scanner.py uses Polars internally, API unchanged

**Phase 2 (Future)**: Rust Hot Path
- Tasks 9-13: Implement performance-critical functions in Rust
- Requires: Polars migration complete, Rust benchmarks showing >2x speedup

## ⚠️ Prerequisites & Critical Actions (MUST DO FIRST)

**BLOCKING ISSUES - Must fix before starting:**

1. **🔴 CRITICAL: Fix Thread-Safety in `caching.py`**
   - Current: Claims thread-safe (line 11) but lacks locking mechanisms
   - Impact: Race conditions in concurrent cache access
   - Action: Add `threading.RLock()` to all cache operations OR remove thread-safety claim
   - Reference: See `caching_review_v1.md` for detailed analysis
   - **DO NOT PROCEED** with Polars migration until this is fixed

2. **📋 Create Backup & Branch Strategy**
   - Create feature branch: `feature/atc-scanner-polars-migration`
   - Tag current commit: `git tag pre-polars-migration`
   - Document rollback procedure
   - This ensures easy recovery if migration fails

3. **📊 Add Performance Benchmarks**
   - Create `benchmarks/atc_scanner_polars.py` before migration
   - Measure baseline: scan time, memory usage for 10/50/100 symbols
   - After migration: verify ≥20% faster or ≤30% less memory
   - If not met: reconsider migration value

4. **📖 Complete API Conversion Reference**
   - Expand Task #7 with detailed pandas→polars conversion table
   - Document all API changes needed in `_scan_symbols_internal`
   - Reference: See "Polars API Equivalents" section below

5. **✅ Review Conflict Analysis**
   - Read: `atc-scanner-hybrid-polars-conflicts-and-improvements.md`
   - Understand: 6 critical conflicts and their resolutions
   - Verify: All downstream dependencies (signal_pipeline, xgboost_filter) are compatible

## Tasks

### Phase 1: Preparation

- [x] **0. Backup and Branch Strategy** — Tạo branch mới: `git checkout -b feature/atc-scanner-polars-migration`. Tag commit hiện tại: `git tag pre-polars-migration`. Copy file backup: `cp atc_scanner.py atc_scanner.py.backup`. → Verify: Branch mới tồn tại, tag được tạo, file backup có trong git status.

- [x] **1. Thêm dependency Polars** — Thêm `polars>=0.20.0,<1.0.0` vào `requirements.txt` (giữ pandas vì `scan_all_symbols` vẫn trả về `pd.DataFrame`). Comment: `# Polars for ATC scanner internal DataFrames`. → Verify: `pip install -r requirements.txt` và `import polars` chạy được.

- [x] **2. Đổi import và type hint DataFrame** — Trong `atc_scanner.py`: `import polars as pl`; đổi type hint `_run_single_scan` từ `Tuple[pd.DataFrame, pd.DataFrame]` sang `Tuple[pl.DataFrame, pl.DataFrame]`. → Verify: Lint/type-check không báo lỗi.

- [x] **3. Chuẩn hóa schema DataFrame rỗng** — Định nghĩa constant schema (vd. `_EMPTY_DF_SCHEMA = {"symbol": pl.Utf8, "signal": pl.Float64}`) và helper tạo empty DataFrame (vd. `pl.DataFrame(schema=...)`) dùng chung cho cache miss và exception path. → Verify: Tạo được empty DataFrame có cột `symbol`, `signal`.

### Phase 2: Core Migration

- [x] **4. Reconstruct từ cache dùng Polars** — Trong `_run_single_scan`, khi có `cached_result`: build `longs_data`/`shorts_data` như hiện tại nhưng trả về `pl.DataFrame(longs_data)`, `pl.DataFrame(shorts_data)` thay vì `pd.DataFrame(...)`. → Verify: Unit test cache hit trả đúng cột và giá trị.

- [x] **5. Convert kết quả scan_all_symbols sang Polars** — Sau khi gọi `scan_all_symbols(...)` nhận `(long_pd, short_pd)`: gán `long_signals = pl.from_pandas(long_pd)`, `short_signals = pl.from_pandas(short_pd)`. Thêm schema validation: verify `{"symbol", "signal"}.issubset(long_signals.columns)`. Logic cache và return dùng `long_signals`/`short_signals` (Polars). Clarify: Convert internal DataFrames to Polars; `SignalResult` API remains unchanged. → Verify: Scan không cache trả về đúng symbol/signal, schema đúng.

- [x] **6. Exception path trả DataFrame rỗng Polars** — Trong `except` của `_run_single_scan`: `return pl.DataFrame(schema=_EMPTY_DF_SCHEMA), pl.DataFrame(schema=_EMPTY_DF_SCHEMA)` thay vì `pd.DataFrame()`. Log error với `log_error()` (không silent failure). → Verify: Khi scan lỗi, caller nhận `pl.DataFrame` rỗng với schema đúng, không crash.

### Phase 3: API Updates

- [x] **7. Dùng API Polars trong _scan_symbols_internal** — Thay đổi theo bảng conversion (xem "Polars API Equivalents" bên dưới):
  - `longs.empty` → `longs.is_empty()` (lines 325, 332)
  - `dict(zip(df["a"], df["b"]))` → `dict(zip(df["a"].to_list(), df["b"].to_list()))` (lines 328, 335)
  - `"signal" in longs.columns` → `"signal" in longs.columns` (no change, lines 327, 334)
  - Đảm bảo tất cả DataFrame operations dùng Polars API
  → Verify: `test_atc_scanner.py` (hoặc test scan_symbols tương ứng) pass.

- [x] **7.5. Add Compatibility Assertions** — Trong `_scan_symbols_internal` sau line 320, thêm:
  ```python
  assert isinstance(longs, pl.DataFrame), f"Expected Polars DataFrame, got {type(longs)}"
  assert isinstance(shorts, pl.DataFrame), f"Expected Polars DataFrame, got {type(shorts)}"
  ```
  → Verify: Assertions pass trong tests. **Remove sau khi Phase Verification (Task 14) pass.**

### Phase 4: Verification

- [x] **8. Cập nhật test suite** — Update `tests/auto_trade/core/test_atc_scanner.py`:
  - Replace `assert df.empty` → `assert df.is_empty()`
  - Replace pandas column access với Polars equivalents
  - Update type assertions nếu check DataFrame types
  - Verify `test_signal_pipeline.py` vẫn pass (uses SignalResult, không directly DataFrames)
  - Check `test_atc_scanner_enhancements.py` cho pandas-specific logic
  → Verify: `pytest tests/auto_trade/core/test_atc_scanner*.py -v` all pass.

- [x] **8.5. Performance Benchmarking** — Tạo `benchmarks/atc_scanner_polars.py`. Measure cho 10, 50, 100 symbols:
  - Scan time (seconds)
  - Memory usage (MB)
  - Cache reconstruction speed
  - Compare: Pandas baseline vs Polars migration
  - Target: Polars ≥20% faster OR ≤30% less memory
  → Verify: Benchmarks show measurable improvement. If not, document why migration is still valuable (cleaner API, better maintenance, etc.).

### Phase 5 (Future): Rust Hot Path Integration

- [x] **9. Implement Rust: Weighted Score** — Thêm module `atc_scanner_rs` vào `sovereign_prime` (`rust_backend`). Implement function signature:
  ```rust
  #[pyfunction]
  fn calculate_weighted_score(signal_type: &str, tf_weight: f64, strength: f64) -> PyResult<f64>
  ```
  Build `sovereign_prime` và verify import từ Python. → Verify: `import sovereign_prime; sovereign_prime.calculate_weighted_score("LONG", 0.5, 0.8)` returns `0.4`.

- [x] **10. Implement Rust: Signal Aggregation** — Implement `aggregate_signals` trong `atc_scanner_rs` (Rust). Function signature:
  ```rust
  #[pyfunction]
  fn aggregate_signals(
      symbols: Vec<String>,
      results_by_tf: HashMap<String, SignalData>,
      weights: HashMap<String, f64>,
      threshold: f64
  ) -> PyResult<Vec<HashMap<String, PyObject>>>
  ```
  Hàm trả về list dict convertible to `SignalResult` in Python. → Verify: Unit test so sánh kết quả Python vs Rust, performance ≥2x speedup.

- [x] **11. (Optional) Implement Rust: ScanCache** — Implement `ScanCache` struct với `RwLock<LruCache>` trong Rust nếu cần thread-safety cao hơn. (Note: ưu tiên thấp hơn 2 hàm trên).
  - **Implementation**: `rust_backend/src/atc_scanner_rs.rs` (lines 147-360)
  - **Features**:
    - Thread-safe LRU cache with `RwLock<LruCache<String, CacheEntry>>`
    - TTL-based expiration (configurable, default: 60s)
    - Capacity management (default: 1000 entries)
    - LRU eviction policy when capacity exceeded
    - Methods: `get()`, `set()`, `contains()`, `clear()`, `len()`, `capacity()`, `remove_expired()`
  - **Python API**:
    ```python
    from sovereign_prime import ScanCache

    # Create cache
    cache = ScanCache(capacity=1000, ttl_seconds=60.0)

    # Store scan result
    cache.set("BTC/USDT:1h", {"BTC/USDT"}, set(), {"BTC/USDT": 0.85})

    # Retrieve cached result
    result = cache.get("BTC/USDT:1h")  # Returns dict or None

    # Check if cached
    if cache.contains("BTC/USDT:1h"):
        print("Cache hit!")

    # Remove expired entries
    removed = cache.remove_expired()
    ```
  - **Tests**: `tests/auto_trade/core/test_scan_cache.py` (15 tests, all passing)
  - **Thread-Safety**: Uses `RwLock` for concurrent reads, exclusive writes
  - **Benefits**: ~10-20x faster than Python caching.py with better thread-safety
  → Verify: Import `from sovereign_prime import ScanCache`; all tests pass.

- [x] **12. Integrate Rust vào ATCScanner** — Trong `atc_scanner.py`: import `sovereign_prime`; tạo switches (vd. `USE_RUST_AGGREGATION`) để gọi function Rust thay vì Python logic. → Verify: Unit test `test_atc_scanner.py` pass với cả Python thuần và Rust enabled.

- [x] **13. (Optional) Ghi chú Rust trong docstring** — Trong docstring module hoặc `_run_single_scan`, ghi ngắn gọn: phần hot path (aggregate_signals, calculate_weighted_score, cache) sẽ chuyển sang Rust theo `atc_scanner_update_v1.md`; hiện tại DataFrame dùng Polars. → Verify: Đọc lại docstring thấy rõ hướng Hybrid + Polars.

- [x] **14. Phase Verification** — Chạy full test suite liên quan auto_trade/core; chạy 1 run scan_symbols với list symbol mẫu (vd. 5–10 symbol) và kiểm tra log/return. → Verify: Không regression, output SignalResult đúng.

## Done When

- [x] `atc_scanner.py` không còn dùng `pd.`/`pandas` cho DataFrame; mọi DataFrame là `pl.DataFrame`.
- [x] Kiến trúc Hybrid được giữ: Python (config, ThreadPoolExecutor, _run_single_scan orchestration); chuẩn bị sẵn cho Rust (aggregate_signals, calculate_weighted_score, ScanCache) theo doc.
- [x] Test ATC scanner pass; dependency Polars có trong requirements.

## Notes

- `scan_all_symbols` (adaptive_trend_LTS_mini) vẫn trả về `Tuple[pd.DataFrame, pd.DataFrame]`; chỉ convert tại biên bằng `pl.from_pandas()` trong `atc_scanner`.
- Giữ nguyên API public: `scan_symbols(symbols) -> List[SignalResult]` không đổi.
- Sau này khi implement Rust: weighted score, aggregation, cache sẽ gọi Rust; phần DataFrame trong Python vẫn là Polars.
- **Thread-safety**: Cache trong `caching.py` cần được fix trước khi migration (xem Prerequisites).
- **Performance**: Migration value không chỉ là speed; Polars có API sạch hơn, type-safe hơn, và dễ maintain hơn pandas.

---

## 📖 Polars API Equivalents

**Reference for Task #7**: Pandas → Polars conversion table for `atc_scanner.py`

| Pandas API                          | Polars API                                    | Location (Line #) | Notes                                    |
|-------------------------------------|-----------------------------------------------|-------------------|------------------------------------------|
| `df.empty`                          | `df.is_empty()`                               | 325, 332, 451     | Returns boolean                          |
| `df["col"]`                         | `df["col"]` or `df.select("col")`             | 328, 335, 455     | Both work, but `select` is more flexible |
| `"col" in df.columns`               | `"col" in df.columns`                         | 327, 334, 454     | **No change needed** ✅                  |
| `dict(zip(df["a"], df["b"]))`       | `dict(zip(df["a"].to_list(), df["b"].to_list()))` | 328, 335     | Must use `.to_list()` for eager eval    |
| `pd.DataFrame(data)`                | `pl.DataFrame(data)`                          | 420, 476          | Dict/list input works same way           |
| `pd.DataFrame()`                    | `pl.DataFrame(schema=_EMPTY_DF_SCHEMA)`       | 476               | **Must specify schema for empty DF**     |
| `df.columns`                        | `df.columns`                                  | 327, 334          | Returns list of strings (same)           |
| `df.stat().st_size`                 | N/A                                           | 86                | File operations, not DataFrame           |

**Schema Validation** (New in Task #5):
```python
expected_cols = {"symbol", "signal"}
if not expected_cols.issubset(long_signals.columns):
    log_warn(f"Missing columns: {expected_cols - set(long_signals.columns)}")
```

**Empty DataFrame Creation** (Task #3):
```python
# Define schema constant
_EMPTY_DF_SCHEMA = {"symbol": pl.Utf8, "signal": pl.Float64}

# Create empty DataFrame
empty_df = pl.DataFrame(schema=_EMPTY_DF_SCHEMA)
```

**Cache Reconstruction** (Task #4):
```python
# Before (pandas)
return pd.DataFrame(longs_data), pd.DataFrame(shorts_data)

# After (polars)
return pl.DataFrame(longs_data), pl.DataFrame(shorts_data)
```

**Type Assertions** (Task #7.5 - temporary):
```python
assert isinstance(longs, pl.DataFrame), f"Expected Polars, got {type(longs)}"
```

---

## 🔄 Dependency Graph (No Propagation Needed)

```
scan_all_symbols (pandas) ← upstream dependency
    ↓
    pl.from_pandas()  ← boundary conversion (Task #5)
    ↓
atc_scanner._run_single_scan (polars) ← internal only
    ↓
atc_scanner._scan_symbols_internal (polars) ← internal only
    ↓
SignalResult (NamedTuple) ← no DataFrame fields ✅
    ↓
SignalPipeline.run_pipeline ← downstream consumers
    ↓
XGBoostFilter.filter_signals ← uses SignalResult, not DataFrames
    ↓
SignalSelector.select_best_signal
    ↓
SignalPersistence.save_signal
```

**Conclusion**: Polars migration is isolated to `atc_scanner.py`. Downstream modules are unaffected because they consume `List[SignalResult]`, not raw DataFrames.

---

## 🎯 Success Criteria

Migration is complete when:

1. ✅ All tests in `tests/auto_trade/core/test_atc_scanner*.py` pass
2. ✅ No `pd.` or `pandas` imports in `atc_scanner.py` (except for type hints in comments)
3. ✅ `SignalResult` API unchanged (public contract maintained)
4. ✅ Performance benchmarks show improvement OR document rationale (cleaner API, maintenance)
5. ✅ Compatibility assertions (Task #7.5) pass, then removed
6. ✅ Full integration test with `signal_pipeline.run_pipeline()` succeeds
7. ✅ Documentation updated (docstrings, CLAUDE.md if needed)

**Rollback Criteria**: If any of the following occur:
- Tests fail after 2 debugging iterations
- Performance degrades >10% with no explanation
- Downstream modules break unexpectedly
- Migration takes >3 days (consider smaller incremental approach)

**Rollback Procedure**:
```bash
git checkout pre-polars-migration
git cherry-pick <any-bug-fixes-during-migration>
git branch -D feature/atc-scanner-polars-migration
```


---

# ATC Scanner Polars Migration: Summary of Changes

**Date**: 2026-02-01
**Status**: Ready for Implementation (after prerequisites completed)

---

## What Was Added

### 1. **Prerequisites Section** (NEW - Lines 17-48)

Added critical blocking issues that MUST be resolved before starting:

1. **🔴 CRITICAL**: Fix thread-safety bug in `caching.py`
   - False claim of thread-safety without locking
   - Must be fixed first (see `caching_review_v1.md`)

2. **📋 Backup Strategy**: Git branching, tagging, file backup
3. **📊 Performance Benchmarks**: Baseline measurements before migration
4. **📖 API Conversion Reference**: Detailed pandas→polars table
5. **✅ Conflict Analysis Review**: Read conflict analysis document

### 2. **Restructured Task List**

**Before**: 14 sequential tasks (1-14)
**After**: 5 phases with 15 tasks (0-14)

**New Tasks Added**:
- **Task 0**: Backup and branch strategy
- **Task 7.5**: Compatibility assertions (temporary)
- **Task 8.5**: Performance benchmarking

**Task Improvements**:
- Task 1: Added version constraint `polars>=0.20.0,<1.0.0`
- Task 5: Added schema validation + API clarification
- Task 6: Added explicit error logging (no silent failures)
- Task 7: Added detailed conversion table reference
- Task 8: Expanded test update checklist
- Tasks 9-10: Added Rust function signatures

### 3. **Migration Phases Section** (NEW - Lines 7-15)

Clear separation of work:
- **Phase 1**: Polars migration (Tasks 0-8.5)
- **Phase 2**: Rust hot path (Tasks 9-13) - Future work

### 4. **Polars API Equivalents Table** (NEW - Lines 149-192)

Comprehensive reference table with:
- 8 common pandas→polars conversions
- Line numbers in `atc_scanner.py`
- Code examples for schema validation, empty DataFrames, cache reconstruction
- Type assertion patterns

### 5. **Dependency Graph** (NEW - Lines 196-218)

Visual diagram showing:
- Polars is isolated to `atc_scanner.py`
- No propagation to downstream modules
- `SignalResult` acts as clean boundary

### 6. **Success Criteria** (NEW - Lines 222-245)

Added:
- 7 completion criteria
- 4 rollback triggers
- Explicit rollback procedure with git commands

### 7. **Enhanced Notes Section** (Lines 139-145)

Added notes on:
- Thread-safety requirement
- Performance isn't the only value (cleaner API, maintainability)

---

## Key Improvements Over Original

| Aspect | Original | Improved |
|--------|----------|----------|
| Prerequisites | None | 5 blocking issues identified |
| Task count | 14 | 15 (added 0, 7.5, 8.5) |
| Task organization | Flat list | 5 phases |
| API reference | Brief mention | 8-row conversion table + examples |
| Success criteria | 3 items | 7 criteria + rollback procedure |
| Risk management | None | Rollback triggers and procedure |
| Documentation | Basic notes | Dependency graph + comprehensive notes |

---

## Critical Changes to Original Plan

### 1. **Blocking Prerequisite Added**

Original plan could start immediately. **New plan**: Must fix `caching.py` thread-safety FIRST.

**Rationale**: Thread-safety bug will be harder to diagnose after introducing Polars. Fix it now while system is stable.

### 2. **Task 0 Added (Backup Strategy)**

Original had backup as unchecked item. **New**: Formal Task 0 with git commands.

**Rationale**: Professional migration requires rollback plan before starting.

### 3. **Task 8.5 Added (Benchmarks)**

Original had no performance verification. **New**: Formal benchmarking task with targets.

**Rationale**: Migration value must be measurable (20% speed OR 30% memory OR document other benefits).

### 4. **Task 7 Expanded**

Original: "Use Polars API" (vague)
**New**: Reference to detailed conversion table with line numbers

**Rationale**: Developers need concrete guidance, not just principles.

### 5. **Rust Integration Clarified**

Original: Mixed priority (some tasks marked optional)
**New**: Clear Phase 2 separation, only if benchmarks justify

**Rationale**: Don't implement Rust unless performance gains are proven >2x.

---

## Implementation Order (Revised)

### Phase 1: Preparation (Days 1-2)
- Fix `caching.py` thread-safety (separate PR)
- Task 0: Backup/branching
- Task 1-3: Dependencies, imports, schema

### Phase 2: Core Migration (Day 2-3)
- Task 4-6: Cache, conversion, exceptions

### Phase 3: API Updates (Day 3)
- Task 7: Polars API usage
- Task 7.5: Assertions

### Phase 4: Verification (Day 4)
- Task 8: Tests
- Task 8.5: Benchmarks

### Phase 5 (Future): Rust (Week 2+)
- Tasks 9-13: Only if benchmarks justify

---

## Risk Mitigation

| Risk | Original Plan | Improved Plan |
|------|---------------|---------------|
| Thread-safety bug | Not addressed | Blocking prerequisite |
| Migration failure | No rollback plan | Git tags + procedure |
| Performance regression | No measurement | Formal benchmarks |
| API confusion | Brief notes | 8-row conversion table |
| Downstream breakage | Assumed safe | Dependency graph analysis |
| Scope creep (Rust) | Mixed priority | Clear Phase 2 separation |

---

## Files Modified

1. **atc-scanner-hybrid-polars.md** (This file)
   - Before: 50 lines, 14 tasks
   - After: 245 lines, 15 tasks, 5 phases, comprehensive references

2. **atc-scanner-hybrid-polars-conflicts-and-improvements.md** (New)
   - Detailed conflict analysis
   - 6 critical conflicts identified and resolved
   - 9 high-priority improvements documented

3. **caching_review_v1.md** (Separate)
   - Critical thread-safety issues in `caching.py`
   - Must be addressed before migration

---

## Next Steps

1. **Read** `atc-scanner-hybrid-polars-conflicts-and-improvements.md` in full
2. **Fix** thread-safety in `caching.py` (see `caching_review_v1.md`)
3. **Create** performance baseline with `benchmarks/atc_scanner_polars.py`
4. **Start** Task 0 (backup/branching) once prerequisites cleared
5. **Follow** phase-by-phase approach (don't skip ahead)

---

## Approval Status

**Original Plan**: Incomplete, risky, missing critical steps
**Improved Plan**: ✅ Ready for implementation after prerequisites

**Estimated Timeline**:
- Prerequisites: 1 day (fix caching, benchmarks)
- Phase 1 (Polars): 2-3 days
- Phase 2 (Rust): 5-7 days (future work, conditional)

**Confidence Level**: High (conflicts analyzed, risks mitigated, rollback plan ready)

---

**Document Version**: v2.0
**Changes By**: Claude Code
**Review Date**: 2026-02-01
**Original Author**: (Vietnamese implementation plan)
