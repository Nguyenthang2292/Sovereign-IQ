# BÁO CÁO XÁC MINH CÁC FIX - MODULES XGBOOST_LTS & ADAPTIVE_TREND_LTS

**Ngày xác minh:** 2026-01-30  
**Người xác minh:** Claude Code  
**Trạng thái:** Đã xem xét các thay đổi sau khi người dùng sửa thủ công

---

## ✅ XGBOOST_LTS - CÁC FIX ĐÃ XÁC NHẬN

### 1. [FIXED] Logic Gap Không Nhất Quán Trong CV
**File:** `core/model.py:336-337` và `utils/cv_parallel.py:55-59`

**Trạng thái:** ✅ ĐÃ SỬA

**Thay đổi đã phát hiện:**
- model.py:336: `test_idx_filtered = test_idx_array[test_idx_array >= min_test_start]` - Luôn lọc thay vì chỉ khi điều kiện đầu tiên thỏa mãn
- cv_parallel.py:57: `test_idx_filtered = test_idx_array[test_idx_array >= min_test_start]` - Tương tự cho parallel CV

**Xác minh:**
```python
# Trước:
if test_idx_array[0] < min_test_start:
    test_idx_filtered = test_idx_array[test_idx_array >= min_test_start]
else:
    test_idx_filtered = test_idx_array  # Không lọc!

# Sau:
test_idx_filtered = test_idx_array[test_idx_array >= min_test_start]  # Luôn lọc
```

**Kết luận:** Data leakage issue đã được khắc phục trong cả sequential và parallel CV.

---

### 2. [NOT FIXED] Lỗi Caching Trong `apply_directional_labels`
**File:** `core/labeling.py:363-364`

**Trạng thái:** ⚠️ CHƯA SỬA

**Vấn đề vẫn tồn tại:**
```python
# labeling.py:363-364
if use_cache and cache_manager is not None and cache_config is not None:
    cache_manager.save_labels(df, df, cache_config)  # Vẫn dùng df đã thay đổi!
```

**Vấn đề:** 
- `df` đã được thêm các cột `Target`, `TargetLabel`, `DynamicThreshold`
- Hash của `df` đã thay đổi sẽ khác với DataFrame ban đầu khi load
- Gây ra cache miss hoặc lưu trữ không đúng

**Khuyến nghị sửa:**
```python
# Lưu original_df từ đầu hàm
def apply_directional_labels(df: pd.DataFrame, use_cache: bool = True) -> pd.DataFrame:
    original_df = df.copy()  # Lưu bản gốc
    
    # ... thực hiện labeling trên df ...
    
    # Lưu cache với original_df
    if use_cache and cache_manager is not None and cache_config is not None:
        cache_manager.save_labels(df, original_df, cache_config)  # Dùng original_df
```

---

### 3. [PARTIALLY FIXED] File Lock Không Đáng Tin Cậy
**File:** `core/optimization.py:52-60`

**Trạng thái:** 🟡 MỘT PHẦN

**Thay đổi đã phát hiện:**
```python
# optimization.py:52-60
lock_file = None
try:
    lock_file = open(lock_file_path, "w")
except (PermissionError, OSError) as e:
    logging.warning(f"Cannot create lock file {lock_file_path}: {e}. Continuing without file lock.")
    yield
    return
```

**Đánh giá:**
- ✅ Đã thêm error handling cho PermissionError/OSError
- ❌ Vẫn sử dụng advisory lock trên file thay vì database
- ❌ Không xử lý `sqlite3.OperationalError: database is locked`

**Khuyến nghị bổ sung:**
```python
# Thêm retry logic với exponential backoff
try:
    study.optimize(...)
except sqlite3.OperationalError as e:
    if "database is locked" in str(e):
        time.sleep(random.uniform(0.1, 0.5))
        retry_count += 1
```

---

### 4. [FIXED] Side Effect Trong `_resolve_xgb_classifier`
**File:** `core/model.py:107-109`

**Trạng thái:** ✅ ĐÃ SỬA

**Thay đổi đã phát hiện:**
```python
# Trước (dòng 107-109 cũ):
sklearn_classifier = _GradientBoostingWrapper
xgb.XGBClassifier = sklearn_classifier  # Side effect nguy hiểm!
return sklearn_classifier

# Sau (dòng 107-109 hiện tại):
# Return the resolved classifier without modifying global state
# This prevents side effects on other modules that import xgboost
return sklearn_classifier  # Không còn gán vào xgb.XGBClassifier!
```

**Kết luận:** Đã loại bỏ side effect nguy hiểm.

---

### 5. [FIXED] Threshold Calculation Cực Đoan
**File:** `core/labeling.py:314`

**Trạng thái:** ✅ ĐÃ SỬA

**Thay đổi đã phát hiện:**
```python
# Trước:
base_threshold = historical_pct.abs().fillna(TARGET_BASE_THRESHOLD).clip(lower=TARGET_BASE_THRESHOLD)

# Sau (dòng 314):
# Add upper bound to prevent extreme thresholds from pump/dump events
# Lower: TARGET_BASE_THRESHOLD (default ~1%), Upper: 10% to keep labels meaningful
base_threshold = historical_pct.abs().fillna(TARGET_BASE_THRESHOLD).clip(lower=TARGET_BASE_THRESHOLD, upper=0.1)
```

**Kết luận:** Threshold giờ có upper bound 10%, tránh tình trạng 100% sau pump/dump.

---

## ✅ ADAPTIVE_TREND_LTS - CÁC FIX ĐÃ XÁC NHẬN

### 1. [PARTIALLY FIXED] Logic Cutout Trong `equity_series`
**File:** `core/compute_equity/equity_series.py:109-116`

**Trạng thái:** 🟡 MỘT PHẦN

**Thay đổi đã phát hiện:**
```python
# equity_series.py:109-116
# NOTE: cutout is always 0 now as slicing happens early in compute_atc_signals
# If equity_series is called directly with cutout > 0, warn user
cutout = 0
if verbose and cutout > 0:
    log_warn(
        f"cutout parameter ({cutout}) is ignored in equity_series. "
        f"Cutout should be applied at compute_atc_signals level."
    )
```

**Đánh giá:**
- ✅ Đã thêm warning khi cutout > 0
- ❌ Vẫn hard-code cutout = 0
- ❌ Không giải quyết được vấn đề nếu gọi trực tiếp

**Khuyến nghị:** Thêm tham số để cho phép override hoặc throw error nếu cutout > 0.

---

### 2. [FIXED] Double NaN Check Trong Scan All Symbols
**File:** `core/scanner/scan_all_symbols.py:258-263`

**Trạng thái:** ✅ ĐÃ SỬA

**Thay đổi đã phát hiện:**
- Đã xóa bỏ duplicate check (trước có 2 lần check `if results_df.empty`, giờ chỉ còn 1)
- Đã thêm xử lý neutral signals trong comment (line 261)

**Kết luận:** Code sạch hơn, không còn duplicate logic.

---

### 3. [PARTIALLY FIXED] Memory Leak Trong Series Pool
**File:** `core/process_layer1/layer1_signal.py:167-169`

**Trạng thái:** 🟡 MỘT PHẦN

**Thay đổi đã phát hiện:**
```python
# layer1_signal.py:167-169
try:
    # ... calculation ...
    e_values_array = _calculate_equity_vectorized(...)
finally:
    # Always release input buffer - safe because output is separate allocation
    pool.release(sig_prev_values)
```

**Đánh giá:**
- ✅ Đã sử dụng try/finally để release array pool buffer
- ❌ Không thấy series_tuple và equity_tuple release trong layer1_signal.py (file chỉ có 201 lines)
- ❌ Trong `compute_atc_signals.py:321-329` vẫn còn try/except (không phải try/finally) cho series release

**Vấn đề còn lại trong compute_atc_signals.py:**
```python
# compute_atc_signals.py:321-329
try:
    for s in signals_tuple:
        series_pool.release(s)
    for e in equity_tuple:
        series_pool.release(e)
except Exception as e:
    log_warn(f"Error releasing series to pool for {ma_type}: {e}")
```

**Khuyến nghị:** Chuyển sang try/finally hoặc context manager để đảm bảo release.

---

### 4. [FIXED] Zero Denominator Trong Average Signal
**File:** `core/compute_atc_signals/average_signal.py:162-169`

**Trạng thái:** ✅ ĐÃ SỬA

**Thay đổi đã phát hiện:**
```python
# average_signal.py:162-169
# Handle zero denominator case (when all equity weights are zero)
zero_den_mask = den_array == 0
if np.any(zero_den_mask):
    zero_count = np.sum(zero_den_mask)
    log_warn(f"Sum of equity weights is zero for {zero_count} bars, returning neutral signal (0.0)")
    # Replace zero denominators with 1.0 to avoid division by zero
    # Since nominator is also 0 when all equities are 0, result will be 0/1 = 0 (neutral)
    den_array = np.where(zero_den_mask, 1.0, den_array)

# Calculate final average (no special error handling needed now)
avg_signal_array = nom_array / den_array
```

**Kết luận:** Đã xử lý zero denominator bằng cách thay thế bằng 1.0 trước khi chia.

---

### 5. [NOT FIXED] Race Condition Trong MA Calculation
**File:** `core/compute_moving_averages/set_of_moving_averages_enhanced.py:114-119`

**Trạng thái:** ⚠️ CHƯA SỬA

**Vấn đề vẫn tồn tại:**
```python
# set_of_moving_averages_enhanced.py:114-119
with ThreadPoolExecutor(max_workers=config.num_threads) as executor:
    futures = [
        executor.submit(ma_calculation_enhanced, source, ma_len, ma_type, use_cache, use_rust_backend)
        for ma_len in ma_lengths
    ]
    mas = [f.result() for f in futures]
```

**Vấn đề:**
- Vẫn sử dụng `use_cache=True` trong parallel execution
- Nếu `ma_calculation_enhanced` sử dụng file-based cache, có thể xảy ra race condition
- Chưa có cơ chế lock hoặc thread-safe cache

**Khuyến nghị:**
```python
# Tắt cache khi dùng parallel execution
use_cache_safe = use_cache and not use_parallel
# Hoặc sử dụng thread-local cache
```

---

## 📊 TỔNG KẾT

### XGBOOST_LTS:
| Issue | Trạng thái | Mức độ |
|-------|-----------|--------|
| #1 Logic gap | ✅ Fixed | Critical |
| #2 Caching | ⚠️ Not Fixed | Critical |
| #3 File lock | 🟡 Partial | High |
| #4 Side effect | ✅ Fixed | High |
| #5 Threshold | ✅ Fixed | High |

**Tỷ lệ hoàn thành:** 3/5 issues (60%) - 1 partial

### ADAPTIVE_TREND_LTS:
| Issue | Trạng thái | Mức độ |
|-------|-----------|--------|
| #1 Cutout | 🟡 Partial | Critical |
| #2 Duplicate check | ✅ Fixed | Critical |
| #3 Memory leak | 🟡 Partial | Critical |
| #4 Zero denominator | ✅ Fixed | High |
| #5 Race condition | ⚠️ Not Fixed | High |

**Tỷ lệ hoàn thành:** 2/5 issues (40%) - 2 partial

---

## ⚠️ ISSUES CẦN SỬA TIẾP

### Cần sửa ngay:
1. **xgboost_LTS #2:** Caching trong `apply_directional_labels` - cần lưu original_df
2. **adaptive_trend_LTS #5:** Race condition trong MA calculation - cần thread-safe cache

### Cần cải thiện:
1. **xgboost_LTS #3:** File lock - thêm retry logic cho SQLite
2. **adaptive_trend_LTS #1:** Cutout - cho phép override hoặc throw error
3. **adaptive_trend_LTS #3:** Memory leak - chuyển sang try/finally trong compute_atc_signals

---

## 🧪 KIỂM TRA REGRESSION

### Không phát hiện regression mới:
- ✅ Code vẫn chạy được sau khi sửa
- ✅ Không có syntax errors
- ✅ Logic flow vẫn đúng

### Lỗi LSP cũ vẫn tồn tại (không liên quan đến fixes):
- Type mismatch trong equity_series (ArrayLike vs ndarray)
- Import không resolve trong incremental_atc.py
- Function declaration obscured trong compute_atc_signals.py

---

**Kết luận:** Các fix đã cải thiện đáng kể, nhưng vẫn còn 2 critical issues chưa được sửa. Khuyến nghị xử lý tiếp các issues còn lại trước khi deploy.
