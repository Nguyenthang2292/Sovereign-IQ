# BÁO CÁO KIỂM TRA TOÀN DIỆN MODULE `modules\xgboost_LTS`

## Tóm Tắt

Module `xgboost_LTS` được thiết kế để dự đoán hướng giá cryptocurrency sử dụng XGBoost. Sau khi phân tích toàn diện, phát hiện **15 lỗi/vấn đề tiềm ẩn** bao gồm lỗi logic nghiêm trọng, vấn đề về data leakage, và lỗi đồng bộ hóa.

---

## CÁC LỖI LOGIC NGHIÊM TRỌNG

### 1. [CRITICAL] Logic Gap Không Nhất Quán - Data Leakage Tiềm ẩn
**File:** `core/model.py:314-339`, `utils/cv_parallel.py:42-60`

**Mô tả:**
- Trong `model.py`, logic kiểm tra gap giữa train và test set sử dụng biến `min_test_start` được tính từ `train_idx_filtered[-1]`
- Nhưng điều kiện kiểm tra `test_idx_array[0] < min_test_start` có thể không đủ
- Khi `test_idx_array[0]` nằm trong khoảng gap, nó sẽ bị lọc ra, nhưng các index tiếp theo trong `test_idx_array` có thể vẫn nằm trong gap

**Code có vấn đề:**
```python
# model.py:328-336
min_test_start = train_idx_filtered[-1] + TARGET_HORIZON + 1
if test_idx_array[0] < min_test_start:
    test_idx_filtered = test_idx_array[test_idx_array >= min_test_start]  # Chỉ lọc phần đầu
    if len(test_idx_filtered) == 0:
        continue
else:
    test_idx_filtered = test_idx_array  # Không kiểm tra overlap!
```

**Giả thuyết kiểm tra:**
- Nếu `train_idx` kết thúc tại index 100, `TARGET_HORIZON=24` → `min_test_start = 125`
- Nếu `test_idx_array = [110, 111, 112, ...]` → 110 < 125, nên chỉ các index >=125 được giữ lại
- **Vấn đề:** Nếu `test_idx_array = [130, 131, ...]` → 130 >= 125, toàn bộ test set được giữ nguyên mà không kiểm tra xem có index nào < 125 không

**Mức độ nghiêm trọng:** 🔴 **CAO** - Có thể gây data leakage trong CV

**Giải pháp:**
```python
test_idx_filtered = test_idx_array[test_idx_array >= min_test_start]
if len(test_idx_filtered) == 0:
    continue
```
Luôn lọc test set, không chỉ khi điều kiện đầu tiên thỏa mãn.

---

### 2. [CRITICAL] Lỗi Caching Trong `apply_directional_labels`
**File:** `core/labeling.py:361-363`

**Mô tả:**
- Hàm lưu cache sử dụng `cache_manager.save_labels(df, df, cache_config)`
- Nhưng `df` đã được sửa đổi (đã thêm cột Target, TargetLabel, DynamicThreshold)
- Việc hash DataFrame đã thay đổi sẽ tạo ra hash khác với DataFrame ban đầu
- Điều này có thể gây ra cache miss hoặc lưu trữ không đúng

**Code có vấn đề:**
```python
# labeling.py:361-363
if use_cache and cache_manager is not None and cache_config is not None:
    cache_manager.save_labels(df, df, cache_config)  # df đã thay đổi!
```

**Giải pháp:**
- Lưu `source_df` (bản gốc) vào cache, không phải `df` đã thay đổi

---

### 3. [HIGH] Lỗi Đồng Bộ Hóa File Lock Không Hoạt Động Đúng
**File:** `core/optimization.py:43-65`

**Mô tả:**
- File lock sử dụng `msvcrt.locking` trên Windows và `fcntl` trên Unix
- Vấn đề: Lock được áp dụng trên file handle của lock_file, không phải trên SQLite database
- Nếu nhiều process cùng mở studies.db, lock file chỉ ngăn chặn các process khác mở lock_file, không ngăn chúng ghi vào database
- Ngoài ra, `msvcrt.locking` có thể không hoạt động đúng với các ứng dụng Python đa luồng

**Giả thuyết kiểm tra:**
```python
# Kiểm tra xem lock có thực sự ngăn chặn concurrent access không
# Chạy 2 process song song, cùng optimize cùng symbol/timeframe
# Nếu không có lỗi "database is locked", lock không hoạt động
```

**Giải pháp:**
- Sử dụng Optuna's built-in locking mechanism hoặc xử lý `sqlite3.OperationalError: database is locked`
- Hoặc sử dụng RDBMS khác (PostgreSQL, MySQL) cho concurrent access

---

### 4. [HIGH] Lỗi Fallback Trong `_resolve_xgb_classifier` Cache Sai Đối Tượng
**File:** `core/model.py:107-109`

**Mô tả:**
- Khi fallback sang `GradientBoostingClassifier`, code cache kết quả vào `xgb.XGBClassifier`
- Điều này gây ra side effect: tất cả các import sau này của `xgb.XGBClassifier` sẽ trỏ đến wrapper class
- Có thể gây lỗi cho các module khác sử dụng xgboost

**Code có vấn đề:**
```python
# model.py:107-109
sklearn_classifier = _GradientBoostingWrapper
xgb.XGBClassifier = sklearn_classifier  # Side effect nguy hiểm!
return sklearn_classifier
```

**Giải pháp:**
- Không cache vào `xgb.XGBClassifier`, chỉ return classifier
- Hoặc sử dụng singleton pattern riêng cho module này

---

### 5. [HIGH] Threshold Calculation Có Thể Tạo Ra Giá Trị Cực Đoan
**File:** `core/labeling.py:310-323`

**Mô tả:**
- `base_threshold` được tính từ `historical_pct.abs().fillna(TARGET_BASE_THRESHOLD).clip(lower=TARGET_BASE_THRESHOLD)`
- Sau đó nhân với `atr_ratio` và clip lại: `(base_threshold * atr_ratio).clip(lower=TARGET_BASE_THRESHOLD)`
- Vấn đề: Nếu `historical_pct` có giá trị rất lớn (ví dụ: sau một cây nến pump/dump mạnh), `base_threshold` sẽ rất lớn
- Sau khi nhân với `atr_ratio` (tối đa 2.0), threshold có thể trở nên quá lớn, gây ra ít nhãn UP/DOWN và nhiều nhãn NEUTRAL

**Giả thuyết kiểm tra:**
```python
# Giả sử historical_pct = 0.5 (50% thay đổi)
# base_threshold = 0.5
# atr_ratio = 2.0
# final_threshold = 1.0 (100% thay đổi cần để được label UP/DOWN)
```

**Giải pháp:**
- Thêm upper bound cho `base_threshold`: `.clip(lower=TARGET_BASE_THRESHOLD, upper=0.1)` (10%)
- Hoặc sử dụng rolling percentile thay vì giá trị tuyệt đối

---

## CÁC LỖI LOGIC TRUNG BÌNH

### 6. [MEDIUM] Memory Leak Trong `apply_directional_labels`
**File:** `core/labeling.py:355-358`

**Mô tả:**
- Các biến intermediate được xóa và `gc.collect()` được gọi
- Tuy nhiên, các Series và DataFrame tạo ra trong hàm vẫn có tham chiếu đến dữ liệu gốc
- `gc.collect()` không đảm bảo giải phóng bộ nhớ ngay lập tức
- Ngoài ra, `df.loc[:, "DynamicThreshold"] = threshold_series` tạo ra view/copy ambiguity

**Giải pháp:**
- Sử dụng `df = df.copy()` ở đầu hàm để tránh ảnh hưởng DataFrame gốc
- Hoặc sử dụng context manager để đảm bảo giải phóng

---

### 7. [MEDIUM] Float32 Conversion Mất Thông Tin
**File:** `core/model.py:156-157`

**Mô tả:**
- `X = X.astype(np.float32)` chuyển đổi tất cả features sang float32
- Vấn đề: Nếu features có giá trị rất nhỏ hoặc rất lớn, float32 có thể mất precision
- Đặc biệt nguy hiểm nếu có các indicator với giá trị normalized (ví dụ: z-score với giá trị > 1e6)

**Giả thuyết kiểm tra:**
```python
# Giá trị float32 có precision ~7 decimal digits
# Nếu close price = 100000.123456789, float32 sẽ là ~100000.125
```

**Giải pháp:**
- Chỉ chuyển đổi các cột không cần precision cao
- Hoặc kiểm tra range của features trước khi chuyển đổi

---

### 8. [MEDIUM] Không Kiểm Tra Feature Importance Trước Khi Cache
**File:** `utils/cache_manager.py:38-50`

**Mô tả:**
- `_compute_df_hash` sử dụng `pd.util.hash_pandas_object` để hash toàn bộ DataFrame
- Vấn đề: Nếu DataFrame có cột không dùng cho training (ví dụ: timestamp, symbol), thay đổi các cột này sẽ tạo ra hash khác dù features giống nhau
- Gây ra cache miss không cần thiết

**Giải pháp:**
- Chỉ hash các cột trong `MODEL_FEATURES` + `Target`

---

### 9. [MEDIUM] Không Xử Lý Trường Hợp `predict_proba` Trả Về Sai Shape
**File:** `core/model.py:426-433`

**Mô tả:**
- `predict_next_move` giả định `model.predict_proba(X_new)[0]` luôn trả về array với 3 phần tử
- Nếu model được train với ít hơn 3 classes (do class diversity issues), `predict_proba` sẽ trả về array với ít hơn 3 phần tử
- Gây ra IndexError hoặc logic sai khi sử dụng

**Code có vấn đề:**
```python
proba = model.predict_proba(X_new)[0]  # Giả định luôn có 3 classes
# Nếu chỉ có 2 classes, proba chỉ có 2 phần tử
# proba[2] sẽ raise IndexError
```

**Giải pháp:**
- Kiểm tra `len(proba)` và pad với 0 nếu cần
- Hoặc raise warning khi số classes không khớp

---

### 10. [MEDIUM] Race Condition Trong Parallel CV
**File:** `utils/cv_parallel.py:146-171`

**Mô tả:**
- `ProcessPoolExecutor` sử dụng `as_completed` để xử lý kết quả
- Vấn đề: Không có cơ chế để đảm bảo thứ tự folds (fold 1, 2, 3, 4, 5)
- Kết quả có thể đến theo thứ tự bất kỳ tùy thuộc vào thời gian xử lý
- Điều này không ảnh hưởng đến accuracy trung bình, nhưng làm cho logs khó theo dõi

**Giải pháp:**
- Sử dụng `sorted(futures.items())` hoặc `enumerate` để giữ thứ tự

---

## CÁC LỖI LOGIC NHẸ

### 11. [LOW] Không Kiểm Tra `test_idx_filtered` Trước Khi Sử Dụng
**File:** `core/optimization.py:309-320`

**Mô tả:**
- Sau khi lọc test indices, code sử dụng `test_idx_filtered` để đánh giá
- Nhưng không kiểm tra xem `test_idx_filtered` có rỗng không trước khi tính accuracy
- Nếu tất cả test indices đều nằm trong gap, `test_idx_filtered` sẽ rỗng và `accuracy_score` sẽ raise ValueError

**Giải pháp:**
```python
if len(test_idx_filtered) == 0:
    continue
```

---

### 12. [LOW] Thiếu Xử Lý Lỗi Khi `best_params` Không Tồn Tại
**File:** `core/optimization.py:460-475`

**Mô tả:**
- Sau khi `study.optimize()`, code truy cập `study.best_params` và `study.best_value`
- Nếu tất cả các trial đều fail (raise exception trong objective), study sẽ không có best trial
- Truy cập `study.best_params` sẽ raise `RuntimeError`

**Giải pháp:**
```python
if len(study.trials) == 0 or study.best_trial is None:
    log_warn("No successful trials, using default parameters")
    return XGBOOST_PARAMS.copy()
```

---

### 13. [LOW] `file_lock` Không Xử Lý Trường Hợp Lock File Tồn Tại Nhưng Không Thể Ghi
**File:** `core/optimization.py:43-65`

**Mô tả:**
- `file_lock` mở lock file với mode "w" (write)
- Nếu lock file tồn tại và được tạo bởi user khác, hoặc không có quyền ghi, sẽ raise PermissionError
- Không có xử lý exception cho trường hợp này

**Giải pháp:**
```python
try:
    lock_file = open(lock_file_path, "w")
except (PermissionError, OSError) as e:
    log_warn(f"Cannot create lock file: {e}")
    yield  # Continue without lock
    return
```

---

### 14. [LOW] Không Reset Random State Trong CV
**File:** `core/model.py:357`, `utils/cv_parallel.py:75`

**Mô tả:**
- Mỗi fold trong CV tạo model mới với `random_state=42`
- Điều này đảm bảo reproducibility cho từng fold riêng lẻ
- Nhưng khi chạy parallel CV, các process khác nhau có thể có random state giống nhau
- Điều này không phải là vấn đề nghiêm trọng, nhưng có thể gây ra kết quả không đa dạng

**Giải pháp:**
- Sử dụng `random_state=42 + fold_num` để đảm bảo mỗi fold có seed khác nhau

---

### 15. [LOW] Thiếu Kiểm Tra `last_row` Trong `predict_next_move`
**File:** `core/model.py:408-433`

**Mô tả:**
- `predict_next_move` không kiểm tra xem `last_row` có chứa tất cả `MODEL_FEATURES` không
- Nếu thiếu features, `X_new = last_row[MODEL_FEATURES]` sẽ raise KeyError
- Hoặc nếu features có giá trị NaN/Inf, prediction sẽ không chính xác

**Giải pháp:**
```python
missing_features = set(MODEL_FEATURES) - set(last_row.index if isinstance(last_row, pd.Series) else last_row.columns)
if missing_features:
    raise ValueError(f"Missing features: {missing_features}")

if not np.isfinite(last_row[MODEL_FEATURES]).all():
    raise ValueError("Features contain NaN or Inf values")
```

---

## TỔNG KẾT VÀ KHUYẾN NGHỊ

### Lỗi Cần Sửa Ngay (Critical + High):
1. ✅ Logic gap không nhất quán trong CV
2. ✅ Lỗi caching trong `apply_directional_labels`
3. ✅ Cơ chế file lock không đáng tin cậy
4. ✅ Side effect trong `_resolve_xgb_classifier`
5. ✅ Threshold calculation có thể tạo giá trị cực đoan

### Lỗi Cần Sửa Trong Phiên Bản Tiếp Theo (Medium):
6. Memory leak trong labeling
7. Float32 conversion mất precision
8. Hash toàn bộ DataFrame thay vì chỉ features cần thiết
9. Không xử lý predict_proba shape không đúng
10. Race condition trong parallel CV logging

### Cải Tiến Có Thể Thực Hiện (Low):
11-15. Các kiểm tra đầu vào và xử lý edge cases

### Test Cases Cần Thêm:
1. Test gap prevention với các scenario khác nhau
2. Test concurrent access vào study database
3. Test threshold calculation với dữ liệu pump/dump
4. Test cache hit/miss với DataFrame thay đổi
5. Test predict_next_move với missing features
6. Test class diversity handling trong CV
7. Test memory usage với large DataFrame
8. Test reproducibility của CV với parallel execution

---

## MÃ CODE KIỂM TRA TỰ ĐỘNG

```python
# Test gap prevention logic
def test_gap_prevention():
    train_idx = np.array([0, 1, 2, ..., 100])
    test_idx = np.array([110, 111, 112])  # Một phần trong gap
    
    # Kiểm tra logic hiện tại
    min_test_start = 100 + 24 + 1  # 125
    
    # Test case 1: test_idx bắt đầu trong gap
    assert test_idx[0] < min_test_start
    test_idx_filtered = test_idx[test_idx >= min_test_start]
    assert len(test_idx_filtered) == 0  # Nên skip fold này
    
    # Test case 2: test_idx bắt đầu sau gap nhưng có phần trong gap
    test_idx = np.array([125, 126, 110, 111])  # Lộn xộn thứ tự
    # Logic hiện tại chỉ kiểm tra test_idx[0], có thể miss các index sau

# Test threshold calculation
def test_threshold_extreme_values():
    # Giả lập pump 50%
    historical_pct = pd.Series([0.0, 0.5, 0.0, 0.0])
    base_threshold = historical_pct.abs().clip(lower=0.01)
    # base_threshold = [0.01, 0.5, 0.01, 0.01]
    
    atr_ratio = pd.Series([2.0, 2.0, 2.0, 2.0])
    final_threshold = base_threshold * atr_ratio
    # final_threshold = [0.02, 1.0, 0.02, 0.02]
    
    # Threshold 100% quá lớn, sẽ tạo ra toàn NEUTRAL labels
    assert (final_threshold <= 0.1).all(), "Threshold quá lớn!"
```

---

**Ngày tạo báo cáo:** 2026-01-30  
**Người kiểm tra:** Claude Code  
**Phiên bản module:** Latest
