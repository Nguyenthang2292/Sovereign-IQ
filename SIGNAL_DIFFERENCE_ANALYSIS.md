# Phân Tích Sự Khác Biệt Signal Values: Python vs Rust

**Ngày phân tích**: 12 tháng 2, 2026

---

## 🔍 Vấn Đề Phát Hiện

Signal values giữa Python và Rust có sự khác biệt đáng kể:

| Symbol  | TF  | Python   | Rust     | Diff     |
|---------|-----|----------|----------|----------|
| BTCUSDT | 15m | -0.742469| -0.625000| 0.117469 |
| BTCUSDT | 1h  | 0.575693 | 0.750000 | 0.174307 |
| ETHUSDT | 1h  | 0.530193 | 0.833333 | 0.303140 |
| XMRUSDT | 15m | -0.469053| -0.083333| **0.385720** ← Max diff |

**Quan sát quan trọng**: 
- Rust values thường là phân số đơn giản: 0.75 (3/4), -0.625 (-5/8), 0.833333 (5/6), 0.416667 (5/12)
- Python values là continuous: -0.742469, 0.575693, 0.530193

---

## 🧐 Nguyên Nhân Gốc Rễ

### ❌ **Sai lầm trong Rust Implementation**

Rust implementation **THIẾU bước discretization** của Layer 1 signals!

### Python Implementation (ĐÚNG):

**File**: `modules/adaptive_trend_LTS_mini/core/compute_atc_signals/average_signal.py`

```python
# Line 127-129: DISCRETIZE Layer 1 signals TRƯỚC KHI weight
C = np.where(S_np > long_threshold, 1.0,      # > 0.1 → 1.0
             np.where(S_np < short_threshold, -1.0,  # < -0.1 → -1.0
                      0.0))                           # else → 0.0

# Line 132-133: Weighted average với DISCRETE values
nom_array = np.sum(C * E_np, axis=0)  # Tử số: sum(discrete_signal * equity)
den_array = np.sum(E_np, axis=0)       # Mẫu số: sum(equity)
avg_signal_array = nom_array / den_array
```

**Quy trình Python**:
1. Tính Layer 1 signals (raw values trong [-1, 1])
2. **DISCRETIZE** → chỉ giữ 3 giá trị: -1.0, 0.0, 1.0
3. Weight bằng Layer 2 equities
4. Tính weighted average

### Rust Implementation (SAI):

**File**: `modules/adaptive_trend_LTS_serverless/src/signal_detection.rs`

```rust
// Line 268-278: KHÔNG CÓ discretization!
let last_signal = if n > 0 { signal_series[n - 1] } else { 0.0 };
let combined_weight = ma_config.weight * equity_weight;

weighted_score_sum += last_signal * combined_weight;  // ← Sử dụng RAW signal!
total_weight += combined_weight;

// Line 280-283: Final score
let final_score = if total_weight > 0.0 {
    weighted_score_sum / total_weight
} else {
    0.0
};
```

**Quy trình Rust (SAI)**:
1. Tính Layer 1 signals (raw values trong [-1, 1])
2. ~~DISCRETIZE~~ ← **THIẾU BƯỚC NÀY!**
3. Weight bằng Layer 2 equities với **raw continuous values**
4. Tính weighted average

---

## 📊 Ví Dụ Minh Họa

Giả sử có 6 MAs với signals:
- EMA: 0.85, equity: 1.2
- HMA: 0.92, equity: 1.1
- WMA: 0.78, equity: 1.0
- DEMA: 0.65, equity: 0.9
- LSMA: 0.05, equity: 0.8
- KAMA: -0.02, equity: 0.7

### Python (Đúng):
1. **Discretize** (threshold = 0.1):
   - EMA: 0.85 → 1.0
   - HMA: 0.92 → 1.0
   - WMA: 0.78 → 1.0
   - DEMA: 0.65 → 1.0
   - LSMA: 0.05 → 0.0
   - KAMA: -0.02 → 0.0

2. **Weighted average**:
   ```
   numerator = (1.0×1.2 + 1.0×1.1 + 1.0×1.0 + 1.0×0.9 + 0.0×0.8 + 0.0×0.7)
             = 4.2
   denominator = (1.2 + 1.1 + 1.0 + 0.9 + 0.8 + 0.7) = 5.7
   average = 4.2 / 5.7 = 0.737
   ```

### Rust (Sai):
1. **KHÔNG discretize** - sử dụng raw values

2. **Weighted average**:
   ```
   numerator = (0.85×1.2 + 0.92×1.1 + 0.78×1.0 + 0.65×0.9 + 0.05×0.8 + -0.02×0.7)
             = 1.02 + 1.012 + 0.78 + 0.585 + 0.04 - 0.014 = 3.423
   denominator = 5.7
   average = 3.423 / 5.7 = 0.600
   ```

**Kết quả**: 0.737 (Python) vs 0.600 (Rust) - Sai số 0.137!

---

## 🔧 Giải Pháp

### Option 1: Sửa Rust để match Python (Khuyến nghị)

Thêm discretization step trong `compute_symbol_score()`:

**File**: `modules/adaptive_trend_LTS_serverless/src/signal_detection.rs`

```rust
pub fn compute_symbol_score(prices: &[f64], config: &ATCConfig) -> (f64, String) {
    let prices_arr = ArrayView1::from(prices);
    let n = prices.len();

    let mut weighted_score_sum = 0.0;
    let mut total_weight = 0.0;

    for ma_config in &config.ma_configs {
        let (signal_series, equity_weight) = calculate_layer1_signal(
            prices_arr,
            &ma_config.ma_type,
            ma_config.length,
            config.lambda_param,
            config.decay,
        );

        let last_signal = if n > 0 { signal_series[n - 1] } else { 0.0 };

        // ===== THÊM DISCRETIZATION (GIỐNG PYTHON) =====
        let discrete_signal = if last_signal > config.threshold {
            1.0
        } else if last_signal < -config.threshold {
            -1.0
        } else {
            0.0
        };
        // ===============================================

        let combined_weight = ma_config.weight * equity_weight;

        // Sử dụng discrete_signal thay vì last_signal
        weighted_score_sum += discrete_signal * combined_weight;
        total_weight += combined_weight;
    }

    let final_score = if total_weight > 0.0 {
        weighted_score_sum / total_weight
    } else {
        0.0
    };

    let signal_type = if final_score > config.threshold {
        "LONG".to_string()
    } else if final_score < -config.threshold {
        "SHORT".to_string()
    } else {
        "NEUTRAL".to_string()
    };

    (final_score, signal_type)
}
```

### Option 2: Document sự khác biệt

Nếu muốn giữ continuous values trong Rust (có thể có lý do kỹ thuật), cần:
1. Document rõ ràng sự khác biệt này
2. Thêm parameter `discretize: bool` để người dùng chọn
3. Update benchmark để compare cả 2 modes

---

## 📈 Tác Động

### Trading Impact:
- **Signal consistency**: 88.9% (8/9) - tương đối tốt
- **Signal value difference**: Max 0.386 - có thể ảnh hưởng đến:
  - Position sizing (nếu sử dụng signal strength)
  - Risk management
  - Backtesting results

### Khuyến nghị:
**🔴 CRITICAL**: Cần sửa để đảm bảo consistency với Python implementation

**Lý do**:
1. Python implementation đã được test trong production
2. Discretization là phần quan trọng của thuật toán ATC
3. Sự khác biệt này làm mất ý nghĩa của "serverless port"

---

## ✅ Action Items

1. **Immediate** (30 phút):
   - [ ] Thêm discretization vào Rust `compute_symbol_score()`
   - [ ] Chạy lại benchmark
   - [ ] Verify signal consistency = 100%

2. **Follow-up** (1 giờ):
   - [ ] Thêm unit tests cho discretization
   - [ ] Thêm integration tests so sánh Python vs Rust
   - [ ] Update documentation

3. **Long-term**:
   - [ ] Xem xét có nên hỗ trợ both modes (discrete vs continuous)
   - [ ] Performance impact của discretization step

---

## 📝 Kết Luận

**Root cause**: Rust implementation thiếu bước discretization của Layer 1 signals trước khi tính weighted average.

**Fix**: Thêm discretization step như Python implementation.

**Expected result**: Signal consistency tăng từ 88.9% lên ~100%, max difference giảm từ 0.386 xuống <0.001.

**Priority**: 🔴 **CRITICAL** - Cần sửa ngay để đảm bảo correctness.
