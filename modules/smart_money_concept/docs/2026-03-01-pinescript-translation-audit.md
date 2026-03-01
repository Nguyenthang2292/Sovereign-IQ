# Audit: PineScript → Python Translation

**Date:** 2026-03-01  
**Source:** `docs/source_pine.txt` (LuxAlgo Smart Money Concepts v5, 848 lines)  
**Scope:** `core/` and `models/` sub-modules

---

## 1. SWING DETECTION — Khác cách tiếp cận nhưng chấp nhận được

| Aspect | PineScript | Python |
|---|---|---|
| Method | `leg()` — bar-by-bar rolling: `high[size] > ta.highest(size)` | `scipy.signal.argrelextrema` — local extrema so sánh 2 phía |
| Internal size | `5` | `internal_order=5` ✅ |
| Swing size | `swingsLengthInput` = **50** | `external_order=` **30** ❌ |

### Bug #1 — Medium
`external_order` mặc định **30** nhưng Pine mặc định **50**.  
**Fix:** Đổi `external_order=30` thành `external_order=50` trong `SMCAnalyzer.__init__()` và `detect_swings()`.

---

## 2. BOS DETECTION — Sai logic nghiêm trọng

### PineScript (trong `displayStructure()`)
```pinescript
if ta.crossover(close, p_ivot.currentLevel)  → Bullish BOS/CHoCH
if ta.crossunder(close, p_ivot.currentLevel) → Bearish BOS/CHoCH
```
- Chỉ dùng **`close`** để xác định crossover (`close > level` AND `close[1] <= level`).
- Kiểm tra real-time trên pivot **chưa bị crossed** (`not p_ivot.crossed`).

### Python (`core/bos.py`)
```python
breakout = df_range[
    (df_range["High"] > current.level)
    | (df_range["Open"] > current.level)
    | (df_range["Close"] > current.level)
    | (df_range["Low"] > current.level)
]
```

### Bug #2 — Critical
Dùng OR trên tất cả OHLC thay vì chỉ **close crossover**. Điều kiện quá lỏng — hầu như bất kỳ bar nào chạm level đều count là breakout.  
Pine yêu cầu `close` phải **cross qua** level (trước đó `close ≤ level`, bây giờ `close > level`).

### Bug #3 — Critical
Logic lặp giữa 2 swing liên tiếp cũng sai. Pine kiểm tra pivot gần nhất chưa crossed trên **mỗi bar**, không phải giữa 2 swing kế nhau.

---

## 3. CHOCH DETECTION — Sai hoàn toàn

### PineScript
CHoCH vs BOS phân biệt dựa trên **trend hiện tại tại thời điểm cross**:
- Nếu `trend = BEARISH` mà `close` cross over swing high → **Bullish CHoCH** (đổi hướng trend)
- Nếu `trend = BULLISH` mà `close` cross under swing low → **Bearish CHoCH**
- Còn lại → **BOS** (tiếp diễn trend)

### Python (`core/choch.py`)
```python
# Kiểm tra xem có swing_low nằm giữa 2 BOS timestamps không
for swing in swing_lows:
    if t_prev < swing.bar_time < t_curr:
        bullish_choch.append(t_prev)
```

### Bug #4 — Critical
Thuật toán hoàn toàn khác Pine. Python tìm swing nằm giữa 2 BOS event, trong khi Pine phân biệt CHoCH/BOS dựa trên **hướng trend tại thời điểm cross**. Kết quả sẽ rất khác nhau.

**Fix:** CHoCH và BOS phải được xác định cùng lúc trong hàm detect BOS, không tách riêng. Logic cần:
1. Track `current_trend` thay đổi sau mỗi crossover.
2. Nếu cross ngược chiều trend hiện tại → CHoCH, cùng chiều → BOS.

---

## 4. TREND DETECTION — Sai phương pháp

### PineScript
Trend được set bởi **structure break cuối cùng**:
```pinescript
// Khi bullish cross (BOS hoặc CHoCH) xảy ra:
t_rend.bias := BULLISH
// Khi bearish cross xảy ra:
t_rend.bias := BEARISH
```

### Python (`core/trend.py`)
```python
if last_high > prev_high and last_low > prev_low:
    return BULLISH
elif last_high < prev_high and last_low < prev_low:
    return BEARISH
```

### Bug #5 — Critical
Python dùng pattern HH+HL / LH+LL để xác định trend. Pine dùng **hướng structure break cuối cùng**. Hai phương pháp cho kết quả khác nhau, và trend sai sẽ làm CHoCH detection bị sai theo.

---

## 5. ORDER BLOCKS — Sai logic cơ bản

### PineScript (`storeOrdeBlock()`)
- OB được tạo **khi BOS/CHoCH xảy ra** (gọi `storeOrdeBlock()` ngay trong `displayStructure()`).
- Tìm bar có `parsedHigh` max (bearish OB) hoặc `parsedLow` min (bullish OB) trong range từ pivot → current bar.
- Áp dụng **high volatility bar filtering**: nếu `(high - low) >= 2 * volatilityMeasure` → swap `parsedHigh = low`, `parsedLow = high`.

Mitigation (`deleteOrderBlocks()`):
- Bearish OB bị xóa khi `mitigationSource > ob.barHigh`
- Bullish OB bị xóa khi `mitigationSource < ob.barLow`
- Mặc định `mitigationSource` = `high` (bearish) / `low` (bullish)

### Python (`core/order_block.py`)
- OB được tạo từ **cặp swing liên tiếp**, KHÔNG gắn với sự kiện BOS/CHoCH.
- Không có high volatility bar filtering (`parsedHigh` / `parsedLow`).
- Mitigation logic ngược: Python kiểm tra `Low < ob.level_y0` (bullish), Pine kiểm tra `mitigationSource < ob.barLow`.

### Bug #6 — Critical
Order block không kết nối với structure break. Trong SMC, OB chỉ có ý nghĩa khi tạo ra cùng lúc với BOS/CHoCH — đây là nguyên tắc cốt lõi của LuxAlgo SMC.

**Fix:** `storeOrderBlock()` cần được gọi bên trong hàm detect BOS/CHoCH, ngay khi phát hiện crossover.

---

## 6. EQUAL HIGHS/LOWS — Gần đúng nhưng sai source data

### PineScript
Dùng pivots riêng phát hiện với `equalHighsLowsLengthInput` (default **3**):
```pinescript
getCurrentStructure(equalHighsLowsLengthInput, true) // size = 3
```

### Python (`core/equal_hl.py`)
Dùng `internal_highs` / `internal_lows` (order=**5**).

### Bug #7 — Medium
Dùng sai source pivots. Nên detect pivots riêng với order=3 (equal HL length) thay vì tái sử dụng internal pivots order=5. Điều này khiến phát hiện EQH/EQL kém nhạy hơn so với Pine.

---

## 7. Features thiếu (có trong Pine, chưa implement)

| Feature | Pine Lines | Status |
|---|---|---|
| Fair Value Gaps (FVG) | 615–640 | ❌ Missing |
| Premium / Discount Zones | 770–780 | ❌ Missing |
| MTF Levels (Daily / Weekly / Monthly) | 660–710 | ❌ Missing |
| Strong / Weak High / Low | 720–745 | ❌ Missing |
| Trailing Extremes tracking | 715–720 | ❌ Missing |
| High Volatility Bar filtering for OB | 320–325 | ❌ Missing |
| Confluence Filter (internal structure) | 555–557 | ❌ Missing |
| Color Candles (trend coloring) | 785–788 | ➖ Display only, skip |

---

## Tổng kết mức độ nghiêm trọng

| # | Vấn đề | File | Mức độ |
|---|---|---|---|
| 1 | `external_order` 30 thay vì 50 | `core/swing.py`, `core/analyzer.py` | Medium |
| 2 | BOS dùng OR trên OHLC thay vì close crossover | `core/bos.py` | **Critical** |
| 3 | BOS lặp giữa 2 swing thay vì check per-bar | `core/bos.py` | **Critical** |
| 4 | CHoCH algorithm hoàn toàn khác Pine | `core/choch.py` | **Critical** |
| 5 | Trend dùng HH/HL pattern thay vì last structure break | `core/trend.py` | **Critical** |
| 6 | Order Block không gắn với BOS/CHoCH event | `core/order_block.py` | **Critical** |
| 7 | Equal HL dùng sai pivot source (order=5 vs 3) | `core/equal_hl.py` | Medium |

---

## Đề xuất thứ tự sửa

1. **Refactor `bos.py` + `choch.py`** thành một hàm duy nhất `detect_structure()` track trend state và phân loại BOS/CHoCH cùng lúc — đây là root cause của bugs 2, 3, 4, 5.
2. **Sửa `order_block.py`** để OB được tạo từ sự kiện structure break, thêm volatility filter.
3. **Sửa `swing.py`** default `external_order=50`.
4. **Sửa `equal_hl.py`** detect pivot riêng với order=3.
5. **Implement FVG** nếu cần dùng cho trading signal.
