# Fix Plan: Gann Square Module — Từ Horizontal Bands → Gann Fan Lines

**Ngày:** 2026-02-26  
**Vấn đề:** Module vẽ horizontal bands (giống Fibonacci) thay vì Gann Fan Lines chéo từ pivot point.

**Thiết kế đã xác nhận:**

- Góc: **1×1** (đường chéo 45° theo tỷ lệ price/time của chart)
- DOWN: fan lines từ **Swing High** hướng xuống
- UP: fan lines từ **Swing Low** hướng lên
- Scale: **tự động** theo price_range và số candle trên chart matplotlib

---

## Vấn Đề 1 — `gann_calculator.py`: Boundaries tính sai

### Hiện tại (SAI)

```python
boundaries = [
    high_price,
    high_price - price_range * 0.25,
    high_price - price_range * 0.50,
    high_price - price_range * 0.75,
    low_price,
]
```

Code đang chia price range đều thành 4 phần bằng nhau — đây là **Fibonacci Retracement**, không phải Gann Fan.

### Cần sửa

`GannZone` không còn có thể dùng `upper_price`/`lower_price` cố định vì boundary của fan line **thay đổi theo thời gian** (là hàm của `t`). Cần đổi model:

- **DOWN trend**: từ pivot `(t_high, p_high)`, slope = `-price_per_candle`
  - Zone 1 upper = đường ngang `p_high` (flat top)
  - Zone 1 lower = 1×1 line: `p_high - slope * (t - t_high)`
  - Zone 2 lower = 2×1 line (dốc gấp đôi)
  - Zone 3 lower = 3×1 line
  - Zone 4 lower = 4×1 line (đến swing_low tại `t_low`)

- **UP trend**: từ pivot `(t_low, p_low)`, slope = `+price_per_candle`
  - Zone 1 upper = 1×1 line lên
  - Zone 2 upper = 2×1 line
  - ...

### Công thức slope (1×1)

```
price_per_candle = price_range / (t_low - t_high)   # cho DOWN
```

Các slope của 4 đường fan:

```
line_0: slope × (1/4)  →  chia 4 vùng đều về thời gian
line_1: slope × (2/4)
line_2: slope × (3/4)
line_3: slope × (4/4) = slope  ← đường 1×1 thực sự
```

> Cách này chia thời gian từ `t_pivot` đến `t_close` thành 4 phần đều nhau để tạo ra 4 fan zones tương tự ảnh reference TradingView.

### Thay đổi `GannZone` dataclass

Cần thêm field để biểu diễn đường fan:

```python
@dataclass
class GannZone:
    zone_number: int
    label: str
    is_tradeable: bool
    signal: SignalCode
    # Fan line định nghĩa boundary dưới của zone (nếu DOWN) hoặc trên (nếu UP)
    pivot_index: int        # index của pivot point (t_high hoặc t_low)
    pivot_price: float      # giá tại pivot
    slope: float            # delta_price / candle (âm=DOWN, dương=UP)

    def price_at(self, candle_index: int) -> float:
        """Giá của đường fan tại candle index bất kỳ."""
        return self.pivot_price + self.slope * (candle_index - self.pivot_index)
```

### `GannSquareResult` cũng cần update

Thêm `pivot_index` và `price_per_candle` vào result để chart generator dùng.

---

## Vấn Đề 2 — `gann_chart_generator.py`: Vẽ sai loại đường

### Hiện tại (SAI)

```python
ax.axhspan(zone.lower_price, zone.upper_price, ...)  # dải ngang
ax.axhline(y=zone.upper_price, ...)                  # đường ngang
```

### Cần sửa: Vẽ fan lines chéo + fill vùng tam giác

**Bước 1** — Sinh ra mảng `x` (candle indices) từ pivot đến cuối chart:

```python
x_range = np.arange(pivot_index, len(df))
timestamps = df.index[x_range]
```

**Bước 2** — Tính giá của từng fan line tại mỗi candle:

```python
fan_prices = [zone.price_at(i) for i in x_range]
```

**Bước 3** — Vẽ đường fan:

```python
ax.plot(timestamps, fan_prices, color=color, linewidth=1.2, linestyle='--', alpha=0.7)
```

**Bước 4** — Fill giữa 2 fan lines liền kề để tạo vùng màu:

```python
ax.fill_between(timestamps, upper_fan_prices, lower_fan_prices,
                alpha=zone_alpha, color=color)
```

**Bước 5** — Đặt zone label ở giữa vùng (ở candle cuối cùng):

```python
mid_price = (upper_fan[-1] + lower_fan[-1]) / 2
ax.text(timestamps[-1], mid_price, zone.label, ...)
```

### Xác định current zone (thay `contains()`)

Vì zone không còn là dải ngang, `GannZone.contains(price)` cần dùng giá tại **candle index hiện tại**:

```python
def contains_at(self, price: float, current_index: int) -> bool:
    upper = self.upper_fan.price_at(current_index)
    lower = self.lower_fan.price_at(current_index)
    return lower < price <= upper
```

---

## Vấn Đề 3 — `gann_signal_engine.py`: Prompt cũ mô tả sai

### File: `prompts/gann_analysis.txt`

Dòng 25 nói `"4 horizontal Gann zones"` — cần đổi thành mô tả fan lines:

```
- 4 diagonal Gann Fan zones radiating from the pivot point
  (Zone 1 & 2 are active trade zones, Zone 3 & 4 are skip zones)
```

Phần Zone Breakdown trong prompt cũng cần thêm note rằng boundaries là **dynamic** (giá tại thời điểm hiện tại):

```
Zone Breakdown (prices at current candle):
  Zone 1: {ZONE1_UPPER} → {ZONE1_LOWER}  ({ZONE1_SIGNAL})
  ...
```

---

## Vấn Đề 4 — Tests: Cần update sau khi đổi model

Các tests trong `test_gann_calculator.py` hiện assert `upper_price`/`lower_price` là float cố định — sẽ fail sau khi đổi sang fan model. Cần viết lại:

| Test cũ | Test mới |
|---------|----------|
| `z1.upper_price == 100.0` | `z1.price_at(pivot_index) == pytest.approx(100.0)` |
| `z1.lower_price == 90.0` | Fan line tại pivot = 100.0, tại `t_low` = 60.0 |
| `zone.contains(95.0)` | `zone.contains_at(95.0, current_index)` |

---

## Thứ Tự Implement

```
1. [gann_calculator.py]      Thiết kế lại GannZone + GannCalculator
2. [gann_chart_generator.py] Vẽ fan lines thay vì axhspan
3. [prompts/gann_analysis.txt] Cập nhật mô tả zones
4. [tests/test_gann_calculator.py] Rewrite tests theo model mới
5. [tests/test_gann_chart_generator.py] Thêm smoke test cho fan lines
```

---

## Không Thay Đổi

- `swing_detector.py` — giữ nguyên, logic detect pivot vẫn đúng
- `gann_signal_engine.py` (phần orchestration) — giữ nguyên pipeline
- `cli/` — không thay đổi
- `GannSquareResult.trend`, `.signal_code`, `.current_zone` — giữ nguyên interface public
