# ATR Targets Module

Module tính toán các target prices dựa trên ATR (Average True Range) multiples.

## Tổng quan

Module này cung cấp các công cụ để tính toán và hiển thị các target prices dựa trên ATR multiples, thường được sử dụng trong phân tích kỹ thuật để xác định các mức giá mục tiêu tiềm năng.

## Cấu trúc Module

```
modules/atr_targets/
├── __init__.py
├── core/
│   ├── __init__.py
│   └── target_calculator.py    # Logic tính toán ATR targets
└── README.md
```

## Sử dụng

### Python API

```python
from modules.atr_targets import calculate_atr_targets, format_atr_target_display
from modules.common.utils import format_price

# Tính toán ATR targets
current_price = 100.0
atr = 2.0
direction = "UP"  # "UP", "DOWN", hoặc "NEUTRAL"
multiples = [1, 2, 3]

targets = calculate_atr_targets(
    current_price=current_price,
    atr=atr,
    direction=direction,
    multiples=multiples,
)

# Hiển thị kết quả
for target in targets:
    display_text = format_atr_target_display(target, format_price_func=format_price)
    print(display_text)
```

### Ví dụ Output

```
  ATR x1: $102.00 | Delta $2.00 (2.00%)
  ATR x2: $104.00 | Delta $4.00 (4.00%)
  ATR x3: $106.00 | Delta $6.00 (6.00%)
```

## API Reference

### `calculate_atr_targets()`

Tính toán các target prices dựa trên ATR multiples.

**Parameters:**
- `current_price` (float): Giá hiện tại
- `atr` (float): Giá trị ATR (Average True Range)
- `direction` (str): Hướng di chuyển dự kiến ("UP", "DOWN", "NEUTRAL")
- `multiples` (List[int], optional): Danh sách ATR multiples để tính toán (mặc định: [1, 2, 3])

**Returns:**
- `List[ATRTargetResult]`: Danh sách các kết quả tính toán

### `ATRTargetResult`

Dataclass chứa thông tin về một ATR target.

**Attributes:**
- `multiple` (int): ATR multiple (1, 2, 3, ...)
- `target_price` (float): Giá target tính toán
- `delta` (float): Delta (move_abs) - khoảng cách tuyệt đối từ giá hiện tại
- `delta_pct` (float): Delta theo phần trăm

### `format_atr_target_display()`

Format một ATR target result thành string để hiển thị.

**Parameters:**
- `result` (ATRTargetResult): Kết quả cần format
- `format_price_func` (Callable): Function để format giá

**Returns:**
- `str`: String đã được format

## Logic Tính Toán

1. **Xác định sign dựa trên direction:**
   - `"UP"` → `atr_sign = 1` (tăng giá)
   - `"DOWN"` → `atr_sign = -1` (giảm giá)
   - `"NEUTRAL"` hoặc khác → `atr_sign = 0` (không di chuyển)

2. **Tính target price:**
   ```
   target_price = current_price + atr_sign * multiple * atr
   ```

3. **Tính delta:**
   ```
   delta = abs(target_price - current_price)
   delta_pct = (delta / current_price) * 100
   ```

## Integration

Module này được sử dụng bởi:
- `modules.xgboost.cli.main`: Hiển thị ATR targets trong XGBoost prediction results

## Dependencies

- `dataclasses`: Cho ATRTargetResult dataclass
- `typing`: Cho type hints

## License

Same as main project.

