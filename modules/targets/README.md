# Targets Module

Module tính toán các target prices với nhiều phương pháp khác nhau.

## Tổng quan

Module này cung cấp các công cụ để tính toán và hiển thị các target prices dựa trên nhiều phương pháp khác nhau, thường được sử dụng trong phân tích kỹ thuật để xác định các mức giá mục tiêu tiềm năng.

Hiện tại module hỗ trợ:
- **ATR (Average True Range) multiples**: Tính target dựa trên bội số của ATR

Module được thiết kế để dễ dàng mở rộng với các phương pháp khác như:
- Fibonacci retracements/extensions
- Support/Resistance levels
- Pivot points
- Và các phương pháp khác...

## Cấu trúc Module

```
modules/targets/
├── __init__.py
├── core/
│   ├── __init__.py
│   ├── base.py              # Base classes và interfaces
│   └── atr.py               # ATR target implementation
└── README.md
```

## Kiến trúc

Module sử dụng pattern Strategy với base class `TargetCalculator`:

- **`TargetCalculator`**: Abstract base class định nghĩa interface chung
- **`ATRTargetCalculator`**: Implementation cụ thể cho ATR targets
- **`TargetResult`**: Base dataclass cho kết quả tính toán
- **`ATRTargetResult`**: Dataclass cụ thể cho ATR results

## Sử dụng

### Python API - ATR Targets

```python
from modules.targets import calculate_atr_targets, format_atr_target_display
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

### Sử dụng với Calculator class (Recommended cho mở rộng)

```python
from modules.targets import ATRTargetCalculator
from modules.common.utils import format_price

calculator = ATRTargetCalculator()

targets = calculator.calculate(
    current_price=100.0,
    atr=2.0,
    direction="UP",
    multiples=[1, 2, 3],
)

for target in targets:
    display_text = calculator.format_display(target, format_price_func=format_price)
    print(display_text)
```

### Ví dụ Output

```
  ATR x1: $102.00 | Delta $2.00 (2.00%)
  ATR x2: $104.00 | Delta $4.00 (4.00%)
  ATR x3: $106.00 | Delta $6.00 (6.00%)
```

## API Reference

### Convenience Functions (Backward Compatible)

#### `calculate_atr_targets()`

Tính toán các target prices dựa trên ATR multiples.

**Parameters:**
- `current_price` (float): Giá hiện tại
- `atr` (float): Giá trị ATR (Average True Range)
- `direction` (str): Hướng di chuyển dự kiến ("UP", "DOWN", "NEUTRAL")
- `multiples` (List[int], optional): Danh sách ATR multiples để tính toán (mặc định: [1, 2, 3])

**Returns:**
- `List[ATRTargetResult]`: Danh sách các kết quả tính toán

#### `format_atr_target_display()`

Format một ATR target result thành string để hiển thị.

**Parameters:**
- `result` (ATRTargetResult): Kết quả cần format
- `format_price_func` (Callable): Function để format giá

**Returns:**
- `str`: String đã được format

### Classes

#### `TargetCalculator` (Abstract Base Class)

Base class cho tất cả các target calculator.

**Methods:**
- `calculate(current_price, direction, **kwargs) -> List[TargetResult]`: Tính toán targets
- `format_display(result, format_price_func) -> str`: Format kết quả để hiển thị

#### `ATRTargetCalculator`

Implementation cụ thể cho ATR targets.

**Methods:**
- `calculate(current_price, atr, direction, multiples=None) -> List[ATRTargetResult]`
- `format_display(result, format_price_func) -> str`

#### `TargetResult`

Base dataclass chứa thông tin về một target.

**Attributes:**
- `target_price` (float): Giá target tính toán
- `delta` (float): Delta (move_abs) - khoảng cách tuyệt đối từ giá hiện tại
- `delta_pct` (float): Delta theo phần trăm
- `label` (str): Nhãn mô tả target

#### `ATRTargetResult`

Dataclass chứa thông tin về một ATR target.

**Attributes:**
- `multiple` (int): ATR multiple (1, 2, 3, ...)
- `target_price` (float): Giá target tính toán
- `delta` (float): Delta (move_abs) - khoảng cách tuyệt đối từ giá hiện tại
- `delta_pct` (float): Delta theo phần trăm
- `label` (str): Nhãn mô tả target (ví dụ: "ATR x1")

## Logic Tính Toán - ATR Targets

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

## Mở rộng Module

Để thêm một phương pháp tính target mới (ví dụ: Fibonacci):

1. Tạo file mới trong `core/` (ví dụ: `fibonacci.py`)
2. Implement class kế thừa từ `TargetCalculator`:

```python
from modules.targets.core.base import TargetCalculator, TargetResult

class FibonacciTargetCalculator(TargetCalculator):
    def calculate(self, current_price, direction, **kwargs):
        # Implementation logic
        pass
    
    def format_display(self, result, format_price_func):
        # Format logic
        pass
```

3. Export trong `core/__init__.py` và `__init__.py`
4. Thêm convenience functions nếu cần

## Integration

Module này được sử dụng bởi:
- `modules.xgboost.cli.main`: Hiển thị ATR targets trong XGBoost prediction results

## Dependencies

- `dataclasses`: Cho TargetResult dataclass
- `typing`: Cho type hints
- `abc`: Cho abstract base classes

## License

Same as main project.

