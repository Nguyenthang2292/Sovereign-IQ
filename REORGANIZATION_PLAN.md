# Kế hoạch tổ chức lại pairs_trading module

## Cấu trúc mới

```
modules/pairs_trading/
├── __init__.py                    # Main exports (đã cập nhật)
├── core/                          # Core analysis components
│   ├── __init__.py                # ✅ Đã tạo
│   ├── pairs_analyzer.py          # ⏳ Cần di chuyển
│   ├── pair_metrics_computer.py   # ⏳ Cần di chuyển
│   └── opportunity_scorer.py      # ⏳ Cần di chuyển
├── metrics/                       # Metrics calculations
│   ├── __init__.py                # ✅ Đã tạo
│   ├── statistical_tests.py       # ⏳ Cần di chuyển
│   ├── hedge_ratio.py             # ⏳ Cần di chuyển
│   ├── zscore_metrics.py         # ⏳ Cần di chuyển
│   └── risk_metrics.py            # ⏳ Cần di chuyển
├── analysis/                      # Performance analysis
│   ├── __init__.py                # ⏳ Cần tạo (có file 'analysis' cần xóa trước)
│   └── performance_analyzer.py    # ⏳ Cần di chuyển
└── ui/                            # User interface
    ├── __init__.py                # ✅ Đã tạo
    ├── cli.py                     # ⏳ Cần di chuyển
    ├── display.py                 # ⏳ Cần di chuyển
    └── utils.py                   # ⏳ Cần di chuyển
```

## Các bước thực hiện

### 1. Di chuyển files (đã có script `reorganize_pairs_trading.py`)
- Chạy script hoặc di chuyển thủ công

### 2. Cập nhật imports trong các file đã di chuyển

#### core/pairs_analyzer.py
- Tìm: `from modules.pairs_trading.pair_metrics_computer`
- Đổi: `from modules.pairs_trading.core.pair_metrics_computer`
- Tìm: `from modules.pairs_trading.opportunity_scorer`
- Đổi: `from modules.pairs_trading.core.opportunity_scorer`
- Tìm: `from modules.pairs_trading.metrics` (nếu có)
- Đổi: `from modules.pairs_trading.metrics` (giữ nguyên vì metrics là sub-package)

#### core/pair_metrics_computer.py
- Tìm: `from modules.pairs_trading.metrics`
- Đổi: `from modules.pairs_trading.metrics` (giữ nguyên)

#### core/opportunity_scorer.py
- Kiểm tra imports liên quan đến metrics

#### metrics/*.py
- Các file metrics thường không import lẫn nhau, nhưng cần kiểm tra

#### analysis/performance_analyzer.py
- Kiểm tra imports

#### ui/cli.py, ui/display.py, ui/utils.py
- Kiểm tra imports giữa các file UI

### 3. Cập nhật imports trong main_pairs_trading.py
```python
# Trước:
from modules.pairs_trading.performance_analyzer import PerformanceAnalyzer
from modules.pairs_trading.pairs_analyzer import PairsTradingAnalyzer
from modules.pairs_trading.display import ...
from modules.pairs_trading.utils import ...
from modules.pairs_trading.cli import ...

# Sau:
from modules.pairs_trading.analysis import PerformanceAnalyzer
from modules.pairs_trading.core import PairsTradingAnalyzer
from modules.pairs_trading.ui import display_performers, display_pairs_opportunities
from modules.pairs_trading.ui import select_top_unique_pairs, ...
from modules.pairs_trading.ui import parse_args, ...
```

### 4. Cập nhật imports trong test files
- Tìm tất cả `from modules.pairs_trading.` trong tests/
- Cập nhật theo cấu trúc mới

### 5. Xóa file reorganize_pairs_trading.py sau khi hoàn thành

## Lưu ý
- File `analysis` (không phải thư mục) cần được xóa trước khi tạo thư mục `analysis/`
- Đảm bảo tất cả imports được cập nhật để tránh lỗi
- Chạy tests sau khi hoàn thành để đảm bảo mọi thứ hoạt động

