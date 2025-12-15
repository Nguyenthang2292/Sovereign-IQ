# HMM (Hidden Markov Model) Module

Module HMM cung cấp các triển khai Hidden Markov Model cho phân tích và dự đoán xu hướng thị trường cryptocurrency. Module này tích hợp nhiều chiến lược HMM khác nhau và kết hợp chúng để tạo ra tín hiệu giao dịch có độ tin cậy cao.

## 📋 Mục lục

- [Tổng quan](#tổng-quan)
- [Cấu trúc Module](#cấu-trúc-module)
- [Các HMM Strategies](#các-hmm-strategies)
- [Signal Processing](#signal-processing)
- [Cách sử dụng](#cách-sử-dụng)
- [Configuration](#configuration)
- [Ví dụ](#ví-dụ)

## 🎯 Tổng quan

Module HMM bao gồm 3 chiến lược HMM chính:

1. **HMM-Swings**: Sử dụng swing detection (điểm cao/thấp) để xác định trạng thái thị trường
2. **HMM-KAMA**: Sử dụng Kaufman Adaptive Moving Average (KAMA) làm đặc trưng cho HMM
3. **True High-Order HMM**: HMM bậc cao thực sự, dự đoán dựa trên k trạng thái trước đó

Tất cả các chiến lược được kết hợp thông qua hệ thống voting/scoring để tạo ra tín hiệu giao dịch cuối cùng.

## 📁 Cấu trúc Module

```
modules/hmm/
├── __init__.py              # Module exports
├── core/                    # Core HMM implementations
│   ├── __init__.py         # Core module exports
│   ├── swings/             # HMM-Swings strategy (modular)
│   │   ├── __init__.py
│   │   ├── models.py       # HMM_SWINGS, HighOrderHMM class
│   │   ├── state_conversion.py  # convert_swing_to_state
│   │   ├── optimization.py      # optimize_n_states
│   │   ├── model_creation.py    # create_hmm_model, train_model
│   │   ├── prediction.py        # predict functions
│   │   ├── swing_utils.py       # Utilities (timeout, safe_forward_backward)
│   │   ├── workflow.py          # hmm_swings function
│   │   └── strategy.py          # SwingsHMMStrategy class
│   ├── kama/               # HMM-KAMA strategy (modular)
│   │   ├── __init__.py
│   │   ├── models.py       # HMM_KAMA, model operations
│   │   ├── features.py     # prepare_observations
│   │   ├── analysis.py     # Secondary analysis (ARM, clustering)
│   │   ├── workflow.py     # hmm_kama function
│   │   ├── strategy.py     # KamaHMMStrategy class
│   │   └── utils.py        # Utilities (prevent_infinite_loop, timeout)
│   └── high_order/         # True High-Order HMM strategy (modular)
│       ├── __init__.py
│       ├── models.py       # TrueHighOrderHMM class
│       ├── state_expansion.py  # State space expansion functions
│       ├── model_creation.py   # create_high_order_hmm_model
│       ├── optimization.py    # optimize_order_k, optimize_n_states
│       ├── prediction.py       # Prediction functions
│       ├── workflow.py         # true_high_order_hmm function
│       └── strategy.py        # TrueHighOrderHMMStrategy class
├── signals/                 # Signal processing & combination
│   ├── strategy.py         # Strategy interface & result dataclass
│   ├── registry.py         # Strategy registry for dynamic loading
│   ├── combiner.py         # Signal combiner & voting
│   ├── voting.py           # Voting mechanisms
│   ├── scoring.py          # Signal scoring logic
│   ├── confidence.py       # Confidence calculation
│   ├── resolution.py       # Conflict resolution
│   └── utils.py            # Utility functions
├── cli/                     # CLI utilities
│   └── test_high_order.py  # High-order HMM testing
└── utils/                   # General utilities
```

## 🔬 Các HMM Strategies

### 1. HMM-Swings (`core/swings/`)

**Mô tả**: Sử dụng swing detection để xác định các điểm cao/thấp trong giá, sau đó chuyển đổi thành chuỗi trạng thái thị trường (Bullish, Neutral, Bearish).

**Đặc điểm**:
- Sử dụng `scipy.signal.argrelextrema` để phát hiện swing points
- Chuyển đổi swing highs/lows thành trạng thái: 0 (Downtrend), 1 (Sideways), 2 (Uptrend)
- Hỗ trợ strict mode và non-strict mode cho việc chuyển đổi swing-to-state
- Sử dụng `pomegranate.hmm.DenseHMM` cho mô hình HMM

**Class**: `SwingsHMMStrategy`

**Tham số chính**:
- `orders_argrelextrema`: Tham số order cho argrelextrema (mặc định: 5)
- `strict_mode`: Sử dụng strict mode cho swing-to-state conversion (mặc định: True)

### 2. HMM-KAMA (`core/kama/`)

**Mô tả**: Sử dụng KAMA (Kaufman Adaptive Moving Average) làm đặc trưng cho HMM, kết hợp với Association Rule Mining (ARM) và K-Means clustering để phân tích trạng thái.

**Đặc điểm**:
- Tính toán KAMA từ giá đóng cửa
- Sử dụng `hmmlearn.GaussianHMM` cho mô hình HMM
- Phân tích trạng thái bằng ARM (Apriori, FP-Growth) và K-Means
- Tạo nhiều tín hiệu từ các phương pháp khác nhau và kết hợp chúng

**Class**: `KamaHMMStrategy`

**Tham số chính**:
- `window_kama`: Kích thước cửa sổ cho KAMA (mặc định: 10)
- `fast_kama`: Fast period cho KAMA (mặc định: 2)
- `slow_kama`: Slow period cho KAMA (mặc định: 30)
- `window_size`: Kích thước cửa sổ cho HMM analysis (mặc định: 100)

### 3. True High-Order HMM (`core/high_order/`)

**Mô tả**: HMM bậc cao thực sự, sử dụng state space expansion để dự đoán trạng thái tiếp theo dựa trên k trạng thái trước đó thay vì chỉ 1 trạng thái.

**Đặc điểm**:
- **State Space Expansion**: Mở rộng không gian trạng thái từ n_base_states thành n_base_states^k
- **Tự động tối ưu order k**: Sử dụng BIC (Bayesian Information Criterion) để chọn order tối ưu
- **Cross-validation**: Sử dụng TimeSeriesSplit để đánh giá mô hình
- Hỗ trợ order từ min_order đến max_order (mặc định: 2-4)

**Class**: `TrueHighOrderHMMStrategy`

**Tham số chính**:
- `min_order`: Order tối thiểu (mặc định: 2)
- `max_order`: Order tối đa (mặc định: 4)
- `orders_argrelextrema`: Tham số cho swing detection (mặc định: 5)
- `strict_mode`: Strict mode cho swing-to-state (mặc định: True)

**Cách hoạt động**:
1. Phát hiện swing points và chuyển đổi thành base states (0, 1, 2)
2. Với mỗi order k từ min_order đến max_order:
   - Mở rộng state space: mỗi expanded state đại diện cho chuỗi k base states
   - Train HMM với expanded states
   - Đánh giá mô hình bằng BIC
3. Chọn order có BIC thấp nhất (mô hình tốt nhất)
4. Dự đoán trạng thái tiếp theo dựa trên k trạng thái gần nhất

## 🔄 Signal Processing

### Strategy Interface (`signals/strategy.py`)

Tất cả HMM strategies đều implement interface `HMMStrategy`:

```python
class HMMStrategy(ABC):
    def analyze(self, df: pd.DataFrame, **kwargs) -> HMMStrategyResult:
        """Phân tích dữ liệu và trả về tín hiệu giao dịch."""
        pass
```

**HMMStrategyResult**:
- `signal`: Tín hiệu giao dịch (LONG=1, HOLD=0, SHORT=-1)
- `probability`: Độ tin cậy (0.0 đến 1.0)
- `state`: Trạng thái nội bộ (strategy-specific)
- `metadata`: Dữ liệu bổ sung

### Strategy Registry (`signals/registry.py`)

Registry quản lý và load strategies động từ config:

- **HMMStrategyRegistry**: Singleton registry để quản lý strategies
- Load strategies từ `config/hmm.py` (`HMM_STRATEGIES`)
- Hỗ trợ enable/disable strategies
- Quản lý weights cho mỗi strategy

### Signal Combiner (`signals/combiner.py`)

**HMMSignalCombiner** kết hợp tín hiệu từ tất cả enabled strategies:

**Voting Mechanisms** (`signals/voting.py`):
1. **Simple Majority**: Đếm số lượng strategies đồng ý
2. **Weighted**: Tính tổng weighted scores
3. **Confidence Weighted**: Sử dụng probability * weight
4. **Threshold Based**: Yêu cầu tỷ lệ strategies đồng ý với confidence >= threshold

**Output**:
```python
{
    "signals": {
        "swings": 1,
        "kama": 1,
        "true_high_order": -1
    },
    "combined_signal": 1,  # LONG
    "confidence": 0.75,
    "votes": {1: 2, -1: 1, 0: 0},  # LONG: 2, SHORT: 1, HOLD: 0
    "metadata": {...}
}
```

### Conflict Resolution (`signals/resolution.py`)

Xử lý xung đột khi các strategies đưa ra tín hiệu khác nhau:
- So sánh confidence/probability
- Sử dụng dynamic threshold dựa trên volatility
- Ưu tiên strategies có weight cao hơn

### Confidence Calculation (`signals/confidence.py`)

Tính toán độ tin cậy cho tín hiệu:
- `calculate_kama_confidence`: Confidence cho HMM-KAMA
- `calculate_combined_confidence`: Confidence tổng hợp từ nhiều strategies

## 💻 Cách sử dụng

### Sử dụng cơ bản

```python
import pandas as pd
from modules.hmm.signals.combiner import combine_signals

# Chuẩn bị dữ liệu OHLCV
df = pd.DataFrame({
    'open': [...],
    'high': [...],
    'low': [...],
    'close': [...],
    'volume': [...]
})

# Kết hợp tín hiệu từ tất cả strategies
result = combine_signals(df)

# Lấy tín hiệu cuối cùng
signal = result["combined_signal"]  # 1 (LONG), 0 (HOLD), -1 (SHORT)
confidence = result["confidence"]   # 0.0 đến 1.0

# Xem tín hiệu từ từng strategy
for strategy_name, signal_value in result["signals"].items():
    print(f"{strategy_name}: {signal_value}")
```

### Sử dụng từng strategy riêng lẻ

```python
from modules.hmm.core.swings import hmm_swings
from modules.hmm.core.kama import hmm_kama
from modules.hmm.core.high_order import true_high_order_hmm

# HMM-Swings
swings_result = hmm_swings(df)

# HMM-KAMA
kama_result = hmm_kama(df, window_kama=10, fast_kama=2, slow_kama=30)

# True High-Order HMM
high_order_result = true_high_order_hmm(
    df,
    min_order=2,
    max_order=4,
    train_ratio=0.8
)
```

### Sử dụng Strategy Classes trực tiếp

```python
from modules.hmm.core.swings import SwingsHMMStrategy
from modules.hmm.core.kama import KamaHMMStrategy
from modules.hmm.core.high_order import TrueHighOrderHMMStrategy

# Tạo strategy instance
swings_strategy = SwingsHMMStrategy(
    name="swings",
    weight=1.0,
    enabled=True,
    orders_argrelextrema=5,
    strict_mode=True
)

# Phân tích dữ liệu
result = swings_strategy.analyze(df)
print(f"Signal: {result.signal}, Probability: {result.probability}")
```

### Sử dụng Strategy Registry

```python
from modules.hmm.signals.registry import HMMStrategyRegistry

# Lấy default registry
registry = HMMStrategyRegistry()

# Lấy tất cả enabled strategies
enabled_strategies = registry.get_enabled()

# Chạy từng strategy
for strategy in enabled_strategies:
    result = strategy.analyze(df)
    print(f"{strategy.name}: {result.signal} (prob: {result.probability:.3f})")
```

## ⚙️ Configuration

Cấu hình được định nghĩa trong `config/hmm.py`:

### Strategy Configuration

```python
HMM_STRATEGIES = {
    "swings": {
        "enabled": True,
        "weight": 1.0,
        "class": "modules.hmm.core.swings.SwingsHMMStrategy",
        "params": {
            "orders_argrelextrema": 5,
            "strict_mode": True,
        }
    },
    "kama": {
        "enabled": True,
        "weight": 1.5,
        "class": "modules.hmm.core.kama.KamaHMMStrategy",
        "params": {
            "window_kama": 10,
            "fast_kama": 2,
            "slow_kama": 30,
            "window_size": 100,
        }
    },
    "true_high_order": {
        "enabled": True,
        "weight": 1.0,
        "class": "modules.hmm.core.high_order.TrueHighOrderHMMStrategy",
        "params": {
            "min_order": 2,
            "max_order": 4,
        }
    },
}
```

### Voting Configuration

```python
HMM_VOTING_MECHANISM = "weighted"  # Options: "simple_majority", "weighted", "confidence_weighted", "threshold_based"
HMM_VOTING_THRESHOLD = 0.5  # Used for threshold_based voting
```

### KAMA Configuration

```python
HMM_WINDOW_KAMA_DEFAULT = 10
HMM_FAST_KAMA_DEFAULT = 2
HMM_SLOW_KAMA_DEFAULT = 30
HMM_WINDOW_SIZE_DEFAULT = 100
```

### High-Order HMM Configuration

```python
HMM_HIGH_ORDER_MIN_ORDER_DEFAULT = 2
HMM_HIGH_ORDER_MAX_ORDER_DEFAULT = 4
HMM_HIGH_ORDER_ORDERS_ARGRELEXTREMA_DEFAULT = 5
HMM_HIGH_ORDER_STRICT_MODE_DEFAULT = True
```

## 📊 Ví dụ

### Ví dụ 1: Sử dụng Signal Combiner

```python
import pandas as pd
from modules.hmm.signals.combiner import combine_signals

# Load dữ liệu
df = pd.read_csv("btc_data.csv")

# Kết hợp tín hiệu
result = combine_signals(df)

# In kết quả
print(f"Combined Signal: {result['combined_signal']}")
print(f"Confidence: {result['confidence']:.3f}")
print(f"Votes: LONG={result['votes'][1]}, SHORT={result['votes'][-1]}, HOLD={result['votes'][0]}")

# Xem chi tiết từng strategy
for name, signal in result["signals"].items():
    metadata = result["metadata"][name]
    print(f"{name}: {signal} (metadata: {metadata})")
```

### Ví dụ 2: Tùy chỉnh Strategy Configuration

```python
from modules.hmm.signals.registry import HMMStrategyRegistry
from modules.hmm.core.swings import SwingsHMMStrategy

# Tạo custom registry
registry = HMMStrategyRegistry()

# Tạo custom strategy
custom_strategy = SwingsHMMStrategy(
    name="custom_swings",
    weight=2.0,  # Weight cao hơn
    enabled=True,
    orders_argrelextrema=7,  # Tham số tùy chỉnh
    strict_mode=False
)

# Đăng ký strategy
registry.register(custom_strategy)

# Sử dụng combiner với custom registry
from modules.hmm.signals.combiner import HMMSignalCombiner
combiner = HMMSignalCombiner(registry=registry)
result = combiner.combine(df)
```

### Ví dụ 3: So sánh các Voting Mechanisms

```python
from modules.hmm.signals.combiner import HMMSignalCombiner
from modules.hmm.signals.registry import HMMStrategyRegistry
from modules.hmm.signals.voting import VotingMechanism

registry = HMMStrategyRegistry()

# Test với simple majority
combiner = HMMSignalCombiner(registry=registry)
combiner.voting_mechanism = VotingMechanism.simple_majority
result1 = combiner.combine(df)

# Test với weighted voting
combiner.voting_mechanism = VotingMechanism.weighted_voting
result2 = combiner.combine(df)

# Test với confidence weighted
combiner.voting_mechanism = VotingMechanism.confidence_weighted
result3 = combiner.combine(df)

print(f"Simple Majority: {result1['combined_signal']}")
print(f"Weighted: {result2['combined_signal']}")
print(f"Confidence Weighted: {result3['combined_signal']}")
```

## 🔍 Chi tiết kỹ thuật

### State Space Expansion (High-Order HMM)

Với order k, không gian trạng thái được mở rộng từ n_base_states thành n_base_states^k:

- **Base states**: 0 (Down), 1 (Side), 2 (Up) → 3 states
- **Order 2**: 3² = 9 expanded states
  - State 0: (0, 0)
  - State 1: (0, 1)
  - State 2: (0, 2)
  - State 3: (1, 0)
  - ...
- **Order 3**: 3³ = 27 expanded states

Mỗi expanded state đại diện cho một chuỗi k base states, cho phép HMM "nhớ" k trạng thái trước đó.

### BIC (Bayesian Information Criterion)

BIC được sử dụng để chọn order tối ưu:

```
BIC = -2 * log_likelihood + k * log(n)
```

Trong đó:
- `log_likelihood`: Log-likelihood của mô hình
- `k`: Số lượng tham số (tăng theo order)
- `n`: Số lượng observations

Order có BIC thấp nhất được chọn làm order tối ưu.

### Cross-Validation

Sử dụng `TimeSeriesSplit` để đảm bảo:
- Training data luôn trước test data (temporal order)
- Không có data leakage
- Đánh giá mô hình một cách công bằng

## 📝 Notes

- Tất cả strategies đều implement `HMMStrategy` interface để đảm bảo tính nhất quán
- Strategy registry cho phép thêm/bớt strategies mà không cần sửa code
- Voting mechanisms có thể được thay đổi trong config hoặc runtime
- Module hỗ trợ cả backward compatibility và extensibility

## 🔗 Liên kết

- [Pomegranate HMM Documentation](https://pomegranate.readthedocs.io/)
- [HMMLearn Documentation](https://hmmlearn.readthedocs.io/)
- [Kaufman Adaptive Moving Average](https://www.investopedia.com/terms/k/kaufman-adaptive-moving-average-kama.asp)

