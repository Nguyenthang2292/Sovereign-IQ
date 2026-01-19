# Decision Matrix Implementation - Hướng dẫn sử dụng

## Tổng quan

Decision Matrix module cung cấp 2 hệ thống classification:

1. **Voting System**: Hệ thống voting dựa trên weighted impact và feature importance.
2. **Random Forest Algorithm**: Implementation đúng theo Pine Script (`source_pine.txt`), sử dụng pattern-based classification với historical data.

Cả hai hệ thống có thể được sử dụng độc lập hoặc kết hợp để đưa ra dự đoán chính xác hơn.

## Files cấu trúc

### Main Files (Workflows)
- `main_hybrid.py` - **Phương án 1 (Hybrid)**: Kết hợp ATC scan, Range Oscillator filter và Decision Matrix voting.
- `main_voting.py` - **Phương án 2 (Pure Voting)**: Thay thế hoàn toàn bằng voting system, tính toán song song.

### Module Files
- `modules/decision_matrix/classifier.py`: `DecisionMatrixClassifier` class (wrapper chính).
- `modules/decision_matrix/random_forest_core.py`: `RandomForestCore` (thuật toán cốt lõi).
- `modules/decision_matrix/training_data.py`: Storage cho training data.
- `modules/decision_matrix/pattern_matcher.py`: Logic so khớp pattern.
- `modules/decision_matrix/shuffle.py`: Fisher-Yates shuffle.
- `modules/decision_matrix/config.py`: Configuration classes.
- `modules/decision_matrix/threshold.py`: Tính toán threshold động.

## Cài đặt

Module tự quản lý dependencies (chủ yếu là `numpy`), không cần cài đặt thêm packages ngoài môi trường dự án chuẩn.

---

## Hướng dẫn nhanh (Quick Start)

### 1. Minimal Example (Random Forest)

```python
from modules.decision_matrix import (
    DecisionMatrixClassifier,
    FeatureType,
)

# Initialize classifier
classifier = DecisionMatrixClassifier()

# Collect training data (accumulate over time)
classifier.add_training_sample(x1=50.0, x2=1000.0, y=1)
classifier.add_training_sample(x1=45.0, x2=800.0, y=0)
# ... add more samples

# Classify
results = classifier.classify_with_random_forest(
    x1=52.0,
    x2=1200.0,
    y=1,
    x1_type="RSI",
    x2_type="Volume",
)

print(f"Prediction: {results['vote']}")
print(f"Accuracy: {results['accuracy']:.2f}%")
```

### 2. Minimal Example (Voting System)

```python
from modules.decision_matrix import DecisionMatrixClassifier

# Initialize with indicators
classifier = DecisionMatrixClassifier(indicators=["atc", "oscillator"])

# Add votes from indicators
classifier.add_node_vote("atc", vote=1, signal_strength=0.7, accuracy=0.65)
classifier.add_node_vote("oscillator", vote=1, signal_strength=0.8, accuracy=0.72)

# Calculate weighted impact
classifier.calculate_weighted_impact()

# Get cumulative vote
vote, weighted_score, breakdown = classifier.calculate_cumulative_vote(
    threshold=0.5, min_votes=2
)

print(f"Vote: {vote}, Score: {weighted_score:.2f}")
```

### 3. Using Individual Components

Nếu bạn muốn sử dụng từng component riêng lẻ:

```python
from modules.decision_matrix import (
    TrainingDataStorage,
    ThresholdCalculator,
    RandomForestCore,
)

# Training Data Storage
storage = TrainingDataStorage(training_length=850)
storage.add_sample(1.0, 2.0, 1)

# Random Forest Core
rf = RandomForestCore()
x1_matrix = storage.get_x1_matrix()
x2_matrix = storage.get_x2_matrix()

results = rf.classify(
    x1_matrix, x2_matrix, 
    current_x1=1.5, current_x2=2.5,
    x1_threshold=0.5, x2_threshold=1.0
)
```

---

## Chi tiết thuật toán (Algorithm Details)

### Random Forest Algorithm
Implementation dựa trên `source_pine.txt` (lines 84-170):

1. **Collect Training Data**: Lưu trữ cặp [feature, label] lịch sử.
2. **Shuffle Data**: Randomize training matrices bằng Fisher-Yates (đảm bảo tính ngẫu nhiên như Pine Script).
3. **Match Patterns**: Tìm dữ liệu lịch sử nằm trong khoảng `±threshold` của giá trị hiện tại.
   - Volume: `stdev(volume, 14)`
   - Z-Score: `0.05`
   - Khác (RSI, MFI, v.v.): `0.5`
4. **Count Pass/Fail**: Đếm số lượng match có label=1 vs label=0.
5. **Vote**: So sánh total passes vs fails.

### Voting System
Hệ thống bỏ phiếu có trọng số:

1. **Weight Capping**: Tự động giới hạn trọng số để tránh việc một chỉ báo chiếm ưu thế quá lớn.
   - **N=2**: Cap ở mức 60% (tỷ lệ tối đa 60-40).
   - **N>=3**: Cap ở mức 40% (tỷ lệ tối đa 40-30-30...).
2. **Thresholds**:
   - `threshold`: Điểm số trọng số tối thiểu để vote 1 (default 0.5).
   - `min_votes`: Số lượng chỉ báo tối thiểu phải đồng thuận.

---

## Workflows thực tế

Module đã tích hợp sẵn 2 workflows chính trong thư mục gốc:

### Phương án 1: Hybrid Approach (`main_hybrid.py`)
Kết hợp tuần tự: ATC Scan → Range Oscillator Filter → SPC Filter → Decision Matrix Voting.

```bash
python main_hybrid.py --timeframe 1h --enable-spc --use-decision-matrix --voting-threshold 0.6
```

### Phương án 2: Pure Voting System (`main_voting.py`)
Tính toán song song tất cả signals và đưa vào Voting System.

```bash
python main_voting.py --timeframe 1h --voting-threshold 0.5 --min-votes 2
```

---

## Cấu hình (Configuration)

Sử dụng `RandomForestConfig` để quản lý tham số:

```python
from modules.decision_matrix import RandomForestConfig, FeatureType, TargetType

config = RandomForestConfig(
    training_length=850,
    x1_type=FeatureType.RSI,
    x2_type=FeatureType.VOLUME,
    target_type=TargetType.RED_GREEN_CANDLE,
    # Các tham số optional khác:
    rsi_length=14,
    atr_length=14
)

# Export/Import dict
config_dict = config.to_dict()
new_config = RandomForestConfig.from_dict(config_dict)
```

## Best Practices

1. **Training Data Collection**: 
   - Thu thập data liên tục qua từng nến (time-series).
   - Không add toàn bộ data một lần nếu muốn mô phỏng đúng quá trình học.

2. **Voting Sequence**:
   - Luôn gọi `calculate_weighted_impact()` trước khi `calculate_cumulative_vote()`.
   - Nếu thiếu bước này sẽ gây ra lỗi `ValueError`.

3. **Choosing Thresholds**:
   - Conservative: `threshold=0.6`, `min_votes=3` (ít tín hiệu, độ chính xác cao hơn).
   - Aggressive: `threshold=0.3`, `min_votes=1` (nhiều tín hiệu hơn).

## Troubleshooting

- **Lỗi "Missing feature importance data"**:
  - Nguyên nhân: Gọi `calculate_weighted_impact` khi chưa add votes hoặc thiếu tham số accuracy.
  - Khắc phục: Đảm bảo gọi `add_node_vote` cho các indicators đã khai báo.

- **Lỗi Import**:
  - Đảm bảo đang chạy python từ root directory của project.
  - Kiểm tra `__init__.py` đã export đúng class.

## Testing

```bash
# Run all tests
pytest modules/decision_matrix/test_decision_matrix.py -v
```

Tests cover: Shuffle mechanism, Threshold calculation, Pattern matching, và Storage logic.
