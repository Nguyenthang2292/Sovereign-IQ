# XGBoost Prediction Documentation

Tài liệu cho XGBoost prediction component.

## Overview

XGBoost prediction component sử dụng machine learning (XGBoost) để dự đoán hướng di chuyển tiếp theo của cryptocurrency pairs.

## Components

### Model
- **Location:** `modules/xgboost/model.py`
- XGBoost model training và prediction
- Multi-class classification (UP, NEUTRAL, DOWN)

### Labeling
- **Location:** `modules/xgboost/labeling.py`
- Dynamic labeling dựa trên volatility
- Triple-barrier method với adaptive thresholds

### CLI
- **Location:** `modules/xgboost/cli.py`
- Command-line interface parser
- Input validation và prompts

### Display
- **Location:** `modules/xgboost/display.py`
- Classification report formatting
- Confusion matrix visualization

### Optimization
- **Location:** `modules/xgboost/optimization.py`
- Hyperparameter optimization với Optuna
- Study management và caching
- Time-series cross-validation với gap prevention

## Usage

```bash
python xgboost_prediction_main.py
```

## Configuration

Tất cả config được định nghĩa trong `modules/config.py` section **XGBoost Prediction Configuration**:
- `TARGET_HORIZON` - Số candles để predict ahead
- `TARGET_BASE_THRESHOLD` - Base threshold cho labeling
- `XGBOOST_PARAMS` - Model hyperparameters
- `MODEL_FEATURES` - List các features sử dụng

## Features

- Multi-exchange support với fallback
- Advanced technical indicators
- Dynamic threshold adjustment
- Real-time prediction với confidence scores

## Hyperparameter Optimization

Module này cung cấp công cụ tự động tối ưu hyperparameters cho XGBoost model sử dụng Optuna.

### Tính năng

- **HyperparameterTuner**: Tự động tìm kiếm bộ tham số tối ưu với Optuna
- **StudyManager**: Quản lý và lưu trữ kết quả optimization studies
- **Time-Series Cross-Validation**: Sử dụng TimeSeriesSplit với gap prevention để tránh data leakage
- **Caching**: Tự động load cached parameters nếu study còn hợp lệ

### Cài đặt

Optuna đã được bao gồm trong `requirements-ml.txt`. Cài đặt bằng:

```bash
pip install -r requirements-ml.txt
```

### Sử dụng cơ bản

#### 1. Tối ưu hyperparameters cho một symbol/timeframe

```python
from modules.xgboost.optimization import HyperparameterTuner
import pandas as pd

# Chuẩn bị dữ liệu (DataFrame với MODEL_FEATURES và "Target" column)
df = prepare_data()  # Your data preparation function

# Tạo tuner
tuner = HyperparameterTuner(
    symbol="BTCUSDT",
    timeframe="1h"
)

# Chạy optimization
best_params = tuner.optimize(
    df=df,
    n_trials=100,  # Số lượng trials
    n_splits=5,    # Số folds cho cross-validation
)

print(f"Best parameters: {best_params}")
```

#### 2. Sử dụng cached parameters

```python
# Tự động load cached params nếu có (trong vòng 30 ngày)
tuner = HyperparameterTuner(symbol="BTCUSDT", timeframe="1h")
best_params = tuner.get_best_params(df=df, use_cached=True)
```

#### 3. Quản lý studies

```python
from modules.xgboost.optimization import StudyManager

# Tạo StudyManager
manager = StudyManager(storage_dir="artifacts/xgboost/optimization")

# Load best params từ study gần nhất
best_params = manager.load_best_params(
    symbol="BTCUSDT",
    timeframe="1h",
    max_age_days=30  # Chỉ load nếu study < 30 ngày
)
```

### Tích hợp với model training

Sau khi có best parameters, bạn có thể sử dụng chúng để train model:

```python
from modules.xgboost.optimization import HyperparameterTuner
from modules.xgboost.model import train_and_predict
from config import XGBOOST_PARAMS

# Lấy best parameters
tuner = HyperparameterTuner(symbol="BTCUSDT", timeframe="1h")
best_params = tuner.get_best_params(df=df)

# Cập nhật config (tùy chọn)
XGBOOST_PARAMS.update(best_params)

# Train model với best parameters
model = train_and_predict(df)
```

### Search Space

HyperparameterTuner tự động tìm kiếm trong các ranges sau:

- `n_estimators`: 50-500 (step 50)
- `learning_rate`: 0.01-0.3 (log scale)
- `max_depth`: 3-10
- `subsample`: 0.6-1.0
- `colsample_bytree`: 0.6-1.0
- `gamma`: 0.0-0.5
- `min_child_weight`: 1-10

Các parameters cố định:
- `random_state`: 42
- `objective`: "multi:softprob"
- `eval_metric`: "mlogloss"
- `n_jobs`: -1
- `num_class`: len(TARGET_LABELS)

### Lưu trữ

Studies được lưu tại:
- **SQLite database**: `artifacts/xgboost/optimization/studies.db`
- **JSON metadata**: `artifacts/xgboost/optimization/study_{symbol}_{timeframe}_{timestamp}.json`

Mỗi study JSON chứa:
- Best parameters và best score
- Trial history
- Timestamp và metadata

### Lưu ý

1. **Data Requirements**: Cần ít nhất 100 samples để chạy optimization
2. **Time-Series Gap**: Tự động áp dụng gap = TARGET_HORIZON để tránh data leakage
3. **Class Diversity**: Chỉ sử dụng folds có đủ tất cả target classes
4. **Study Persistence**: Studies được lưu trong SQLite database, có thể tiếp tục optimization sau

### Ví dụ nâng cao

#### Tùy chỉnh search space

Để tùy chỉnh search space, bạn có thể chỉnh sửa method `_objective` trong `HyperparameterTuner`:

```python
# Trong _objective method
params = {
    "n_estimators": trial.suggest_int("n_estimators", 100, 300),  # Custom range
    "learning_rate": trial.suggest_float("learning_rate", 0.05, 0.2),  # Custom range
    # ... other parameters
}
```

#### Multi-objective optimization

Hiện tại module chỉ optimize accuracy. Để optimize multiple metrics, bạn có thể mở rộng `_objective` method để return dictionary:

```python
def _objective(self, trial, X, y, n_splits=5):
    # ... training code ...
    return {
        "accuracy": mean_accuracy,
        "f1_score": mean_f1,
    }
```

Và sử dụng `optuna.create_study` với `directions=["maximize", "maximize"]`.

### Troubleshooting

#### Study not found

Nếu study chưa tồn tại, module sẽ tự động tạo study mới.

#### Insufficient data

Nếu có ít hơn 100 samples, module sẽ trả về default parameters từ `XGBOOST_PARAMS`.

#### No valid folds

Nếu không có fold nào hợp lệ sau khi áp dụng gap, hãy:
- Tăng số lượng data
- Giảm `n_splits`
- Kiểm tra class distribution

## Related Documentation

- [Common Utilities](../common/) - DataFetcher, ExchangeManager
- [Config](../../config/xgboost.py) - XGBoost configuration

