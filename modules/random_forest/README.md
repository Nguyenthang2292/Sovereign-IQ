# Random Forest Module

Module Random Forest cung cấp chức năng train và sử dụng Random Forest model để dự đoán trading signals dựa trên các technical indicators.

## Tổng quan

Module này sử dụng Random Forest Classifier để phân loại trading signals (LONG/SHORT/NEUTRAL) dựa trên các technical features được tính toán từ dữ liệu OHLCV. Module bao gồm:

- **Model Training**: Train Random Forest model với SMOTE để xử lý class imbalance
- **Signal Generation**: Tạo trading signals từ dữ liệu market mới nhất
- **Model Evaluation**: Đánh giá model với nhiều confidence thresholds
- **CLI Tools**: Command-line interface để train và test model

## Cấu trúc Module

```
random_forest/
├── __init__.py                 # Module exports
├── README.md                   # Tài liệu này
├── core/
│   ├── __init__.py             # Core exports
│   ├── model.py                # Model training, loading, saving
│   ├── signals.py              # Signal generation
│   └── evaluation.py           # Model evaluation functions
├── utils/
│   ├── __init__.py             # Utils exports
│   ├── data_preparation.py     # Data preparation utilities
│   └── training.py             # Training helpers (SMOTE, model creation)
└── cli/
    ├── __init__.py             # CLI exports
    ├── argument_parser.py      # Argument parsing
    └── main.py                 # Main CLI function
```

## Cấu trúc Chi tiết

### Core Module

**`core/model.py`**
- `load_random_forest_model()` - Load model đã train từ file
- `train_random_forest_model()` - Train và save model
- `train_and_save_global_rf_model()` - Train global model từ multiple symbols

**`core/signals.py`**
- `get_latest_random_forest_signal()` - Generate trading signal từ latest market data

**`core/evaluation.py`**
- `evaluate_model_with_confidence()` - Đánh giá model với nhiều confidence thresholds
- `apply_confidence_threshold()` - Áp dụng confidence threshold cho predictions
- `calculate_and_display_metrics()` - Tính toán và hiển thị performance metrics

### Utils Module

**`utils/data_preparation.py`**
- `prepare_training_data()` - Chuẩn bị features và target variable từ OHLCV data

**`utils/training.py`**
- `apply_smote()` - Áp dụng SMOTE để balance classes
- `create_model_and_weights()` - Tạo Random Forest model với class weights

### CLI Module

**`cli/argument_parser.py`**
- `parse_args()` - Parse command-line arguments

**`cli/main.py`**
- `main()` - Main CLI function để train model và test signal generation

## Sử dụng

### Import Module

```python
from modules.random_forest import (
    train_random_forest_model,
    get_latest_random_forest_signal,
    load_random_forest_model,
    train_and_save_global_rf_model,
    evaluate_model_with_confidence,
)
```

### Ví dụ Cơ bản

#### 1. Train Model

```python
import pandas as pd
from modules.random_forest import train_random_forest_model

# Chuẩn bị dữ liệu OHLCV
df = pd.DataFrame({
    'open': [...],
    'high': [...],
    'low': [...],
    'close': [...],
    'volume': [...]
})

# Train model
model = train_random_forest_model(df, save_model=True)
```

#### 2. Generate Signal

```python
from modules.random_forest import get_latest_random_forest_signal

# Load model đã train
from modules.random_forest import load_random_forest_model
model = load_random_forest_model()

# Hoặc sử dụng model vừa train
# model = train_random_forest_model(df, save_model=False)

# Generate signal từ latest market data
latest_data = pd.DataFrame({
    'open': [50000],
    'high': [51000],
    'low': [49000],
    'close': [50500],
    'volume': [1000]
})

signal, confidence = get_latest_random_forest_signal(latest_data, model)
print(f"Signal: {signal}, Confidence: {confidence:.4f}")
# Output: Signal: LONG, Confidence: 0.8234
```

#### 3. Train Global Model

```python
from modules.random_forest import train_and_save_global_rf_model

# Combined data từ multiple symbols
combined_df = pd.DataFrame({
    'open': [...],
    'high': [...],
    'low': [...],
    'close': [...],
    'volume': [...]
})

# Train và save global model
model, model_path = train_and_save_global_rf_model(combined_df)
print(f"Model saved to: {model_path}")
```

#### 4. Evaluate Model

```python
from modules.random_forest import evaluate_model_with_confidence
from sklearn.model_selection import train_test_split

# Split data
X_train, X_test, y_train, y_test = train_test_split(features, target, test_size=0.2)

# Train model
model = train_random_forest_model(df, save_model=False)

# Evaluate với nhiều confidence thresholds
evaluate_model_with_confidence(model, X_test, y_test)
```

### Sử dụng CLI

```bash
# Train model với default settings
python -m modules.random_forest.cli.main

# Train với custom pairs
python -m modules.random_forest.cli.main --pairs "BTC/USDT,ETH/USDT"

# Set log level
python -m modules.random_forest.cli.main --log-level DEBUG
```

## Tính năng

### 1. Feature Engineering

Module sử dụng `IndicatorEngine` từ `modules.common.core.indicator_engine` để tính toán các technical indicators:

- **Moving Averages**: SMA_20, SMA_50, SMA_200
- **Momentum**: RSI_9, RSI_14, RSI_25
- **Volatility**: ATR_14
- **Trend**: MACD_12_26_9, MACDh, MACDs
- **Bollinger Bands**: BBP_5_2.0
- **Stochastic**: STOCHRSIk, STOCHRSId
- **Volume**: OBV
- **Candlestick Patterns**: DOJI, HAMMER, ENGULFING, v.v.

### 2. Class Imbalance Handling

Module sử dụng **SMOTE** (Synthetic Minority Over-sampling Technique) để xử lý class imbalance:

- Kiểm tra memory trước khi apply SMOTE
- Tự động skip SMOTE nếu memory không đủ
- Sử dụng reduced k_neighbors cho large datasets

### 3. Model Training

- **Class Weights**: Tự động tính toán balanced class weights
- **Data Sampling**: Tự động sample nếu dataset quá lớn
- **Train/Test Split**: Chia data thành training và testing sets
- **Model Persistence**: Save và load model với joblib

### 4. Signal Generation

- **Confidence Threshold**: Chỉ trả về signal nếu confidence >= threshold
- **Feature Filtering**: Tự động filter features có sẵn trong DataFrame
- **Error Handling**: Graceful handling khi feature calculation fails

### 5. Model Evaluation

- **Multiple Thresholds**: Đánh giá với nhiều confidence thresholds
- **Comprehensive Metrics**: Accuracy, Precision, Recall, F1-score
- **Per-class Metrics**: Metrics cho từng class (LONG/SHORT/NEUTRAL)

## Configuration

Module sử dụng configuration từ `config/random_forest.py` và `config/model_features.py`:

- `MODEL_FEATURES`: Danh sách features sử dụng cho training
- `CONFIDENCE_THRESHOLD`: Default confidence threshold
- `CONFIDENCE_THRESHOLDS`: List thresholds để evaluate
- `MAX_TRAINING_ROWS`: Số rows tối đa để training
- `MODEL_RANDOM_STATE`: Random state cho reproducibility
- `MODEL_TEST_SIZE`: Test size ratio
- `MIN_TRAINING_SAMPLES`: Số samples tối thiểu để train
- `MIN_MEMORY_GB`: Memory tối thiểu để apply SMOTE

## Testing

Module có test suite đầy đủ trong `tests/random_forest/test_signals_random_forest.py`:

- Test model training với nhiều scenarios
- Test signal generation với edge cases
- Test model evaluation functions
- Test error handling

Chạy tests:

```bash
pytest tests/random_forest/test_signals_random_forest.py -v
```

## Dependencies

- `scikit-learn`: Random Forest Classifier
- `imbalanced-learn`: SMOTE
- `pandas`, `numpy`: Data manipulation
- `modules.common.core.indicator_engine`: Feature engineering
- `config`: Configuration constants

## Notes

- Module tự động filter `MODEL_FEATURES` để chỉ sử dụng features có sẵn trong DataFrame sau khi tính toán indicators
- SMOTE sẽ được skip nếu memory không đủ (< MIN_MEMORY_GB)
- Model training sẽ fail nếu không đủ samples sau khi chuẩn bị data
- Signal generation trả về "NEUTRAL" nếu confidence < CONFIDENCE_THRESHOLD hoặc có lỗi

