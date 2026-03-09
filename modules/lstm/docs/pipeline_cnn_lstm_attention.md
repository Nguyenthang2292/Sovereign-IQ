# CNN-LSTM-Attention Model Pipeline

## 📋 Tổng quan

Pipeline này mô tả quy trình hoàn chỉnh để train và sử dụng mô hình **CNN-LSTM với Attention Mechanism** cho việc dự đoán tín hiệu trading (LONG/SHORT/NONE) hoặc dự đoán return.

**Main Entry Point:** `modules/lstm/models/unified_trainer.py` - `LSTMTrainer` class

---

## 📁 Cấu trúc Modules

```text
modules/lstm/
├── core/                    # Core components và utilities
│   ├── cnn_1d_extractor.py          # CNN feature extraction
│   ├── create_balanced_target.py    # Target creation cho classification
│   ├── evaluate_models.py           # Model evaluation utilities
│   ├── feed_forward.py              # Feed-forward layers
│   ├── focal_loss.py                # Focal Loss implementation
│   ├── multi_head_attention.py      # Multi-head attention mechanism
│   ├── positional_encoding.py       # Positional encoding
│   └── threshold_optimizer.py       # Threshold optimization
│
├── models/                  # Model architectures và trainers
│   ├── unified_trainer.py           # LSTMTrainer - Unified trainer for all variants
│   ├── trainer/                     # Trainer components
│   │   ├── base_trainer.py          # BaseLSTMTrainer - Common training logic
│   │   ├── cnn_mixin.py             # CNNFeatureMixin - CNN-specific logic
│   │   └── attention_mixin.py       # AttentionFeatureMixin - Attention-specific logic
│   ├── model_utils.py               # Utility functions (loading, inference)
│   ├── lstm_models.py               # Model definitions
│   └── model_factory.py             # Model factory functions
│
├── utils/                   # Utility functions
│   ├── batch_size.py                # Batch size optimization
│   ├── data_utils.py                # Data splitting utilities
│   ├── indicator_features.py        # Technical indicator generation
│   └── preprocessing.py             # Data preprocessing
│
└── cli/                     # Command-line interface
    └── main.py                     # CLI entry point
```

---

## 🔄 Workflow

```text
INPUT: OHLC DataFrame
    ↓
STEP 1: Preprocessing
    - generate_indicator_features()
    - create_balanced_target() hoặc tính future return
    - preprocess_cnn_lstm_data()
    ↓
STEP 2: Data Splitting
    - split_train_test_data()
    ↓
STEP 3: Model Creation
    - create_cnn_lstm_attention_model()
    ↓
STEP 4: Training
    - LSTMTrainer.train()
    - _setup_device()
    - _prepare_tensors()
    - _setup_training_components()
    - _train_epoch()
    - _validate_epoch()
    ↓
STEP 5: Threshold Optimization
    - GridSearchThresholdOptimizer.optimize_classification_threshold()
    - GridSearchThresholdOptimizer.optimize_regression_threshold()
    ↓
STEP 6: Evaluation & Saving
    - _evaluate_model()
    - _save_model()
    ↓
OUTPUT: Trained Model + Metadata
```

---

## 🔀 Unified Trainer Architecture

Tất cả 4 model variants đều sử dụng cùng một `LSTMTrainer` class với các flags khác nhau:

### Examples cho các variants

**LSTM (không CNN, không Attention):**
```python
trainer = LSTMTrainer(use_cnn=False, use_attention=False)
```

**LSTM-Attention (không CNN, có Attention):**
```python
trainer = LSTMTrainer(use_cnn=False, use_attention=True, attention_heads=8)
```

**CNN-LSTM (có CNN, không Attention):**
```python
trainer = LSTMTrainer(use_cnn=True, use_attention=False, look_back=60)
```

**CNN-LSTM-Attention (có CNN, có Attention):**
```python
trainer = LSTMTrainer(use_cnn=True, use_attention=True, attention_heads=8, look_back=60)
```

Tất cả đều dùng cùng interface `train()` method.

---

## 📝 Chi tiết các bước

### **STEP 1: Preprocessing**

**Functions:**

- `modules.lstm.utils.indicator_features.generate_indicator_features()` - Tạo technical indicators
- `modules.lstm.core.create_balanced_target.create_balanced_target()` - Tạo classification targets (cho classification mode)
- `modules.lstm.utils.preprocessing.preprocess_cnn_lstm_data()` - Chuẩn bị sequences và scaling

**Output:**

- `X_sequences`: Sequences array `(n_samples, look_back, num_features)`
- `y_targets`: Target array `(n_samples,)`
- `scaler`: Fitted scaler
- `feature_names`: List feature names

---

### **STEP 2: Data Splitting**

**Function:**

- `modules.lstm.utils.data_utils.split_train_test_data()` - Chia train/validation/test

**Output:**

- `X_train, X_val, X_test, y_train, y_val, y_test`

---

### **STEP 3: Model Creation**

**Function:**

- `modules.lstm.models.model_factory.create_cnn_lstm_attention_model()` - Tạo model architecture

**Models (từ `modules.lstm.models.lstm_models`):**

- `CNNLSTMAttentionModel` (khi `use_cnn=True`)
- `LSTMAttentionModel` (khi `use_attention=True` và `use_cnn=False`)
- `LSTMModel` (standard LSTM)

---

### **STEP 4: Training**

**Class:** `LSTMTrainer` (từ `modules/lstm/models/unified_trainer.py`)

Unified trainer hỗ trợ tất cả 4 variants:
- LSTM (use_cnn=False, use_attention=False)
- LSTM-Attention (use_cnn=False, use_attention=True)
- CNN-LSTM (use_cnn=True, use_attention=False)
- CNN-LSTM-Attention (use_cnn=True, use_attention=True)

**Methods:**

- `_setup_device()` - Setup GPU và mixed precision (từ BaseLSTMTrainer)
- `_prepare_tensors()` - Chuyển data thành tensors (từ BaseLSTMTrainer)
- `create_model()` - Tạo model dựa trên flags (use_cnn, use_attention)
- `_setup_training_components()` - Setup optimizer, scheduler, loss function
- `_train_epoch()` - Train một epoch (từ BaseLSTMTrainer)
- `_validate_epoch()` - Validate một epoch (từ BaseLSTMTrainer)
- `train()` - Main training loop với early stopping

**Helper Functions:**

- `modules.common.utils.system.detect_pytorch_gpu_availability()` - Detect GPU
- `modules.lstm.utils.batch_size.get_optimal_batch_size()` - Tối ưu batch size
- `CNNFeatureMixin._adjust_batch_size_for_cnn()` - Điều chỉnh batch size cho CNN models

---

### **STEP 5: Threshold Optimization**

**Class:** `modules.lstm.core.threshold_optimizer.GridSearchThresholdOptimizer`

**Methods:**

- `optimize_classification_threshold()` - Tối ưu cho classification mode
- `optimize_regression_threshold()` - Tối ưu cho regression mode

---

### **STEP 6: Evaluation & Saving**

**Methods:**

- `_evaluate_model()` - Đánh giá model trên test set
- `_save_model()` - Lưu model và metadata

---

## 🎯 Main Functions

### 1. `preprocess_cnn_lstm_data()`

**Location:** `modules/lstm/utils/preprocessing.py`

Chuẩn bị dữ liệu cho training.

### 2. `split_train_test_data()`

**Location:** `modules/lstm/utils/data_utils.py`

Chia dữ liệu thành train/validation/test.

### 3. `create_cnn_lstm_attention_model()`

**Location:** `modules/lstm/models/model_factory.py`

Tạo model architecture.

### 4. `LSTMTrainer.train()`

**Location:** `modules/lstm/models/unified_trainer.py`

Train model hoàn chỉnh cho tất cả variants.

### 5. `GridSearchThresholdOptimizer`

**Location:** `modules/lstm/core/threshold_optimizer.py`

Tối ưu threshold cho trading signals.

---

## ⚙️ Configuration

Các constants từ `config.lstm`:

- `WINDOW_SIZE_LSTM` - Default look_back
- `TARGET_THRESHOLD_LSTM` - Threshold cho classification targets
- `NEUTRAL_ZONE_LSTM` - Neutral zone cho balanced targets
- `TRAIN_TEST_SPLIT` - Train ratio (0.7)
- `VALIDATION_SPLIT` - Validation ratio (0.15)
- `DEFAULT_EPOCHS` - Default số epochs

---

## 🚀 Example Usage

```python
from modules.lstm.models import LSTMTrainer
import pandas as pd

# Load data
df = pd.read_csv('price_data.csv')

# Create unified trainer for CNN-LSTM-Attention
trainer = LSTMTrainer(
    use_cnn=True,
    use_attention=True,
    output_mode='classification',
    look_back=60
)

# Train
model, threshold_optimizer, model_path = trainer.train(
    df_input=df,
    epochs=100,
    save_model=True
)
```

---

## 📊 Model Architecture

```text
Input: (batch_size, look_back, num_features)
    ↓
CNN1DExtractor (core.cnn_1d_extractor) - nếu use_cnn=True
    ↓
LSTM Layers (models.lstm_models)
    ↓
PositionalEncoding (core.positional_encoding) - nếu use_attention=True
    ↓
MultiHeadAttention (core.multi_head_attention) - nếu use_attention=True
    ↓
FeedForward (core.feed_forward)
    ↓
Output: 
  - Classification: (batch_size, 3) - [SELL, NEUTRAL, BUY]
  - Regression: (batch_size, 1) - [Return]
```

**Loss Functions:**

- Classification: `FocalLoss` (core.focal_loss)
- Regression: `nn.MSELoss` (PyTorch)

---

## 🔗 Module Dependencies

### Core Modules (`modules/lstm/core/`)

- `core.cnn_1d_extractor` - CNN1DExtractor class cho feature extraction
- `core.create_balanced_target` - `create_balanced_target()` function
- `core.evaluate_models` - `evaluate_model_with_confidence()`, `apply_confidence_threshold()`
- `core.feed_forward` - FeedForward class
- `core.focal_loss` - FocalLoss class
- `core.multi_head_attention` - MultiHeadAttention class
- `core.positional_encoding` - PositionalEncoding class
- `core.threshold_optimizer` - GridSearchThresholdOptimizer class

### Model Modules (`modules/lstm/models/`)

- `models.unified_trainer` - `LSTMTrainer` class (unified trainer cho tất cả variants)
- `models.trainer.base_trainer` - `BaseLSTMTrainer` class (common training logic)
- `models.trainer.cnn_mixin` - `CNNFeatureMixin` class (CNN-specific logic)
- `models.trainer.attention_mixin` - `AttentionFeatureMixin` class (attention-specific logic)
- `models.model_utils` - Utility functions:
  - `load_lstm_model()` - Load trained model từ checkpoint
  - `get_latest_signal()` - Generate trading signals từ model
- `models.lstm_models` - `LSTMModel`, `LSTMAttentionModel`, `CNNLSTMAttentionModel` classes
- `models.model_factory` - `create_cnn_lstm_attention_model()` function

### Utility Modules (`modules/lstm/utils/`)

- `utils.preprocessing` - `preprocess_cnn_lstm_data()` function
- `utils.data_utils` - `split_train_test_data()` function
- `utils.indicator_features` - `generate_indicator_features()` function
- `utils.batch_size` - `get_optimal_batch_size()` function

### CLI (`modules/lstm/cli/`)

- `cli.main` - Command-line interface entry point

### External Dependencies

- PyTorch
- NumPy, Pandas
- Scikit-learn
- Custom modules:
  - `modules.common.utils.system` - GPU detection utilities
  - `modules.common.ui.logging` - Logging utilities
  - `config.lstm` - Configuration constants
  - `config.model_features` - Feature definitions
