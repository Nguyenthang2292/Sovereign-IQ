# CNN-LSTM-Attention Model Pipeline

## 📋 Tổng quan

Pipeline này mô tả quy trình hoàn chỉnh để train và sử dụng mô hình **CNN-LSTM với Attention Mechanism** cho việc dự đoán tín hiệu trading (LONG/SHORT/NONE) hoặc dự đoán return.

**File:** `modules/lstm/signals_cnn_lstm_attention.py`

---

## 🔄 Workflow Diagram

```
┌─────────────────────────────────────────────────────────────────┐
│                    INPUT: OHLC DataFrame                        │
└────────────────────────────┬────────────────────────────────────┘
                             │
                             ▼
┌─────────────────────────────────────────────────────────────────┐
│  STEP 1: Preprocessing                                          │
│  - Generate Technical Indicators                                │
│  - Create Targets (Classification/Regression)                  │
│  - Scale Features (MinMax/Standard)                            │
│  - Create Sliding Window Sequences                              │
└────────────────────────────┬────────────────────────────────────┘
                             │
                             ▼
┌─────────────────────────────────────────────────────────────────┐
│  STEP 2: Data Splitting                                         │
│  - Train Set (70%)                                              │
│  - Validation Set (15%)                                         │
│  - Test Set (15%)                                               │
└────────────────────────────┬────────────────────────────────────┘
                             │
                             ▼
┌─────────────────────────────────────────────────────────────────┐
│  STEP 3: Model Creation                                          │
│  - CNN-LSTM-Attention Model                                     │
│  - LSTM-Attention Model                                         │
│  - Standard LSTM Model                                          │
└────────────────────────────┬────────────────────────────────────┘
                             │
                             ▼
┌─────────────────────────────────────────────────────────────────┐
│  STEP 4: Training                                               │
│  - GPU Setup & Mixed Precision                                  │
│  - Loss Function (FocalLoss/MSELoss)                           │
│  - Optimizer (AdamW)                                           │
│  - Learning Rate Scheduler                                      │
│  - Early Stopping                                              │
└────────────────────────────┬────────────────────────────────────┘
                             │
                             ▼
┌─────────────────────────────────────────────────────────────────┐
│  STEP 5: Threshold Optimization                                 │
│  - Grid Search for Optimal Threshold                            │
│  - Maximize Sharpe Ratio                                       │
└────────────────────────────┬────────────────────────────────────┘
                             │
                             ▼
┌─────────────────────────────────────────────────────────────────┐
│  OUTPUT: Trained Model + Metadata                               │
│  - Model State Dict                                             │
│  - Model Config                                                │
│  - Training History                                            │
│  - Optimal Threshold                                           │
└─────────────────────────────────────────────────────────────────┘
```

---

## 📝 Chi tiết các bước

### **STEP 1: Preprocessing** (`preprocess_cnn_lstm_data`)

#### Input:
- `df_input`: DataFrame chứa OHLC data
- `look_back`: Số time steps để tạo sequence (default: `WINDOW_SIZE_LSTM`)
- `output_mode`: `'classification'` hoặc `'regression'`
- `scaler_type`: `'minmax'` hoặc `'standard'`

#### Quy trình:

1. **Generate Technical Indicators**
   ```python
   df = _generate_indicator_features(df_input.copy())
   ```
   - Tính toán các technical indicators từ OHLC data
   - Sử dụng các features từ `MODEL_FEATURES`

2. **Create Targets**
   - **Classification Mode:**
     ```python
     df = create_balanced_target(df, threshold=TARGET_THRESHOLD_LSTM, 
                                 neutral_zone=NEUTRAL_ZONE_LSTM)
     ```
     - Tạo 3 classes: LONG (-1), NONE (0), SHORT (1)
     - Dựa trên future return và threshold
   
   - **Regression Mode:**
     ```python
     df['Target'] = df['close'].pct_change().shift(-1)
     ```
     - Target là future return

3. **Data Cleaning**
   - Drop NaN values
   - Validate data sufficiency
   - Handle invalid values (NaN, Inf)

4. **Feature Scaling**
   ```python
   scaler = MinMaxScaler()  # hoặc StandardScaler()
   scaled_features = scaler.fit_transform(features)
   ```

5. **Create Sliding Window Sequences**
   ```python
   for i in range(look_back, len(scaled_features)):
       sequence = scaled_features[i-look_back:i]
       X_sequences.append(sequence)
       y_targets.append(target_values[i])
   ```
   - Mỗi sequence có shape: `(look_back, num_features)`
   - Tạo sequences liên tiếp từ data

#### Output:
- `X_sequences`: Array shape `(n_samples, look_back, num_features)`
- `y_targets`: Array shape `(n_samples,)`
- `scaler`: Fitted scaler object
- `feature_names`: List các features được sử dụng

---

### **STEP 2: Data Splitting** (`split_train_test_data`)

#### Quy trình:

```python
X_train, X_val, X_test, y_train, y_val, y_test = split_train_test_data(
    X, y, 
    train_ratio=0.7,      # 70% train
    validation_ratio=0.15 # 15% validation
)
# Test set: 15% còn lại
```

#### Validation:
- Kiểm tra X và y có cùng length
- Đảm bảo đủ samples (tối thiểu 10)
- Validate ratios hợp lệ

---

### **STEP 3: Model Creation** (`create_cnn_lstm_attention_model`)

#### Các loại model:

1. **CNNLSTMAttentionModel** (khi `use_cnn=True`)
   - CNN layers để extract features
   - LSTM layers để capture temporal patterns
   - Attention mechanism để focus vào important time steps
   - Output: Classification (3 classes) hoặc Regression (1 value)

2. **LSTMAttentionModel** (khi `use_attention=True` và `use_cnn=False`)
   - LSTM layers với Multi-Head Attention
   - Không có CNN layers

3. **LSTMModel** (khi cả `use_cnn=False` và `use_attention=False`)
   - Standard LSTM model

#### Parameters:
- `input_size`: Số lượng features
- `look_back`: Sequence length
- `output_mode`: `'classification'` hoặc `'regression'`
- `num_heads`: Số attention heads (default từ `GPU_MODEL_CONFIG['nhead']`)
- `cnn_features`: 64
- `lstm_hidden`: 32
- `dropout`: 0.3

---

### **STEP 4: Training** (`train_cnn_lstm_attention_model`)

#### GPU Setup:

```python
gpu_available = check_gpu_availability()
device = torch.device('cuda:0' if gpu_available else 'cpu')
use_mixed_precision = gpu_available and torch.cuda.get_device_capability(0)[0] >= 7
```

- Tự động detect GPU
- Sử dụng mixed precision (FP16) nếu GPU hỗ trợ (compute capability >= 7)
- Cấu hình GPU memory

#### Training Configuration:

1. **Loss Function:**
   - Classification: `FocalLoss(alpha=0.25, gamma=2.0)`
   - Regression: `nn.MSELoss()`

2. **Optimizer:**
   ```python
   optimizer = optim.AdamW(
       model.parameters(), 
       lr=0.001, 
       weight_decay=0.01, 
       eps=1e-8
   )
   ```

3. **Learning Rate Scheduler:**
   ```python
   scheduler = optim.lr_scheduler.CosineAnnealingWarmRestarts(
       optimizer, 
       T_0=10,      # Initial period
       T_mult=2,    # Period multiplier
       eta_min=1e-6 # Minimum learning rate
   )
   ```

4. **Batch Size Optimization:**
   ```python
   optimal_batch_size = get_optimal_batch_size(device, input_size, look_back)
   if use_cnn:
       optimal_batch_size = max(4, optimal_batch_size // 8)
   ```
   - Tự động tối ưu batch size dựa trên GPU memory
   - CNN models sử dụng batch size nhỏ hơn

#### Training Loop:

**Training Phase:**
```python
for batch_X, batch_y in train_loader:
    # Forward pass
    outputs = model(batch_X)
    loss = criterion(outputs, batch_y)
    
    # Backward pass (với mixed precision nếu có)
    if use_mixed_precision:
        scaler_amp.scale(loss).backward()
        scaler_amp.step(optimizer)
    else:
        loss.backward()
        optimizer.step()
    
    # Gradient clipping
    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
```

**Validation Phase:**
```python
model.eval()
with torch.no_grad():
    for batch_X, batch_y in val_loader:
        outputs = model(batch_X)
        loss = criterion(outputs, batch_y)
        val_loss += loss.item()
```

**Early Stopping:**
- Patience: 10 epochs
- Monitor validation loss
- Restore best model state khi dừng sớm

#### Metrics Logging:

- **Classification:**
  - Train/Validation Loss
  - Train/Validation Accuracy
  - Learning Rate

- **Regression:**
  - Train/Validation Loss
  - Learning Rate

---

### **STEP 5: Threshold Optimization**

#### Classification Mode:

```python
best_confidence, best_sharpe = threshold_optimizer.optimize_classification_threshold(
    test_predictions, test_returns
)
```

- Grid search các confidence thresholds
- Tính Sharpe ratio cho mỗi threshold
- Chọn threshold tối ưu

#### Regression Mode:

```python
best_threshold, best_sharpe = threshold_optimizer.optimize_regression_threshold(
    test_predictions.flatten(), test_returns, prices
)
```

- Grid search các return thresholds
- Tính Sharpe ratio
- Chọn threshold tối ưu

---

### **STEP 6: Model Saving**

#### Saved Information:

```python
save_dict = {
    'model_state_dict': model.state_dict(),
    'model_config': {
        'input_size': input_size,
        'look_back': look_back,
        'output_mode': output_mode,
        'use_cnn': use_cnn,
        'use_attention': use_attention,
        'attention_heads': attention_heads,
        'num_classes': num_classes
    },
    'training_info': {
        'epochs_trained': epoch + 1,
        'best_val_loss': best_val_loss,
        'final_lr': optimizer.param_groups[0]['lr']
    },
    'data_info': {
        'scaler': scaler,
        'feature_names': feature_names,
        'train_samples': len(X_train),
        'val_samples': len(X_val),
        'test_samples': len(X_test)
    },
    'optimization_results': {
        'optimal_threshold': threshold_optimizer.best_threshold,
        'best_sharpe': threshold_optimizer.best_sharpe
    },
    'training_history': {
        'train_loss': [...],
        'val_loss': [...]
    }
}
```

#### File Location:
- `MODELS_DIR / "cnn_lstm_attention_{output_mode}_model.pth"`

---

## 🎯 Các Functions chính

### 1. `preprocess_cnn_lstm_data()`

**Mục đích:** Chuẩn bị dữ liệu cho training

**Input:**
- `df_input`: DataFrame với OHLC data
- `look_back`: Số time steps (default: `WINDOW_SIZE_LSTM`)
- `output_mode`: `'classification'` hoặc `'regression'`
- `scaler_type`: `'minmax'` hoặc `'standard'`

**Output:**
- `X_sequences`: Sequences array `(n_samples, look_back, num_features)`
- `y_targets`: Target array `(n_samples,)`
- `scaler`: Fitted scaler
- `feature_names`: List feature names

---

### 2. `split_train_test_data()`

**Mục đích:** Chia dữ liệu thành train/validation/test

**Input:**
- `X`: Input sequences
- `y`: Targets
- `train_ratio`: 0.7 (70%)
- `validation_ratio`: 0.15 (15%)

**Output:**
- `X_train, X_val, X_test, y_train, y_val, y_test`

---

### 3. `create_cnn_lstm_attention_model()`

**Mục đích:** Tạo model architecture

**Input:**
- `input_size`: Số features
- `use_attention`: Có dùng attention không
- `use_cnn`: Có dùng CNN không
- `look_back`: Sequence length
- `output_mode`: `'classification'` hoặc `'regression'`

**Output:**
- Model object (CNNLSTMAttentionModel, LSTMAttentionModel, hoặc LSTMModel)

---

### 4. `train_cnn_lstm_attention_model()`

**Mục đích:** Train model hoàn chỉnh

**Input:**
- `df_input`: Input DataFrame
- `save_model`: Có lưu model không (default: True)
- `epochs`: Số epochs (default: `DEFAULT_EPOCHS`)
- `use_early_stopping`: Có dùng early stopping không (default: True)
- `use_attention`: Có dùng attention không (default: True)
- `use_cnn`: Có dùng CNN không (default: True)
- `look_back`: Sequence length (default: `WINDOW_SIZE_LSTM`)
- `output_mode`: `'classification'` hoặc `'regression'` (default: `'classification'`)
- `attention_heads`: Số attention heads (default: từ `GPU_MODEL_CONFIG`)

**Output:**
- `(trained_model, threshold_optimizer)`

---

### 5. `train_and_save_global_cnn_lstm_attention_model()`

**Mục đích:** Wrapper function để train và lưu global model

**Input:**
- `combined_df`: Combined DataFrame từ nhiều trading pairs
- `model_filename`: Optional custom filename
- Các parameters tương tự `train_cnn_lstm_attention_model()`

**Output:**
- `(trained_model, model_path_string)`

---

## ⚙️ Configuration Constants

Các constants được import từ `livetrade.config`:

- `MODEL_FEATURES`: List các features sử dụng trong model
- `WINDOW_SIZE_LSTM`: Default look_back window size
- `DEFAULT_EPOCHS`: Default số epochs để train
- `TARGET_THRESHOLD_LSTM`: Threshold để tạo classification targets
- `NEUTRAL_ZONE_LSTM`: Neutral zone cho balanced targets
- `TRAIN_TEST_SPLIT`: Train ratio (default: 0.7)
- `VALIDATION_SPLIT`: Validation ratio (default: 0.15)
- `GPU_MODEL_CONFIG`: GPU configuration (bao gồm `nhead` cho attention)
- `MODELS_DIR`: Directory để lưu models
- `COL_CLOSE`: Column name cho close price

---

## 🚀 Ví dụ sử dụng

### Example 1: Train Classification Model

```python
import pandas as pd
from modules.lstm.signals_cnn_lstm_attention import train_cnn_lstm_attention_model

# Load price data
df = pd.read_csv('price_data.csv')

# Train model
model, threshold_optimizer = train_cnn_lstm_attention_model(
    df_input=df,
    epochs=100,
    use_cnn=True,
    use_attention=True,
    output_mode='classification',
    look_back=60
)

print(f"Optimal confidence threshold: {threshold_optimizer.best_threshold}")
print(f"Best Sharpe ratio: {threshold_optimizer.best_sharpe}")
```

### Example 2: Train Regression Model

```python
model, threshold_optimizer = train_cnn_lstm_attention_model(
    df_input=df,
    epochs=100,
    use_cnn=True,
    use_attention=True,
    output_mode='regression',
    look_back=60
)
```

### Example 3: Train Global Model

```python
from modules.lstm.signals_cnn_lstm_attention import train_and_save_global_cnn_lstm_attention_model

# Combined data from multiple symbols
combined_df = pd.concat([df_btc, df_eth, df_bnb], ignore_index=True)

model, model_path = train_and_save_global_cnn_lstm_attention_model(
    combined_df=combined_df,
    use_cnn=True,
    use_attention=True,
    output_mode='classification'
)

print(f"Model saved to: {model_path}")
```

---

## 🔧 Advanced Features

### 1. Mixed Precision Training

- Tự động enable khi GPU hỗ trợ (compute capability >= 7)
- Sử dụng FP16 để tăng tốc và giảm memory
- Gradient scaling để tránh underflow

### 2. Early Stopping

- Patience: 10 epochs
- Monitor validation loss
- Tự động restore best model state

### 3. Gradient Clipping

- Max norm: 1.0
- Tránh gradient explosion

### 4. Batch Size Optimization

- Tự động tối ưu dựa trên GPU memory
- CNN models sử dụng batch size nhỏ hơn (÷8)

### 5. Error Handling & Fallback

- Kiểm tra data sufficiency
- Fallback to minimal features nếu thiếu indicators
- Comprehensive error logging

---

## 📊 Model Architecture

### CNN-LSTM-Attention Model:

```
Input: (batch_size, look_back, num_features)
    ↓
CNN Layers (Feature Extraction)
    ↓
LSTM Layers (Temporal Patterns)
    ↓
Multi-Head Attention (Important Time Steps)
    ↓
Dense Layers
    ↓
Output: 
  - Classification: (batch_size, 3) - [LONG, NONE, SHORT]
  - Regression: (batch_size, 1) - [Return]
```

---

## 📈 Performance Metrics

### Classification Mode:
- **Accuracy**: Tỷ lệ dự đoán đúng
- **Loss**: Focal Loss
- **Sharpe Ratio**: Từ threshold optimization

### Regression Mode:
- **MSE Loss**: Mean Squared Error
- **Sharpe Ratio**: Từ threshold optimization

---

## 💾 Model Loading

```python
import torch
from modules.lstm.signals_cnn_lstm_attention import create_cnn_lstm_attention_model

# Load saved model
checkpoint = torch.load('cnn_lstm_attention_classification_model.pth')

# Recreate model
model = create_cnn_lstm_attention_model(
    input_size=checkpoint['model_config']['input_size'],
    look_back=checkpoint['model_config']['look_back'],
    output_mode=checkpoint['model_config']['output_mode'],
    use_cnn=checkpoint['model_config']['use_cnn'],
    use_attention=checkpoint['model_config']['use_attention'],
    attention_heads=checkpoint['model_config']['attention_heads']
)

# Load weights
model.load_state_dict(checkpoint['model_state_dict'])

# Get scaler and feature names
scaler = checkpoint['data_info']['scaler']
feature_names = checkpoint['data_info']['feature_names']
optimal_threshold = checkpoint['optimization_results']['optimal_threshold']
```

---

## ⚠️ Lưu ý quan trọng

1. **Data Requirements:**
   - Tối thiểu `look_back + 50` rows để train
   - Cần có đủ OHLC columns
   - Technical indicators sẽ được generate tự động

2. **GPU Requirements:**
   - CUDA-compatible GPU cho training nhanh
   - Mixed precision yêu cầu compute capability >= 7
   - CPU mode vẫn hoạt động nhưng chậm hơn

3. **Memory Management:**
   - Batch size được tự động optimize
   - CNN models sử dụng nhiều memory hơn
   - Gradient accumulation có thể cần thiết cho GPU nhỏ

4. **Model Selection:**
   - **CNN-LSTM-Attention**: Best performance, nhiều memory
   - **LSTM-Attention**: Balance giữa performance và memory
   - **LSTM**: Fastest, ít memory nhất

---

## 🔗 Dependencies

- PyTorch
- NumPy
- Pandas
- Scikit-learn
- Custom modules:
  - `signals._components.LSTM__class__models`
  - `signals._components.LSTM__class__focal_loss`
  - `signals._components.LSTM__class__grid_search_threshold_optimizer`
  - `signals._components._generate_indicator_features`
  - `signals._components.LSTM__function__create_balanced_target`
  - `signals._components.LSTM__function__get_optimal_batch_size`
  - `signals._components._gpu_check_availability`

---

## 📝 Notes

- Model được train với Focal Loss để handle class imbalance
- Threshold optimization sử dụng Sharpe ratio làm metric
- Training history được lưu để analyze overfitting
- Scaler và feature names được lưu để inference sau này

