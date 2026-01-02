# LSTM Trainer Migration Guide

## Overview

The LSTM trainer architecture has been refactored from separate trainers to a unified `LSTMTrainer` class that supports all model variants through configuration flags.

## Removed Exports

The following exports have been **removed** from `modules.lstm.models`:

1. `CNNLSTMAttentionTrainer` - Old class-based trainer for CNN-LSTM models
2. `train_lstm_attention_model` - Function-based trainer for LSTM models
3. `train_and_save_global_lstm_attention_model` - Global model training function

## New Unified API

All functionality is now available through the unified `LSTMTrainer` class:

```python
from modules.lstm.models import LSTMTrainer

# Create trainer for any variant
trainer = LSTMTrainer(
    use_cnn=False,           # Set True for CNN-based models
    use_attention=False,     # Set True for attention models
    look_back=60,            # Sequence length
    output_mode='classification',  # or 'regression'
    attention_heads=8,       # If use_attention=True
)

# Train model
model, threshold_optimizer, model_path = trainer.train(
    df_input=df,
    epochs=100,
    save_model=True
)
```

## Migration Examples

### Before: CNNLSTMAttentionTrainer

```python
# OLD
from modules.lstm.models import CNNLSTMAttentionTrainer

trainer = CNNLSTMAttentionTrainer(
    use_cnn=True,
    use_attention=True,
    look_back=60,
    output_mode='classification'
)
model, threshold_opt, path = trainer.train(df_input=df, epochs=100)
```

```python
# NEW
from modules.lstm.models import LSTMTrainer

trainer = LSTMTrainer(
    use_cnn=True,
    use_attention=True,
    look_back=60,
    output_mode='classification'
)
model, threshold_opt, path = trainer.train(df_input=df, epochs=100)
```

### Before: train_lstm_attention_model

```python
# OLD
from modules.lstm.models.lstm_attention_trainer import train_lstm_attention_model

model, results_df = train_lstm_attention_model(
    df_input=df,
    use_attention=True,
    epochs=100
)
```

```python
# NEW
from modules.lstm.models import LSTMTrainer

trainer = LSTMTrainer(
    use_cnn=False,
    use_attention=True,
    output_mode='classification'
)
model, threshold_opt, path = trainer.train(df_input=df, epochs=100)
```

### Before: train_and_save_global_lstm_attention_model

```python
# OLD
from modules.lstm.models.lstm_attention_trainer import train_and_save_global_lstm_attention_model

model, path = train_and_save_global_lstm_attention_model(
    combined_df=df,
    use_attention=True
)
```

```python
# NEW
from modules.lstm.models import LSTMTrainer

trainer = LSTMTrainer(
    use_cnn=False,
    use_attention=True,
    output_mode='classification'
)
model, threshold_opt, path = trainer.train(
    df_input=df,
    epochs=100,
    save_model=True
)
```

## Model Variants Supported

The unified `LSTMTrainer` supports all 4 model variants:

1. **LSTM** (`use_cnn=False, use_attention=False`)
2. **LSTM-Attention** (`use_cnn=False, use_attention=True`)
3. **CNN-LSTM** (`use_cnn=True, use_attention=False`)
4. **CNN-LSTM-Attention** (`use_cnn=True, use_attention=True`)

## Utility Functions

Utility functions have been moved to `modules.lstm.models.model_utils`:

- `load_lstm_model()` - Load trained model (replaces `load_lstm_attention_model`)
- `get_latest_signal()` - Generate trading signals (replaces `get_latest_lstm_attention_signal`)

Backward compatibility aliases are still available:
- `load_lstm_attention_model()` → `load_lstm_model()`
- `get_latest_lstm_attention_signal()` → `get_latest_signal()`

## Verification Status

✅ **All internal usages have been migrated:**
- `modules/lstm/cli/main.py` - Updated to use `LSTMTrainer`
- All imports verified - No remaining references to removed exports

✅ **Documentation updated:**
- `modules/lstm/PIPELINE_CNN_LSTM_ATTENTION.md` - Updated with new API examples

✅ **No external dependencies found:**
- No wildcard imports affected
- No test files using removed exports
- No other modules depending on removed exports

## Benefits of Unified Trainer

1. **Single API** - One interface for all model variants
2. **Better code reuse** - Common logic shared, no duplication
3. **Consistent behavior** - All variants use same training pipeline
4. **Easier maintenance** - Changes in one place affect all variants
5. **Clearer organization** - Logic organized by features (base, CNN, attention)

## Breaking Changes

⚠️ **Breaking Changes:**
- `CNNLSTMAttentionTrainer` class no longer exists
- `train_lstm_attention_model()` function no longer exists
- `train_and_save_global_lstm_attention_model()` function no longer exists

All code using these must be migrated to `LSTMTrainer`.

