# 📚 Deep Learning Dataset Documentation

## Mục lục
1. [Tổng quan](#tổng-quan)
2. [Khởi tạo](#khởi-tạo)
3. [Phương thức chính](#phương-thức-chính)
4. [TimeSeriesDataSet Creation](#timeseriesdataset-creation)
5. [Ví dụ sử dụng](#ví-dụ-sử-dụng)
6. [Best Practices](#best-practices)
7. [Xử lý Missing Candles](#xử-lý-missing-candles)
8. [Time Index Calculation](#time-index-calculation)

---

## Tổng quan

`TFTDataModule` là một PyTorch Lightning DataModule để tạo TimeSeriesDataSet cho Temporal Fusion Transformer (TFT). Module này cung cấp:

- ✅ **TimeSeriesDataSet Creation** - Tự động tạo dataset với đúng format cho TFT
- ✅ **Time Index Handling** - Xử lý `time_idx` với missing candle resampling
- ✅ **Multi-asset Support** - Hỗ trợ training nhiều symbols cùng lúc
- ✅ **Missing Candle Handling** - Resample và fill gaps thông minh
- ✅ **Feature Categorization** - Tự động phân loại known/unknown future features
- ✅ **DataLoaders** - Train/validation/test DataLoaders với batch size configurable
- ✅ **Metadata Persistence** - Lưu dataset metadata để dùng cho inference

### Khi nào dùng TFTDataModule?

| Mục đích | Dùng TFTDataModule? | Phương thức |
|----------|---------------------|-------------|
| Tạo TimeSeriesDataSet cho TFT | ✅ Có | `setup()` + `_create_dataset()` |
| Cần train/val/test DataLoaders | ✅ Có | `train_dataloader()`, `val_dataloader()`, `test_dataloader()` |
| Cần xử lý missing candles | ✅ Có | Tự động trong `prepare_data()` |
| Cần multi-asset training | ✅ Có | `group_ids=["symbol"]` |
| Cần lưu dataset metadata | ✅ Có | `save_dataset_metadata()` |

---

## Khởi tạo

### Cú pháp

```python
from modules.deeplearning_dataset import TFTDataModule, create_tft_datamodule

# Cách 1: Sử dụng convenience function (khuyến nghị)
datamodule = create_tft_datamodule(
    train_df=train_df,
    val_df=val_df,
    test_df=test_df,
    target_col="future_log_return",
    task_type="regression",
    timeframe="1h"
)

# Cách 2: Khởi tạo trực tiếp
datamodule = TFTDataModule(
    train_df=train_df,
    val_df=val_df,
    test_df=test_df,
    target_col="future_log_return",
    task_type="regression",
    max_encoder_length=64,
    max_prediction_length=24,
    batch_size=64,
    timeframe="1h"
)
```

### Tham số chính

- `train_df` (pd.DataFrame, **bắt buộc**): Training DataFrame (đã preprocess)
- `val_df` (pd.DataFrame, **bắt buộc**): Validation DataFrame (đã preprocess)
- `test_df` (pd.DataFrame, **tùy chọn**): Test DataFrame (đã preprocess)
- `target_col` (str, **mặc định**: `"future_log_return"`): Target column name
- `task_type` (str, **mặc định**: `"regression"`): `"regression"` hoặc `"classification"`
- `max_encoder_length` (int, **mặc định**: `64`): Lookback window length (64-128 recommended)
- `max_prediction_length` (int, **mặc định**: `24`): Prediction horizon (align với TARGET_HORIZON)
- `batch_size` (int, **mặc định**: `64`): Batch size cho DataLoaders
- `num_workers` (int, **mặc định**: `4`): Số workers cho DataLoader
- `timeframe` (str, **tùy chọn**): Timeframe string (ví dụ: `"1h"`, `"4h"`) để tính time_idx chính xác
- `allow_missing_timesteps` (bool, **mặc định**: `False`): Cho phép missing timesteps (nếu False, sẽ resample)
- `max_ffill_limit` (int, **mặc định**: `5`): Giới hạn forward fill để tránh infinite fill
- `use_interpolation` (bool, **mặc định**: `True`): Dùng linear interpolation cho gaps ngắn
- `max_gap_candles` (int, **mặc định**: `10`): Kích thước gap tối đa để dùng interpolation

### Ví dụ khởi tạo

```python
from modules.deeplearning_dataset import create_tft_datamodule

# Cách 1: Mặc định
datamodule = create_tft_datamodule(
    train_df=train_df,
    val_df=val_df,
    test_df=test_df
)

# Cách 2: Tùy chỉnh
datamodule = create_tft_datamodule(
    train_df=train_df,
    val_df=val_df,
    test_df=test_df,
    target_col="triple_barrier_label",
    task_type="classification",
    max_encoder_length=128,
    max_prediction_length=24,
    batch_size=32,
    timeframe="1h",
    use_interpolation=True,
    max_ffill_limit=3
)

# Cách 3: Cho prediction (không cần target)
datamodule = create_tft_datamodule(
    train_df=train_df,
    val_df=val_df,
    test_df=None  # Không có test set
)
```

---

## Phương thức chính

### `prepare_data() -> None`

Chuẩn bị data: resample missing candles và tạo time_idx.

**Lưu ý:** Được gọi tự động bởi PyTorch Lightning, nhưng có thể gọi thủ công.

**Ví dụ:**

```python
datamodule.prepare_data()  # Resample và tạo time_idx
```

### `setup(stage=None) -> None`

Setup datasets cho training, validation, và testing.

**Tham số:**
- `stage` (str, **tùy chọn**): `"fit"` hoặc `"test"`

**Ví dụ:**

```python
# Setup cho training
datamodule.setup("fit")

# Setup cho testing
datamodule.setup("test")
```

### `train_dataloader() -> DataLoader`

Trả về training DataLoader.

**Lưu ý:** Phải gọi `setup("fit")` trước.

**Ví dụ:**

```python
datamodule.setup("fit")
train_loader = datamodule.train_dataloader()

for batch in train_loader:
    # Training loop
    pass
```

### `val_dataloader() -> DataLoader`

Trả về validation DataLoader.

**Ví dụ:**

```python
val_loader = datamodule.val_dataloader()
```

### `test_dataloader() -> DataLoader`

Trả về test DataLoader.

**Lưu ý:** Phải có `test_df` và gọi `setup("test")` trước.

**Ví dụ:**

```python
datamodule.setup("test")
test_loader = datamodule.test_dataloader()
```

### `save_dataset_metadata(filepath=None) -> None`

Lưu dataset metadata để dùng cho inference sau này.

**Tham số:**
- `filepath` (str, **tùy chọn**): Đường dẫn file (mặc định: `artifacts/deep/datasets/dataset_metadata.pkl`)

**Ví dụ:**

```python
datamodule.setup("fit")
datamodule.save_dataset_metadata()
```

### `load_dataset_metadata(filepath=None) -> Dict`

Load dataset metadata đã lưu.

**Ví dụ:**

```python
metadata = datamodule.load_dataset_metadata()
print(metadata["target_col"])
print(metadata["max_encoder_length"])
```

### `get_dataset_info() -> Dict`

Lấy thông tin về datasets.

**Ví dụ:**

```python
info = datamodule.get_dataset_info()
print(info)
# {
#     "train_samples": 1000,
#     "val_samples": 200,
#     "test_samples": 200,
#     "max_encoder_length": 64,
#     "max_prediction_length": 24,
#     ...
# }
```

---

## TimeSeriesDataSet Creation

### Feature Categorization

Module tự động phân loại features thành:

1. **Time-varying Known Reals**: Features biết trước (known future)
   - `hour_sin`, `hour_cos`
   - `day_sin`, `day_cos`
   - `day_of_month_sin`, `day_of_month_cos`
   - `hours_to_funding`, `is_funding_time`
   - `candle_index`

2. **Time-varying Unknown Reals**: Features không biết trước
   - Price features: `open`, `high`, `low`, `close`, `volume`
   - Technical indicators: `SMA_20`, `RSI_14`, `MACD_12_26_9`, etc.
   - Volatility metrics: `volatility_20`, `volatility_50`

3. **Static Reals**: Features không đổi theo time series
   - Hiện tại empty (có thể thêm sau nếu cần)

4. **Categorical Features**: Categorical variables
   - Hiện tại empty (có thể thêm sau nếu cần)

### Target Normalization

- **Regression**: Sử dụng `GroupNormalizer` với `transformation="softplus"` per symbol
- **Classification**: Không normalize (None)

---

## Ví dụ sử dụng

### Ví dụ 1: Basic Workflow

```python
from modules.deeplearning_dataset import create_tft_datamodule
from modules.deeplearning_data_pipeline import DeepLearningDataPipeline

# 1. Prepare data
pipeline = DeepLearningDataPipeline(data_fetcher)
df = pipeline.fetch_and_prepare(symbols=["BTC/USDT"], timeframe="1h")
train_df, val_df, test_df = pipeline.split_chronological(df)

# 2. Create DataModule
datamodule = create_tft_datamodule(
    train_df=train_df,
    val_df=val_df,
    test_df=test_df,
    timeframe="1h"
)

# 3. Setup và sử dụng
datamodule.prepare_data()
datamodule.setup("fit")

# 4. Get DataLoaders
train_loader = datamodule.train_dataloader()
val_loader = datamodule.val_dataloader()

# 5. Training loop
for batch in train_loader:
    # Your training code
    pass
```

### Ví dụ 2: Với PyTorch Lightning Trainer

```python
import pytorch_lightning as pl
from pytorch_forecasting import TemporalFusionTransformer

# Create DataModule
datamodule = create_tft_datamodule(
    train_df=train_df,
    val_df=val_df,
    test_df=test_df,
    timeframe="1h"
)

# Create model
model = TemporalFusionTransformer.from_dataset(
    datamodule.training,
    learning_rate=0.03,
    hidden_size=16,
    attention_head_size=4,
    dropout=0.1
)

# Train
trainer = pl.Trainer(max_epochs=10)
trainer.fit(model, datamodule=datamodule)
```

### Ví dụ 3: Multi-asset Training

```python
# Fetch nhiều symbols
df = pipeline.fetch_and_prepare(
    symbols=["BTC/USDT", "ETH/USDT", "BNB/USDT"],
    timeframe="4h"
)
train_df, val_df, test_df = pipeline.split_chronological(df)

# Create DataModule (group_ids=["symbol"] tự động)
datamodule = create_tft_datamodule(
    train_df=train_df,
    val_df=val_df,
    test_df=test_df,
    timeframe="4h"
)

datamodule.setup("fit")
# Model sẽ train trên tất cả symbols cùng lúc
```

### Ví dụ 4: Classification Task

```python
# Với triple barrier labels
pipeline = DeepLearningDataPipeline(
    data_fetcher=data_fetcher,
    use_triple_barrier=True
)

df = pipeline.fetch_and_prepare(symbols=["BTC/USDT"])
train_df, val_df, test_df = pipeline.split_chronological(
    df,
    target_col="triple_barrier_label",
    task_type="classification"
)

datamodule = create_tft_datamodule(
    train_df=train_df,
    val_df=val_df,
    test_df=test_df,
    target_col="triple_barrier_label",
    task_type="classification",
    timeframe="1h"
)
```

### Ví dụ 5: Tùy chỉnh Gap Handling

```python
# Tắt interpolation, chỉ dùng limited ffill
datamodule = create_tft_datamodule(
    train_df=train_df,
    val_df=val_df,
    test_df=test_df,
    use_interpolation=False,
    max_ffill_limit=3,  # Chỉ fill tối đa 3 candles
    timeframe="1h"
)

# Hoặc cho phép missing timesteps
datamodule = create_tft_datamodule(
    train_df=train_df,
    val_df=val_df,
    test_df=test_df,
    allow_missing_timesteps=True,  # Không resample
    timeframe="1h"
)
```

---

## Best Practices

### 1. Timeframe Specification

**Luôn chỉ định `timeframe`** để đảm bảo `time_idx` chính xác:

```python
datamodule = create_tft_datamodule(
    train_df=train_df,
    val_df=val_df,
    timeframe="1h"  # Quan trọng!
)
```

### 2. Missing Candle Handling

- **Mặc định (`allow_missing_timesteps=False`)**: Resample và fill gaps
- **Với gaps lớn**: Sử dụng `max_ffill_limit` để tránh infinite fill
- **Với gaps nhỏ**: Bật `use_interpolation=True` để smooth hơn

### 3. Multi-asset Training

- Đảm bảo data đã được normalize per symbol (từ pipeline)
- `group_ids=["symbol"]` tự động được set
- Mỗi symbol có `time_idx` bắt đầu từ 0

### 4. Encoder/Prediction Length

- **max_encoder_length**: 64-128 bars (khuyến nghị)
- **max_prediction_length**: Phải align với `TARGET_HORIZON`
- Đảm bảo có đủ data: `len(df) >= max_encoder_length + max_prediction_length`

### 5. Batch Size

- **GPU**: 32-128 (tùy GPU memory)
- **CPU**: 16-64
- Lớn hơn = nhanh hơn nhưng cần nhiều memory hơn

---

## Xử lý Missing Candles

### Strategy

Module sử dụng chiến lược thông minh để xử lý missing candles:

1. **Short gaps (≤ max_gap_candles)**: Linear interpolation (nếu `use_interpolation=True`)
2. **Medium gaps**: Limited forward fill (`max_ffill_limit`)
3. **Large gaps**: Giữ NaN (tốt hơn là tạo artificial flat data)

### Ví dụ

```python
# Với interpolation cho gaps ngắn
datamodule = create_tft_datamodule(
    train_df=train_df,
    val_df=val_df,
    use_interpolation=True,
    max_gap_candles=10,  # Interpolate cho gaps ≤ 10 candles
    max_ffill_limit=5    # Ffill tối đa 5 candles
)

# Chỉ dùng ffill (không interpolation)
datamodule = create_tft_datamodule(
    train_df=train_df,
    val_df=val_df,
    use_interpolation=False,
    max_ffill_limit=3
)
```

### Tại sao cần limit ffill?

- **Vô hạn ffill**: Tạo artificial flat data khi có gaps lớn (ví dụ: exchange maintenance)
- **Limited ffill**: Chỉ fill gaps nhỏ, giữ NaN cho gaps lớn (an toàn hơn)

---

## Time Index Calculation

### Sử dụng candle_index từ Pipeline

Module ưu tiên sử dụng `candle_index` từ `DeepLearningDataPipeline`:

1. **Nếu có `candle_index`**: Sử dụng và normalize per symbol
2. **Nếu không có**: Tính từ timestamps (fallback)

### Normalization Per Symbol

Mỗi symbol có `time_idx` bắt đầu từ 0:

```python
# Symbol 1: candle_index = [100, 101, 102, ...]
# → time_idx = [0, 1, 2, ...]

# Symbol 2: candle_index = [200, 201, 202, ...]
# → time_idx = [0, 1, 2, ...]  # Cũng bắt đầu từ 0
```

### Tính nhất quán

- `time_idx`: Dùng cho TimeSeriesDataSet ordering
- `candle_index`: Dùng làm known future feature
- Cả hai đều dựa trên cùng logic tính toán (từ timestamp)

---

## Configuration

Các config constants trong `modules/config.py`:

```python
# Dataset Configuration
DEEP_MAX_ENCODER_LENGTH = 64  # Lookback window
DEEP_MAX_PREDICTION_LENGTH = 24  # Prediction horizon (TARGET_HORIZON)
DEEP_BATCH_SIZE = 64  # Batch size
DEEP_NUM_WORKERS = 4  # DataLoader workers
DEEP_TARGET_COL = "future_log_return"  # Default target
DEEP_TARGET_COL_CLASSIFICATION = "triple_barrier_label"  # Classification target
DEEP_DATASET_DIR = "artifacts/deep/datasets"  # Metadata directory
```

---

## Troubleshooting

### Lỗi: "Must call setup('fit') first"

**Nguyên nhân:** Chưa gọi `setup()` trước khi dùng DataLoader

**Giải pháp:**
```python
datamodule.setup("fit")  # Phải gọi trước
train_loader = datamodule.train_dataloader()
```

### Lỗi: "Target column not found"

**Nguyên nhân:** Target column không tồn tại trong DataFrame

**Giải pháp:**
- Kiểm tra `target_col` parameter
- Đảm bảo đã preprocess data đúng cách
- Với classification, dùng `"triple_barrier_label"`

### Lỗi: "No valid features after filtering"

**Nguyên nhân:** Không có features hợp lệ sau khi filter

**Giải pháp:**
- Kiểm tra data quality
- Đảm bảo có numeric features
- Kiểm tra có target leakage columns không

### Time_idx issues

**Nguyên nhân:** `time_idx` không liên tục hoặc có gaps

**Giải pháp:**
- Đảm bảo `allow_missing_timesteps=False` (mặc định)
- Chỉ định `timeframe` để tính chính xác
- Kiểm tra data có missing candles không

---

## Tham khảo

- [PyTorch Forecasting Documentation](https://pytorch-forecasting.readthedocs.io/)
- [Temporal Fusion Transformer Paper](https://arxiv.org/abs/1912.09363)
- [PyTorch Lightning DataModule](https://pytorch-lightning.readthedocs.io/en/stable/data/datamodule.html)

