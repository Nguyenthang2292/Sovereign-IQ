## Roadmap: PyTorch Forecasting + Temporal Fusion Transformer

### 1. Environment Setup
- Install dependencies: `torch`, `torchvision`, `torchaudio` (matching CUDA), `pytorch-lightning`, `pytorch-forecasting`, `tensorboard`, `pandas`, `scikit-learn`.
- Update `requirements.txt` or create `requirements-deep.txt`.
- Verify GPU availability (`torch.cuda.is_available()`); fall back to CPU if needed.

### 2. Data Preparation Pipeline
- Build `modules/data_pipeline_deep.py`:
  - Fetch OHLCV via existing `DataFetcher`.
  - **Target Engineering**:
    - Use **Log Returns** or **% Change** instead of absolute price for better stationarity.
    - For Classification: Implement **Triple Barrier Method** (TP, SL, Time Limit) for robust labeling.
  - **Stationarity**: Apply Fractional Differentiation to preserve memory while ensuring stationarity.
  - Join with technical indicators, volatility metrics.
  - Generate known-future features: time-of-day, day-of-week, funding schedule.
  - **Normalization**: Scale numeric columns (e.g., `StandardScaler`) *per symbol* to handle multi-asset scale differences; persist scaler params.
- Split chronologically into train/validation/test ranges.

### 2.5. Feature Selection & Engineering
- **Filter Features**: Avoid "Garbage In, Garbage Out".
- Use **Mutual Information** or **Boruta** to select top 20-30 most relevant features.
- Remove highly collinear features to improve model stability.

### 3. Dataset & DataModule
- Use `pytorch_forecasting.TimeSeriesDataSet`:
  - `time_idx`: monotonically increasing integer. **Critical**: Handle missing candles (resample/ffill) to ensure no gaps in integer index.
  - `group_ids`: `["symbol"]` for multi-asset training. **Note**: Ensure data is normalized per-symbol (e.g. Z-Score) before feeding in.
  - `target`: Log-return (Regression) or Triple-Barrier Label (Classification).
  - `max_encoder_length`: lookback window (e.g., 64–128 bars).
  - `max_prediction_length`: prediction horizon (align with `TARGET_HORIZON`).
  - Define `time_varying_known_reals`, `time_varying_unknown_reals`, `static_reals`, categorical columns.
- Wrap in a Lightning `DataModule` to create deterministic train/val/test DataLoaders.

### 4. Model Configuration
**Phased Implementation Strategy:**

#### Phase 1: Vanilla TFT (MVP)
- Instantiate standard `TemporalFusionTransformer.from_dataset(...)`.
- **Loss**: Use `QuantileLoss` to generate confidence intervals (trade only when CI is narrow).
- Key hyperparameters: `hidden_size`, `attention_head_size`, `dropout`, `learning_rate`.
- Callbacks: `EarlyStopping`, `ModelCheckpoint`, `LearningRateMonitor`.

#### Phase 2: Optimization
- Use **Optuna** for hyperparameter tuning.

#### Phase 3: Hybrid LSTM + TFT Architecture (Advanced)
*Implement only if Phase 2 plateaus.*
- **Dual Branch**:
  - **LSTM branch**: Process raw price/volume series.
  - **TFT branch**: Process complex features (static + known future).
- **Fusion**: Gated fusion (GLU) of latent vectors.
- **Multi-task Head**: 
    - Task 1: Direction (Classification/Softmax).
    - Task 2: Magnitude (Regression/QuantileLoss).
- **Huấn luyện**: tối ưu loss tổng hợp có trọng số (λ_class * CE + λ_reg * QuantileLoss).

### 5. Training Script (`deep_prediction_main.py`)
- Parse CLI args: symbol filter, timeframe, epochs, batch size, GPU flag.
- Build dataset/datamodule, instantiate TFT, and run `pl.Trainer`.
- Log metrics to TensorBoard/W&B; track validation loss, MAE/RMSE, class accuracy if applicable.
- Save:
  - Best checkpoint (`artifacts/deep/tft.ckpt`).
  - Dataset metadata (`TimeSeriesDataSet.to_dataset()`).
  - Scaler/config JSON for reproducibility.

### 6. Inference Wrapper
- Create `modules/deep_prediction_model.py`:
  - Load scaler + dataset metadata + TFT checkpoint.
  - Prepare latest window of data, apply same preprocessing, and call `model.predict()`.
  - Convert forecast to actionable signal (e.g., expected move vs dynamic threshold → UP/DOWN/NEUTRAL probabilities).
- Add CLI option `--model deep` in `xgboost_prediction_main.py` (or new script) to switch between models.

### 7. Evaluation & Backtesting
- **Walk-Forward Validation**: Use Rolling Window Backtest (e.g., Train 2020-2022 -> Test Q1 2023; Train 2020-Q1 2023 -> Test Q2 2023) to adapt to regime changes.
- Compare against XGBoost baseline:
  - Metrics: Accuracy, F1, Directional Hit-Rate.
  - **Calibration**: Check if predicted probabilities match actual frequencies.
- Integrate with backtesting (vectorbt/backtrader):
  - Feed TFT signals (and Confidence Intervals) into strategy.
  - Evaluate PnL, Sharpe, Drawdown.

### 8. Operationalization
- Schedule periodic retraining (e.g., weekly) with sliding windows.
- Implement drift monitoring: track live accuracy/funding vs expectation.
- For real-time inference:
  - Run a service that keeps model loaded on GPU.
  - Consume WebSocket data, update window, emit predictions via message bus.
- Document deployment steps and create CI hook to validate model artifacts.

