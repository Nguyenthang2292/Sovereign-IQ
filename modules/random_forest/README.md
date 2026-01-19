# Random Forest Module

Module Random Forest cung cấp chức năng train và sử dụng Random Forest model để dự đoán trading signals dựa trên các technical indicators.

## Tổng quan

Module này sử dụng Random Forest Classifier để phân loại trading signals (LONG/SHORT/NEUTRAL) dựa trên các technical features được tính toán từ dữ liệu OHLCV. Module bao gồm:

- **Model Training**: Train Random Forest model với advanced sampling strategies để xử lý class imbalance
- **Signal Generation**: Tạo trading signals từ dữ liệu market mới nhất
- **Model Evaluation**: Đánh giá model với nhiều confidence thresholds
- **Hyperparameter Optimization**: Optuna-based hyperparameter tuning
- **Model Ensemble**: Hỗ trợ VotingClassifier và StackingClassifier
- **Walk-Forward Optimization**: Time-series validation với expanding/rolling windows
- **Feature Selection**: Automated feature selection (mutual information, RF importance)
- **Probability Calibration**: CalibratedClassifierCV để cải thiện probability reliability
- **CLI Tools**: Command-line interface để train và test model
- **Analyzer Integration**: Tích hợp với VotingAnalyzer và HybridAnalyzer

⚠️ **IMPORTANT**: Module này sử dụng sklearn's RandomForestClassifier (ML-based signals), KHÔNG phải `modules.decision_matrix.RandomForestCore` (Pine Script pattern matching).

## Cấu trúc Module

```
random_forest/
├── __init__.py                      # Module exports
├── README.md                        # Tài liệu này
├── optimization.py                 # Hyperparameter optimization với Optuna
├── core/
│   ├── __init__.py                  # Core exports
│   ├── model.py                     # Model training, loading, saving
│   ├── signals.py                   # Signal generation
│   ├── evaluation.py                # Model evaluation functions
│   ├── ensemble.py                   # Model ensemble (Voting/Stacking)
│   └── decision_matrix_integration.py  # Integration với Decision Matrix
├── utils/
│   ├── __init__.py                  # Utils exports
│   ├── data_preparation.py          # Data preparation utilities
│   ├── training.py                  # Training helpers (sampling, model creation)
│   ├── features.py                  # Feature engineering (price-derived, advanced)
│   ├── advanced_labeling.py         # Advanced target labeling strategies
│   ├── feature_selection.py         # Feature selection methods
│   ├── walk_forward.py               # Walk-forward validation, drift detection
│   └── calibration.py               # Probability calibration utilities
└── cli/
    ├── __init__.py                  # CLI exports
    ├── argument_parser.py           # Argument parsing
    └── main.py                      # Main CLI function
```

## Cấu trúc Chi tiết

### Core Module

**`core/model.py`**
- `load_random_forest_model()` - Load model đã train từ file
- `train_random_forest_model()` - Train và save model (hỗ trợ ensemble, calibration, feature selection)
- `train_and_save_global_rf_model()` - Train global model từ multiple symbols
- `_time_series_split_with_gap()` - Time-series split với gap validation để prevent data leakage

**`core/signals.py`**
- `get_latest_random_forest_signal()` - Generate trading signal từ latest market data
- ⚠️ **Chỉ support models trained với derived features** (returns_1, returns_5, log_volume, etc.)
- **Rejects models với raw OHLCV features** (open, high, low, close, volume)

**`core/evaluation.py`**
- `evaluate_model_with_confidence()` - Đánh giá model với nhiều confidence thresholds
- `apply_confidence_threshold()` - Áp dụng confidence threshold cho predictions
- `calculate_and_display_metrics()` - Tính toán và hiển thị performance metrics

**`core/ensemble.py`**
- `create_ensemble()` - Tạo VotingClassifier hoặc StackingClassifier
- `LSTMWrapper` - Wrapper cho PyTorch LSTM models để tích hợp với sklearn
- `load_xgboost_model()` - Load pre-trained XGBoost models
- `load_lstm_model_wrapped()` - Load và wrap LSTM models

**`core/decision_matrix_integration.py`**
- `calculate_random_forest_vote()` - Tính vote cho Decision Matrix
- `get_random_forest_signal_for_decision_matrix()` - Get signal cho Decision Matrix

### Utils Module

**`utils/data_preparation.py`**
- `prepare_training_data()` - Chuẩn bị features và target variable từ OHLCV data
- **Preserves feature order** từ MODEL_FEATURES để ensure consistency
- Integrates advanced labeling và advanced feature engineering

**`utils/training.py`**
- `apply_sampling()` - Áp dụng sampling strategies (SMOTE, ADASYN, BorderlineSMOTE, NONE, BALANCED_RF)
- Returns `sampling_applied` flag để conditionally apply class weights
- `create_model_and_weights()` - Tạo Random Forest model với class weights (nếu sampling không được apply)

**`utils/features.py`**
- `add_price_derived_features()` - Tạo price-derived features (returns_1, returns_5, log_volume, high_low_range, close_open_diff)
- `add_advanced_features()` - Advanced feature engineering (momentum, volatility, lag features, time-based)
- `get_enhanced_feature_names()` - Identify enhanced features

**`utils/advanced_labeling.py`**
- `create_volatility_adjusted_labels()` - Dynamic thresholds based on volatility
- `create_multi_horizon_targets()` - Multi-horizon target generation
- `create_trend_based_labels()` - Trend-based filtering
- `create_advanced_target()` - Unified advanced labeling function

**`utils/feature_selection.py`**
- `select_features_mutual_info()` - Select top-K features using mutual information
- `select_features_rf_importance()` - Select features based on RF importance threshold
- `select_features()` - Unified feature selection entry point

**`utils/walk_forward.py`**
- `WalkForwardValidator` - Time-series validation với expanding/rolling windows
- `ModelDriftDetector` - Monitor và detect model performance degradation
- `ModelVersionManager` - Generate versioned model filenames và track metadata
- `should_retrain_model()` - Check periodic retraining schedule

**`utils/calibration.py`**
- `calibrate_model()` - Wrap model với CalibratedClassifierCV (sigmoid/isotonic)
- `evaluate_calibration()` - Evaluate calibration quality (Brier score, ECE)

### Optimization Module

**`optimization.py`**
- `HyperparameterTuner` - Optuna-based hyperparameter optimization
- **Expanded search space**: n_estimators, max_depth, min_samples_split, min_samples_leaf, max_features, criterion, min_impurity_decrease, bootstrap
- `StudyManager` - Manage Optuna studies

## Sử dụng

### Import Module

```python
from modules.random_forest import (
    train_random_forest_model,
    get_latest_random_forest_signal,
    load_random_forest_model,
    train_and_save_global_rf_model,
    evaluate_model_with_confidence,
    calculate_random_forest_vote,
    HyperparameterTuner,
    StudyManager,
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

# Train model (tự động tạo derived features và advanced features)
model = train_random_forest_model(df, save_model=True)
```

#### 2. Generate Signal

```python
from modules.random_forest import get_latest_random_forest_signal, load_random_forest_model

# Load model đã train
model = load_random_forest_model()

# Generate signal từ latest market data
# ⚠️ Model phải được train với derived features (returns_1, returns_5, log_volume, etc.)
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

#### 3. Train với Advanced Features

```python
from modules.random_forest import train_random_forest_model
from config.random_forest import (
    RANDOM_FOREST_USE_ENSEMBLE,
    RANDOM_FOREST_USE_FEATURE_SELECTION,
    RANDOM_FOREST_USE_PROBABILITY_CALIBRATION,
)

# Enable advanced features (configure in config/random_forest.py)
# RANDOM_FOREST_USE_ENSEMBLE = True
# RANDOM_FOREST_USE_FEATURE_SELECTION = True
# RANDOM_FOREST_USE_PROBABILITY_CALIBRATION = True

model = train_random_forest_model(df, save_model=True)
```

#### 4. Hyperparameter Optimization

```python
from modules.random_forest import HyperparameterTuner

tuner = HyperparameterTuner()
study = tuner.optimize(
    df_input=df,
    n_trials=100,
    timeout=None,
    n_jobs=1,
)

# Get best parameters
best_params = study.best_params
print(f"Best parameters: {best_params}")
```

#### 5. Integration với Analyzers

```python
from core.signal_calculators import get_random_forest_signal
from modules.common.core.data_fetcher import DataFetcher

# Sử dụng trong VotingAnalyzer hoặc HybridAnalyzer
rf_result = get_random_forest_signal(
    data_fetcher=data_fetcher,
    symbol="BTC/USDT",
    timeframe="1h",
    limit=500,
    model_path=None,  # Uses default model path
)

if rf_result is not None:
    signal, confidence = rf_result
    # signal: 1 (LONG), -1 (SHORT), 0 (NEUTRAL)
    # confidence: 0.0 to 1.0
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

#### Price-Derived Features (Required)

Module tự động tạo các price-derived features từ OHLCV data:

- **returns_1**: 1-period return (pct_change)
- **returns_5**: 5-period return (pct_change)
- **log_volume**: Log-normalized volume (handles 0 volume gracefully)
- **high_low_range**: Normalized range `(high - low) / close`
- **close_open_diff**: Normalized price change `(close - open) / open`

⚠️ **IMPORTANT**: Models trained với raw OHLCV features (open, high, low, close, volume) **KHÔNG được support** và sẽ bị reject với error message.

#### Technical Indicators

Module sử dụng `IndicatorEngine` từ `modules.common.core.indicator_engine` để tính toán:

- **Moving Averages**: SMA_20, SMA_50, SMA_200
- **Momentum**: RSI_9, RSI_14, RSI_25
- **Volatility**: ATR_14
- **Trend**: MACD_12_26_9, MACDh, MACDs
- **Bollinger Bands**: BBP_5_2.0
- **Stochastic**: STOCHRSIk, STOCHRSId
- **Volume**: OBV
- **Candlestick Patterns**: DOJI, HAMMER, ENGULFING, v.v.

#### Advanced Features

- **Price Momentum**: ROC (Rate of Change) cho multiple periods
- **Volatility Ratios**: ATR normalized by price
- **Relative Strength**: Price vs Moving Averages ratios
- **Rolling Statistics**: Rolling std và skew trên returns
- **Lag Features**: Temporal context (t-1, t-2, t-3)
- **Time-based Features**: Hour, day of week, month (nếu DatetimeIndex available)

### 2. Advanced Target Labeling

#### Volatility-Adjusted Thresholds

Dynamic buy/sell thresholds based on rolling volatility:

```python
# High volatility → higher thresholds
# Low volatility → lower thresholds
```

#### Multi-Horizon Targets

Generate targets cho multiple horizons (1, 3, 5 periods) và combine:

```python
# Combines signals from multiple horizons
# More robust to short-term noise
```

#### Trend-Based Filtering

Filter targets based on trend direction:

```python
# Only label LONG in uptrend
# Only label SHORT in downtrend
```

### 3. Class Imbalance Handling

Module sử dụng **multiple sampling strategies** để xử lý class imbalance:

- **SMOTE**: Synthetic Minority Over-sampling Technique
- **ADASYN**: Adaptive Synthetic Sampling
- **BorderlineSMOTE**: Borderline SMOTE variant
- **NONE**: No sampling (use class weights instead)
- **BALANCED_RF**: Use balanced Random Forest (class_weight='balanced')

**Smart Behavior**:
- Kiểm tra memory trước khi apply sampling
- Tự động skip sampling nếu memory không đủ
- Sử dụng reduced k_neighbors cho large datasets
- **Class weights chỉ được apply nếu sampling KHÔNG được apply** (tránh double-counting)

### 4. Model Training

#### Time-Series Split với Gap Validation

- **Time-series split** thay vì random split (preserves temporal order)
- **Gap validation**: Total gap = target_horizon + safety_gap để prevent data leakage
- **No lookahead bias**: Training labels at index N use data from N+target_horizon

#### Feature Order Preservation

- Features được preserve trong **MODEL_FEATURES order** để ensure consistency
- Enhanced features được add **sau** MODEL_FEATURES
- **No sorting**: Preserves insertion order để match training/inference

### 5. Model Ensemble

Hỗ trợ combining multiple models:

- **VotingClassifier**: Hard hoặc soft voting
- **StackingClassifier**: Meta-learner approach
- **Base Estimators**: RandomForest, XGBoost, LSTM (wrapped)
- **Graceful Fallback**: Falls back to single RandomForest nếu other models không available

Configuration:
```python
RANDOM_FOREST_USE_ENSEMBLE = True
RANDOM_FOREST_ENSEMBLE_METHOD = "voting"  # or "stacking"
RANDOM_FOREST_ENSEMBLE_VOTING = "soft"    # or "hard"
RANDOM_FOREST_ENSEMBLE_INCLUDE_XGBOOST = True
RANDOM_FOREST_ENSEMBLE_INCLUDE_LSTM = True
```

### 6. Walk-Forward Optimization

Time-series validation với:

- **Expanding Window**: Training set grows over time
- **Rolling Window**: Fixed-size training window
- **Gap Validation**: Prevents data leakage between train/test
- **Periodic Retraining**: Automatic retraining based on schedule
- **Model Drift Detection**: Monitor performance degradation
- **Model Versioning**: Timestamped model filenames và metadata tracking

### 7. Feature Selection

Automated feature selection:

- **Mutual Information**: SelectKBest với mutual_info_classif
- **RF Importance**: Select features based on importance threshold
- **Fallback Handling**: Graceful handling khi k quá large hoặc threshold quá high

### 8. Probability Calibration

Cải thiện reliability của predicted probabilities:

- **CalibratedClassifierCV**: Sigmoid (Platt scaling) hoặc Isotonic regression
- **Cross-Validation**: 5-fold CV by default
- **Evaluation Metrics**: Brier score, Expected Calibration Error (ECE)

### 9. Signal Generation

- **Confidence Threshold**: Chỉ trả về signal nếu confidence >= threshold
- **Feature Validation**: Tự động validate và reject models với raw OHLCV features
- **Error Handling**: Graceful handling khi feature calculation fails
- **Feature Order Matching**: Ensures features match model's expected order

### 10. Analyzer Integration

Tích hợp với VotingAnalyzer và HybridAnalyzer:

- **VotingAnalyzer**: Random Forest signals được include trong voting system
- **HybridAnalyzer**: Random Forest signals được add sau SPC processing
- **Error Handling**: Analyzers handle Random Forest errors gracefully
- **Configuration**: Enable/disable via `enable_random_forest` flag

## Configuration

Module sử dụng configuration từ `config/random_forest.py` và `config/model_features.py`:

### Basic Configuration

- `MODEL_FEATURES`: Danh sách features sử dụng cho training
- `CONFIDENCE_THRESHOLD`: Default confidence threshold
- `CONFIDENCE_THRESHOLDS`: List thresholds để evaluate
- `MAX_TRAINING_ROWS`: Số rows tối đa để training
- `MODEL_RANDOM_STATE`: Random state cho reproducibility
- `MODEL_TEST_SIZE`: Test size ratio
- `MIN_TRAINING_SAMPLES`: Số samples tối thiểu để train
- `MIN_MEMORY_GB`: Memory tối thiểu để apply sampling

### Advanced Configuration

#### Target Labeling

- `RANDOM_FOREST_TARGET_HORIZON`: Number of periods to look ahead for labels
- `RANDOM_FOREST_TOTAL_GAP`: Total gap for time-series split (target_horizon + safety_gap)
- `RANDOM_FOREST_USE_VOLATILITY_ADJUSTED_THRESHOLDS`: Enable volatility-adjusted thresholds
- `RANDOM_FOREST_MULTI_HORIZON_ENABLED`: Enable multi-horizon targets
- `RANDOM_FOREST_TREND_BASED_LABELING_ENABLED`: Enable trend-based filtering

#### Sampling Strategies

- `RANDOM_FOREST_SAMPLING_STRATEGY`: "SMOTE", "ADASYN", "BORDERLINE_SMOTE", "NONE", "BALANCED_RF"

#### Model Ensemble

- `RANDOM_FOREST_USE_ENSEMBLE`: Enable model ensemble
- `RANDOM_FOREST_ENSEMBLE_METHOD`: "voting" or "stacking"
- `RANDOM_FOREST_ENSEMBLE_VOTING`: "hard" or "soft"
- `RANDOM_FOREST_ENSEMBLE_INCLUDE_XGBOOST`: Include XGBoost in ensemble
- `RANDOM_FOREST_ENSEMBLE_INCLUDE_LSTM`: Include LSTM in ensemble

#### Walk-Forward Optimization

- `RANDOM_FOREST_USE_WALK_FORWARD`: Enable walk-forward validation
- `RANDOM_FOREST_WALK_FORWARD_N_SPLITS`: Number of splits
- `RANDOM_FOREST_WALK_FORWARD_EXPANDING_WINDOW`: Use expanding window (True) or rolling window (False)
- `RANDOM_FOREST_RETRAIN_PERIOD_DAYS`: Days between retraining
- `RANDOM_FOREST_DRIFT_DETECTION_ENABLED`: Enable drift detection
- `RANDOM_FOREST_DRIFT_THRESHOLD`: Performance degradation threshold
- `RANDOM_FOREST_MODEL_VERSIONING_ENABLED`: Enable model versioning

#### Feature Selection

- `RANDOM_FOREST_USE_FEATURE_SELECTION`: Enable feature selection
- `RANDOM_FOREST_FEATURE_SELECTION_METHOD`: "mutual_info" or "rf_importance"
- `RANDOM_FOREST_FEATURE_SELECTION_K`: Number of features to select (for mutual_info)
- `RANDOM_FOREST_FEATURE_IMPORTANCE_THRESHOLD`: Importance threshold (for rf_importance)

#### Probability Calibration

- `RANDOM_FOREST_USE_PROBABILITY_CALIBRATION`: Enable probability calibration
- `RANDOM_FOREST_CALIBRATION_METHOD`: "sigmoid" or "isotonic"
- `RANDOM_FOREST_CALIBRATION_CV`: Cross-validation folds

## Breaking Changes

### ⚠️ Backward Compatibility Removed

**Models trained với raw OHLCV features (open, high, low, close, volume) KHÔNG được support nữa.**

- Old models sẽ bị **reject** với error message
- **Phải retrain models** với derived features (returns_1, returns_5, log_volume, high_low_range, close_open_diff)
- Error message sẽ hướng dẫn retrain model

### Feature Order Changes

- Features giờ được preserve trong **MODEL_FEATURES order** (không còn sorted)
- Enhanced features được add **sau** MODEL_FEATURES
- Ensures consistency giữa training và inference

## Testing

Module có comprehensive test suite:

### Unit Tests

- `tests/random_forest/test_signals_random_forest.py`: Signal generation tests
- `tests/random_forest/test_optimization.py`: Hyperparameter optimization tests
- `tests/random_forest/test_advanced_labeling.py`: Advanced labeling tests
- `tests/random_forest/test_ensemble.py`: Model ensemble tests
- `tests/random_forest/test_walk_forward.py`: Walk-forward validation tests
- `tests/random_forest/test_feature_selection.py`: Feature selection tests
- `tests/random_forest/test_calibration.py`: Probability calibration tests
- `tests/random_forest/test_price_derived_features.py`: Price-derived features tests
- `tests/random_forest/test_feature_order_consistency.py`: Feature order consistency tests
- `tests/random_forest/test_no_legacy_compatibility.py`: Legacy model rejection tests

### Integration Tests

- `tests/random_forest/test_analyzer_integration.py`: Integration với VotingAnalyzer và HybridAnalyzer

Chạy tests:

```bash
# All tests
pytest tests/random_forest/ -v

# Specific test file
pytest tests/random_forest/test_signals_random_forest.py -v

# Integration tests
pytest tests/random_forest/test_analyzer_integration.py -v
```

## Dependencies

- `scikit-learn`: Random Forest Classifier, ensemble methods, feature selection, calibration
- `imbalanced-learn`: SMOTE, ADASYN, BorderlineSMOTE
- `optuna`: Hyperparameter optimization
- `pandas`, `numpy`: Data manipulation
- `modules.common.core.indicator_engine`: Feature engineering
- `config`: Configuration constants

## Notes

### Feature Engineering

- Module tự động tạo **price-derived features** từ OHLCV data
- **Raw OHLCV features không được sử dụng** trong training hoặc inference
- Features được preserve trong **MODEL_FEATURES order** để ensure consistency

### Data Leakage Prevention

- **Time-series split** với gap validation thay vì random split
- **Total gap** = target_horizon (for label lookahead) + safety_gap (for independence)
- **No lookahead bias**: Training labels use future data correctly

### Sampling và Class Weights

- **Sampling strategies** (SMOTE, ADASYN, etc.) được apply chỉ trên training set
- **Class weights** chỉ được apply nếu sampling **KHÔNG được apply** (tránh double-counting)
- `apply_sampling()` returns `sampling_applied` flag để conditionally apply class weights

### Model Compatibility

- ⚠️ **Models trained với raw OHLCV features sẽ bị reject**
- Models phải được train với **derived features** (returns_1, returns_5, log_volume, etc.)
- Error message sẽ hướng dẫn retrain model

### Signal Generation

- Signal generation trả về "NEUTRAL" nếu:
  - Confidence < CONFIDENCE_THRESHOLD
  - Model uses deprecated raw OHLCV features
  - Missing critical derived features
  - Feature calculation fails

### Integration với Analyzers

- Random Forest signals được integrate vào VotingAnalyzer và HybridAnalyzer
- Errors được handle gracefully (không crash analyzer)
- Results include `random_forest_signal`, `random_forest_vote`, `random_forest_confidence`

## Migration Guide

### Từ Raw OHLCV Features sang Derived Features

1. **Retrain models** với code mới (tự động tạo derived features)
2. **Old models sẽ bị reject** - phải retrain
3. **Feature order** được preserve - không cần thay đổi code

### Từ apply_smote sang apply_sampling

- Function name changed: `apply_smote()` → `apply_sampling()`
- Returns tuple: `(features, target, sampling_applied)` thay vì `(features, target)`
- `sampling_applied` flag được sử dụng để conditionally apply class weights

## Related Modules

- **modules.decision_matrix**: Decision Matrix voting system (sử dụng Random Forest signals)
- **core.voting_analyzer**: Pure voting analyzer (tích hợp Random Forest)
- **core.hybrid_analyzer**: Sequential filtering + voting analyzer (tích hợp Random Forest)
- **core.signal_calculators**: Signal calculator functions (includes `get_random_forest_signal()`)
