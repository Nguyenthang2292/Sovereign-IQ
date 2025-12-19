"""
XGBoost Prediction Configuration.

Configuration constants for XGBoost model training and prediction.
Includes target configuration, model features, and hyperparameters.
"""

# Prediction Target Configuration
TARGET_HORIZON = 24  # Number of candles to predict ahead
TARGET_BASE_THRESHOLD = 0.01  # Base threshold for directional labeling (1%)
TARGET_LABELS = ["DOWN", "NEUTRAL", "UP"]
LABEL_TO_ID = {label: idx for idx, label in enumerate(TARGET_LABELS)}
ID_TO_LABEL = {idx: label for label, idx in LABEL_TO_ID.items()}

# Prediction Windows Mapping
# Maps input timeframes to prediction horizons
PREDICTION_WINDOWS = {
    "30m": "12h",
    "45m": "18h",
    "1h": "24h",
    "2h": "36h",
    "4h": "48h",
    "6h": "72h",
    "12h": "7d",
    "1d": "7d",
}

# Dynamic Lookback Configuration
# Controls how historical reference prices are weighted based on volatility
DYNAMIC_LOOKBACK_SHORT_MULTIPLIER = 1.5  # Short lookback: TARGET_HORIZON * 1.5
DYNAMIC_LOOKBACK_MEDIUM_MULTIPLIER = 2.0  # Medium lookback: TARGET_HORIZON * 2.0 (original)
DYNAMIC_LOOKBACK_LONG_MULTIPLIER = 2.5  # Long lookback: TARGET_HORIZON * 2.5

# Volatility Thresholds for Weight Adjustment
DYNAMIC_LOOKBACK_VOL_LOW_THRESHOLD = 1.8  # Below this = low volatility
DYNAMIC_LOOKBACK_VOL_HIGH_THRESHOLD = 2.2  # Above this = high volatility

# Weight Configuration for Different Volatility Regimes
# Format: [weight_short, weight_medium, weight_long]
DYNAMIC_LOOKBACK_WEIGHTS_LOW_VOL = [0.4, 0.4, 0.2]  # Low volatility: favor short-medium
DYNAMIC_LOOKBACK_WEIGHTS_MEDIUM_VOL = [0.2, 0.5, 0.3]  # Medium volatility: balanced
DYNAMIC_LOOKBACK_WEIGHTS_HIGH_VOL = [0.2, 0.3, 0.5]  # High volatility: favor medium-long

# Model Features - imported from shared configuration
from .model_features import MODEL_FEATURES  # noqa: F401

# XGBoost Model Hyperparameters
XGBOOST_PARAMS = {
    "n_estimators": 200,  # Số lượng cây quyết định (trees). Càng nhiều càng phức tạp, dễ overfitting.
    "learning_rate": 0.05,  # Tốc độ học (eta). Kiểm soát mức đóng góp của mỗi cây.
    "max_depth": 5,  # Độ sâu tối đa của mỗi cây. Kiểm soát độ phức tạp của model.
    "subsample": 0.9,  # Tỷ lệ mẫu dữ liệu dùng cho mỗi cây (giảm overfitting).
    "colsample_bytree": 0.9,  # Tỷ lệ đặc trưng (features) dùng cho mỗi cây.
    "gamma": 0.1,  # Mức giảm loss tối thiểu để chia nút (cắt tỉa cây).
    "min_child_weight": 3,  # Tổng trọng số tối thiểu tại nút con (tránh học nhiễu).
    "random_state": 42,  # Hạt giống ngẫu nhiên để tái lập kết quả.
    "objective": "multi:softprob",  # Hàm mục tiêu: phân loại đa lớp trả về xác suất.
    "eval_metric": "mlogloss",  # Thước đo đánh giá lỗi: Multi-class Log Loss.
    "n_jobs": -1,  # Sử dụng tất cả lõi CPU.
}

