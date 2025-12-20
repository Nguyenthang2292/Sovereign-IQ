"""
Random Forest Model Configuration.

Configuration constants for Random Forest model training and prediction.
"""

from pathlib import Path

# Model Features - imported from shared configuration
from .model_features import MODEL_FEATURES  # noqa: F401

# Random Forest specific features
# Only includes features computed by IndicatorProfile.CORE (no candlestick patterns)
# This ensures consistency between training and prediction
RANDOM_FOREST_FEATURES = [
    # Basic OHLCV data
    "open",
    "high",
    "low",
    "close",
    "volume",
    # Trend Indicators (from TrendIndicators.apply)
    "SMA_20",
    "SMA_50",
    "SMA_200",
    "ADX_14",
    # Momentum Indicators (from MomentumIndicators.apply)
    "RSI_9",
    "RSI_14",
    "RSI_25",
    "MACD_12_26_9",
    "MACDh_12_26_9",
    "MACDs_12_26_9",
    "BBP_5_2.0",
    "STOCHRSIk_14_14_3_3",
    "STOCHRSId_14_14_3_3",
    "KAMA_10",
    # Volatility Indicators (from VolatilityIndicators.apply)
    "ATR_14",
    "ATR_50",
    "ATR_RATIO_14_50",
    # Volume Indicators (from VolumeIndicators.apply)
    "OBV",
]

# Confidence Thresholds
CONFIDENCE_THRESHOLD = 0.6  # Default confidence threshold for signal generation
CONFIDENCE_THRESHOLDS = [0.5, 0.6, 0.7, 0.8, 0.9]  # Thresholds for model evaluation

# Training Configuration
MAX_TRAINING_ROWS = 100000  # Maximum rows for training (will sample if exceeded)
MODEL_RANDOM_STATE = 42  # Random state for reproducibility
MODEL_TEST_SIZE = 0.2  # Test set size ratio (20%)

# Model Storage
MODELS_DIR = Path("artifacts/models")  # Directory to save models
RANDOM_FOREST_MODEL_FILENAME = "random_forest_model.joblib"  # Default model filename

# Target Label Creation
BUY_THRESHOLD = 0.01  # Threshold for buy signal (1% future return)
SELL_THRESHOLD = -0.01  # Threshold for sell signal (-1% future return)

# SMOTE Configuration
LARGE_DATASET_THRESHOLD_FOR_SMOTE = 50000  # Threshold to reduce k_neighbors in SMOTE
MIN_MEMORY_GB = 2.0  # Minimum available memory (GB) to run SMOTE
MIN_TRAINING_SAMPLES = 100  # Minimum samples required for training

