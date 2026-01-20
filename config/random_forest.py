"""
Random Forest Model Configuration.

Configuration constants for Random Forest model training and prediction.
"""

from pathlib import Path

# Calculate project root: config/random_forest.py -> config/ -> project root
_PROJECT_ROOT = Path(__file__).parent.parent

# Training Configuration
MAX_TRAINING_ROWS = 100000  # Maximum rows for training (will sample if exceeded)
MODEL_RANDOM_STATE = 42  # Random state for reproducibility
MODEL_TEST_SIZE = 0.2  # Test set size ratio (20%)

# Model Storage
# Note: Model file is stored directly in artifacts/models/ (not in subfolder)
MODELS_DIR = _PROJECT_ROOT / "artifacts" / "models"  # Directory to save models
RANDOM_FOREST_MODEL_FILENAME = "rf_v1_20260120_112819.joblib"  # Default model filename

# Target Prediction Configuration
RANDOM_FOREST_TARGET_HORIZON = 5  # Number of periods to predict ahead for target variable creation
# This must match the shift value used in data_preparation.py to prevent data leakage
# When creating targets, we use shift(-RANDOM_FOREST_TARGET_HORIZON) to look ahead

# Gap Prevention Configuration
# Total gap = target_horizon (for label lookahead) + safety_gap (for independence)
# Training labels at index N use data from N+target_horizon, so we need:
# - target_horizon periods to account for the lookahead in training labels
# - safety_gap additional periods to ensure true independence between train/test windows
RANDOM_FOREST_SAFETY_GAP = 5  # Additional safety margin beyond target_horizon for gap prevention
RANDOM_FOREST_TOTAL_GAP = RANDOM_FOREST_TARGET_HORIZON + RANDOM_FOREST_SAFETY_GAP  # Total gap = 10 periods

# Target Label Creation - Basic (Legacy)
BUY_THRESHOLD = 0.01  # Threshold for buy signal (1% future return) - used as fallback
SELL_THRESHOLD = -0.01  # Threshold for sell signal (-1% future return) - used as fallback

# Advanced Target Labeling Configuration
# Volatility-adjusted thresholds
RANDOM_FOREST_VOLATILITY_WINDOW = 20  # Rolling window for volatility calculation
RANDOM_FOREST_VOLATILITY_MULTIPLIER = 0.5  # Multiplier for dynamic thresholds (0.5 * volatility)
RANDOM_FOREST_USE_VOLATILITY_ADJUSTED_THRESHOLDS = True  # Enable volatility-adjusted thresholds

# Multi-horizon labeling
RANDOM_FOREST_MULTI_HORIZON_ENABLED = True  # Enable multi-horizon target creation
RANDOM_FOREST_HORIZON_1H = 5  # Periods for 1-hour equivalent horizon
RANDOM_FOREST_HORIZON_4H = 20  # Periods for 4-hour equivalent horizon
RANDOM_FOREST_HORIZON_1D = 100  # Periods for 1-day equivalent horizon

# Trend-based labeling
RANDOM_FOREST_TREND_BASED_LABELING_ENABLED = True  # Enable trend-based filtering
RANDOM_FOREST_MIN_TREND_STRENGTH = 0.7  # Minimum trend strength (0-1) to generate signal
RANDOM_FOREST_TREND_STRENGTH_WINDOW = 20  # Window for trend strength calculation (SMA slope)

# SMOTE and Resampling Configuration
RANDOM_FOREST_SAMPLING_STRATEGY = "SMOTE"  # Options: "SMOTE", "ADASYN", "BorderlineSMOTE", "BALANCED_RF", "NONE"
LARGE_DATASET_THRESHOLD_FOR_SMOTE = 50000  # Threshold to reduce k_neighbors in SMOTE
MIN_MEMORY_GB = 2.0  # Minimum available memory (GB) to run SMOTE
MIN_TRAINING_SAMPLES = 100  # Minimum samples required for training

# Model Ensemble Configuration
RANDOM_FOREST_USE_ENSEMBLE = False  # Enable model ensemble (VotingClassifier or StackingClassifier)
RANDOM_FOREST_ENSEMBLE_METHOD = "voting"  # "voting" or "stacking"
RANDOM_FOREST_ENSEMBLE_VOTING = "soft"  # "hard" or "soft" (for VotingClassifier)
RANDOM_FOREST_ENSEMBLE_INCLUDE_XGBOOST = True  # Include XGBoost in ensemble
RANDOM_FOREST_ENSEMBLE_INCLUDE_LSTM = True  # Include LSTM in ensemble
RANDOM_FOREST_ENSEMBLE_FINAL_ESTIMATOR = (
    "RandomForest"  # Final estimator for StackingClassifier ("RandomForest", "XGBoost", "LogisticRegression")
)

# Walk-Forward Optimization Configuration
RANDOM_FOREST_USE_WALK_FORWARD = False  # Enable walk-forward validation instead of single split
RANDOM_FOREST_WALK_FORWARD_N_SPLITS = 5  # Number of splits for walk-forward validation
RANDOM_FOREST_WALK_FORWARD_EXPANDING_WINDOW = True  # Use expanding window (False = rolling window)
RANDOM_FOREST_RETRAIN_PERIOD_DAYS = 30  # Retrain model every N days (0 = disable periodic retraining)
RANDOM_FOREST_DRIFT_DETECTION_ENABLED = True  # Enable model drift detection
RANDOM_FOREST_DRIFT_THRESHOLD = 0.05  # Performance degradation threshold to trigger retraining (5%)
RANDOM_FOREST_DRIFT_WINDOW_SIZE = 100  # Number of recent predictions to monitor for drift
RANDOM_FOREST_MODEL_VERSIONING_ENABLED = True  # Enable model versioning with timestamp

# Feature Selection Configuration
RANDOM_FOREST_USE_FEATURE_SELECTION = False  # Enable feature selection
RANDOM_FOREST_FEATURE_SELECTION_METHOD = "mutual_info"  # "mutual_info" or "rf_importance"
RANDOM_FOREST_FEATURE_SELECTION_K = 20  # Number of top features for SelectKBest (mutual_info method)
RANDOM_FOREST_FEATURE_IMPORTANCE_THRESHOLD = 0.01  # Minimum feature importance threshold (rf_importance method)

# Probability Calibration Configuration
RANDOM_FOREST_USE_PROBABILITY_CALIBRATION = False  # Enable probability calibration using CalibratedClassifierCV
RANDOM_FOREST_CALIBRATION_METHOD = "sigmoid"  # "sigmoid" (Platt scaling) or "isotonic" (isotonic regression)
RANDOM_FOREST_CALIBRATION_CV = 5  # Number of cross-validation folds for calibration

# CLI Default Configuration
DATA_PROCESSING_WAIT_TIME_IN_SECONDS = 2  # Wait time for data processing (seconds)
DEFAULT_CRYPTO_SYMBOLS = ["BTC/USDT", "ETH/USDT", "BNB/USDT"]  # Default crypto symbols for analysis
DEFAULT_TEST_SYMBOL = "BTC/USDT"  # Default symbol for testing
DEFAULT_TEST_TIMEFRAME = "1h"  # Default timeframe for testing
DEFAULT_TIMEFRAMES = ["1h", "4h", "1d"]  # Default list of timeframes for analysis
DEFAULT_TOP_SYMBOLS = 10  # Default number of top symbols to analyze
