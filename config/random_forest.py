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
RANDOM_FOREST_MODEL_FILENAME = "random_forest_model.joblib"  # Default model filename

# Target Label Creation
BUY_THRESHOLD = 0.01  # Threshold for buy signal (1% future return)
SELL_THRESHOLD = -0.01  # Threshold for sell signal (-1% future return)

# SMOTE Configuration
LARGE_DATASET_THRESHOLD_FOR_SMOTE = 50000  # Threshold to reduce k_neighbors in SMOTE
MIN_MEMORY_GB = 2.0  # Minimum available memory (GB) to run SMOTE
MIN_TRAINING_SAMPLES = 100  # Minimum samples required for training

# CLI Default Configuration
DATA_PROCESSING_WAIT_TIME_IN_SECONDS = 2  # Wait time for data processing (seconds)
DEFAULT_CRYPTO_SYMBOLS = ["BTC/USDT", "ETH/USDT", "BNB/USDT"]  # Default crypto symbols for analysis
DEFAULT_TEST_SYMBOL = "BTC/USDT"  # Default symbol for testing
DEFAULT_TEST_TIMEFRAME = "1h"  # Default timeframe for testing
DEFAULT_TIMEFRAMES = ["1h", "4h", "1d"]  # Default list of timeframes for analysis
DEFAULT_TOP_SYMBOLS = 10  # Default number of top symbols to analyze
