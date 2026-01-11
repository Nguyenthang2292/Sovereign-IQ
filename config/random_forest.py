
from pathlib import Path

from .evaluation import CONFIDENCE_THRESHOLD, CONFIDENCE_THRESHOLDS  # noqa: F401
from .model_features import MODEL_FEATURES  # noqa: F401
from .model_features import MODEL_FEATURES  # noqa: F401

"""
Random Forest Model Configuration.

Configuration constants for Random Forest model training and prediction.
"""


# Confidence Thresholds - imported from shared evaluation configuration
from .evaluation import CONFIDENCE_THRESHOLD, CONFIDENCE_THRESHOLDS  # noqa: F401

# Model Features - imported from shared configuration
from .model_features import MODEL_FEATURES  # noqa: F401

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
