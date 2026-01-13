"""
Deep Learning Configuration.

Configuration constants for deep learning models (TFT, LSTM, etc.).
Includes data pipeline, feature selection, model architecture, and training settings.
"""

from config.xgboost import TARGET_HORIZON

# Triple Barrier Method Configuration
DEEP_TRIPLE_BARRIER_TP_THRESHOLD = 0.02  # 2% take profit threshold
DEEP_TRIPLE_BARRIER_SL_THRESHOLD = 0.01  # 1% stop loss threshold

# Fractional Differentiation Configuration
DEEP_FRACTIONAL_DIFF_D = 0.5  # Fractional differentiation order (0 < d < 1)
DEEP_FRACTIONAL_DIFF_WINDOW = 100  # Window size for fractional differentiation

# Pipeline Configuration
DEEP_USE_FRACTIONAL_DIFF = True  # Whether to apply fractional differentiation
DEEP_USE_TRIPLE_BARRIER = False  # Whether to use Triple Barrier Method for labeling
DEEP_SCALER_DIR = "artifacts/deep/scalers"  # Directory to save/load scaler parameters

# Data Split Configuration
DEEP_TRAIN_RATIO = 0.7  # Proportion for training set
DEEP_VAL_RATIO = 0.15  # Proportion for validation set
DEEP_TEST_RATIO = 0.15  # Proportion for test set

# Feature Selection & Engineering Configuration
DEEP_FEATURE_SELECTION_METHOD = "combined"  # 'mutual_info', 'boruta', 'f_test', or 'combined'
DEEP_FEATURE_SELECTION_TOP_K = 25  # Number of top features to select (20-30 recommended)
DEEP_FEATURE_COLLINEARITY_THRESHOLD = 0.85  # Correlation threshold for removing collinear features (0.8-0.95)
DEEP_FEATURE_SELECTION_DIR = "artifacts/deep/feature_selection"  # Directory to save/load feature selection results
DEEP_USE_FEATURE_SELECTION = True  # Whether to apply feature selection

# Dataset & DataModule Configuration
# Note: DEEP_MAX_PREDICTION_LENGTH depends on TARGET_HORIZON from xgboost config

DEEP_MAX_ENCODER_LENGTH = 64  # Lookback window (64-128 bars recommended)
DEEP_MAX_PREDICTION_LENGTH = TARGET_HORIZON  # Prediction horizon (align with TARGET_HORIZON)
DEEP_BATCH_SIZE = 64  # Batch size for training
DEEP_NUM_WORKERS = 4  # Number of workers for DataLoader
DEEP_TARGET_COL = "future_log_return"  # Default target column for regression
DEEP_TARGET_COL_CLASSIFICATION = "triple_barrier_label"  # Target column for classification
DEEP_DATASET_DIR = "artifacts/deep/datasets"  # Directory to save/load dataset metadata

# Model Configuration - Phase 1: Vanilla TFT (MVP)
DEEP_MODEL_HIDDEN_SIZE = 16  # Hidden size of TFT model
DEEP_MODEL_ATTENTION_HEAD_SIZE = 4  # Size of attention heads
DEEP_MODEL_DROPOUT = 0.1  # Dropout rate
DEEP_MODEL_LEARNING_RATE = 0.03  # Learning rate
DEEP_MODEL_QUANTILES = [0.02, 0.1, 0.25, 0.5, 0.75, 0.9, 0.98]  # Quantiles for QuantileLoss
DEEP_MODEL_REDUCE_ON_PLATEAU_PATIENCE = 4  # Patience for learning rate reduction
DEEP_CHECKPOINT_DIR = "artifacts/deep/checkpoints"  # Directory to save model checkpoints
DEEP_EARLY_STOPPING_PATIENCE = 10  # Early stopping patience
DEEP_CHECKPOINT_SAVE_TOP_K = 3  # Number of top checkpoints to save

# Model Configuration - Phase 2: Optuna Optimization
DEEP_OPTUNA_N_TRIALS = 20  # Number of optimization trials
DEEP_OPTUNA_TIMEOUT = None  # Timeout in seconds (None = no timeout)
DEEP_OPTUNA_N_JOBS = 1  # Number of parallel jobs
DEEP_OPTUNA_DIR = "artifacts/deep/optuna"  # Directory to save Optuna results
DEEP_OPTUNA_MAX_EPOCHS = 50  # Maximum epochs per trial

# Model Configuration - Phase 3: Hybrid LSTM + TFT (Advanced)
DEEP_HYBRID_LSTM_HIDDEN_SIZE = 32  # Hidden size for LSTM branch
DEEP_HYBRID_LSTM_NUM_LAYERS = 2  # Number of LSTM layers
DEEP_HYBRID_FUSION_SIZE = 64  # Size of fused representation
DEEP_HYBRID_NUM_CLASSES = 3  # Number of classes for classification (UP, NEUTRAL, DOWN)
DEEP_HYBRID_LAMBDA_CLASS = 1.0  # Weight for classification loss
DEEP_HYBRID_LAMBDA_REG = 1.0  # Weight for regression loss
DEEP_HYBRID_LEARNING_RATE = 0.001  # Learning rate for hybrid model

# Training Configuration
DEEP_MAX_EPOCHS = 100  # Maximum number of training epochs
DEEP_ACCELERATOR = "auto"  # Accelerator: 'auto', 'gpu', 'cpu'
DEEP_DEVICES = 1  # Number of devices to use
DEEP_PRECISION = 32  # Precision: 16, 32, or 'bf16'
DEEP_GRADIENT_CLIP_VAL = 0.5  # Gradient clipping value (None to disable)

# PyTorch Environment Configuration
# These environment variables are set before importing PyTorch to ensure proper behavior
# Production-safe environment variables (minimal, performance-optimized)
PYTORCH_ENV = {
    # WARNING: KMP_DUPLICATE_LIB_OK suppresses warnings about duplicate OpenMP libraries.
    # This flag masks underlying dependency conflicts and should be used with caution.
    #
    # Root cause: Multiple packages (PyTorch, NumPy, SciPy, etc.) may ship different
    # versions of OpenMP libraries, causing conflicts when loaded simultaneously.
    #
    # Recommended solutions (in order of preference):
    # 1. Use conda environments with conda-forge packages (most compatible)
    # 2. Ensure all packages use compatible OpenMP versions via package manager
    # 3. Set LD_LIBRARY_PATH (Linux) or DYLD_LIBRARY_PATH (macOS) to prioritize one OpenMP version
    # 4. Use virtual environments with carefully managed dependencies
    # 5. As a last resort, keep this flag enabled but monitor for runtime issues
    #
    # This is currently enabled to allow the application to run, but investigate
    # the root cause if you encounter:
    # - Performance degradation
    # - Unexpected crashes during multithreaded operations
    # - Memory issues or resource leaks
    "KMP_DUPLICATE_LIB_OK": "True",
}

# Debug-only environment variables (only applied when DEBUG or DEV env var is set)
# These settings hurt production performance but help with debugging
PYTORCH_DEBUG_ENV = {
    "OMP_NUM_THREADS": "1",  # Limit OpenMP threads to avoid oversubscription
    "CUDA_LAUNCH_BLOCKING": "1",  # Synchronous CUDA launches for better error messages
    "TORCH_USE_CUDA_DSA": "1",  # Enable CUDA Device-Side Assertions for debugging
}
