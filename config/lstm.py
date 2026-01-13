"""
LSTM Model Configuration.

Configuration constants for LSTM/CNN-LSTM models including:
- Directory paths
- Data preprocessing settings
- Model architecture parameters
- Training hyperparameters
- Target creation thresholds
"""

from pathlib import Path

# ============================================================================
# Directory Configuration
# ============================================================================

# Calculate project root: config/lstm.py -> config/ -> project root
_PROJECT_ROOT = Path(__file__).parent.parent

MODELS_DIR = _PROJECT_ROOT / "artifacts" / "models" / "lstm"  # Directory to save/load LSTM models

# ============================================================================
# Data Preprocessing Configuration
# ============================================================================

WINDOW_SIZE_LSTM = 60  # Number of time steps to look back for sequence creation (sliding window)
TRAIN_TEST_SPLIT = 0.7  # Proportion for training set (70%)
VALIDATION_SPLIT = 0.15  # Proportion for validation set (15%)
# Test split = 1 - TRAIN_TEST_SPLIT - VALIDATION_SPLIT = 15%

# ============================================================================
# Target Creation Configuration
# ============================================================================

TARGET_THRESHOLD_LSTM = 0.01  # Threshold for creating classification targets (1%)
NEUTRAL_ZONE_LSTM = 0.005  # Neutral zone around zero for balanced target creation (0.5%)
FUTURE_RETURN_SHIFT_MULTIPLIER = 0.4  # Multiplier for calculating future return shift from window size
FUTURE_RETURN_SHIFT = int(
    WINDOW_SIZE_LSTM * FUTURE_RETURN_SHIFT_MULTIPLIER
)  # Number of periods to shift forward for future return calculation

# ============================================================================
# Model Architecture Configuration
# ============================================================================

# LSTM Layer Dimensions
# Note: First layer's hidden dimension is passed as `hidden_size` parameter (default: 64)
# See modules/lstm/models/lstm_models.py:LSTMModel.__init__ for details
LSTM_HIDDEN_DIM_L2 = 32  # Hidden dimension for second LSTM layer (layer index 2)
LSTM_HIDDEN_DIM_L3 = 16  # Hidden dimension for third LSTM layer (layer index 3)
LSTM_ATTENTION_DIM = 16  # Attention dimension for LSTM-Attention models

# Classifier Configuration
CLASSIFIER_HIDDEN_DIM = 8  # Hidden dimension for classifier layers

# Dropout Configuration
DROPOUT_FINAL_LAYER = 0.2  # Dropout rate for final layers (classifier/regressor)

# GPU Model Configuration
# Transformer-style parameters for hybrid LSTM-attention/Transformer-augmented models.
# These parameters configure transformer components integrated into LSTM architectures:
# - nhead: Used by MultiHeadAttention (modules/lstm/core/multi_head_attention.py) as num_heads
# - d_model: Used by PositionalEncoding (modules/lstm/core/positional_encoding.py) and
#   FeedForward (modules/lstm/core/feed_forward.py) for model dimension
# - dim_feedforward: Used by FeedForward as d_ff (feedforward network hidden dimension)
# - num_layers: Reserved for future transformer encoder/decoder layers
# - dropout: Applied across attention and feedforward components
# Consumed by: LSTMAttentionModel, CNNLSTMAttentionModel (modules/lstm/models/lstm_models.py),
#             cnn_lstm_attention_trainer.py, lstm_attention_trainer.py
GPU_MODEL_CONFIG = {
    "nhead": 8,  # Number of attention heads for multi-head attention
    "d_model": 128,  # Model dimension
    "num_layers": 2,  # Number of encoder/decoder layers
    "dropout": 0.1,  # Dropout rate
    "dim_feedforward": 512,  # Feedforward network dimension
}

# ============================================================================
# Training Configuration
# ============================================================================

DEFAULT_EPOCHS = 50  # Default number of training epochs

# Batch Size Configuration
CPU_BATCH_SIZE = 32  # Default batch size for CPU training
GPU_BATCH_SIZE = 64  # Default batch size for GPU training

# ============================================================================
# Batch Size Optimization Configuration
# ============================================================================

# Memory usage constants
GPU_MEMORY_USAGE_RATIO = 0.8  # Use max 80% of GPU memory to leave room for other operations
CPU_MAX_USABLE_MEMORY_MB = 2048  # Safe cap for CPU batch memory (2GB)

# Batch size bounds
MIN_BATCH_SIZE = 4  # Minimum batch size for all model types

# Maximum batch sizes by model type (GPU)
MAX_BATCH_SIZE_CNN_LSTM = 256
MAX_BATCH_SIZE_LSTM_ATTENTION = 512
MAX_BATCH_SIZE_LSTM = 1024

# CPU batch size divisors for model-specific limits
CPU_BATCH_DIVISOR_CNN_LSTM = 4
CPU_BATCH_DIVISOR_LSTM_ATTENTION = 2

# Minimum batch sizes for CPU model-specific limits
CPU_MIN_BATCH_CNN_LSTM = 8
CPU_MIN_BATCH_LSTM_ATTENTION = 16

# Fallback batch size divisors
FALLBACK_BATCH_DIVISOR_CNN_LSTM = 4
FALLBACK_BATCH_DIVISOR_LSTM_ATTENTION = 2
FALLBACK_MIN_BATCH_CNN_LSTM = 8
FALLBACK_MIN_BATCH_LSTM_ATTENTION = 16

# Training vs inference memory multipliers
# Training requires additional memory for gradients, optimizer states, and activations
TRAINING_MEMORY_MULTIPLIER = 3.0  # Training typically needs ~3x more memory than inference
INFERENCE_MEMORY_MULTIPLIER = 1.0

# Model complexity multipliers (consistent for both CPU and GPU)
# These represent relative memory overhead of each model type
COMPLEXITY_MULTIPLIER = {
    "lstm": 1.0,
    "lstm_attention": 2.5,  # Attention mechanism requires more memory
    "cnn_lstm": 3.0,  # CNN + LSTM requires most memory
}

# ============================================================================
# Kalman Filter Configuration
# ============================================================================

ENABLE_KALMAN_FILTER = False  # Default: disabled for backward compatibility
KALMAN_PROCESS_VARIANCE = 1e-5  # Process noise covariance (Q) - smaller = smoother output
KALMAN_OBSERVATION_VARIANCE = 1.0  # Observation noise covariance (R) - smaller = less smoothing
KALMAN_METHOD = "univariate"  # 'univariate' or 'multivariate' (currently only univariate implemented)

# ============================================================================
# Model Features
# ============================================================================
# Note: MODEL_FEATURES should be imported directly from config.model_features
# to ensure consistency across all models (XGBoost, Random Forest, LSTM)
