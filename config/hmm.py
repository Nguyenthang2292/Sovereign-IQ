"""
HMM (Hidden Markov Model) Configuration.

Configuration constants for HMM-based signal generation.
Includes KAMA, High-Order HMM, signal scoring, and confidence settings.
"""

# KAMA (Kaufman Adaptive Moving Average) Configuration
HMM_WINDOW_KAMA_DEFAULT = 10  # Default window size for KAMA calculation
HMM_FAST_KAMA_DEFAULT = 2  # Default fast period for KAMA
HMM_SLOW_KAMA_DEFAULT = 30  # Default slow period for KAMA
HMM_WINDOW_SIZE_DEFAULT = 100  # Default window size for HMM analysis

# High-Order HMM Configuration
HMM_HIGH_ORDER_ORDERS_ARGRELEXTREMA_DEFAULT = 5  # Order parameter for argrelextrema swing detection
HMM_HIGH_ORDER_STRICT_MODE_DEFAULT = True  # Whether to use strict mode for swing-to-state conversion

# Signal Configuration
# Note: Signal values (LONG=1, HOLD=0, SHORT=-1) are now constants in modules.hmm.signal_resolution
HMM_PROBABILITY_THRESHOLD = 0.5  # Minimum probability threshold for signal generation

# Signal Scoring Configuration
HMM_SIGNAL_PRIMARY_WEIGHT = 2  # Weight for primary signal (next_state_with_hmm_kama)
HMM_SIGNAL_TRANSITION_WEIGHT = 1  # Weight for transition states
HMM_SIGNAL_ARM_WEIGHT = 1  # Weight for ARM-based states
HMM_SIGNAL_MIN_THRESHOLD = 3  # Minimum score threshold for signal generation

# Confidence & Normalization Configuration
HMM_HIGH_ORDER_MAX_SCORE = 1.0  # Max score from High-Order HMM (normalized)
HMM_HIGH_ORDER_WEIGHT = 0.4  # Weight for High-Order HMM in combined confidence
HMM_KAMA_WEIGHT = 1.0 - HMM_HIGH_ORDER_WEIGHT  # Weight for KAMA (calculated automatically)
HMM_AGREEMENT_BONUS = 1.2  # Bonus multiplier when signals agree

# Feature Flags
HMM_FEATURES = {
    "confidence_enabled": True,
    "normalization_enabled": True,
    "combined_confidence_enabled": True,
    "high_order_scoring_enabled": True,
    "conflict_resolution_enabled": True,
    "dynamic_threshold_enabled": True,
    "state_strength_enabled": True,
}

# High-Order HMM Scoring Configuration
HMM_HIGH_ORDER_STRENGTH = {
    "bearish": 1.0,  # Strength multiplier for bearish signals
    "bullish": 1.0,  # Strength multiplier for bullish signals
}

# Conflict Resolution Configuration
HMM_CONFLICT_RESOLUTION_THRESHOLD = 1.2  # Ratio to prioritize High-Order over KAMA

# Dynamic Threshold Configuration
HMM_VOLATILITY_CONFIG = {
    "high_threshold": 0.03,  # Volatility threshold (3% std)
    "adjustments": {
        "high": 1.2,   # Multiplier for high volatility (conservative)
        "low": 0.9,   # Multiplier for low volatility (aggressive)
    }
}

# State Strength Multipliers Configuration
HMM_STATE_STRENGTH = {
    "strong": 1.0,  # Multiplier for strong states (0, 3)
    "weak": 0.7,   # Multiplier for weak states (1, 2)
}

