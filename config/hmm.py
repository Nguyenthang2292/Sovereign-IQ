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
HMM_HIGH_ORDER_USE_DATA_DRIVEN_INIT = (
    True  # Use data-driven transition matrix and emissions instead of hardcoded values
)
HMM_HIGH_ORDER_MIN_ORDER_DEFAULT = 2  # Minimum order k for High-Order HMM optimization
HMM_HIGH_ORDER_MAX_ORDER_DEFAULT = 4  # Maximum order k for High-Order HMM optimization

# High-Order HMM State Constants
# Base number of states (0=Down, 1=Side, 2=Up)
HMM_HIGH_ORDER_N_BASE_STATES = 3
# Number of observable symbols (0=Down, 1=Side, 2=Up)
HMM_HIGH_ORDER_N_SYMBOLS = 3

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
        "high": 1.2,  # Multiplier for high volatility (conservative)
        "low": 0.9,  # Multiplier for low volatility (aggressive)
    },
}

# State Strength Multipliers Configuration
HMM_STATE_STRENGTH = {
    "strong": 1.0,  # Multiplier for strong states (0, 3)
    "weak": 0.7,  # Multiplier for weak states (1, 2)
}

# Strategy Registry Configuration
# Defines all available HMM strategies with their configuration
HMM_STRATEGIES = {
    "swings": {
        "enabled": True,
        "weight": 1.0,
        "class": "modules.hmm.core.swings.SwingsHMMStrategy",
        "params": {
            "orders_argrelextrema": HMM_HIGH_ORDER_ORDERS_ARGRELEXTREMA_DEFAULT,
            "strict_mode": HMM_HIGH_ORDER_STRICT_MODE_DEFAULT,
        },
    },
    "kama": {
        "enabled": True,
        "weight": 1.5,  # Higher weight for KAMA
        "class": "modules.hmm.core.kama.KamaHMMStrategy",
        "params": {
            "window_kama": HMM_WINDOW_KAMA_DEFAULT,
            "fast_kama": HMM_FAST_KAMA_DEFAULT,
            "slow_kama": HMM_SLOW_KAMA_DEFAULT,
            "window_size": HMM_WINDOW_SIZE_DEFAULT,
        },
    },
    "true_high_order": {
        "enabled": True,
        "weight": 1.0,
        "class": "modules.hmm.core.high_order.TrueHighOrderHMMStrategy",
        "params": {
            "min_order": HMM_HIGH_ORDER_MIN_ORDER_DEFAULT,
            "max_order": HMM_HIGH_ORDER_MAX_ORDER_DEFAULT,
            "orders_argrelextrema": HMM_HIGH_ORDER_ORDERS_ARGRELEXTREMA_DEFAULT,
            "strict_mode": HMM_HIGH_ORDER_STRICT_MODE_DEFAULT,
        },
    },
    # Future strategies can be added here:
    # "new_strategy": {
    #     "enabled": True,
    #     "weight": 1.0,
    #     "class": "modules.hmm.core.new_strategy.NewStrategy",
    #     "params": {...}
    # },
}

# Voting Mechanism Configuration
HMM_VOTING_MECHANISM = (
    "confidence_weighted"  # Options: "simple_majority", "weighted_voting", "confidence_weighted", "threshold_based"
)
HMM_VOTING_THRESHOLD = 0.5  # Threshold for threshold_based voting mechanism
