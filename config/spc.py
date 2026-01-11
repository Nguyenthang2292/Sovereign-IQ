"""
Simplified Percentile Clustering (SPC) Configuration.

Configuration constants for SPC clustering and signal generation.
Includes strategy-specific parameters and vote aggregation settings.
"""

# SPC General Configuration
SPC_LOOKBACK = 500  # Default historical bars for SPC clustering (lookback window)
SPC_P_LOW = 5.0  # Default lower percentile for SPC clustering
SPC_P_HIGH = 95.0  # Default upper percentile for SPC clustering

# SPC Strategy-Specific Parameters
# These parameters control signal generation for each SPC strategy
# Adjust based on backtesting results or market conditions
SPC_STRATEGY_PARAMETERS = {
    "cluster_transition": {
        "min_signal_strength": 0.3,  # Minimum signal strength threshold
        "min_rel_pos_change": 0.1,  # Minimum relative position change
    },
    "regime_following": {
        "min_regime_strength": 0.7,  # Minimum regime strength threshold
        "min_cluster_duration": 2,  # Minimum bars in same cluster
    },
    "mean_reversion": {
        "extreme_threshold": 0.2,  # Real_clust threshold for extreme detection
        "min_extreme_duration": 3,  # Minimum bars in extreme state
    },
}

# SPC Vote Aggregation Configuration
# Controls how votes from 3 SPC strategies are aggregated into a single vote

# SPC Aggregation Consensus Mode
# "threshold": Requires minimum fraction of strategies to agree
# "weighted": Uses weighted voting with minimum total and difference thresholds
SPC_AGGREGATION_MODE = "weighted"  # "threshold" or "weighted"

# Threshold Mode Parameters
SPC_AGGREGATION_THRESHOLD = 0.5  # Minimum fraction of strategies that must agree (0.0-1.0)

# Weighted Mode Parameters
SPC_AGGREGATION_WEIGHTED_MIN_TOTAL = (
    0.33  # Minimum total weight required for signal (reduced from 0.5 to accept single strong strategy)
)
SPC_AGGREGATION_WEIGHTED_MIN_DIFF = 0.1  # Minimum difference between LONG and SHORT weights

# Adaptive Weights Configuration
SPC_AGGREGATION_ENABLE_ADAPTIVE_WEIGHTS = False  # Enable performance-based weight adjustment
SPC_AGGREGATION_ADAPTIVE_PERFORMANCE_WINDOW = 10  # Lookback window for performance calculation

# Signal Strength Filtering
SPC_AGGREGATION_MIN_SIGNAL_STRENGTH = 0.0  # Minimum strength required (0.0 = disabled)

# Simple Mode Fallback Configuration
SPC_AGGREGATION_ENABLE_SIMPLE_FALLBACK = True  # Enable fallback to simple mode when weighted/threshold fail
SPC_AGGREGATION_SIMPLE_MIN_ACCURACY_TOTAL = (
    0.65  # Minimum total accuracy for simple mode (reduced from 1.5 to accept single strategy)
)

# Custom Strategy Weights (optional, overrides accuracy-based weights if provided)
# If None, uses DECISION_MATRIX_SPC_STRATEGY_ACCURACIES
SPC_AGGREGATION_STRATEGY_WEIGHTS = None  # Dict[str, float] or None
