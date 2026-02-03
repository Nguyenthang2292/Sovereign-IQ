"""
Decision Matrix Configuration.

Configuration constants for Decision Matrix voting system.
Indicator accuracy values and voting thresholds.
"""

# Indicator Accuracy Values (for Decision Matrix voting system)
# These values represent historical accuracy/performance of each indicator
# Adjust based on backtesting results or actual performance data

# Main Indicators Accuracy
DECISION_MATRIX_ATC_ACCURACY = 0.65  # Adaptive Trend Classification accuracy
DECISION_MATRIX_OSCILLATOR_ACCURACY = 0.70  # Range Oscillator accuracy (highest)
DECISION_MATRIX_XGBOOST_ACCURACY = 0.68  # XGBoost prediction accuracy
DECISION_MATRIX_HMM_ACCURACY = 0.67  # HMM (Hidden Markov Model) accuracy
DECISION_MATRIX_RANDOM_FOREST_ACCURACY = 0.66  # Random Forest prediction accuracy

# SPC Strategy Accuracies (for weighted aggregation)
DECISION_MATRIX_SPC_STRATEGY_ACCURACIES = {
    "cluster_transition": 0.68,  # Cluster Transition strategy accuracy
    "regime_following": 0.66,  # Regime Following strategy accuracy
    "mean_reversion": 0.64,  # Mean Reversion strategy accuracy
}

# SPC Aggregated Accuracy (weighted average of 3 strategies)
DECISION_MATRIX_SPC_AGGREGATED_ACCURACY = sum(DECISION_MATRIX_SPC_STRATEGY_ACCURACIES.values()) / len(
    DECISION_MATRIX_SPC_STRATEGY_ACCURACIES
)

# Dictionary for all indicator accuracies
DECISION_MATRIX_INDICATOR_ACCURACIES = {
    "atc": DECISION_MATRIX_ATC_ACCURACY,
    "oscillator": DECISION_MATRIX_OSCILLATOR_ACCURACY,
    "spc": DECISION_MATRIX_SPC_AGGREGATED_ACCURACY,
    "xgboost": DECISION_MATRIX_XGBOOST_ACCURACY,
    "hmm": DECISION_MATRIX_HMM_ACCURACY,
    "random_forest": DECISION_MATRIX_RANDOM_FOREST_ACCURACY,
}

# Decision Matrix Voting Configuration
DECISION_MATRIX_VOTING_THRESHOLD = 0.5  # Minimum weighted score for positive vote (0.0-1.0)
DECISION_MATRIX_MIN_VOTES = 2  # Minimum number of indicators that must agree
