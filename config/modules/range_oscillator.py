"""
Range Oscillator Configuration.

Configuration constants for Range Oscillator signal generation.
"""

# Strategy categories for dynamic selection
TRENDING_STRATEGIES = [3, 4, 6, 8]
RANGE_BOUND_STRATEGIES = [2, 7, 9]
VOLATILE_STRATEGIES = [6, 7]
STABLE_STRATEGIES = [2, 3, 9]

# Constants for performance scoring weights
AGREEMENT_WEIGHT = 0.6
STRENGTH_WEIGHT = 0.4

# Normalization constant for oscillator extreme calculation
OSCILLATOR_NORMALIZATION = 100.0

# Valid strategy IDs
VALID_STRATEGY_IDS = {2, 3, 4, 6, 7, 8, 9}

# Range Oscillator Default Parameters
# These parameters control Range Oscillator signal generation
# Adjust based on backtesting results or market conditions
RANGE_OSCILLATOR_LENGTH = 50  # Oscillator length parameter
RANGE_OSCILLATOR_MULTIPLIER = 2.0  # Oscillator multiplier
