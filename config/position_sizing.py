"""
Position Sizing Configuration.

Configuration constants for position sizing calculation using
Bayesian Kelly Criterion and Regime Switching.
"""

# Backtest Configuration
DEFAULT_LOOKBACK_DAYS = 15  # Number of days to look back for backtesting
DEFAULT_TIMEFRAME = "15m"  # Default timeframe for backtesting (aligned with common.py)
DEFAULT_LIMIT = 1500  # Default number of candles to fetch for backtesting

# Kelly Criterion Configuration
DEFAULT_FRACTIONAL_KELLY = 0.25  # Use 25% of full Kelly to reduce risk
DEFAULT_CONFIDENCE_LEVEL = 0.95  # Confidence level for Bayesian estimation
DEFAULT_MIN_WIN_RATE = 0.4  # Minimum win rate to consider (below this, position size = 0)
DEFAULT_MIN_TRADES = 10  # Minimum number of trades required for reliable Kelly calculation

# Position Size Constraints
DEFAULT_MAX_POSITION_SIZE = 1.0  # Max 100% of account per symbol
DEFAULT_MIN_POSITION_SIZE = 0.1  # Min 10% of account per symbol (if signal is valid)
DEFAULT_MAX_PORTFOLIO_EXPOSURE = 0.8  # Max 80% of account across all positions

# Bayesian Prior Parameters (Beta distribution)
# Prior belief about win rate before seeing data
# Higher alpha, beta = stronger prior belief
KELLY_PRIOR_ALPHA = 2.0  # Prior success count
KELLY_PRIOR_BETA = 2.0  # Prior failure count
# This gives a prior mean of alpha/(alpha+beta) = 0.5 (50% win rate)

# Kelly Fraction Bounds
KELLY_MIN_FRACTION = 0.0  # Minimum Kelly fraction (0%)
KELLY_MAX_FRACTION = 0.30  # Maximum Kelly fraction (30% of account)

# Kelly Conservative Estimation Thresholds
# Used when determining whether to use lower bound or posterior mean
KELLY_MIN_LOWER_BOUND_THRESHOLD = 0.1  # Minimum acceptable lower bound (below this, use posterior mean)
KELLY_SMALL_SAMPLE_SIZE = 20  # Sample size threshold for small sample adjustment
KELLY_LOWER_BOUND_MEAN_RATIO = 0.7  # Ratio threshold: if lower_bound < posterior_mean * this, use posterior mean
KELLY_POSTERIOR_MEAN_DISCOUNT = 0.9  # Discount factor applied to posterior mean when used instead of lower bound

# Backtest Entry/Exit Rules
BACKTEST_STOP_LOSS_PCT = 0.02  # 2% stop loss
BACKTEST_TAKE_PROFIT_PCT = 0.04  # 4% take profit
BACKTEST_TRAILING_STOP_PCT = 0.015  # 1.5% trailing stop
BACKTEST_MAX_HOLD_PERIODS = 100  # Maximum periods to hold a position
BACKTEST_RISK_PER_TRADE = 0.01  # 1% risk per trade for equity curve calculation

# Performance Metrics Thresholds
MIN_SHARPE_RATIO = 0.5  # Minimum Sharpe ratio to consider position
MIN_WIN_RATE_THRESHOLD = 0.45  # Minimum win rate threshold
MAX_DRAWDOWN_THRESHOLD = -0.3  # Maximum acceptable drawdown (-30%)

# Data Export/Import
DEFAULT_EXPORT_FORMAT = "csv"  # Default format for exporting results
SUPPORTED_EXPORT_FORMATS = ["csv", "json"]  # Supported export formats

# Hybrid Signal Configuration
# Enabled indicators for hybrid signal calculation
ENABLED_INDICATORS = [
    "adaptive_trend",
    "hmm",
    "range_oscillator",
    "random_forest",
    "spc",
    "xgboost",
]

# Signal calculation mode
SIGNAL_CALCULATION_MODE = "precomputed"  # "precomputed" (default) or "incremental" (skip when position open)

# Signal combination mode
SIGNAL_COMBINATION_MODE = "majority_vote"  # "majority_vote", "weighted_voting", "consensus"

# Minimum number of indicators that must agree for a valid signal
MIN_INDICATORS_AGREEMENT = 3  # At least 3 indicators must agree

# Whether to use confidence scores to weight votes
USE_CONFIDENCE_WEIGHTING = True  # Weight votes by indicator confidence scores

# Range Oscillator parameters for hybrid signals
HYBRID_OSC_LENGTH = 50  # Range Oscillator length
HYBRID_OSC_MULT = 2.0  # Range Oscillator multiplier
HYBRID_OSC_STRATEGIES = [2, 3, 4, 6, 7, 8, 9]  # Enabled strategies

# SPC parameters for hybrid signals (optional, uses defaults if None)
HYBRID_SPC_PARAMS = None  # Can be set to a dict with strategy-specific parameters

# Parallel Processing Configuration
ENABLE_PARALLEL_PROCESSING = True  # Enable multiprocessing for batch processing
NUM_WORKERS = None  # Number of workers (None = auto-detect CPU count)
BATCH_SIZE = None  # Batch size for periods (None = auto-calculate)
USE_GPU = True  # Enable GPU acceleration if available (auto-detect)
ENABLE_MULTITHREADING = True  # Enable multithreading for I/O operations

# Debug and Performance Configuration
ENABLE_DEBUG_LOGGING = False  # Enable debug logging (set to False for production to improve performance)
OPTIMIZE_BATCH_SIZE = True  # Automatically optimize batch size based on DataFrame size and CPU count

# Lazy evaluation - only check when actually needed
GPU_AVAILABLE = None  # Will be set on first use

# Performance Monitoring
ENABLE_PERFORMANCE_PROFILING = False  # Enable detailed performance profiling
LOG_PERFORMANCE_METRICS = True  # Log performance metrics to console

# Cache Configuration
SIGNAL_CACHE_MAX_SIZE = 200  # Maximum number of cached signals (LRU cache)
INDICATOR_CACHE_MAX_SIZE = 500  # Maximum number of cached indicator results (LRU cache)
DATA_CACHE_MAX_SIZE = 10  # Maximum number of cached DataFrames

# Memory Optimization
USE_EFFICIENT_DTYPES = True  # Use float32 instead of float64 where precision allows
CLEAR_CACHE_ON_COMPLETE = False  # Clear caches after backtest completes (saves memory)

# Performance Tuning
MIN_BATCH_SIZE = 50  # Minimum batch size for parallel processing
MAX_BATCH_SIZE = 5000  # Maximum batch size for parallel processing
BATCH_SIZE_OVERHEAD_FACTOR = 4  # Factor for calculating optimal batch size (higher = smaller batches)
