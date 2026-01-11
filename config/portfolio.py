"""
Portfolio Manager Configuration.

Configuration constants for portfolio analysis, risk management, and hedging.
"""

# Benchmark Configuration
BENCHMARK_SYMBOL = "BTC/USDT"  # Default benchmark for beta calculation

# Beta Calculation Configuration
DEFAULT_BETA_MIN_POINTS = 50  # Minimum data points required for beta calculation
DEFAULT_BETA_LIMIT = 1000  # Default limit for beta calculation OHLCV fetch
DEFAULT_BETA_TIMEFRAME = "1h"  # Default timeframe for beta calculation

# Correlation Analysis Configuration
DEFAULT_CORRELATION_MIN_POINTS = 10  # Minimum data points for correlation analysis
DEFAULT_WEIGHTED_CORRELATION_MIN_POINTS = 10  # Minimum data points for weighted correlation

# Hedge Correlation Thresholds
HEDGE_CORRELATION_HIGH_THRESHOLD = 0.7  # High correlation threshold (>= 0.7 = excellent for hedging)
HEDGE_CORRELATION_MEDIUM_THRESHOLD = 0.4  # Medium correlation threshold (0.4-0.7 = moderate hedging effect)
HEDGE_CORRELATION_DIFF_THRESHOLD = 0.1  # Maximum difference between methods for consistency check

# VaR (Value at Risk) Configuration
DEFAULT_VAR_CONFIDENCE = 0.95  # Default confidence level for VaR calculation (95%)
DEFAULT_VAR_LOOKBACK_DAYS = 60  # Default lookback period for VaR calculation (days)
DEFAULT_VAR_MIN_HISTORY_DAYS = 20  # Minimum history required for reliable VaR
DEFAULT_VAR_MIN_PNL_SAMPLES = 10  # Minimum PnL samples required for VaR
