"""
Pairs Trading Configuration.

Configuration constants for pairs trading analysis, validation, and scoring.
Includes hedge ratio calculation, cointegration testing, and opportunity scoring.
"""

# Performance Analysis Weights (Default)
PAIRS_TRADING_WEIGHTS = {
    "1d": 0.5,  # Weight for 1 day (24 candles)
    "3d": 0.3,  # Weight for 3 days (72 candles)
    "1w": 0.2,  # Weight for 1 week (168 candles)
}

# Performance Analysis Weight Presets
# Named presets for CLI selection with different trading strategies
PAIRS_TRADING_WEIGHT_PRESETS = {
    "momentum": {"1d": 0.5, "3d": 0.3, "1w": 0.2},  # Favor short-term signals
    "balanced": {"1d": 0.3, "3d": 0.4, "1w": 0.3},  # Balanced short-medium-long term
    "short_term_bounce": {"1d": 0.7, "3d": 0.2, "1w": 0.1},  # Very sensitive to 1d volatility
    "trend_follower": {"1d": 0.2, "3d": 0.3, "1w": 0.5},  # Follow longer trends
    "mean_reversion": {"1d": 0.25, "3d": 0.5, "1w": 0.25},  # Emphasize medium-term for mean reversion
    "volatility_buffer": {"1d": 0.2, "3d": 0.4, "1w": 0.4},  # Reduce short-term noise, increase stability
}

# Hedge Ratio Calculation Configuration
PAIRS_TRADING_OLS_FIT_INTERCEPT = True  # Whether to fit intercept in OLS regression
PAIRS_TRADING_KALMAN_DELTA = 1e-5  # Default delta for Kalman filter
PAIRS_TRADING_KALMAN_OBS_COV = 1.0  # Default observation covariance for Kalman filter

# Kalman Filter Presets
PAIRS_TRADING_KALMAN_PRESETS = {
    "fast_react": {
        "description": "Fast reaction – beta changes quickly, suitable for volatile markets",
        "delta": 5e-5,
        "obs_cov": 0.5,
    },
    "balanced": {
        "description": "Balanced – default setting, moderate reaction",
        "delta": 1e-5,
        "obs_cov": 1.0,
    },
    "stable": {
        "description": "Stable – beta changes slowly, reduces noise",
        "delta": 5e-6,
        "obs_cov": 2.0,
    },
}

# Opportunity Scoring Presets
# Multipliers for different scoring factors
PAIRS_TRADING_OPPORTUNITY_PRESETS = {
    "balanced": {
        "description": "Default balanced between rewards/penalties",
        "hedge_ratio_strategy": "best",  # 'ols', 'kalman', 'best', or 'avg'
        "corr_good_bonus": 1.20,
        "corr_low_penalty": 0.80,
        "corr_high_penalty": 0.90,
        "cointegration_bonus": 1.15,
        "weak_cointegration_bonus": 1.05,
        "half_life_bonus": 1.10,
        "zscore_divisor": 5.0,
        "zscore_cap": 0.20,
        "hurst_good_bonus": 1.08,
        "hurst_ok_bonus": 1.02,
        "hurst_ok_threshold": 0.60,
        "sharpe_good_bonus": 1.08,
        "sharpe_ok_bonus": 1.03,
        "maxdd_bonus": 1.05,
        "calmar_bonus": 1.05,
        "johansen_bonus": 1.08,
        "f1_high_bonus": 1.05,
        "f1_mid_bonus": 1.02,
        # Momentum-specific bonuses
        "momentum_cointegration_penalty": 0.95,
        "momentum_zscore_high_bonus": 1.15,  # Bonus for |z-score| > 2.0
        "momentum_zscore_moderate_bonus": 1.08,  # Bonus for |z-score| > 1.0
        "momentum_zscore_high_threshold": 2.0,  # Threshold for high z-score bonus
        "momentum_zscore_moderate_threshold": 1.0,  # Threshold for moderate z-score bonus
    },
    "aggressive": {
        "description": "Large rewards for strong signals, accept volatility",
        "hedge_ratio_strategy": "best",  # Use best of OLS/Kalman
        "corr_good_bonus": 1.30,
        "corr_low_penalty": 0.70,
        "corr_high_penalty": 0.85,
        "cointegration_bonus": 1.25,
        "weak_cointegration_bonus": 1.10,
        "half_life_bonus": 1.15,
        "zscore_divisor": 4.0,
        "zscore_cap": 0.30,
        "hurst_good_bonus": 1.12,
        "hurst_ok_bonus": 1.05,
        "hurst_ok_threshold": 0.65,
        "sharpe_good_bonus": 1.12,
        "sharpe_ok_bonus": 1.05,
        "maxdd_bonus": 1.02,
        "calmar_bonus": 1.02,
        "johansen_bonus": 1.12,
        "f1_high_bonus": 1.08,
        "f1_mid_bonus": 1.04,
        # Momentum-specific bonuses
        "momentum_cointegration_penalty": 0.90,
        "momentum_zscore_high_bonus": 1.20,  # Bonus for |z-score| > 2.0
        "momentum_zscore_moderate_bonus": 1.12,  # Bonus for |z-score| > 1.0
        "momentum_zscore_high_threshold": 2.0,
        "momentum_zscore_moderate_threshold": 1.0,
    },
    "conservative": {
        "description": "Light rewards, prioritize stable pairs",
        "hedge_ratio_strategy": "ols",  # Prefer stable OLS metrics
        "corr_good_bonus": 1.10,
        "corr_low_penalty": 0.90,
        "corr_high_penalty": 0.95,
        "cointegration_bonus": 1.10,
        "weak_cointegration_bonus": 1.02,
        "half_life_bonus": 1.05,
        "zscore_divisor": 6.0,
        "zscore_cap": 0.15,
        "hurst_good_bonus": 1.04,
        "hurst_ok_bonus": 1.01,
        "hurst_ok_threshold": 0.55,
        "sharpe_good_bonus": 1.05,
        "sharpe_ok_bonus": 1.02,
        "maxdd_bonus": 1.08,
        "calmar_bonus": 1.08,
        "johansen_bonus": 1.05,
        "f1_high_bonus": 1.03,
        "f1_mid_bonus": 1.01,
        # Momentum-specific bonuses
        "momentum_cointegration_penalty": 0.98,
        "momentum_zscore_high_bonus": 1.10,  # Bonus for |z-score| > 2.0
        "momentum_zscore_moderate_bonus": 1.05,  # Bonus for |z-score| > 1.0
        "momentum_zscore_high_threshold": 2.0,
        "momentum_zscore_moderate_threshold": 1.0,
    },
}

# Quantitative Score Weights Configuration
# Default weights and thresholds for calculate_quantitative_score (0-100 scale)
PAIRS_TRADING_QUANTITATIVE_SCORE_WEIGHTS = {
    # Cointegration weights
    "cointegration_full_weight": 30.0,  # Full weight if cointegrated
    "cointegration_weak_weight": 15.0,  # Weak weight if pvalue < 0.1
    "cointegration_weak_pvalue_threshold": 0.1,  # Threshold for weak cointegration
    # Half-life weights and thresholds
    "half_life_excellent_weight": 20.0,  # Weight if half_life < excellent_threshold
    "half_life_good_weight": 10.0,  # Weight if half_life < good_threshold
    "half_life_excellent_threshold": 20.0,  # Excellent threshold (periods)
    "half_life_good_threshold": 50.0,  # Good threshold (periods)
    # Hurst exponent weights and thresholds
    "hurst_excellent_weight": 15.0,  # Weight if hurst < excellent_threshold
    "hurst_good_weight": 8.0,  # Weight if hurst < good_threshold
    "hurst_excellent_threshold": 0.4,  # Excellent threshold
    "hurst_good_threshold": 0.5,  # Good threshold
    # Sharpe ratio weights and thresholds
    "sharpe_excellent_weight": 15.0,  # Weight if sharpe > excellent_threshold
    "sharpe_good_weight": 8.0,  # Weight if sharpe > good_threshold
    "sharpe_excellent_threshold": 2.0,  # Excellent threshold
    "sharpe_good_threshold": 1.0,  # Good threshold
    # F1-score weights and thresholds
    "f1_excellent_weight": 10.0,  # Weight if f1 > excellent_threshold
    "f1_good_weight": 5.0,  # Weight if f1 > good_threshold
    "f1_excellent_threshold": 0.7,  # Excellent threshold
    "f1_good_threshold": 0.6,  # Good threshold
    # Max drawdown weights and thresholds
    "maxdd_excellent_weight": 10.0,  # Weight if abs(maxdd) < excellent_threshold
    "maxdd_good_weight": 5.0,  # Weight if abs(maxdd) < good_threshold
    "maxdd_excellent_threshold": 0.2,  # Excellent threshold (20%)
    "maxdd_good_threshold": 0.3,  # Good threshold (30%)
    # Calmar ratio weights and thresholds
    "calmar_excellent_weight": 5.0,  # Weight if calmar >= excellent_threshold
    "calmar_good_weight": 2.5,  # Weight if calmar >= good_threshold
    "calmar_excellent_threshold": 1.0,  # Excellent threshold
    "calmar_good_threshold": 0.5,  # Good threshold
    # Momentum extensions
    "momentum_adx_strong_weight": 10.0,
    "momentum_adx_moderate_weight": 5.0,
    # Maximum score cap
    "max_score": 100.0,  # Maximum quantitative score (capped at 100)
}

# Momentum-specific filters and thresholds
PAIRS_TRADING_ADX_PERIOD = 14
PAIRS_TRADING_MOMENTUM_FILTERS = {
    "min_adx": 18.0,  # Minimum ADX required to consider a leg trending
    "strong_adx": 25.0,  # Strong trend confirmation threshold
    "adx_base_bonus": 1.03,  # Bonus when both legs pass min_adx
    "adx_strong_bonus": 1.08,  # Bonus when both legs exceed strong_adx
    "adx_weak_penalty_factor": 0.5,  # Penalty scaling factor when ADX < min_adx (0.0-1.0, lower = more penalty)
    "adx_very_weak_threshold": 10.0,  # ADX below this gets severe penalty
    "adx_very_weak_penalty": 0.3,  # Severe penalty multiplier for very weak ADX
    "low_corr_threshold": 0.30,  # Prefer divergence / low correlation
    "high_corr_threshold": 0.75,  # Penalize highly correlated legs
    "low_corr_bonus": 1.05,  # Bonus if |corr| below low_corr_threshold
    "negative_corr_bonus": 1.10,  # Bonus if correlation is negative
    "high_corr_penalty": 0.90,  # Penalty if |corr| above high_corr_threshold
}

# Performance Analysis Settings
PAIRS_TRADING_TOP_N = 5  # Number of top/bottom performers to display
PAIRS_TRADING_MIN_CANDLES = 168  # Minimum candles required (1 week = 168h)
PAIRS_TRADING_TIMEFRAME = "1h"  # Timeframe for analysis
PAIRS_TRADING_LIMIT = 200  # Number of candles to fetch (enough for 1 week + buffer)

# Pairs Validation Settings
PAIRS_TRADING_MIN_VOLUME = 1000000  # Minimum volume (USDT) to consider
PAIRS_TRADING_MIN_SPREAD = 0.01  # Minimum spread (%)
PAIRS_TRADING_MAX_SPREAD = 0.50  # Maximum spread (%)
PAIRS_TRADING_MIN_CORRELATION = 0.3  # Minimum correlation to consider
PAIRS_TRADING_MAX_CORRELATION = 0.9  # Maximum correlation (avoid over-correlation)
PAIRS_TRADING_CORRELATION_MIN_POINTS = 50  # Minimum data points for correlation calculation
PAIRS_TRADING_ADF_PVALUE_THRESHOLD = 0.05  # P-value threshold for cointegration confirmation
PAIRS_TRADING_ADF_MAXLAG = 1  # Maximum lag for Augmented Dickey-Fuller test (used with autolag="AIC")
PAIRS_TRADING_MAX_HALF_LIFE = 50  # Maximum candles for half-life (mean reversion)
PAIRS_TRADING_MIN_HALF_LIFE_POINTS = 10  # Minimum number of valid data points required for half-life calculation
PAIRS_TRADING_ZSCORE_LOOKBACK = 60  # Number of candles for rolling z-score calculation
PAIRS_TRADING_HURST_THRESHOLD = 0.5  # Hurst exponent threshold (mean reversion < 0.5)
PAIRS_TRADING_MIN_SPREAD_SHARPE = 1.0  # Minimum Sharpe ratio for spread
PAIRS_TRADING_MAX_DRAWDOWN = 0.3  # Maximum drawdown (30%)
PAIRS_TRADING_MIN_CALMAR = 1.0  # Minimum Calmar ratio
PAIRS_TRADING_JOHANSEN_CONFIDENCE = 0.95  # Confidence level for Johansen test
PAIRS_TRADING_JOHANSEN_DET_ORDER = 0  # Deterministic order: 0=no constant/trend, 1=constant, -1=no constant with trend
PAIRS_TRADING_JOHANSEN_K_AR_DIFF = 1  # Lag order for VAR model (k_ar_diff = number of lags)
PAIRS_TRADING_PERIODS_PER_YEAR = 365 * 24  # Number of periods per year (for 1h timeframe)
PAIRS_TRADING_CLASSIFICATION_ZSCORE = 0.5  # Z-score threshold for direction classification

# Z-score Metrics Configuration
PAIRS_TRADING_MIN_LAG = 2  # Minimum lag for R/S analysis (must be >= 2 for meaningful variance calculation)
PAIRS_TRADING_MAX_LAG_DIVISOR = 2  # Maximum lag is limited to half of series length for stability
PAIRS_TRADING_HURST_EXPONENT_MULTIPLIER = 2.0  # Multiplier to convert R/S slope to Hurst exponent
PAIRS_TRADING_MIN_CLASSIFICATION_SAMPLES = 20  # Minimum number of samples required for reliable classification metrics
PAIRS_TRADING_HURST_EXPONENT_MIN = 0.0  # Theoretical minimum for Hurst exponent
PAIRS_TRADING_HURST_EXPONENT_MAX = 2.0  # Theoretical maximum for Hurst exponent
PAIRS_TRADING_HURST_EXPONENT_MEAN_REVERTING_MAX = 0.5  # Maximum Hurst for mean-reverting behavior

# Pairs DataFrame Column Names
# Complete list of columns for pairs trading DataFrame results
# This includes core pair information and all quantitative metrics
PAIRS_TRADING_PAIR_COLUMNS = [
    # Core pair information
    "long_symbol",
    "short_symbol",
    "long_score",
    "short_score",
    "spread",
    "correlation",
    "opportunity_score",
    "quantitative_score",
    # OLS-based metrics
    "hedge_ratio",
    "adf_pvalue",
    "is_cointegrated",
    "half_life",
    "mean_zscore",
    "std_zscore",
    "skewness",
    "kurtosis",
    "current_zscore",
    "hurst_exponent",
    "spread_sharpe",
    "max_drawdown",
    "calmar_ratio",
    "classification_f1",
    "classification_precision",
    "classification_recall",
    "classification_accuracy",
    # Johansen test (independent of hedge ratio method)
    "johansen_trace_stat",
    "johansen_critical_value",
    "is_johansen_cointegrated",
    # Kalman hedge ratio
    "kalman_hedge_ratio",
    # Kalman-based metrics
    "kalman_half_life",
    "kalman_mean_zscore",
    "kalman_std_zscore",
    "kalman_skewness",
    "kalman_kurtosis",
    "kalman_current_zscore",
    "kalman_hurst_exponent",
    "kalman_spread_sharpe",
    "kalman_max_drawdown",
    "kalman_calmar_ratio",
    "kalman_classification_f1",
    "kalman_classification_precision",
    "kalman_classification_recall",
    "kalman_classification_accuracy",
]
