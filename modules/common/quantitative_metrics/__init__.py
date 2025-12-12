"""
Quantitative metrics for technical analysis.

This module provides advanced quantitative indicators and metrics that are distinct from
basic technical indicators. These include:
- Mathematical transformations (Fisher Transform) - in transformations/
- Price ratios (MAR) - in ratios/
- Statistical transformations (Z-Score) - in mean_reversion/
- Risk metrics (Sharpe ratio, max drawdown, Calmar ratio) - in risk/
- Statistical tests (correlation, ADF test, Johansen test) - in statistical_tests/
- Mean reversion metrics (half-life, Hurst exponent, z-score stats) - in mean_reversion/
- Hedge ratio calculations (OLS, Kalman filter) - in hedge_ratios/
- Classification metrics (direction prediction) - in classification/

Key differences from basic indicators:
- More complex calculations (e.g., Fisher Transform with recursive smoothing)
- Statistical transformations (e.g., Z-Score standardization)
- Price ratios (e.g., MAR)
- Risk-adjusted performance metrics
- Cointegration and stationarity tests

Note: CCI and DMI difference have been moved to modules.common.indicators.trend
as they are standard trend indicators.
"""

# Mathematical transformations
from modules.common.quantitative_metrics.transformations import (
    calculate_fisher_transform,
    NUMBA_AVAILABLE,
)

# Price ratios
from modules.common.quantitative_metrics.ratios import (
    calculate_mar,
)

# Risk metrics
from modules.common.quantitative_metrics.risk import (
    calculate_calmar_ratio,
    calculate_max_drawdown,
    calculate_sharpe_ratio,
    calculate_spread_sharpe,  # Alias for calculate_sharpe_ratio
)

# Statistical tests
from modules.common.quantitative_metrics.statistical_tests import (
    calculate_adf_test,
    calculate_correlation,
    calculate_johansen_test,
)

# Mean reversion metrics
from modules.common.quantitative_metrics.mean_reversion import (
    calculate_half_life,
    calculate_hurst_exponent,
    calculate_zscore,
    calculate_zscore_stats,
)

# Hedge ratio calculations
from modules.common.quantitative_metrics.hedge_ratios import (
    calculate_kalman_hedge_ratio,
    calculate_ols_hedge_ratio,
)

# Classification metrics
from modules.common.quantitative_metrics.classification import (
    calculate_direction_metrics,
)

__all__ = [
    # Core quantitative indicators
    "calculate_fisher_transform",
    "calculate_mar",
    "NUMBA_AVAILABLE",
    # Risk metrics
    "calculate_calmar_ratio",
    "calculate_max_drawdown",
    "calculate_sharpe_ratio",
    "calculate_spread_sharpe",  # Alias for calculate_sharpe_ratio
    # Statistical tests
    "calculate_adf_test",
    "calculate_correlation",
    "calculate_johansen_test",
    # Mean reversion metrics
    "calculate_half_life",
    "calculate_hurst_exponent",
    "calculate_zscore",
    "calculate_zscore_stats",
    # Hedge ratio calculations
    "calculate_kalman_hedge_ratio",
    "calculate_ols_hedge_ratio",
    # Classification metrics
    "calculate_direction_metrics",
]

