"""
Pairs trading analysis component.

This package provides comprehensive tools for pairs trading analysis including:
- Core analysis components (PairsTradingAnalyzer, PairMetricsComputer, OpportunityScorer)
- Performance analysis across multiple timeframes
- Statistical and quantitative metrics (cointegration, correlation, Hurst exponent, etc.)
- Risk metrics (Sharpe ratio, max drawdown, Calmar ratio)
- Hedge ratio calculations (OLS and Kalman filter)
- Utility functions for pair selection and manipulation
- CLI tools for interactive analysis
"""

# Core components
from modules.pairs_trading.core import (
    PairsTradingAnalyzer,
    PairMetricsComputer,
    OpportunityScorer,
)

# Analysis components
from modules.pairs_trading.analysis import (
    PerformanceAnalyzer,
)

# Metrics (imported from common.quantitative_metrics)
from modules.common.quantitative_metrics import (
    calculate_adf_test,
    calculate_half_life,
    calculate_johansen_test,
    calculate_sharpe_ratio,
    calculate_max_drawdown,
    calculate_calmar_ratio,
    calculate_ols_hedge_ratio,
    calculate_kalman_hedge_ratio,
    calculate_zscore_stats,
    calculate_hurst_exponent,
    calculate_direction_metrics,
)

# Utility functions
from modules.pairs_trading.utils import (
    select_top_unique_pairs,
    select_pairs_for_symbols,
    ensure_symbols_in_candidate_pools,
)

# CLI components
from modules.pairs_trading.cli import (
    # Display formatters
    display_performers,
    display_pairs_opportunities,
    # Argument parsing
    parse_args,
    # Interactive prompts
    prompt_interactive_mode,
    prompt_weight_preset_selection,
    prompt_kalman_preset_selection,
    prompt_opportunity_preset_selection,
    prompt_target_pairs,
    prompt_candidate_depth,
    # Input parsers
    parse_weights,
    parse_symbols,
)

__all__ = [
    # Core classes
    'PairsTradingAnalyzer',
    'PairMetricsComputer',
    'OpportunityScorer',
    'PerformanceAnalyzer',
    # Statistical tests
    'calculate_adf_test',
    'calculate_half_life',
    'calculate_johansen_test',
    # Risk metrics
    'calculate_sharpe_ratio',
    'calculate_max_drawdown',
    'calculate_calmar_ratio',
    # Hedge ratio
    'calculate_ols_hedge_ratio',
    'calculate_kalman_hedge_ratio',
    # Z-score metrics
    'calculate_zscore_stats',
    'calculate_hurst_exponent',
    'calculate_direction_metrics',
    # Utility functions
    'select_top_unique_pairs',
    'select_pairs_for_symbols',
    'ensure_symbols_in_candidate_pools',
    # Display functions
    'display_performers',
    'display_pairs_opportunities',
    # CLI functions
    'parse_args',
    'prompt_interactive_mode',
    'prompt_weight_preset_selection',
    'prompt_kalman_preset_selection',
    'prompt_opportunity_preset_selection',
    'prompt_target_pairs',
    'prompt_candidate_depth',
    'parse_weights',
    'parse_symbols',
]
