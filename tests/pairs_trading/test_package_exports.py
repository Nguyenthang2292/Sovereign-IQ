from modules.pairs_trading import __all__ as exported_names


def test_package_exports_expected_symbols():
    expected = {
        # Main classes
        "PairsTradingAnalyzer",
        "PairMetricsComputer",
        "OpportunityScorer",
        # Display functions
        "display_performers",
        "display_pairs_opportunities",
        # Utility functions
        "select_top_unique_pairs",
        "ensure_symbols_in_candidate_pools",
        "select_pairs_for_symbols",
        "reverse_pairs",
        # Statistical tests
        "calculate_adf_test",
        "calculate_half_life",
        "calculate_johansen_test",
        # Risk metrics
        "calculate_spread_sharpe",
        "calculate_max_drawdown",
        "calculate_calmar_ratio",
        # Hedge ratio
        "calculate_ols_hedge_ratio",
        "calculate_kalman_hedge_ratio",
        # Z-score metrics
        "calculate_zscore_stats",
        "calculate_hurst_exponent",
        "calculate_direction_metrics",
    }

    assert set(exported_names) == expected

