from modules.pairs_trading import __all__ as exported_names


def test_package_exports_expected_symbols():
    expected = {
        "PairsTradingAnalyzer",
        "PairMetricsComputer",
        "OpportunityScorer",
        "PerformanceAnalyzer",
        "display_performers",
        "display_pairs_opportunities",
        "select_top_unique_pairs",
        "select_pairs_for_symbols",
        "ensure_symbols_in_candidate_pools",
        "calculate_adf_test",
        "calculate_half_life",
        "calculate_johansen_test",
        "calculate_sharpe_ratio",
        "calculate_max_drawdown",
        "calculate_calmar_ratio",
        "calculate_ols_hedge_ratio",
        "calculate_kalman_hedge_ratio",
        "calculate_zscore_stats",
        "calculate_hurst_exponent",
        "calculate_direction_metrics",
        "parse_args",
        "prompt_interactive_mode",
        "prompt_weight_preset_selection",
        "prompt_kalman_preset_selection",
        "prompt_opportunity_preset_selection",
        "prompt_target_pairs",
        "prompt_candidate_depth",
        "parse_weights",
        "parse_symbols",
    }

    assert set(exported_names) == expected
