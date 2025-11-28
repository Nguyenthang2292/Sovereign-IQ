import math

from modules.pairs_trading.core.opportunity_scorer import OpportunityScorer


def test_calculate_opportunity_score_applies_all_adjustments():
    scorer = OpportunityScorer(
        min_correlation=0.3,
        max_correlation=0.9,
        adf_pvalue_threshold=0.05,
        max_half_life=50,
        hurst_threshold=0.5,
        min_spread_sharpe=1.0,
        max_drawdown_threshold=0.3,
        min_calmar=1.0,
    )

    quant_metrics = {
        "is_cointegrated": True,
        "half_life": 25,
        "current_zscore": 1.5,
        "hurst_exponent": 0.45,
        "spread_sharpe": 1.2,
        "max_drawdown": -0.2,
        "calmar_ratio": 1.3,
        "is_johansen_cointegrated": True,
        "classification_f1": 0.75,
    }

    score = scorer.calculate_opportunity_score(0.1, correlation=0.6, quant_metrics=quant_metrics)

    assert score > 0.1 * 1.2  # correlation boost
    assert score > 0.1  # definitely boosted overall


def test_calculate_opportunity_score_penalizes_low_correlation():
    scorer = OpportunityScorer(min_correlation=0.4, max_correlation=0.9)
    score = scorer.calculate_opportunity_score(0.2, correlation=0.1, quant_metrics={})
    assert math.isclose(score, 0.2 * 0.8)


def test_calculate_quantitative_score_combines_metrics():
    scorer = OpportunityScorer()

    metrics = {
        "is_cointegrated": True,
        "half_life": 10,
        "hurst_exponent": 0.35,
        "spread_sharpe": 2.5,
        "classification_f1": 0.8,
        "max_drawdown": -0.1,
    }

    result = scorer.calculate_quantitative_score(metrics)

    # Expect full credit for each bucket
    assert math.isclose(result, 30 + 20 + 15 + 15 + 10 + 10)


def test_momentum_correlation_and_adx_filters():
    scorer = OpportunityScorer(strategy="momentum")

    strong_metrics = {"long_adx": 28.0, "short_adx": 30.0}
    weak_metrics = {"long_adx": 10.0, "short_adx": 12.0}

    strong_score = scorer.calculate_opportunity_score(
        spread=0.2,
        correlation=-0.2,
        quant_metrics=strong_metrics,
    )
    weak_score = scorer.calculate_opportunity_score(
        spread=0.2,
        correlation=0.8,
        quant_metrics=weak_metrics,
    )

    assert strong_score > 0.2  # negative correlation + ADX bonuses
    assert weak_score == 0.0  # fails ADX filter


def test_momentum_high_correlation_penalty():
    scorer = OpportunityScorer(strategy="momentum")
    metrics = {"long_adx": 26.0, "short_adx": 27.0}

    low_corr_score = scorer.calculate_opportunity_score(
        spread=0.2,
        correlation=0.1,
        quant_metrics=metrics,
    )
    high_corr_score = scorer.calculate_opportunity_score(
        spread=0.2,
        correlation=0.95,
        quant_metrics=metrics,
    )

    assert high_corr_score < low_corr_score


def test_momentum_cointegration_penalty():
    scorer = OpportunityScorer(strategy="momentum")
    base_metrics = {"long_adx": 26.0, "short_adx": 27.0}
    coint_metrics = {**base_metrics, "is_cointegrated": True}

    base_score = scorer.calculate_opportunity_score(0.3, 0.2, base_metrics)
    penalized_score = scorer.calculate_opportunity_score(0.3, 0.2, coint_metrics)

    assert penalized_score < base_score


def test_momentum_risk_penalties_only_for_bad_cases():
    scorer = OpportunityScorer(strategy="momentum")
    good_metrics = {"long_adx": 26.0, "short_adx": 27.0, "spread_sharpe": 1.5}
    bad_metrics = {**good_metrics, "spread_sharpe": 0.1, "kalman_spread_sharpe": 0.1}

    good_score = scorer.calculate_opportunity_score(0.25, 0.2, good_metrics)
    bad_score = scorer.calculate_opportunity_score(0.25, 0.2, bad_metrics)

    assert bad_score < good_score


def test_quantitative_score_adds_momentum_adx_bonus():
    scorer = OpportunityScorer(strategy="momentum")
    strong_trend = {"long_adx": 30.0, "short_adx": 28.0}
    moderate_trend = {"long_adx": 20.0, "short_adx": 19.0}

    strong_score = scorer.calculate_quantitative_score(strong_trend)
    moderate_score = scorer.calculate_quantitative_score(moderate_trend)

    assert strong_score >= 10.0
    assert moderate_score >= 5.0
    assert strong_score > moderate_score