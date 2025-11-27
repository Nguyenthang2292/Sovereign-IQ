"""
Opportunity scoring logic for pairs trading.
"""

import numpy as np
from typing import Dict, Optional

try:
    from modules.config import (
        PAIRS_TRADING_ADF_PVALUE_THRESHOLD,
        PAIRS_TRADING_MAX_HALF_LIFE,
        PAIRS_TRADING_HURST_THRESHOLD,
        PAIRS_TRADING_MIN_SPREAD_SHARPE,
        PAIRS_TRADING_MAX_DRAWDOWN,
        PAIRS_TRADING_MIN_CALMAR,
        PAIRS_TRADING_OPPORTUNITY_PRESETS,
    )
except ImportError:
    PAIRS_TRADING_ADF_PVALUE_THRESHOLD = 0.05
    PAIRS_TRADING_MAX_HALF_LIFE = 50
    PAIRS_TRADING_HURST_THRESHOLD = 0.5
    PAIRS_TRADING_MIN_SPREAD_SHARPE = 1.0
    PAIRS_TRADING_MAX_DRAWDOWN = 0.3
    PAIRS_TRADING_MIN_CALMAR = 1.0
    PAIRS_TRADING_OPPORTUNITY_PRESETS = {
        "balanced": {
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
        }
    }


class OpportunityScorer:
    """Calculates opportunity scores for pairs trading opportunities."""

    def __init__(
        self,
        min_correlation: float = 0.3,
        max_correlation: float = 0.9,
        adf_pvalue_threshold: float = PAIRS_TRADING_ADF_PVALUE_THRESHOLD,
        max_half_life: float = PAIRS_TRADING_MAX_HALF_LIFE,
        hurst_threshold: float = PAIRS_TRADING_HURST_THRESHOLD,
        min_spread_sharpe: float = PAIRS_TRADING_MIN_SPREAD_SHARPE,
        max_drawdown_threshold: float = PAIRS_TRADING_MAX_DRAWDOWN,
        min_calmar: float = PAIRS_TRADING_MIN_CALMAR,
        scoring_multipliers: Optional[Dict[str, float]] = None,
    ):
        """
        Initialize OpportunityScorer.
        
        Args:
            min_correlation: Minimum correlation threshold
            max_correlation: Maximum correlation threshold
            adf_pvalue_threshold: P-value threshold for ADF test
            max_half_life: Maximum acceptable half-life
            hurst_threshold: Hurst exponent threshold
            min_spread_sharpe: Minimum Sharpe ratio threshold
            max_drawdown_threshold: Maximum drawdown threshold
            min_calmar: Minimum Calmar ratio threshold
        """
        self.min_correlation = min_correlation
        self.max_correlation = max_correlation
        self.adf_pvalue_threshold = adf_pvalue_threshold
        self.max_half_life = max_half_life
        self.hurst_threshold = hurst_threshold
        self.min_spread_sharpe = min_spread_sharpe
        self.max_drawdown_threshold = max_drawdown_threshold
        self.min_calmar = min_calmar
        default_profile = PAIRS_TRADING_OPPORTUNITY_PRESETS.get("balanced", {})
        self.scoring = {**default_profile, **(scoring_multipliers or {})}

    def calculate_opportunity_score(
        self,
        spread: float,
        correlation: Optional[float] = None,
        quant_metrics: Optional[Dict[str, Optional[float]]] = None,
    ) -> float:
        """
        Calculate opportunity score for a trading pair.
        
        Args:
            spread: Spread between long and short symbols
            correlation: Correlation coefficient (optional)
            quant_metrics: Quantitative metrics dictionary (optional)
            
        Returns:
            Opportunity score (higher is better)
        """
        if quant_metrics is None:
            quant_metrics = {}

        # Start with base spread
        # Spread is the performance gap between the short and long legs:
        # spread = short_score - long_score (absolute). Large spread implies
        # a stronger mean-reversion opportunity, so it becomes the base score.
        opportunity_score = spread

        # Adjust based on correlation
        sc = self.scoring

        if correlation is not None:
            abs_corr = abs(correlation)
            if self.min_correlation <= abs_corr <= self.max_correlation:
                # Bonus for good correlation range
                opportunity_score *= sc.get("corr_good_bonus", 1.2)
            elif abs_corr < self.min_correlation:
                # Penalty for low correlation (may not move together)
                opportunity_score *= sc.get("corr_low_penalty", 0.8)
            elif abs_corr > self.max_correlation:
                # Penalty for over-correlation (may move together too much)
                opportunity_score *= sc.get("corr_high_penalty", 0.9)

        # Boost score if cointegrated and half-life within acceptable range
        if quant_metrics.get("is_cointegrated"):
            opportunity_score *= sc.get("cointegration_bonus", 1.15)
        elif quant_metrics.get("adf_pvalue") is not None and quant_metrics["adf_pvalue"] < (
            self.adf_pvalue_threshold * 1.5
        ):
            opportunity_score *= sc.get("weak_cointegration_bonus", 1.05)

        if (
            quant_metrics.get("half_life") is not None
            and quant_metrics["half_life"] <= self.max_half_life
        ):
            opportunity_score *= sc.get("half_life_bonus", 1.1)

        current_z = quant_metrics.get("current_zscore")
        if current_z is not None and not np.isnan(current_z):
            z_div = sc.get("zscore_divisor", 5.0)
            z_cap = sc.get("zscore_cap", 0.2)
            opportunity_score *= 1 + min(abs(current_z) / max(z_div, 1e-6), z_cap)

        hurst = quant_metrics.get("hurst_exponent")
        if hurst is not None:
            if hurst <= self.hurst_threshold:
                opportunity_score *= sc.get("hurst_good_bonus", 1.08)
            elif hurst < sc.get("hurst_ok_threshold", 0.6):
                opportunity_score *= sc.get("hurst_ok_bonus", 1.02)

        sharpe = quant_metrics.get("spread_sharpe")
        if sharpe is not None:
            if sharpe >= self.min_spread_sharpe:
                opportunity_score *= sc.get("sharpe_good_bonus", 1.08)
            elif sharpe >= self.min_spread_sharpe / 2:
                opportunity_score *= sc.get("sharpe_ok_bonus", 1.03)

        max_dd = quant_metrics.get("max_drawdown")
        if max_dd is not None and abs(max_dd) <= self.max_drawdown_threshold:
            opportunity_score *= sc.get("maxdd_bonus", 1.05)

        calmar = quant_metrics.get("calmar_ratio")
        if calmar is not None and calmar >= self.min_calmar:
            opportunity_score *= sc.get("calmar_bonus", 1.05)

        if quant_metrics.get("is_johansen_cointegrated"):
            opportunity_score *= sc.get("johansen_bonus", 1.08)

        f1_metric = quant_metrics.get("classification_f1")
        if f1_metric is not None:
            if f1_metric >= 0.7:
                opportunity_score *= sc.get("f1_high_bonus", 1.05)
            elif f1_metric >= 0.6:
                opportunity_score *= sc.get("f1_mid_bonus", 1.02)

        return float(opportunity_score)

    def calculate_quantitative_score(
        self, quant_metrics: Optional[Dict[str, Optional[float]]] = None
    ) -> float:
        """
        Calculate combined quantitative score (0-100) based on all metrics.
        
        Weights:
        - Cointegration: 30%
        - Half-life: 20%
        - Hurst: 15%
        - Sharpe: 15%
        - F1-score: 10%
        - Max DD: 10%
        - Calmar ratio: 5% (optional bonus, overall score capped at 100)
        
        Args:
            quant_metrics: Quantitative metrics dictionary
            
        Returns:
            Quantitative score from 0-100 (higher is better)
        """
        if quant_metrics is None:
            quant_metrics = {}
        
        score = 0.0
        
        # Cointegration (30%)
        if quant_metrics.get("is_cointegrated"):
            score += 30.0
        elif quant_metrics.get("is_johansen_cointegrated"):
            score += 30.0
        else:
            adf_pvalue = quant_metrics.get("adf_pvalue")
            if adf_pvalue is not None and adf_pvalue < 0.1:
                score += 15.0
        
        # Half-life (20%)
        half_life = quant_metrics.get("half_life")
        if half_life is not None:
            if half_life < 20:
                score += 20.0
            elif half_life < 50:
                score += 10.0
        
        # Hurst (15%)
        hurst = quant_metrics.get("hurst_exponent")
        if hurst is not None:
            if hurst < 0.4:
                score += 15.0
            elif hurst < 0.5:
                score += 8.0
        
        # Sharpe (15%)
        sharpe = quant_metrics.get("spread_sharpe")
        if sharpe is not None:
            if sharpe > 2.0:
                score += 15.0
            elif sharpe > 1.0:
                score += 8.0
        
        # F1-score (10%)
        f1 = quant_metrics.get("classification_f1")
        if f1 is not None:
            if f1 > 0.7:
                score += 10.0
            elif f1 > 0.6:
                score += 5.0
        
        # Max DD (10%)
        max_dd = quant_metrics.get("max_drawdown")
        if max_dd is not None:
            abs_max_dd = abs(max_dd)
            if abs_max_dd < 0.2:
                score += 10.0
            elif abs_max_dd < 0.3:
                score += 5.0
        
        # Calmar ratio (5%)
        calmar_ratio = quant_metrics.get("calmar_ratio")
        if calmar_ratio is not None:
            if calmar_ratio >= 1.0:
                score += 5.0
            elif calmar_ratio >= 0.5:
                score += 2.5

        return min(100.0, score)

