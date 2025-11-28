"""
Pair metrics computer that orchestrates all quantitative metrics calculation.
"""

import pandas as pd
from typing import Dict, Optional, Union

from modules.pairs_trading.metrics.statistical_tests import (
    calculate_adf_test,
    calculate_half_life,
    calculate_johansen_test,
)
from modules.pairs_trading.metrics.risk_metrics import (
    calculate_spread_sharpe,
    calculate_max_drawdown,
    calculate_calmar_ratio,
)
from modules.pairs_trading.metrics.hedge_ratio import (
    calculate_ols_hedge_ratio,
    calculate_kalman_hedge_ratio,
)
from modules.pairs_trading.metrics.zscore_metrics import (
    calculate_zscore_stats,
    calculate_hurst_exponent,
    calculate_direction_metrics,
)

try:
    from modules.config import (
        PAIRS_TRADING_TIMEFRAME,
        PAIRS_TRADING_ADF_PVALUE_THRESHOLD,
        PAIRS_TRADING_PERIODS_PER_YEAR,
        PAIRS_TRADING_ZSCORE_LOOKBACK,
        PAIRS_TRADING_CLASSIFICATION_ZSCORE,
        PAIRS_TRADING_JOHANSEN_CONFIDENCE,
        PAIRS_TRADING_CORRELATION_MIN_POINTS,
        PAIRS_TRADING_OLS_FIT_INTERCEPT,
        PAIRS_TRADING_KALMAN_DELTA,
        PAIRS_TRADING_KALMAN_OBS_COV,
    )
except ImportError:
    PAIRS_TRADING_TIMEFRAME = "1h"
    PAIRS_TRADING_ADF_PVALUE_THRESHOLD = 0.05
    PAIRS_TRADING_PERIODS_PER_YEAR = 365 * 24
    PAIRS_TRADING_ZSCORE_LOOKBACK = 60
    PAIRS_TRADING_CLASSIFICATION_ZSCORE = 0.5
    PAIRS_TRADING_JOHANSEN_CONFIDENCE = 0.95
    PAIRS_TRADING_CORRELATION_MIN_POINTS = 50
    PAIRS_TRADING_OLS_FIT_INTERCEPT = True
    PAIRS_TRADING_KALMAN_DELTA = 1e-5
    PAIRS_TRADING_KALMAN_OBS_COV = 1.0


class PairMetricsComputer:
    """Computes comprehensive quantitative metrics for trading pairs."""

    def __init__(
        self,
        adf_pvalue_threshold: float = PAIRS_TRADING_ADF_PVALUE_THRESHOLD,
        periods_per_year: int = PAIRS_TRADING_PERIODS_PER_YEAR,
        zscore_lookback: int = PAIRS_TRADING_ZSCORE_LOOKBACK,
        classification_zscore: float = PAIRS_TRADING_CLASSIFICATION_ZSCORE,
        johansen_confidence: float = PAIRS_TRADING_JOHANSEN_CONFIDENCE,
        correlation_min_points: int = PAIRS_TRADING_CORRELATION_MIN_POINTS,
        ols_fit_intercept: bool = PAIRS_TRADING_OLS_FIT_INTERCEPT,
        kalman_delta: float = PAIRS_TRADING_KALMAN_DELTA,
        kalman_obs_cov: float = PAIRS_TRADING_KALMAN_OBS_COV,
    ):
        """
        Initialize PairMetricsComputer.
        
        Args:
            adf_pvalue_threshold: P-value threshold for ADF test
            periods_per_year: Number of periods per year
            zscore_lookback: Lookback period for z-score calculation
            classification_zscore: Z-score threshold for classification
            johansen_confidence: Confidence level for Johansen test
            correlation_min_points: Minimum data points required
        """
        self.adf_pvalue_threshold = adf_pvalue_threshold
        self.periods_per_year = periods_per_year
        self.zscore_lookback = zscore_lookback
        self.classification_zscore = classification_zscore
        self.johansen_confidence = johansen_confidence
        self.correlation_min_points = correlation_min_points
        self.ols_fit_intercept = ols_fit_intercept
        self.kalman_delta = kalman_delta
        self.kalman_obs_cov = kalman_obs_cov

    def compute_pair_metrics(
        self,
        price1: pd.Series,
        price2: pd.Series,
    ) -> Dict[str, Optional[Union[float, bool]]]:
        """
        Compute comprehensive quantitative metrics for a pair.
        
        Calculates metrics for both OLS and Kalman hedge ratio methods:
        - OLS-based metrics: half_life, zscore stats, hurst, sharpe, etc. (based on static hedge ratio)
        - Kalman-based metrics: kalman_half_life, kalman_* metrics (based on dynamic hedge ratio)
        
        Note: ADF test and Johansen test are calculated once as they test cointegration
        between price1 and price2, independent of the hedge ratio method.
        
        Args:
            price1: First price series
            price2: Second price series
            
        Returns:
            Dictionary with all computed metrics (both OLS and Kalman-based)
        """
        metrics: Dict[str, Optional[Union[float, bool]]] = {
            # OLS-based metrics
            "hedge_ratio": None,
            "adf_pvalue": None,
            "is_cointegrated": None,
            "half_life": None,
            "mean_zscore": None,
            "std_zscore": None,
            "skewness": None,
            "kurtosis": None,
            "current_zscore": None,
            "hurst_exponent": None,
            "spread_sharpe": None,
            "max_drawdown": None,
            "calmar_ratio": None,
            "classification_f1": None,
            "classification_precision": None,
            "classification_recall": None,
            "classification_accuracy": None,
            # Johansen test (independent of hedge ratio method)
            "johansen_trace_stat": None,
            "johansen_critical_value": None,
            "is_johansen_cointegrated": None,
            # Kalman hedge ratio
            "kalman_hedge_ratio": None,
            # Kalman-based metrics
            "kalman_half_life": None,
            "kalman_mean_zscore": None,
            "kalman_std_zscore": None,
            "kalman_skewness": None,
            "kalman_kurtosis": None,
            "kalman_current_zscore": None,
            "kalman_hurst_exponent": None,
            "kalman_spread_sharpe": None,
            "kalman_max_drawdown": None,
            "kalman_calmar_ratio": None,
            "kalman_classification_f1": None,
            "kalman_classification_precision": None,
            "kalman_classification_recall": None,
            "kalman_classification_accuracy": None,
        }

        # Calculate hedge ratio
        hedge_ratio = calculate_ols_hedge_ratio(
            price1,
            price2,
            fit_intercept=self.ols_fit_intercept,
        )
        if hedge_ratio is None:
            return metrics

        # Calculate spread
        spread_series = price1 - hedge_ratio * price2
        metrics["hedge_ratio"] = hedge_ratio

        # ADF test
        adf_result = calculate_adf_test(spread_series, self.correlation_min_points)
        if adf_result:
            metrics["adf_pvalue"] = adf_result.get("adf_pvalue")
            metrics["is_cointegrated"] = (
                adf_result.get("adf_pvalue") is not None
                and adf_result["adf_pvalue"] < self.adf_pvalue_threshold
            )

        # Half-life
        half_life = calculate_half_life(spread_series)
        if half_life is not None:
            metrics["half_life"] = half_life

        # Z-score stats
        zscore_stats = calculate_zscore_stats(spread_series, self.zscore_lookback)
        metrics.update(zscore_stats)

        # Hurst exponent
        metrics["hurst_exponent"] = calculate_hurst_exponent(
            spread_series, self.zscore_lookback
        )

        # Risk metrics
        metrics["spread_sharpe"] = calculate_spread_sharpe(
            spread_series, self.periods_per_year
        )
        metrics["max_drawdown"] = calculate_max_drawdown(spread_series)
        metrics["calmar_ratio"] = calculate_calmar_ratio(
            spread_series, self.periods_per_year
        )

        # Johansen test
        johansen = calculate_johansen_test(
            price1,
            price2,
            self.correlation_min_points,
            self.johansen_confidence,
        )
        if johansen:
            metrics.update(johansen)

            # Combine ADF and Johansen cointegration decisions.
            # Johansen is generally stronger, so we treat cointegration as True
            # if either test signals cointegration.
            adf_cointegrated = metrics.get("is_cointegrated")
            johansen_cointegrated = johansen.get("is_johansen_cointegrated")

            if johansen_cointegrated is not None:
                if adf_cointegrated is None:
                    metrics["is_cointegrated"] = bool(johansen_cointegrated)
                else:
                    metrics["is_cointegrated"] = bool(
                        adf_cointegrated or johansen_cointegrated
                    )
        
        # Kalman hedge ratio and Kalman-based metrics
        kalman_beta = calculate_kalman_hedge_ratio(
            price1,
            price2,
            delta=self.kalman_delta,
            observation_covariance=self.kalman_obs_cov,
        )
        if kalman_beta is not None:
            metrics["kalman_hedge_ratio"] = kalman_beta
            
            # Calculate Kalman spread
            kalman_spread_series = price1 - kalman_beta * price2
            
            # Kalman half-life
            kalman_half_life = calculate_half_life(kalman_spread_series)
            if kalman_half_life is not None:
                metrics["kalman_half_life"] = kalman_half_life
            
            # Kalman z-score stats
            kalman_zscore_stats = calculate_zscore_stats(
                kalman_spread_series, self.zscore_lookback
            )
            if kalman_zscore_stats:
                metrics["kalman_mean_zscore"] = kalman_zscore_stats.get("mean_zscore")
                metrics["kalman_std_zscore"] = kalman_zscore_stats.get("std_zscore")
                metrics["kalman_skewness"] = kalman_zscore_stats.get("skewness")
                metrics["kalman_kurtosis"] = kalman_zscore_stats.get("kurtosis")
                metrics["kalman_current_zscore"] = kalman_zscore_stats.get("current_zscore")
            
            # Kalman Hurst exponent
            metrics["kalman_hurst_exponent"] = calculate_hurst_exponent(
                kalman_spread_series, self.zscore_lookback
            )
            
            # Kalman risk metrics
            metrics["kalman_spread_sharpe"] = calculate_spread_sharpe(
                kalman_spread_series, self.periods_per_year
            )
            metrics["kalman_max_drawdown"] = calculate_max_drawdown(kalman_spread_series)
            metrics["kalman_calmar_ratio"] = calculate_calmar_ratio(
                kalman_spread_series, self.periods_per_year
            )
            
            # Kalman direction metrics
            kalman_direction_metrics = calculate_direction_metrics(
                kalman_spread_series, self.zscore_lookback, self.classification_zscore
            )
            if kalman_direction_metrics:
                metrics["kalman_classification_f1"] = kalman_direction_metrics.get("classification_f1")
                metrics["kalman_classification_precision"] = kalman_direction_metrics.get("classification_precision")
                metrics["kalman_classification_recall"] = kalman_direction_metrics.get("classification_recall")
                metrics["kalman_classification_accuracy"] = kalman_direction_metrics.get("classification_accuracy")

        # Direction metrics (OLS-based)
        direction_metrics = calculate_direction_metrics(
            spread_series, self.zscore_lookback, self.classification_zscore
        )
        metrics.update(direction_metrics)

        return metrics

