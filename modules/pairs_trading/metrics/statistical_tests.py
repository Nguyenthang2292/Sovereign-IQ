"""
Statistical tests for pairs trading analysis.
"""

import pandas as pd
import numpy as np
from typing import Dict, Optional, Any, Union

try:
    from statsmodels.tsa.stattools import adfuller  # type: ignore
except ImportError:
    adfuller = None

try:
    from sklearn.linear_model import LinearRegression
except ImportError:
    LinearRegression = None

try:
    from statsmodels.tsa.vector_ar.vecm import coint_johansen  # type: ignore
except ImportError:
    coint_johansen = None

try:
    from modules.config import (
        PAIRS_TRADING_CORRELATION_MIN_POINTS,
        PAIRS_TRADING_JOHANSEN_CONFIDENCE,
        PAIRS_TRADING_JOHANSEN_DET_ORDER,
        PAIRS_TRADING_JOHANSEN_K_AR_DIFF,
        PAIRS_TRADING_ADF_MAXLAG,
        PAIRS_TRADING_MIN_HALF_LIFE_POINTS,
        PAIRS_TRADING_MAX_HALF_LIFE,
    )
except ImportError:
    PAIRS_TRADING_CORRELATION_MIN_POINTS = 50
    PAIRS_TRADING_JOHANSEN_CONFIDENCE = 0.95
    PAIRS_TRADING_JOHANSEN_DET_ORDER = 0
    PAIRS_TRADING_JOHANSEN_K_AR_DIFF = 1
    PAIRS_TRADING_ADF_MAXLAG = 1
    PAIRS_TRADING_MIN_HALF_LIFE_POINTS = 10
    PAIRS_TRADING_MAX_HALF_LIFE = 50


def calculate_adf_test(
    spread: pd.Series, min_points: int = PAIRS_TRADING_CORRELATION_MIN_POINTS
) -> Optional[Dict[str, Union[float, Dict[str, float]]]]:
    """
    Run Augmented Dickey-Fuller (ADF) test to check if spread is stationary (cointegrated).
    
    Tests whether spread has a unit root (non-stationary). Stationary spread = cointegrated assets
    suitable for pairs trading (will revert to mean). Non-stationary = not suitable.
    
    **Interpretation**:
    - **p-value < 0.05**: Reject H0 → Stationary (cointegrated) → Good for pairs trading
    - **p-value >= 0.05**: Fail to reject H0 → Non-stationary → Not suitable
    - More negative adf_statistic (vs critical values) = stronger evidence of stationarity
    
    Args:
        spread: Spread series (price1 - price2 * hedge_ratio). Missing values auto-removed.
        min_points: Minimum data points required. Default: 50.
        
    Returns:
        Dict with test results, or None if test fails:
        {
            'adf_statistic': float,  # Test statistic (more negative = more stationary)
            'adf_pvalue': float,     # p-value (p < 0.05 = stationary)
            'critical_values': dict  # {'1%': float, '5%': float, '10%': float}
        }
        
    Example:
        >>> spread = pd.Series([0.1, -0.05, 0.15, -0.1, 0.08, ...])  # At least 50 points
        >>> result = calculate_adf_test(spread)
        >>> if result and result['adf_pvalue'] < 0.05:
        ...     print("✓ Spread is stationary - suitable for pairs trading")
    
    Note:
        - p-value < 0.05 (preferably < 0.01) indicates cointegrated pairs
        - Uses AIC lag selection with maxlag=PAIRS_TRADING_ADF_MAXLAG
    """
    if adfuller is None or spread is None:
        return None

    spread = spread.dropna()
    if len(spread) < min_points:
        return None

    try:
        adf_result = adfuller(spread, maxlag=PAIRS_TRADING_ADF_MAXLAG, autolag="AIC")
        return {
            "adf_statistic": float(adf_result[0]),
            "adf_pvalue": float(adf_result[1]),
            "critical_values": adf_result[4],
        }
    except Exception:
        return None


def calculate_half_life(spread: pd.Series) -> Optional[float]:
    """
    Calculate half-life of mean reversion for the spread.
    
    Half-life is the expected number of periods for spread deviation to reduce to 50%.
    Shorter half-life = faster mean reversion = better for pairs trading.
    
    **Calculation**: Uses OLS regression Δy(t) = θ * y(t-1) + ε. Formula: half_life = -ln(2) / θ
    where θ (mean reversion coefficient) must be negative for mean reversion to occur.
    
    **Interpretation**:
    - < 10 periods: Very fast (excellent)
    - 10-30 periods: Fast (good)
    - 30-50 periods: Moderate (acceptable)
    - > 50 periods: Slow (not suitable)
    
    Args:
        spread: Spread series (price1 - price2 * hedge_ratio)
        
    Returns:
        Half-life in periods, or None if calculation fails (non-stationary, insufficient data,
        or invalid result exceeding PAIRS_TRADING_MAX_HALF_LIFE).
        
    Example:
        >>> spread = pd.Series([0.1, 0.08, 0.05, 0.02, -0.01, ...])
        >>> half_life = calculate_half_life(spread)
        >>> # Returns number of periods for deviation to halve, e.g., 9.5 periods
    """
    if LinearRegression is None or spread is None:
        return None

    spread_lag = spread.shift(1)
    spread_diff = spread - spread_lag

    valid = spread_lag.notna() & spread_diff.notna()
    if valid.sum() < PAIRS_TRADING_MIN_HALF_LIFE_POINTS:
        return None

    try:
        X = spread_lag[valid].values.reshape(-1, 1)
        y = spread_diff[valid].values
        model = LinearRegression()
        model.fit(X, y)
        theta = model.coef_[0]
        # theta must be negative for mean reversion to occur
        # A negative theta indicates that when spread is above mean, it will decrease back,
        # and when below mean, it will increase back (mean-reverting behavior)
        # If theta >= 0, the spread exhibits random walk or trending behavior (non-stationary)
        if theta >= 0:
            return None
        
        # Half-life formula: half_life = -ln(2) / theta
        # This formula comes from solving the mean-reverting OLS model:
        # Δy(t) = θ * y(t-1) + ε, where θ < 0
        # After half-life periods, the deviation reduces to 50% of original
        # ln(2) ≈ 0.693 is used because we want to find when deviation becomes 1/2
        half_life = -np.log(2) / theta
        
        # Validate half-life: must be positive, finite, and not NaN
        # Also check upper bound: very large half-life indicates slow/non mean-reverting behavior
        if (half_life < 0 or 
            np.isinf(half_life) or 
            np.isnan(half_life) or 
            half_life > PAIRS_TRADING_MAX_HALF_LIFE):
            return None
        
        return float(half_life)
    except Exception:
        return None


def calculate_johansen_test(
    price1: pd.Series,
    price2: pd.Series,
    min_points: int = PAIRS_TRADING_CORRELATION_MIN_POINTS,
    confidence: float = PAIRS_TRADING_JOHANSEN_CONFIDENCE,
    det_order: int = PAIRS_TRADING_JOHANSEN_DET_ORDER,
    k_ar_diff: int = PAIRS_TRADING_JOHANSEN_K_AR_DIFF,
) -> Optional[Dict[str, Union[float, bool]]]:
    """
    Run Johansen cointegration test to determine if two price series are cointegrated.
    
    The Johansen test is a multivariate cointegration test that tests for the presence
    of cointegrating relationships between two or more time series. It is more robust
    than the Engle-Granger test and can handle multiple cointegrating relationships.
    
    **Test Method**:
    The test uses Vector Error Correction Model (VECM) framework and tests the null
    hypothesis of no cointegration against the alternative of cointegration.
    
    **Parameters**:
    - **det_order** (deterministic order): Controls the deterministic terms in the model
      - 0: No constant or trend (mean-reverting around zero)
      - 1: Constant term only (mean-reverting around a constant level)
      - -1: No constant but includes trend
      - Default: 0 (most common for pairs trading where spread mean-reverts to zero)
    
    - **k_ar_diff** (lag order): Number of lags in the Vector Autoregression (VAR) model
      - Controls how many previous periods are used to model the relationship
      - Higher values = more complex model, may capture longer-term relationships
      - Lower values = simpler model, more sensitive to short-term dynamics
      - Default: 1 (one lag, commonly used for pairs trading)
      - Typical range: 1-4 for hourly data, may need more for lower frequency
    
    **Interpretation**:
    - **trace_stat > critical_value**: Reject null hypothesis → Series are cointegrated
    - **is_johansen_cointegrated = True**: Assets have a long-term equilibrium relationship
    - Cointegrated pairs are suitable for pairs trading as they tend to revert to mean
    
    Args:
        price1: First price series (should be aligned with price2)
        price2: Second price series (should be aligned with price1)
        min_points: Minimum number of data points required for reliable test results.
            Default: PAIRS_TRADING_CORRELATION_MIN_POINTS (50)
        confidence: Confidence level for the test. Options: 0.9 (90%), 0.95 (95%), 0.99 (99%).
            Higher confidence = stricter criteria. Default: 0.95 (95%)
        det_order: Deterministic order in the VECM model.
            - 0: No constant/trend (spread mean-reverts around zero)
            - 1: Constant term (spread mean-reverts around a constant level)
            - -1: No constant but includes trend
            Default: PAIRS_TRADING_JOHANSEN_DET_ORDER (0)
        k_ar_diff: Lag order for the VAR model (number of lags to include).
            Typical values: 1-4. Higher = more complex, lower = simpler.
            Default: PAIRS_TRADING_JOHANSEN_K_AR_DIFF (1)
        
    Returns:
        Dictionary containing Johansen test results, or None if test fails or data is insufficient.
        
        Return value structure:
        {
            'johansen_trace_stat': float,        # Trace test statistic
            'johansen_critical_value': float,    # Critical value at specified confidence level
            'is_johansen_cointegrated': bool     # True if trace_stat > critical_value (cointegrated)
        }
        
        **Key Fields**:
        - **johansen_trace_stat**: The trace statistic. Higher values indicate stronger
          evidence of cointegration. Compare with critical_value.
        - **johansen_critical_value**: The critical value at the specified confidence level.
          If trace_stat > critical_value, we reject null hypothesis (cointegrated).
        - **is_johansen_cointegrated**: Boolean indicating if the series are cointegrated
          based on trace_stat > critical_value comparison.
        
        Returns None if:
        - statsmodels is not installed
        - Insufficient data points (< min_points)
        - Test calculation fails
        
    Example:
        >>> import pandas as pd
        >>> from modules.pairs_trading import calculate_johansen_test
        >>>
        >>> # Create two cointegrated price series
        >>> price1 = pd.Series([100, 102, 101, 103, 102, ...])
        >>> price2 = pd.Series([50, 51, 50.5, 51.5, 51, ...])  # Roughly 2x price1
        >>>
        >>> # Run Johansen test with default parameters
        >>> result = calculate_johansen_test(price1, price2)
        >>>
        >>> if result:
        ...     print(f"Trace Statistic: {result['johansen_trace_stat']:.4f}")
        ...     print(f"Critical Value (95%): {result['johansen_critical_value']:.4f}")
        ...     print(f"Cointegrated: {result['is_johansen_cointegrated']}")
        ...
        ...     if result['is_johansen_cointegrated']:
        ...         print("✓ Assets are cointegrated - suitable for pairs trading")
        ...     else:
        ...         print("✗ Assets are not cointegrated - not suitable")
        >>>
        >>> # Customize parameters for different model specification
        >>> result_custom = calculate_johansen_test(
        ...     price1, price2,
        ...     det_order=1,  # Include constant term
        ...     k_ar_diff=2,  # Use 2 lags
        ...     confidence=0.99  # 99% confidence level
        ... )
    
    Note:
        - For pairs trading, typically use det_order=0 (spread mean-reverts to zero)
        - k_ar_diff=1 is usually sufficient for hourly/daily data
        - Higher confidence (0.99) = stricter criteria, fewer pairs will pass
        - Johansen test is more robust than Engle-Granger for multiple series
        - Requires sufficient data points (at least 50, preferably 100+)
    """
    if coint_johansen is None:
        return None

    data = np.column_stack([price1.values, price2.values])
    if data.shape[0] < min_points:
        return None

    try:
        confidence_map = {0.9: 0, 0.95: 1, 0.99: 2}
        confidence_key = round(confidence, 2)
        critical_idx = confidence_map.get(confidence_key, 1)
        
        # Johansen cointegration test with customizable parameters
        # det_order: deterministic order (0=no constant/trend, 1=constant, -1=no constant with trend)
        # k_ar_diff: lag order for VAR model (number of lags to include in the model)
        result = coint_johansen(data, det_order=det_order, k_ar_diff=k_ar_diff)
        trace_stat = result.lr1[0]
        critical_value = result.cvt[0, critical_idx]
        is_cointegrated = trace_stat > critical_value
        return {
            "johansen_trace_stat": float(trace_stat),
            "johansen_critical_value": float(critical_value),
            "is_johansen_cointegrated": is_cointegrated,
        }
    except Exception:
        return None