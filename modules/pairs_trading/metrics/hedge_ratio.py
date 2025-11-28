"""
Hedge ratio calculations for pairs trading.
"""

import pandas as pd
import numpy as np
from typing import Optional

try:
    from modules.config import (
        PAIRS_TRADING_OLS_FIT_INTERCEPT,
        PAIRS_TRADING_KALMAN_DELTA,
        PAIRS_TRADING_KALMAN_OBS_COV,
    )
except ImportError:
    PAIRS_TRADING_OLS_FIT_INTERCEPT = True
    PAIRS_TRADING_KALMAN_DELTA = 1e-5
    PAIRS_TRADING_KALMAN_OBS_COV = 1.0

try:
    from sklearn.linear_model import LinearRegression
except ImportError:
    LinearRegression = None

try:
    from pykalman import KalmanFilter  # type: ignore
except ImportError:
    KalmanFilter = None


def calculate_ols_hedge_ratio(
    price1: pd.Series,
    price2: pd.Series,
    fit_intercept: bool = PAIRS_TRADING_OLS_FIT_INTERCEPT,
) -> Optional[float]:
    """
    Calculate OLS (Ordinary Least Squares) hedge ratio using linear regression.
    
    The hedge ratio (β) determines how many units of asset 2 (price2) to short
    for each unit of asset 1 (price1) to long. It minimizes spread variance:
    spread = price1 - β * price2.
    
    **OLS (Static) vs Kalman (Dynamic)**:
    - OLS: Single constant ratio from all historical data. Best for stable relationships.
    - Kalman: Time-varying ratio that adapts to market changes. Use for evolving relationships.
    
    **Formula**: β = argmin Σ(price1 - β * price2)²
    Regression model: price1 = β * price2 + α (if fit_intercept=True) or price1 = β * price2 (if False)
    
    Args:
        price1: First price series (dependent variable, asset to long)
        price2: Second price series (independent variable, asset to short)
        fit_intercept: Include intercept term. True = price1 = β*price2 + α, False = price1 = β*price2.
            Default: PAIRS_TRADING_OLS_FIT_INTERCEPT (True)
        
    Returns:
        Hedge ratio (β) as float, or None if calculation fails (insufficient data, sklearn not installed).
        
    Example:
        >>> price1 = pd.Series([100, 102, 101, 103, 105])
        >>> price2 = pd.Series([50, 51, 50.5, 51.5, 52.5])
        >>> hedge_ratio = calculate_ols_hedge_ratio(price1, price2)
        >>> # hedge_ratio = 1.92 means: short 1.92 units of price2 per 1 unit of price1 long
    """
    if LinearRegression is None or len(price1) != len(price2) or len(price1) < 10:
        return None

    try:
        # OLS regression: price1 = beta * price2 + alpha (if fit_intercept=True).
        # The hedge ratio is the slope beta. When fit_intercept=False, beta ties directly
        # to price1 ≈ beta * price2 (common for spread = price1 - beta * price2).
        model = LinearRegression(fit_intercept=fit_intercept)
        model.fit(price2.values.reshape(-1, 1), price1.values)
        beta = float(model.coef_[0])

        # Validate beta (avoid NaN/Inf/unrealistic magnitudes).
        if np.isnan(beta) or np.isinf(beta):
            return None
        if abs(beta) > 1e6:
            return None

        return beta
    except Exception:
        return None


def calculate_kalman_hedge_ratio(
    price1: pd.Series,
    price2: pd.Series,
    delta: float = PAIRS_TRADING_KALMAN_DELTA,
    observation_covariance: float = PAIRS_TRADING_KALMAN_OBS_COV,
) -> Optional[float]:
    """
    Estimate dynamic (time-varying) hedge ratio using Kalman filter.
    
    The hedge ratio (β) determines how many units of asset 2 (price2) to short
    for each unit of asset 1 (price1) to long. Unlike OLS (static), Kalman filter
    estimates a time-varying ratio that adapts to changing market conditions.
    
    **Kalman (Dynamic) vs OLS (Static)**:
    - Kalman: Ratio evolves over time, adapting to regime changes and volatility shifts.
      Best for volatile markets or when relationships change.
    - OLS: Single constant ratio from all historical data. Simpler but may be outdated.
    
    **Parameters**:
    - delta: Adaptation speed (1e-6 = slow/stable, 1e-4 = fast/reactive). Default: 1e-5
    - observation_covariance: Trust in new observations vs. previous estimates. Default: 1.0
    
    Args:
        price1: First price series (dependent variable, asset to long)
        price2: Second price series (independent variable, asset to short)
        delta: Transition covariance parameter controlling adaptation speed. Default: 1e-5
        observation_covariance: Observation noise variance. Default: 1.0
        
    Returns:
        Latest (most recent) hedge ratio (β) as float, or None if calculation fails.
        
    Example:
        >>> price1 = pd.Series([100, 102, 101, 103, 105, 104, 106])
        >>> price2 = pd.Series([50, 51, 50.5, 51.5, 52.5, 52, 53])
        >>> hedge_ratio = calculate_kalman_hedge_ratio(price1, price2)
        >>> # Returns current dynamic ratio, e.g., 1.92 = short 1.92 units price2 per 1 unit price1
    """
    if KalmanFilter is None or len(price1) != len(price2) or len(price1) < 10:
        return None

    try:
        # Transition covariance heuristic:
        #   Q = (delta / (1 - delta)) * I
        # Derived from assuming random walk with drift where delta controls
        # process noise scaling. Larger delta -> more noise -> faster adaptation.
        trans_cov = delta / (1 - delta) * np.eye(2)

        # Observation matrix builds relationship: price1_t = beta_t * price2_t + alpha_t
        # Each row is [price2_t, 1], reshaped for KalmanFilter API expectations.
        obs_mat = np.vstack([price2.values, np.ones(len(price2))]).T[:, np.newaxis, :]
        kf = KalmanFilter(
            transition_matrices=np.eye(2),
            observation_matrices=obs_mat,
            transition_covariance=trans_cov,
            observation_covariance=observation_covariance,
        )
        state_means, _ = kf.filter(price1.values)
        beta_series = state_means[:, 0]
        if len(beta_series) == 0:
            return None

        beta = float(beta_series[-1])

        # Validate beta
        if np.isnan(beta) or np.isinf(beta):
            return None
        if abs(beta) > 1e6:
            return None

        return beta
    except Exception:
        return None