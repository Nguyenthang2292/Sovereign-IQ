"""
Risk metrics calculations for pairs trading analysis.
"""

import pandas as pd
import numpy as np
from typing import Optional

try:
    from modules.config import PAIRS_TRADING_PERIODS_PER_YEAR
except ImportError:
    PAIRS_TRADING_PERIODS_PER_YEAR = 365 * 24


def calculate_spread_sharpe(
    spread: pd.Series, 
    periods_per_year: int = PAIRS_TRADING_PERIODS_PER_YEAR,
    risk_free_rate: float = 0.0
) -> Optional[float]:
    """
    Calculate annualized Sharpe ratio of spread returns.
    
    The Sharpe ratio measures risk-adjusted return by comparing excess return
    (above risk-free rate) to volatility. Higher Sharpe ratio indicates better
    risk-adjusted performance.
    
    Formula:
        Sharpe = (Mean_Return - Risk_Free_Rate) / Std_Return * sqrt(periods_per_year)
    
    Args:
        spread: Spread series (price1 - hedge_ratio * price2)
        periods_per_year: Number of periods per year for annualization
                         Examples: 252 (daily), 365*24 (hourly), 365*24*60 (minute)
        risk_free_rate: Annual risk-free rate (default: 0.0)
                       Example: 0.03 for 3% annual rate
        
    Returns:
        Annualized Sharpe ratio, or None if:
        - Insufficient data (< 2 points)
        - Zero volatility (std = 0)
        - Calculation produces NaN or Inf
    """
    if spread is None or len(spread) < 2:
        return None

    # Calculate returns
    returns = spread.pct_change().dropna()
    if returns.empty or returns.std() == 0:
        return None

    try:
        # Convert annual risk-free rate to per-period rate
        period_rf_rate = risk_free_rate / periods_per_year
        
        # Calculate excess return
        mean_return = returns.mean()
        excess_return = mean_return - period_rf_rate
        std_return = returns.std()
        
        # Annualized Sharpe ratio
        sharpe = (excess_return / std_return) * np.sqrt(periods_per_year)
        
        # Validate result
        if np.isnan(sharpe) or np.isinf(sharpe):
            return None
            
        return float(sharpe)
    except Exception:
        return None


def calculate_max_drawdown(spread: pd.Series) -> Optional[float]:
    """
    Calculate maximum drawdown of the cumulative spread.
    
    Maximum drawdown measures the largest peak-to-trough decline in cumulative
    spread value. It represents the worst-case loss from a peak.
    
    Formula:
        Drawdown(t) = (Cumulative(t) - RunningMax(t)) / |RunningMax(t)|
        MaxDrawdown = min(Drawdown(t)) for all t
    
    Args:
        spread: Spread series (price1 - hedge_ratio * price2)
        
    Returns:
        Maximum drawdown as a negative decimal (e.g., -0.15 for 15% drawdown)
        Returns None if calculation fails.
    """
    if spread is None or len(spread) < 2:
        return None
    
    try:
        # Calculate cumulative spread
        cumulative = spread.cumsum()
        
        # Calculate running maximum
        running_max = cumulative.expanding().max()
        
        # Calculate drawdown, avoiding division by zero
        # Use absolute value of running_max to handle negative peaks
        drawdown = np.where(
            running_max != 0,
            (cumulative - running_max) / np.abs(running_max),
            0.0
        )
        
        # Convert to Series for easier handling
        drawdown_series = pd.Series(drawdown, index=spread.index)
        
        # Check if all drawdowns are NaN
        if drawdown_series.isna().all():
            return None
        
        # Return minimum (most negative) drawdown
        max_dd = float(drawdown_series.min())
        
        # Validate result
        if np.isnan(max_dd) or np.isinf(max_dd):
            return None
            
        return max_dd
        
    except Exception:
        return None


def calculate_calmar_ratio(
    spread: pd.Series,
    periods_per_year: int = PAIRS_TRADING_PERIODS_PER_YEAR,
) -> Optional[float]:
    """
    Calculate Calmar ratio (annualized return / absolute max drawdown).
    
    The Calmar ratio measures return per unit of downside risk. It's similar
    to Sharpe ratio but uses max drawdown instead of volatility as the risk measure.
    
    Formula:
        Calmar = Annualized_Return / |Max_Drawdown|
    
    Args:
        spread: Spread series
        periods_per_year: Number of periods per year for annualization
        
    Returns:
        Calmar ratio or None if calculation fails
    """
    if spread is None or len(spread) < 2:
        return None
        
    try:
        # Calculate annualized return
        returns = spread.pct_change().dropna()
        if returns.empty:
            return None
        annual_return = returns.mean() * periods_per_year
        
        # Calculate max drawdown
        max_dd = calculate_max_drawdown(spread)
        if max_dd is None:
            return None
        
        # Take absolute value of max drawdown (it's negative)
        abs_max_dd = abs(max_dd)
        
        # Avoid division by zero
        if abs_max_dd == 0:
            return None
        
        # Calculate Calmar ratio
        calmar = annual_return / abs_max_dd
        
        # Validate result
        if np.isnan(calmar) or np.isinf(calmar):
            return None
            
        return float(calmar)
        
    except Exception:
        return None