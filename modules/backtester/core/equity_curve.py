"""
Equity curve calculation for backtester.

This module contains functions for calculating equity curves from trade PnLs,
using JIT-compiled code when available for performance.
"""

from typing import List, Dict
import pandas as pd
import numpy as np

from .exit_conditions import NUMBA_AVAILABLE

# Try to import Numba for JIT compilation
try:
    from numba import njit
except ImportError:
    # Fallback decorator when numba is not available
    def njit(*args, **kwargs):
        """No-op decorator when numba is not available."""
        def decorator(func):
            return func
        return decorator


@njit(cache=True)
def _calculate_equity_curve_jit(
    trade_pnls: np.ndarray,
    initial_capital: float,
    risk_per_trade: float,
) -> np.ndarray:
    """
    Calculate equity curve from trade PnLs using JIT-compiled code.
    
    Args:
        trade_pnls: Array of trade PnL percentages
        initial_capital: Starting capital
        risk_per_trade: Risk percentage per trade (e.g., 0.01 for 1%)
        
    Returns:
        Array of cumulative equity values
    """
    n = len(trade_pnls)
    equity = np.zeros(n + 1, dtype=np.float64)
    equity[0] = initial_capital
    current_cap = initial_capital
    
    for i in range(n):
        trade_pnl = current_cap * risk_per_trade * trade_pnls[i]
        current_cap += trade_pnl
        equity[i + 1] = current_cap
    
    return equity


def calculate_equity_curve(
    trades: List[Dict],
    initial_capital: float,
    num_periods: int,
    risk_per_trade: float = 0.01,
) -> pd.Series:
    """
    Calculate equity curve from trades.
    
    Args:
        trades: List of trade dictionaries with 'pnl' key
        initial_capital: Starting capital
        num_periods: Number of periods to pad the equity curve to
        risk_per_trade: Risk percentage per trade (default: 0.01 for 1%)
        
    Returns:
        Series with cumulative equity values
    """
    # OPTIMIZATION: Pre-allocate equity array for better performance
    if trades:
        # Extract PnL values as numpy array for faster computation
        # Use contiguous array for better cache performance
        trade_pnls = np.ascontiguousarray([trade['pnl'] for trade in trades], dtype=np.float64)
        
        # Pre-allocate equity array with size = num_periods
        equity_array = np.full(num_periods, initial_capital, dtype=np.float64)
        
        # Use JIT-compiled function to calculate equity curve
        if NUMBA_AVAILABLE:
            capital_values = _calculate_equity_curve_jit(
                trade_pnls=trade_pnls,
                initial_capital=initial_capital,
                risk_per_trade=risk_per_trade,
            )
        else:
            # Fallback to non-JIT version if numba is not available
            # Pre-allocate array
            capital_values = np.zeros(len(trades) + 1, dtype=np.float64)
            capital_values[0] = initial_capital
            current_cap = initial_capital
            for idx, pnl in enumerate(trade_pnls):
                trade_pnl = current_cap * risk_per_trade * pnl
                current_cap += trade_pnl
                capital_values[idx + 1] = current_cap
        
        # FIX: Map capital values to correct periods based on trade exit_index
        # Capital only changes when a trade closes, so we need to map trades to their exit periods
        current_capital = initial_capital
        
        # Sort trades by exit_index to ensure correct order
        sorted_trades = sorted(trades, key=lambda t: t.get('exit_index', 0))
        
        # Track the last period where capital was updated (starts at -1, so first update is at period 0)
        last_exit_period = -1
        
        for trade in sorted_trades:
            # Get exit period (use exit_index if available, otherwise estimate)
            exit_period = trade.get('exit_index', last_exit_period + 1)
            # Ensure exit_period is within bounds
            exit_period = min(max(exit_period, 0), num_periods - 1)
            
            # Calculate capital after this trade closes
            # Use the capital value from the previous period (before this trade closes)
            trade_pnl = current_capital * risk_per_trade * trade['pnl']
            new_capital = current_capital + trade_pnl
            
            # Update equity: periods from (last_exit_period + 1) to exit_period (inclusive)
            # All these periods should have the capital value BEFORE this trade closes
            # Then at exit_period, the capital changes to new_capital
            start_period = last_exit_period + 1
            end_period = exit_period  # Up to but not including exit_period (will set separately)
            
            # Set equity for periods before exit (using current capital)
            for period in range(start_period, min(end_period, num_periods)):
                equity_array[period] = current_capital
            
            # At exit period, capital changes to new value
            if exit_period < num_periods:
                equity_array[exit_period] = new_capital
            
            # Update current capital for next iteration
            current_capital = new_capital
            last_exit_period = exit_period
        
        # Fill remaining periods with final capital value
        if last_exit_period + 1 < num_periods:
            equity_array[last_exit_period + 1:] = current_capital
    else:
        # No trades, equity stays at initial capital
        equity_array = np.full(num_periods, initial_capital, dtype=np.float64)
        current_capital = initial_capital
    
    equity_series = pd.Series(equity_array)
    return equity_series

