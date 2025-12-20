"""
Exit condition checking functions for backtester.

This module contains JIT-compiled functions for checking exit conditions
(stop loss, take profit, trailing stop, max hold period) for both LONG and SHORT positions.
"""

from typing import Tuple

# Try to import Numba for JIT compilation, fallback if not available
try:
    from numba import njit
    NUMBA_AVAILABLE = True
except ImportError:
    NUMBA_AVAILABLE = False
    # Fallback decorator when numba is not available
    def njit(*args, **kwargs):
        """No-op decorator when numba is not available."""
        def decorator(func):
            return func
        return decorator


# OPTIMIZATION: JIT-compiled functions for performance-critical calculations
@njit(cache=True)
def check_long_exit_conditions(
    current_price: float,
    high: float,
    low: float,
    entry_price: float,
    highest_price: float,
    trailing_stop: float,
    stop_loss_pct: float,
    take_profit_pct: float,
    trailing_stop_pct: float,
    hold_periods: int,
    max_hold_periods: int,
) -> Tuple[int, float, float]:
    """
    Check exit conditions for LONG position using JIT-compiled code.
    
    Returns:
        Tuple of (exit_code, exit_price, pnl) where:
        - exit_code: 0=no exit, 1=stop_loss, 2=take_profit, 3=trailing_stop, 4=max_hold
        - exit_price: Price at exit (0.0 if no exit)
        - pnl: Profit/loss percentage (0.0 if no exit)
    """
    # Check stop loss
    stop_loss_price = entry_price * (1.0 - stop_loss_pct)
    if low <= stop_loss_price:
        exit_price = stop_loss_price
        pnl = (exit_price - entry_price) / entry_price
        return (1, exit_price, pnl)
    
    # Check take profit
    take_profit_price = entry_price * (1.0 + take_profit_pct)
    if high >= take_profit_price:
        exit_price = take_profit_price
        pnl = (exit_price - entry_price) / entry_price
        return (2, exit_price, pnl)
    
    # Check trailing stop (only if it's been set)
    if trailing_stop > 0.0:
        if current_price <= trailing_stop:
            exit_price = trailing_stop
            pnl = (exit_price - entry_price) / entry_price
            return (3, exit_price, pnl)
    
    # Check max hold period
    if hold_periods >= max_hold_periods:
        exit_price = current_price
        pnl = (exit_price - entry_price) / entry_price
        return (4, exit_price, pnl)
    
    # No exit
    return (0, 0.0, 0.0)


@njit(cache=True)
def check_short_exit_conditions(
    current_price: float,
    high: float,
    low: float,
    entry_price: float,
    lowest_price: float,
    trailing_stop: float,
    stop_loss_pct: float,
    take_profit_pct: float,
    trailing_stop_pct: float,
    hold_periods: int,
    max_hold_periods: int,
) -> Tuple[int, float, float]:
    """
    Check exit conditions for SHORT position using JIT-compiled code.
    
    Returns:
        Tuple of (exit_code, exit_price, pnl) where:
        - exit_code: 0=no exit, 1=stop_loss, 2=take_profit, 3=trailing_stop, 4=max_hold
        - exit_price: Price at exit (0.0 if no exit)
        - pnl: Profit/loss percentage (0.0 if no exit)
    """
    # Check stop loss
    stop_loss_price = entry_price * (1.0 + stop_loss_pct)
    if high >= stop_loss_price:
        exit_price = stop_loss_price
        pnl = (entry_price - exit_price) / entry_price
        return (1, exit_price, pnl)
    
    # Check take profit
    take_profit_price = entry_price * (1.0 - take_profit_pct)
    if low <= take_profit_price:
        exit_price = take_profit_price
        pnl = (entry_price - exit_price) / entry_price
        return (2, exit_price, pnl)
    
    # Check trailing stop (only if it's been set)
    if trailing_stop > 0.0:
        if current_price >= trailing_stop:
            exit_price = trailing_stop
            pnl = (entry_price - exit_price) / entry_price
            return (3, exit_price, pnl)
    
    # Check max hold period
    if hold_periods >= max_hold_periods:
        exit_price = current_price
        pnl = (entry_price - exit_price) / entry_price
        return (4, exit_price, pnl)
    
    # No exit
    return (0, 0.0, 0.0)


# Backward compatibility aliases (for existing code that uses _check_*)
_check_long_exit_conditions = check_long_exit_conditions
_check_short_exit_conditions = check_short_exit_conditions

