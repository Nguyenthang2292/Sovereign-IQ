
from trades and equity curves.

"""
Performance metrics calculation for backtester.

This module contains functions for calculating performance metrics
"""

from typing import Dict, List

import numpy as np
import pandas as pd

from modules.common.quantitative_metrics.risk import (
    calculate_max_drawdown,
    calculate_sharpe_ratio,
)
from modules.common.utils.domain import timeframe_to_minutes


def calculate_metrics(
    trades: List[Dict],
    equity_curve: pd.Series,
    timeframe: str = "1h",
) -> Dict:
    """
    Calculate performance metrics from trades and equity curve.

    Args:
        trades: List of trade dictionaries with 'pnl' key
        equity_curve: Series with cumulative equity values
        timeframe: Timeframe of the data (default: "1h")

    Returns:
        Dictionary with performance metrics
    """
    if not trades:
        return {
            "win_rate": 0.0,
            "avg_win": 0.0,
            "avg_loss": 0.0,
            "total_return": 0.0,
            "sharpe_ratio": 0.0,
            "max_drawdown": 0.0,
            "num_trades": 0,
            "profit_factor": 0.0,
        }

    # OPTIMIZATION: Vectorize metrics calculation using numpy
    # Extract PnL values as numpy array for faster computation
    # Use contiguous array for better cache performance
    pnls = np.ascontiguousarray([t["pnl"] for t in trades], dtype=np.float64)

    # Calculate win rate using vectorized operations
    winning_mask = pnls > 0
    losing_mask = pnls < 0
    num_trades = len(trades)
    win_rate = np.sum(winning_mask) / num_trades if num_trades > 0 else 0.0

    # Calculate average win/loss using vectorized operations
    winning_pnls = pnls[winning_mask]
    losing_pnls = pnls[losing_mask]
    avg_win = np.mean(winning_pnls) if len(winning_pnls) > 0 else 0.0
    avg_loss = abs(np.mean(losing_pnls)) if len(losing_pnls) > 0 else 0.0

    # Calculate total return
    initial_capital = equity_curve.iloc[0] if len(equity_curve) > 0 else 10000.0
    final_capital = equity_curve.iloc[-1] if len(equity_curve) > 0 else initial_capital
    total_return = (final_capital - initial_capital) / initial_capital if initial_capital > 0 else 0.0

    # Calculate Sharpe ratio with correct frequency
    minutes = timeframe_to_minutes(timeframe)
    periods_per_year = (365 * 24 * 60) // minutes if minutes > 0 else 365 * 24

    returns = equity_curve.pct_change().dropna()
    sharpe_ratio = calculate_sharpe_ratio(returns, periods_per_year=periods_per_year) or 0.0

    # Calculate max drawdown
    # Get absolute drawdown value first
    max_drawdown_absolute = calculate_max_drawdown(equity_curve) or 0.0

    # Convert to percentage: drawdown_pct = (drawdown_absolute / peak_equity) * 100
    # Peak equity is the maximum value in the equity curve
    if max_drawdown_absolute < 0 and len(equity_curve) > 0:
        peak_equity = equity_curve.max()
        if peak_equity > 0:
            # Calculate drawdown as percentage of peak equity
            max_drawdown = (max_drawdown_absolute / peak_equity) * 100.0
        else:
            # If peak equity is 0 or negative, use absolute value
            max_drawdown = max_drawdown_absolute
    else:
        max_drawdown = max_drawdown_absolute

    # Calculate profit factor using vectorized operations
    total_profit = np.sum(winning_pnls) if len(winning_pnls) > 0 else 0.0
    total_loss = abs(np.sum(losing_pnls)) if len(losing_pnls) > 0 else 0.0
    profit_factor = total_profit / total_loss if total_loss > 0 else (total_profit if total_profit > 0 else 0.0)

    return {
        "win_rate": win_rate,
        "avg_win": avg_win,
        "avg_loss": avg_loss,
        "total_return": total_return,
        "sharpe_ratio": sharpe_ratio,
        "max_drawdown": max_drawdown,
        "num_trades": len(trades),
        "profit_factor": profit_factor,
    }


def empty_backtest_result() -> Dict:
    """
    Return empty backtest result structure.

    Returns:
        Dictionary with empty backtest result structure
    """
    return {
        "trades": [],
        "equity_curve": pd.Series([10000.0]),
        "metrics": {
            "win_rate": 0.0,
            "avg_win": 0.0,
            "avg_loss": 0.0,
            "total_return": 0.0,
            "sharpe_ratio": 0.0,
            "max_drawdown": 0.0,
            "num_trades": 0,
            "profit_factor": 0.0,
        },
        "total_time": 0.0,
    }
