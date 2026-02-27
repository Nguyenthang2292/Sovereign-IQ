"""
Performance metrics for StaticBacktester.

Computes summary statistics from a list of TradeRecord objects
and an equity curve Series.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
import pandas as pd

if TYPE_CHECKING:
    from .engine import TradeRecord


def calculate_metrics(
    trades: list[TradeRecord],
    equity_curve: pd.Series,
    initial_capital: float = 10_000.0,
) -> dict[str, float]:
    """
    Calculate performance metrics from completed trades.

    Args:
        trades:           List of :class:`TradeRecord` objects.
        equity_curve:     Equity Index over time (as returned by the engine).
        initial_capital:  Starting capital (used for absolute return scaling).

    Returns:
        Dictionary with keys:
        ``num_trades``, ``win_rate``, ``avg_win_pct``, ``avg_loss_pct``,
        ``total_return_pct``, ``profit_factor``, ``max_drawdown_pct``,
        ``sharpe_ratio``, ``avg_rr``, ``best_trade_pct``, ``worst_trade_pct``,
        ``avg_bars_held``, ``num_long``, ``num_short``.
    """
    if not trades:
        return _empty_metrics()

    pnls = np.array([t.pnl_pct for t in trades], dtype=np.float64)
    num_trades = len(trades)

    # Win / loss split
    win_mask  = pnls > 0
    loss_mask = pnls < 0
    win_rate  = float(np.sum(win_mask)) / num_trades

    winning = pnls[win_mask]
    losing  = pnls[loss_mask]

    avg_win_pct  = float(np.mean(winning)) if len(winning) else 0.0
    avg_loss_pct = float(abs(np.mean(losing))) if len(losing) else 0.0

    # Total return (equity-based, aligned with initial capital)
    final_equity = float(equity_curve.iloc[-1]) if len(equity_curve) else float(initial_capital)
    total_return_pct = (
        (final_equity / float(initial_capital) - 1.0) * 100.0
        if initial_capital > 0
        else 0.0
    )

    # Profit factor
    gross_profit = float(np.sum(winning)) if len(winning) else 0.0
    gross_loss   = float(abs(np.sum(losing))) if len(losing) else 1e-9
    profit_factor = gross_profit / gross_loss if gross_loss > 0 else 0.0

    # Risk-reward ratio (average win / average loss)
    avg_rr = avg_win_pct / avg_loss_pct if avg_loss_pct > 0 else 0.0

    # Best / worst
    best_trade_pct  = float(np.max(pnls))
    worst_trade_pct = float(np.min(pnls))

    # Average bars held
    avg_bars_held = float(np.mean([t.bars_held for t in trades]))

    # Long / short counts
    num_long  = sum(1 for t in trades if t.direction == "LONG")
    num_short = sum(1 for t in trades if t.direction == "SHORT")

    # Equity-curve based metrics
    eq = equity_curve.to_numpy(dtype=np.float64)
    max_drawdown_pct = _calc_max_drawdown(eq)
    sharpe_ratio     = _calc_sharpe(eq)

    return {
        "num_trades":       float(num_trades),
        "win_rate":         win_rate,
        "avg_win_pct":      avg_win_pct,
        "avg_loss_pct":     avg_loss_pct,
        "total_return_pct": total_return_pct,
        "profit_factor":    profit_factor,
        "max_drawdown_pct": max_drawdown_pct,
        "sharpe_ratio":     sharpe_ratio,
        "avg_rr":           avg_rr,
        "best_trade_pct":   best_trade_pct,
        "worst_trade_pct":  worst_trade_pct,
        "avg_bars_held":    avg_bars_held,
        "num_long":         float(num_long),
        "num_short":        float(num_short),
    }


# ---------------------------------------------------------------------------
# Private helpers
# ---------------------------------------------------------------------------

def _calc_max_drawdown(eq: np.ndarray) -> float:
    if len(eq) < 2:
        return 0.0
    peak = eq[0]
    max_dd = 0.0
    for v in eq:
        if v > peak:
            peak = v
        dd = (peak - v) / peak * 100.0 if peak > 0 else 0.0
        if dd > max_dd:
            max_dd = dd
    return max_dd


def _calc_sharpe(eq: np.ndarray, periods_per_year: float = 252.0) -> float:
    if len(eq) < 2:
        return 0.0
    returns = np.diff(eq) / np.where(eq[:-1] != 0, eq[:-1], 1.0)
    std = float(np.std(returns))
    if std == 0:
        return 0.0
    return float(np.mean(returns)) / std * np.sqrt(periods_per_year)


def _empty_metrics() -> dict[str, float]:
    return {
        "num_trades":       0.0,
        "win_rate":         0.0,
        "avg_win_pct":      0.0,
        "avg_loss_pct":     0.0,
        "total_return_pct": 0.0,
        "profit_factor":    0.0,
        "max_drawdown_pct": 0.0,
        "sharpe_ratio":     0.0,
        "avg_rr":           0.0,
        "best_trade_pct":   0.0,
        "worst_trade_pct":  0.0,
        "avg_bars_held":    0.0,
        "num_long":         0.0,
        "num_short":        0.0,
    }
