
from typing import Dict, List

import numpy as np
import pandas as pd

from .exit_conditions import check_long_exit_conditions, check_short_exit_conditions
from modules.common.ui.progress_bar import ProgressBar
from .exit_conditions import check_long_exit_conditions, check_short_exit_conditions
from modules.common.ui.progress_bar import ProgressBar

"""
Trade simulation logic for backtester.

This module contains functions for simulating trades based on signals,
including entry/exit logic and position management.
"""






def simulate_trades(
    df: pd.DataFrame,
    signals: pd.Series,
    signal_type: str,
    initial_capital: float,
    stop_loss_pct: float,
    take_profit_pct: float,
    trailing_stop_pct: float,
    max_hold_periods: int,
) -> List[Dict]:
    """
    Simulate trades based on signals.

    Args:
        df: DataFrame with OHLCV data
        signals: Series with signal values (1 for LONG, -1 for SHORT, 0 for no signal)
        signal_type: "LONG" or "SHORT"
        initial_capital: Initial capital (not used in calculation, but kept for API compatibility)
        stop_loss_pct: Stop loss percentage
        take_profit_pct: Take profit percentage
        trailing_stop_pct: Trailing stop percentage
        max_hold_periods: Maximum periods to hold a position

    Returns:
        List of trade dictionaries with entry/exit information
    """
    trades = []
    position = None  # Current position: None, or dict with entry info
    last_exit_index = -1

    # OPTIMIZATION: Convert DataFrame columns to numpy arrays for faster access
    # This reduces overhead from repeated df.iloc[i] calls
    # Use contiguous arrays for better cache performance
    close_prices = np.ascontiguousarray(df["close"].values, dtype=np.float64)
    high_prices = np.ascontiguousarray(df["high"].values, dtype=np.float64)
    low_prices = np.ascontiguousarray(df["low"].values, dtype=np.float64)
    signals_array = signals.values if hasattr(signals, "values") else signals
    df_index = df.index

    # OPTIMIZATION: Pre-calculate signal array for faster access
    # Use contiguous array for better cache performance
    signals_array = np.ascontiguousarray(np.asarray(signals_array, dtype=np.float64))

    # OPTIMIZATION: Pre-calculate entry signal mask for faster entry detection
    # This allows vectorized entry detection instead of checking in loop
    if signal_type.upper() == "LONG":
        entry_mask = signals_array > 0
    else:  # SHORT
        entry_mask = signals_array < 0

    # Initialize progress bar for trade simulation (only if dataset is large)
    progress = None
    if len(df) > 100:
        print()  # Newline before progress bar
        progress = ProgressBar(total=len(df), label="Simulating trades")

    try:
        for i in range(len(df)):
            current_price = close_prices[i]
            high = high_prices[i]
            low = low_prices[i]

            # Check if we should exit current position
            if position is not None:
                exit_reason = None
                exit_price = None
                pnl = 0.0

                # OPTIMIZATION: Use JIT-compiled functions for exit condition checks
                if signal_type.upper() == "LONG":
                    # Use JIT-compiled function to check exit conditions
                    # FIX: Use None check instead of 'or' operator for highest_price
                    highest_price = position.get("highest_price")
                    if highest_price is None:
                        highest_price = current_price
                    trailing_stop = position.get("trailing_stop")
                    if trailing_stop is None:
                        trailing_stop = 0.0

                    exit_code, exit_price_jit, pnl_jit = check_long_exit_conditions(
                        current_price=current_price,
                        high=high,
                        low=low,
                        entry_price=position["entry_price"],
                        highest_price=highest_price,
                        trailing_stop=trailing_stop,
                        stop_loss_pct=stop_loss_pct,
                        take_profit_pct=take_profit_pct,
                        trailing_stop_pct=trailing_stop_pct,
                        hold_periods=i - position["entry_index"],
                        max_hold_periods=max_hold_periods,
                    )

                    if exit_code > 0:
                        exit_reason_map = {1: "STOP_LOSS", 2: "TAKE_PROFIT", 3: "TRAILING_STOP", 4: "MAX_HOLD"}
                        exit_reason = exit_reason_map[exit_code]
                        exit_price = exit_price_jit
                        pnl = pnl_jit
                    else:
                        # Update trailing stop if price moves favorably (only if no exit)
                        if current_price > position.get("highest_price", current_price):
                            position["highest_price"] = current_price
                            position["trailing_stop"] = current_price * (1 - trailing_stop_pct)
                else:  # SHORT
                    # Use JIT-compiled function to check exit conditions
                    # FIX: Use None check instead of 'or' operator for lowest_price
                    lowest_price = position.get("lowest_price")
                    if lowest_price is None:
                        lowest_price = current_price
                    trailing_stop = position.get("trailing_stop")
                    if trailing_stop is None:
                        trailing_stop = 0.0

                    exit_code, exit_price_jit, pnl_jit = check_short_exit_conditions(
                        current_price=current_price,
                        high=high,
                        low=low,
                        entry_price=position["entry_price"],
                        lowest_price=lowest_price,
                        trailing_stop=trailing_stop,
                        stop_loss_pct=stop_loss_pct,
                        take_profit_pct=take_profit_pct,
                        trailing_stop_pct=trailing_stop_pct,
                        hold_periods=i - position["entry_index"],
                        max_hold_periods=max_hold_periods,
                    )

                    if exit_code > 0:
                        exit_reason_map = {1: "STOP_LOSS", 2: "TAKE_PROFIT", 3: "TRAILING_STOP", 4: "MAX_HOLD"}
                        exit_reason = exit_reason_map[exit_code]
                        exit_price = exit_price_jit
                        pnl = pnl_jit
                    else:
                        # Update trailing stop if price moves favorably (only if no exit)
                        if current_price < position.get("lowest_price", current_price):
                            position["lowest_price"] = current_price
                            position["trailing_stop"] = current_price * (1 + trailing_stop_pct)

                if exit_reason:
                    # Close position
                    trade = {
                        "entry_index": position["entry_index"],
                        "exit_index": i,
                        "entry_price": position["entry_price"],
                        "exit_price": exit_price,
                        "entry_time": df_index[position["entry_index"]],
                        "exit_time": df_index[i],
                        "signal_type": signal_type,
                        "exit_reason": exit_reason,
                        "pnl": pnl,
                        "pnl_pct": pnl * 100,
                        "hold_periods": i - position["entry_index"],
                    }
                    trades.append(trade)
                    position = None
                    last_exit_index = i

            # Check if we should enter a new position
            # OPTIMIZATION: Use pre-calculated entry mask for faster entry detection
            if position is None and i < len(entry_mask) and entry_mask[i] and i > last_exit_index:
                # FIXED: Trailing stop initialization
                # Previously: trailing_stop was initialized immediately on entry, causing premature exits
                # Now: trailing_stop is None initially, only set after price moves favorably
                # This prevents trailing stop from triggering immediately if price moves against position
                position = {
                    "entry_index": i,
                    "entry_price": current_price,
                    "entry_time": df_index[i],
                    "highest_price": current_price if signal_type.upper() == "LONG" else None,
                    "lowest_price": current_price if signal_type.upper() == "SHORT" else None,
                    "trailing_stop": None,  # Will be set after favorable price movement
                }

            # Update progress bar if enabled
            if progress is not None:
                progress.update()
    finally:
        # Ensure progress bar is finished even if there's an error
        if progress is not None:
            progress.finish()

    # Close any remaining position at the end
    if position is not None:
        final_price = close_prices[-1]
        # OPTIMIZATION: Pre-calculate PnL calculation
        if signal_type.upper() == "LONG":
            pnl = (final_price - position["entry_price"]) / position["entry_price"]
        else:  # SHORT
            pnl = (position["entry_price"] - final_price) / position["entry_price"]
        trade = {
            "entry_index": position["entry_index"],
            "exit_index": len(df) - 1,
            "entry_price": position["entry_price"],
            "exit_price": final_price,
            "entry_time": df_index[position["entry_index"]],
            "exit_time": df_index[-1],
            "signal_type": signal_type,
            "exit_reason": "END_OF_DATA",
            "pnl": pnl,
            "pnl_pct": pnl * 100,
            "hold_periods": len(df) - 1 - position["entry_index"],
        }
        trades.append(trade)

    return trades
