"""
Incremental signal calculation functions with position-aware skipping.

These functions combine signal calculation and trade simulation in a single loop,
skipping signal calculation when a position is open to save computation time.
"""

from typing import Dict, List, Tuple

import numpy as np
import pandas as pd

from config.position_sizing import (
    HYBRID_OSC_LENGTH,
    HYBRID_OSC_MULT,
    HYBRID_OSC_STRATEGIES,
    HYBRID_SPC_PARAMS,
)
from modules.common.ui.progress_bar import ProgressBar
from modules.common.utils import (
    log_error,
    log_progress,
    log_warn,
)

from .exit_conditions import check_long_exit_conditions, check_short_exit_conditions


def calculate_signals_incremental(
    df: pd.DataFrame,
    symbol: str,
    timeframe: str,
    limit: int,
    signal_type: str,
    hybrid_signal_calculator,
    stop_loss_pct: float,
    take_profit_pct: float,
    trailing_stop_pct: float,
    max_hold_periods: int,
    initial_capital: float = 10000.0,
) -> Tuple[pd.Series, List[Dict]]:
    """
    Calculate signals incrementally with position-aware skipping.

    OPTIMIZED: Pre-computes all indicators once, then extracts signals incrementally.
    This is much faster than recalculating indicators for each period.

    This function combines signal calculation and trade simulation in a single loop.
    It skips signal calculation when a position is open to save computation time.

    Args:
        df: DataFrame with OHLCV data
        symbol: Trading pair symbol
        timeframe: Timeframe for data
        limit: Number of candles to look back
        signal_type: "LONG" or "SHORT"
        hybrid_signal_calculator: HybridSignalCalculator instance
        stop_loss_pct: Stop loss percentage
        take_profit_pct: Take profit percentage
        trailing_stop_pct: Trailing stop percentage
        max_hold_periods: Maximum periods to hold a position
        initial_capital: Initial capital (for API compatibility)

    Returns:
        Tuple of (signals, trades) where:
        - signals: Series with signal values for all periods
        - trades: List of trade dictionaries
    """
    signals = pd.Series(0, index=df.index)
    trades: List[Dict] = []
    position = None  # Current position: None, or dict with entry info
    last_exit_index = -1

    # Convert DataFrame columns to numpy arrays for faster access
    close_prices = np.ascontiguousarray(df["close"].values, dtype=np.float64)
    high_prices = np.ascontiguousarray(df["high"].values, dtype=np.float64)
    low_prices = np.ascontiguousarray(df["low"].values, dtype=np.float64)
    df_index = df.index

    log_progress(f"  Pre-computing indicators for {len(df)} periods (vectorized)...")
    print()  # Newline before progress bar

    # OPTIMIZATION: Pre-compute all indicators once for entire DataFrame
    # This is much faster than recalculating for each period
    precomputed_indicators = None
    try:
        precomputed_indicators = hybrid_signal_calculator.precompute_all_indicators_vectorized(
            df=df,
            symbol=symbol,
            timeframe=timeframe,
            osc_length=HYBRID_OSC_LENGTH,
            osc_mult=HYBRID_OSC_MULT,
            osc_strategies=HYBRID_OSC_STRATEGIES,
            spc_params=HYBRID_SPC_PARAMS,
        )
        log_progress(
            "  Indicators pre-computed successfully. Starting incremental calculation with position-aware skipping..."
        )
    except Exception as e:
        log_error(f"Error in vectorized indicator pre-computation: {e}")
        log_warn("Falling back to per-period calculation (slower)...")
        precomputed_indicators = None

    print()  # Newline before progress bar

    # Initialize progress bar
    progress = ProgressBar(total=len(df), label="Incremental calculation")

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

                # Use JIT-compiled functions for exit condition checks
                if signal_type.upper() == "LONG":
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
                        # Update trailing stop if price moves favorably
                        if current_price > position.get("highest_price", current_price):
                            position["highest_price"] = current_price
                            position["trailing_stop"] = current_price * (1 - trailing_stop_pct)
                else:  # SHORT
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
                        # Update trailing stop if price moves favorably
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

            # Skip signal calculation if position is open or we just exited
            if position is not None or i <= last_exit_index:
                signals.iloc[i] = 0  # No new signal when position open or just exited
                progress.update()
                continue

            # Calculate signal for this period using precomputed indicators
            try:
                # Need at least some data to calculate signals
                if len(df) < 10:
                    signals.iloc[i] = 0
                    progress.update()
                    continue

                # OPTIMIZATION: Use precomputed indicators instead of recalculating
                if precomputed_indicators is not None:
                    # Extract signal from precomputed data (maintains walk-forward semantics)
                    signal, confidence = hybrid_signal_calculator.calculate_signal_from_precomputed(
                        precomputed_indicators=precomputed_indicators,
                        period_index=i,
                        signal_type=signal_type,
                    )
                else:
                    # Fallback: calculate from scratch (slower)
                    historical_df = df.iloc[: i + 1]
                    signal, confidence = hybrid_signal_calculator.calculate_hybrid_signal(
                        df=historical_df,
                        symbol=symbol,
                        timeframe=timeframe,
                        period_index=i,
                        signal_type=signal_type,
                        osc_length=HYBRID_OSC_LENGTH,
                        osc_mult=HYBRID_OSC_MULT,
                        osc_strategies=HYBRID_OSC_STRATEGIES,
                        spc_params=HYBRID_SPC_PARAMS,
                    )

                signals.iloc[i] = signal

                # Check if we should enter a new position
                if signal != 0:
                    if signal_type.upper() == "LONG" and signal > 0:
                        position = {
                            "entry_index": i,
                            "entry_price": current_price,
                            "entry_time": df_index[i],
                            "highest_price": current_price,
                            "lowest_price": None,
                            "trailing_stop": None,
                        }
                    elif signal_type.upper() == "SHORT" and signal < 0:
                        position = {
                            "entry_index": i,
                            "entry_price": current_price,
                            "entry_time": df_index[i],
                            "highest_price": None,
                            "lowest_price": current_price,
                            "trailing_stop": None,
                        }
            except Exception as e:
                log_warn(f"Error calculating signal for period {i}: {e}")
                signals.iloc[i] = 0

            progress.update()
    finally:
        progress.finish()

    # Close any remaining position at the end
    if position is not None:
        final_price = close_prices[-1]
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

    # Log signal statistics
    long_signals = (signals == 1).sum()
    short_signals = (signals == -1).sum()
    neutral_signals = (signals == 0).sum()
    log_progress(f"  Signal distribution: LONG={long_signals}, SHORT={short_signals}, NEUTRAL={neutral_signals}")
    skipped_count = len(df) - (long_signals + short_signals + neutral_signals)
    if skipped_count > 0:
        log_progress(f"  Skipped {skipped_count} periods due to open positions")

    return signals, trades


def calculate_single_signals_incremental(
    df: pd.DataFrame,
    symbol: str,
    timeframe: str,
    limit: int,
    hybrid_signal_calculator,
    stop_loss_pct: float,
    take_profit_pct: float,
    trailing_stop_pct: float,
    max_hold_periods: int,
    initial_capital: float = 10000.0,
) -> Tuple[pd.Series, List[Dict]]:
    """
    Calculate single signals incrementally with position-aware skipping.

    OPTIMIZED: Pre-computes all indicators once, then extracts signals incrementally.
    This is much faster than recalculating indicators for each period.

    Similar to calculate_signals_incremental() but uses single signal (highest confidence) mode.

    Args:
        df: DataFrame with OHLCV data
        symbol: Trading pair symbol
        timeframe: Timeframe for data
        limit: Number of candles to look back
        hybrid_signal_calculator: HybridSignalCalculator instance
        stop_loss_pct: Stop loss percentage
        take_profit_pct: Take profit percentage
        trailing_stop_pct: Trailing stop percentage
        max_hold_periods: Maximum periods to hold a position
        initial_capital: Initial capital (for API compatibility)

    Returns:
        Tuple of (signals, trades) where:
        - signals: Series with signal values for all periods
        - trades: List of trade dictionaries
    """
    signals = pd.Series(0, index=df.index)
    trades: List[Dict] = []
    position = None  # Current position: None, or dict with entry info
    last_exit_index = -1

    # Convert DataFrame columns to numpy arrays for faster access
    close_prices = np.ascontiguousarray(df["close"].values, dtype=np.float64)
    high_prices = np.ascontiguousarray(df["high"].values, dtype=np.float64)
    low_prices = np.ascontiguousarray(df["low"].values, dtype=np.float64)
    df_index = df.index

    log_progress(f"  Pre-computing indicators for {len(df)} periods (vectorized)...")
    print()  # Newline before progress bar

    # OPTIMIZATION: Pre-compute all indicators once for entire DataFrame
    # This is much faster than recalculating for each period
    precomputed_indicators = None
    try:
        precomputed_indicators = hybrid_signal_calculator.precompute_all_indicators_vectorized(
            df=df,
            symbol=symbol,
            timeframe=timeframe,
            osc_length=HYBRID_OSC_LENGTH,
            osc_mult=HYBRID_OSC_MULT,
            osc_strategies=HYBRID_OSC_STRATEGIES,
            spc_params=HYBRID_SPC_PARAMS,
        )
        log_progress(
            "  Indicators pre-computed successfully. Starting incremental calculation with position-aware skipping..."
        )
    except Exception as e:
        log_error(f"Error in vectorized indicator pre-computation: {e}")
        log_warn("Falling back to per-period calculation (slower)...")
        precomputed_indicators = None

    print()  # Newline before progress bar

    # Initialize progress bar
    progress = ProgressBar(total=len(df), label="Incremental calculation")

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

                # Determine signal type from position
                position_signal_type = position.get("signal_type", "LONG")

                # Use JIT-compiled functions for exit condition checks
                if position_signal_type == "LONG":
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
                        # Update trailing stop if price moves favorably
                        if current_price > position.get("highest_price", current_price):
                            position["highest_price"] = current_price
                            position["trailing_stop"] = current_price * (1 - trailing_stop_pct)
                else:  # SHORT
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
                        # Update trailing stop if price moves favorably
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
                        "signal_type": position_signal_type,
                        "exit_reason": exit_reason,
                        "pnl": pnl,
                        "pnl_pct": pnl * 100,
                        "hold_periods": i - position["entry_index"],
                    }
                    trades.append(trade)
                    position = None
                    last_exit_index = i

            # Skip signal calculation if position is open or we just exited
            if position is not None or i <= last_exit_index:
                signals.iloc[i] = 0  # No new signal when position open or just exited
                progress.update()
                continue

            # Calculate signal for this period using precomputed indicators
            try:
                # Need at least some data to calculate signals
                if len(df) < 10:
                    signals.iloc[i] = 0
                    progress.update()
                    continue

                # OPTIMIZATION: Use precomputed indicators instead of recalculating
                if precomputed_indicators is not None:
                    # Extract signal from precomputed data (maintains walk-forward semantics)
                    signal, confidence = hybrid_signal_calculator.calculate_single_signal_from_precomputed(
                        precomputed_indicators=precomputed_indicators,
                        period_index=i,
                    )
                else:
                    # Fallback: calculate from scratch (slower)
                    historical_df = df.iloc[: i + 1]
                    signal, confidence = hybrid_signal_calculator.calculate_single_signal_highest_confidence(
                        df=historical_df,
                        symbol=symbol,
                        timeframe=timeframe,
                        period_index=i,
                        osc_length=HYBRID_OSC_LENGTH,
                        osc_mult=HYBRID_OSC_MULT,
                        osc_strategies=HYBRID_OSC_STRATEGIES,
                        spc_params=HYBRID_SPC_PARAMS,
                    )

                signals.iloc[i] = signal

                # Check if we should enter a new position (any signal direction)
                if signal != 0:
                    if signal > 0:  # LONG signal
                        position = {
                            "entry_index": i,
                            "entry_price": current_price,
                            "entry_time": df_index[i],
                            "signal_type": "LONG",
                            "highest_price": current_price,
                            "lowest_price": None,
                            "trailing_stop": None,
                        }
                    else:  # SHORT signal
                        position = {
                            "entry_index": i,
                            "entry_price": current_price,
                            "entry_time": df_index[i],
                            "signal_type": "SHORT",
                            "highest_price": None,
                            "lowest_price": current_price,
                            "trailing_stop": None,
                        }
            except Exception as e:
                log_warn(f"Error calculating signal for period {i}: {e}")
                signals.iloc[i] = 0

            progress.update()
    finally:
        progress.finish()

    # Close any remaining position at the end
    if position is not None:
        final_price = close_prices[-1]
        position_signal_type = position.get("signal_type", "LONG")
        if position_signal_type == "LONG":
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
            "signal_type": position_signal_type,
            "exit_reason": "END_OF_DATA",
            "pnl": pnl,
            "pnl_pct": pnl * 100,
            "hold_periods": len(df) - 1 - position["entry_index"],
        }
        trades.append(trade)

    # Log signal statistics
    long_signals = (signals == 1).sum()
    short_signals = (signals == -1).sum()
    neutral_signals = (signals == 0).sum()
    log_progress(f"  Signal distribution: LONG={long_signals}, SHORT={short_signals}, NEUTRAL={neutral_signals}")
    skipped_count = len(df) - (long_signals + short_signals + neutral_signals)
    if skipped_count > 0:
        log_progress(f"  Skipped {skipped_count} periods due to open positions")

    return signals, trades
