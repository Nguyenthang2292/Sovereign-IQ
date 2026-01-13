"""
Display utilities for position sizing CLI.

This module provides formatted display functions for position sizing results.
"""

import pandas as pd
from colorama import Fore, Style

from modules.common.utils import (
    color_text,
    format_price,
)


def display_position_sizing_results(df: pd.DataFrame) -> None:
    """
    Display position sizing results in a formatted table.

    Args:
        df: DataFrame with position sizing results
    """
    if df is None or df.empty:
        print(color_text("\nNo position sizing results to display.", Fore.YELLOW))
        return

    print("\n" + color_text("=" * 120, Fore.CYAN, Style.BRIGHT))
    print(color_text("POSITION SIZING RESULTS", Fore.CYAN, Style.BRIGHT))
    print(color_text("=" * 120, Fore.CYAN, Style.BRIGHT))

    # Select columns to display
    display_columns = [
        "symbol",
        "signal_type",
        "regime",
        "position_size_usdt",
        "position_size_pct",
        "kelly_fraction",
        "adjusted_kelly_fraction",
    ]

    # Add metrics columns if available
    if "metrics" in df.columns:
        # We'll display metrics separately
        pass

    # Create display DataFrame
    display_df = df[display_columns].copy() if all(col in df.columns for col in display_columns) else df.copy()

    # Format columns
    if "position_size_usdt" in display_df.columns:
        display_df["position_size_usdt"] = display_df["position_size_usdt"].apply(lambda x: f"{x:.2f}")
    if "position_size_pct" in display_df.columns:
        display_df["position_size_pct"] = display_df["position_size_pct"].apply(lambda x: f"{x:.2f}%")
    if "kelly_fraction" in display_df.columns:
        display_df["kelly_fraction"] = display_df["kelly_fraction"].apply(lambda x: f"{x:.4f}")
    if "adjusted_kelly_fraction" in display_df.columns:
        display_df["adjusted_kelly_fraction"] = display_df["adjusted_kelly_fraction"].apply(lambda x: f"{x:.4f}")

    # Rename columns for display
    display_df.columns = [
        "Symbol",
        "Signal",
        "Regime",
        "Position Size (USDT)",
        "Position Size (%)",
        "Kelly Fraction",
        "Adjusted Kelly",
    ]

    print(display_df.to_string(index=False))
    print(color_text("=" * 120, Fore.CYAN, Style.BRIGHT))

    # Display trades table for each symbol
    if "backtest_result" in df.columns:
        for _, row in df.iterrows():
            symbol = row.get("symbol", "N/A")
            backtest_result = row.get("backtest_result", {})
            if isinstance(backtest_result, dict):
                trades = backtest_result.get("trades", [])
                if trades and len(trades) > 0:
                    try:
                        display_trades_table(symbol, trades)
                    except Exception as e:
                        print(color_text(f"\nError displaying trades for {symbol}: {e}", Fore.RED))
            elif hasattr(backtest_result, "get"):  # Handle other dict-like objects
                trades = backtest_result.get("trades", []) if hasattr(backtest_result, "get") else []
                if trades and len(trades) > 0:
                    try:
                        display_trades_table(symbol, trades)
                    except Exception as e:
                        print(color_text(f"\nError displaying trades for {symbol}: {e}", Fore.RED))

    # Display metrics summary
    if "metrics" in df.columns:
        print("\n" + color_text("PERFORMANCE METRICS", Fore.CYAN, Style.BRIGHT))
        print(color_text("-" * 120, Fore.CYAN))

        metrics_rows = []
        for _, row in df.iterrows():
            metrics = row.get("metrics", {})
            if isinstance(metrics, dict):
                metrics_row = {
                    "Symbol": row.get("symbol", "N/A"),
                    "Win Rate": f"{metrics.get('win_rate', 0.0) * 100:.2f}%",
                    "Avg Win": f"{metrics.get('avg_win', 0.0) * 100:.2f}%",
                    "Avg Loss": f"{metrics.get('avg_loss', 0.0) * 100:.2f}%",
                    "Sharpe": f"{metrics.get('sharpe_ratio', 0.0):.2f}",
                    "Max DD": f"{metrics.get('max_drawdown', 0.0):.2f}%",
                    "Trades": metrics.get("num_trades", 0),
                }
                metrics_rows.append(metrics_row)

        if metrics_rows:
            metrics_df = pd.DataFrame(metrics_rows)
            print(metrics_df.to_string(index=False))
            print(color_text("-" * 120, Fore.CYAN))

    # Display portfolio summary
    if "position_size_usdt" in df.columns:
        total_position_size = df["position_size_usdt"].sum()
        total_exposure_pct = df["position_size_pct"].sum() if "position_size_pct" in df.columns else 0.0

        print("\n" + color_text("PORTFOLIO SUMMARY", Fore.CYAN, Style.BRIGHT))
        print(color_text("-" * 120, Fore.CYAN))
        print(f"Total Position Size: {color_text(f'{total_position_size:.2f} USDT', Fore.GREEN)}")
        print(f"Total Exposure: {color_text(f'{total_exposure_pct:.2f}%', Fore.YELLOW)}")
        print(f"Number of Positions: {color_text(str(len(df)), Fore.WHITE)}")
        print(color_text("-" * 120, Fore.CYAN))


def get_exit_reason_color(exit_reason: str, pnl: float = 0.0) -> Fore:
    """
    Get color for exit reason.

    Args:
        exit_reason: Exit reason string
        pnl: Profit and loss value (used for TRAILING_STOP coloring)

    Returns:
        Colorama Fore color constant
    """
    # Check if LIGHTBLACK_EX exists, otherwise use WHITE for gray effect
    end_of_data_color = Fore.LIGHTBLACK_EX if hasattr(Fore, "LIGHTBLACK_EX") else Fore.WHITE

    exit_reason_upper = exit_reason.upper()

    # TRAILING_STOP: always yellow
    if exit_reason_upper == "TRAILING_STOP":
        return Fore.YELLOW

    color_map = {
        "STOP_LOSS": Fore.RED,
        "TAKE_PROFIT": Fore.GREEN,
        "MAX_HOLD": Fore.WHITE,
        "END_OF_DATA": end_of_data_color,
    }
    return color_map.get(exit_reason_upper, Fore.WHITE)


def display_trades_table(symbol: str, trades: list) -> None:
    """
    Display trades table with PnL for a symbol.

    Args:
        symbol: Trading pair symbol
        trades: List of trade dictionaries from backtest
    """
    if not trades:
        return

    try:
        import sys

        print("\n" + color_text(f"TRADES DETAIL - {symbol}", Fore.CYAN, Style.BRIGHT))
        print(color_text("-" * 120, Fore.CYAN))

        # Prepare trades data for display
        trades_rows = []
        trades_pnl_map = {}  # Map to store PnL by row index for coloring
        for i, trade in enumerate(trades, 1):
            try:
                entry_time = trade.get("entry_time", "N/A")
                exit_time = trade.get("exit_time", "N/A")
                entry_price = trade.get("entry_price", 0.0)
                exit_price = trade.get("exit_price", 0.0)
                pnl_pct = trade.get("pnl_pct", 0.0)
                pnl = trade.get("pnl", 0.0)
                exit_reason = trade.get("exit_reason", "N/A")
                hold_periods = trade.get("hold_periods", 0)
                signal_type = trade.get("signal_type", "N/A")

                # Format times (if they're timestamps)
                if hasattr(entry_time, "strftime"):
                    entry_time = entry_time.strftime("%Y-%m-%d %H:%M")
                elif entry_time != "N/A":
                    entry_time = str(entry_time)
                if hasattr(exit_time, "strftime"):
                    exit_time = exit_time.strftime("%Y-%m-%d %H:%M")
                elif exit_time != "N/A":
                    exit_time = str(exit_time)

                # Determine PnL color
                pnl_sign = "+" if pnl > 0 else ""

                row_idx = len(trades_rows)  # Index in trades_rows (before append)
                trades_rows.append(
                    {
                        "#": i,
                        "Entry Time": str(entry_time),
                        "Exit Time": str(exit_time),
                        "Direction": signal_type.upper() if signal_type != "N/A" else "N/A",
                        "Entry Price": format_price(float(entry_price)),
                        "Exit Price": format_price(float(exit_price)),
                        "PnL %": f"{pnl_sign}{pnl_pct:.2f}%",
                        "Hold": f"{int(hold_periods)}",
                        "Exit Reason": str(exit_reason),
                    }
                )
                # Store PnL for this row index
                trades_pnl_map[row_idx] = pnl
            except Exception:
                continue  # Skip this trade if there's an error

        if trades_rows:
            trades_df = pd.DataFrame(trades_rows)
            # Display trades table with column-specific coloring
            # First, get the formatted table string to preserve column widths
            table_str = trades_df.to_string(index=False)
            table_lines = table_str.split("\n")

            # Print header (first line)
            if table_lines:
                print(table_lines[0])

            # Print data rows with column-specific coloring
            # For TRAILING_STOP: only Exit Reason column is yellow, other columns follow PnL color
            for idx, row in trades_df.iterrows():
                exit_reason = row.get("Exit Reason", "N/A")
                # Get PnL from trades_pnl_map using DataFrame index
                trade_pnl = trades_pnl_map.get(idx, 0.0)

                # Determine base color for row (based on PnL)
                base_color = Fore.GREEN if trade_pnl > 0 else Fore.RED if trade_pnl < 0 else Fore.WHITE

                # Get the row as a formatted string (skip header, idx+1 because header is at index 0)
                if idx + 1 < len(table_lines):
                    row_str = table_lines[idx + 1]

                    # If TRAILING_STOP, color only the "TRAILING_STOP" text yellow, rest by PnL
                    if exit_reason.upper() == "TRAILING_STOP":
                        # Find the position of "TRAILING_STOP" in the row string
                        trailing_stop_pos = row_str.find("TRAILING_STOP")

                        if trailing_stop_pos >= 0:
                            # Split the row: before TRAILING_STOP, TRAILING_STOP itself, and after
                            before_part = row_str[:trailing_stop_pos]
                            trailing_text = "TRAILING_STOP"
                            after_part = row_str[trailing_stop_pos + len(trailing_text) :]

                            # Color: before and after with base color (based on PnL), TRAILING_STOP with yellow
                            print(
                                color_text(before_part, base_color)
                                + color_text(trailing_text, Fore.YELLOW)
                                + color_text(after_part, base_color)
                            )
                        else:
                            # Fallback: color entire row with base color
                            print(color_text(row_str, base_color))
                    else:
                        # For non-TRAILING_STOP, use exit_reason color for entire row
                        row_color = get_exit_reason_color(exit_reason, trade_pnl)
                        print(color_text(row_str, row_color))

            # Calculate summary
            total_pnl = sum(t.get("pnl", 0.0) for t in trades)
            total_pnl_pct = sum(t.get("pnl_pct", 0.0) for t in trades)
            winning_trades = [t for t in trades if t.get("pnl", 0.0) > 0]
            losing_trades = [t for t in trades if t.get("pnl", 0.0) < 0]

            print(color_text("-" * 120, Fore.CYAN), flush=True)
            # Color summary based on win/loss ratio
            total_trades_str = f"Total Trades: {len(trades)}"
            winning_count = len(winning_trades)
            losing_count = len(losing_trades)

            # Determine overall color: green if more wins, red if more losses, yellow if equal
            if winning_count > losing_count:
                summary_color = Fore.GREEN
            elif losing_count > winning_count:
                summary_color = Fore.RED
            else:
                summary_color = Fore.YELLOW

            print(color_text(total_trades_str, summary_color), end=" | ", flush=True)
            print(f"Winning: {color_text(str(winning_count), Fore.GREEN)} | ", end="", flush=True)
            print(f"Losing: {color_text(str(losing_count), Fore.RED)}", flush=True)

            total_pnl_sign = "+" if total_pnl > 0 else ""
            total_pnl_color = Fore.GREEN if total_pnl > 0 else Fore.RED if total_pnl < 0 else Fore.WHITE
            print(f"Total PnL: {color_text(f'{total_pnl_sign}{total_pnl_pct:.2f}%', total_pnl_color)}", flush=True)
            print(color_text("-" * 120, Fore.CYAN), flush=True)
            sys.stdout.flush()
    except Exception as e:
        print(color_text(f"\nError displaying trades table for {symbol}: {e}", Fore.RED))


def display_configuration(config: dict) -> None:
    """
    Display configuration information.

    Args:
        config: Dictionary with configuration values
    """
    print("\n" + color_text("=" * 80, Fore.CYAN, Style.BRIGHT))
    print(color_text("POSITION SIZING CONFIGURATION", Fore.CYAN, Style.BRIGHT))
    print(color_text("=" * 80, Fore.CYAN, Style.BRIGHT))

    print(color_text("Account Balance:", Fore.WHITE), f"{config.get('account_balance', 0.0):.2f} USDT")
    print(color_text("Timeframe:", Fore.WHITE), config.get("timeframe", "N/A"))
    print(color_text("Lookback Days:", Fore.WHITE), config.get("lookback_days", 0))
    print(
        color_text("Max Position Size:", Fore.WHITE),
        f"{config.get('max_position_size', 0.0) * 100:.1f}% of account balance",
    )

    if "source" in config:
        print(color_text("Source:", Fore.WHITE), config["source"])
    elif "symbols_file" in config:
        print(color_text("Symbols File:", Fore.WHITE), config["symbols_file"])
    elif "symbols" in config:
        print(color_text("Symbols:", Fore.WHITE), config["symbols"])

    print(color_text("=" * 80, Fore.CYAN, Style.BRIGHT))
