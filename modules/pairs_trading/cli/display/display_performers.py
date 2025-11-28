"""
Performance display formatter for pairs trading analysis.

This module provides formatted display functions for showing symbol performance
data in a user-friendly table format with color-coded metrics.
"""

import pandas as pd
from colorama import Fore

try:
    from modules.common.utils import (
        color_text,
        format_price,
        log_warn,
        log_analysis,
        log_data,
    )
except ImportError:
    color_text = None
    format_price = lambda x: f"${x:.2f}" if x else "N/A"
    log_warn = print
    log_analysis = print
    log_data = print


def display_performers(df: pd.DataFrame, title: str, color: str = "") -> None:
    """
    Display top/worst performers in a formatted table.
    
    This function displays symbol performance data including scores, returns across
    multiple timeframes (1d, 3d, 1w), and current prices in a formatted table.
    
    Args:
        df: DataFrame with performance data containing columns:
            - symbol: Trading symbol (e.g., 'BTC/USDT')
            - score: Weighted performance score
            - 1d_return: 1-day return (decimal, e.g., 0.05 for 5%)
            - 3d_return: 3-day return
            - 1w_return: 1-week return
            - current_price: Current price
        title: Title to display for the table
        color: Colorama color code for styling the entire table content (e.g., Fore.GREEN)
        
    Example:
        >>> performance_df = pd.DataFrame({
        ...     'symbol': ['BTC/USDT', 'ETH/USDT'],
        ...     'score': [0.15, -0.10],
        ...     '1d_return': [0.05, -0.03],
        ...     '3d_return': [0.10, -0.05],
        ...     '1w_return': [0.20, -0.08],
        ...     'current_price': [50000, 3000]
        ... })
        >>> display_performers(performance_df, "Top Performers", Fore.GREEN)
    
    Note:
        - Returns are displayed as percentages (multiplied by 100)
        - Scores are also displayed as percentages
        - Uses color-coded output if colorama is available
    """
    if df is None or df.empty:
        if log_warn:
            log_warn(f"No {title.lower()} found.")
        return

    if log_analysis:
        log_analysis("=" * 80)
        # Apply color to title if color is provided and color_text is available
        if color and color_text:
            # Use print directly with color_text since log_analysis may not preserve color
            print(color_text(title, color))
        else:
            log_analysis(title)
        log_analysis("=" * 80)

    # Header row
    header = f"{'Rank':<6} {'Symbol':<15} {'Score':<12} {'1d Return':<12} {'3d Return':<12} {'1w Return':<12} {'Price':<15}"
    separator = "-" * 80
    
    if color and color_text:
        print(color_text(header, color))
        print(color_text(separator, color))
    else:
        if log_data:
            log_data(header)
            log_data(separator)

    # Data rows
    for idx, row in df.iterrows():
        rank = idx + 1
        symbol = row["symbol"]
        score = row["score"] * 100  # Convert to percentage
        return_1d = row["1d_return"] * 100
        return_3d = row["3d_return"] * 100
        return_1w = row["1w_return"] * 100
        price = row["current_price"]

        row_text = (
            f"{rank:<6} {symbol:<15} "
            f"{score:+.2f}%{'':<8} "
            f"{return_1d:+.2f}%{'':<6} {return_3d:+.2f}%{'':<6} {return_1w:+.2f}%{'':<6} {format_price(price):<15}"
        )
        
        if color and color_text:
            print(color_text(row_text, color))
        else:
            if log_data:
                log_data(row_text)

    if log_analysis:
        log_analysis("=" * 80)
