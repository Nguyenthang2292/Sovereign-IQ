"""
Pairs opportunities display formatter for pairs trading analysis.

This module provides formatted display functions for showing pairs trading
opportunities with color-coded metrics and detailed quantitative information.
"""

from typing import Optional

import pandas as pd
from colorama import Fore

try:
    from modules.common.utils import (
        color_text,
        log_analysis,
        log_data,
        log_info,
        log_warn,
    )
except ImportError:
    color_text = None
    log_warn = print
    log_info = print
    log_analysis = print
    log_data = print


def _pad_colored(text: str, width: int, color: str, style: Optional[str] = None) -> str:
    """
    Pad text to fixed width before applying ANSI colors to avoid misalignment.

    Args:
        text: Text to pad and color
        width: Target width for padding
        color: Colorama color code
        style: Optional Colorama style code

    Returns:
        Padded and colored text string
    """
    if color_text is None:
        return text.ljust(width)
    padded = text.ljust(width)
    if style is None:
        return color_text(padded, color)
    return color_text(padded, color, style)


def _format_pair_row(row: pd.Series, rank: int, use_border: bool = True) -> str:
    """
    Format a single pair row with color-coded metrics.

    Args:
        row: DataFrame row containing pair data
        rank: Display rank number
        use_border: Whether to use border characters for table formatting

    Returns:
        Formatted string for display
    """
    long_symbol = row["long_symbol"]
    short_symbol = row["short_symbol"]
    spread = row["spread"] * 100
    correlation = row.get("correlation")
    opportunity_score = row["opportunity_score"] * 100
    quantitative_score = row.get("quantitative_score")
    is_cointegrated = row.get("is_cointegrated")

    # Fallback to Johansen cointegration if ADF not available
    if (is_cointegrated is None or pd.isna(is_cointegrated)) and "is_johansen_cointegrated" in row:
        alt_coint = row.get("is_johansen_cointegrated")
        if alt_coint is not None and not pd.isna(alt_coint):
            is_cointegrated = bool(alt_coint)

    # Get verbose metrics if available
    half_life = row.get("half_life")
    spread_sharpe = row.get("spread_sharpe")
    max_drawdown = row.get("max_drawdown")
    hedge_ratio = row.get("hedge_ratio")

    # Prepare spread text
    spread_text = f"{spread:+.2f}%"

    # Format hedge ratio
    if hedge_ratio is not None and not pd.isna(hedge_ratio):
        hedge_text = f"{hedge_ratio:.4f}"
    else:
        hedge_text = "N/A"

    # Color code based on opportunity score
    if opportunity_score > 20:
        score_color = Fore.GREEN
    elif opportunity_score > 10:
        score_color = Fore.YELLOW
    else:
        score_color = Fore.WHITE
    opp_text = f"{opportunity_score:+.1f}%"
    opp_display = _pad_colored(opp_text, 10, score_color)

    # Color code quantitative score
    if quantitative_score is not None and not pd.isna(quantitative_score):
        if quantitative_score >= 70:
            quant_color = Fore.GREEN
        elif quantitative_score >= 50:
            quant_color = Fore.YELLOW
        else:
            quant_color = Fore.RED
        quant_text = f"{quantitative_score:.1f}"
    else:
        quant_color = Fore.WHITE
        quant_text = "N/A"
    quant_display = _pad_colored(quant_text, 10, quant_color)

    # Cointegration status
    if is_cointegrated is not None and not pd.isna(is_cointegrated):
        coint_status = "OK" if is_cointegrated else "NOT"
        coint_color = Fore.GREEN if is_cointegrated else Fore.RED
    else:
        # If both ADF and Johansen failed or were unavailable, treat as NOT cointegrated
        coint_status = "NOT"
        coint_color = Fore.RED
    coint_display_verbose = _pad_colored(coint_status, 6, coint_color)

    # Color code correlation
    if correlation is not None and not pd.isna(correlation):
        abs_corr = abs(correlation)
        if abs_corr > 0.7:
            corr_color = Fore.GREEN
        elif abs_corr > 0.4:
            corr_color = Fore.YELLOW
        else:
            corr_color = Fore.RED
        corr_text = f"{correlation:+.3f}"
    else:
        corr_color = Fore.WHITE
        corr_text = "N/A"
    corr_display = _pad_colored(corr_text, 8, corr_color)

    # Format verbose metrics
    half_life_text = f"{half_life:.1f}" if half_life is not None and not pd.isna(half_life) else "N/A"
    sharpe_text = f"{spread_sharpe:.2f}" if spread_sharpe is not None and not pd.isna(spread_sharpe) else "N/A"
    maxdd_text = f"{max_drawdown * 100:.1f}%" if max_drawdown is not None and not pd.isna(max_drawdown) else "N/A"

    # Use border characters for better table formatting
    border = "│" if use_border else " "

    return (
        f"{border} {rank:<4} {border} {long_symbol:<14} {border} {short_symbol:<14} {border} "
        f"{spread_text:<9} {border} {corr_display} {border} "
        f"{opp_display} {border} "
        f"{quant_display} {border} "
        f"{coint_display_verbose} {border} "
        f"{hedge_text:<11} {border} "
        f"{half_life_text:<9} {border} {sharpe_text:<9} {border} {maxdd_text:<9} {border}"
    )


def display_pairs_opportunities(
    pairs_df: pd.DataFrame,
    max_display: int = 10,
) -> None:
    """
    Display pairs trading opportunities in a formatted table.

    This function displays pairs trading opportunities with color-coded metrics
    including opportunity scores, quantitative scores, cointegration status,
    correlation, and various risk metrics. Optionally displays reversed pairs
    (swapped long/short) for alternative trading perspectives.

    Args:
        pairs_df: DataFrame with pairs trading opportunities containing columns:
            - long_symbol: Symbol to long (buy)
            - short_symbol: Symbol to short (sell)
            - spread: Performance spread between symbols
            - opportunity_score: Overall opportunity score (0-1)
            - correlation: Correlation coefficient (optional)
            - quantitative_score: Quantitative metrics score 0-100 (optional)
            - is_cointegrated: Cointegration test result (optional)
            - hedge_ratio: OLS hedge ratio (optional)
            - half_life: Mean reversion half-life (optional)
            - spread_sharpe: Spread Sharpe ratio (optional)
            - max_drawdown: Maximum drawdown (optional)
        max_display: Maximum number of pairs to display (default: 10)

    Example:
        >>> pairs = pd.DataFrame({
        ...     'long_symbol': ['BTC/USDT'],
        ...     'short_symbol': ['ETH/USDT'],
        ...     'spread': [0.15],
        ...     'opportunity_score': [0.25],
        ...     'correlation': [0.85],
        ...     'quantitative_score': [75.0],
        ...     'is_cointegrated': [True]
        ... })
        >>> display_pairs_opportunities(pairs, max_display=5)

    Note:
        - Opportunity scores > 20% are green, > 10% are yellow
        - Quantitative scores >= 70 are green, >= 50 are yellow, < 50 are red
        - Correlation |r| > 0.7 is green, > 0.4 is yellow, <= 0.4 is red
        - Cointegrated pairs show "OK" in green, non-cointegrated show "NOT" in red
    """
    if pairs_df is None or pairs_df.empty:
        if log_warn:
            log_warn("No pairs trading opportunities found.")
        return

    # Display original pairs
    if log_analysis:
        log_analysis("=" * 150)
        log_analysis("PAIRS TRADING OPPORTUNITIES")
        log_analysis("=" * 150)

    # Table header with borders
    if log_data:
        header = (
            f"│ {'Rank':<4} │ {'Long':<14} │ {'Short':<14} │ {'Spread':<9} │ {'Corr':<8} │ "
            f"{'OppScore':<10} │ {'QuantScore':<10} │ {'Coint':<6} │ {'HedgeRatio':<11} │ "
            f"{'HalfLife':<9} │ {'Sharpe':<9} │ {'MaxDD':<9} │"
        )
        separator = (
            "├"
            + "─" * 6
            + "┼"
            + "─" * 16
            + "┼"
            + "─" * 16
            + "┼"
            + "─" * 11
            + "┼"
            + "─" * 10
            + "┼"
            + "─" * 12
            + "┼"
            + "─" * 12
            + "┼"
            + "─" * 8
            + "┼"
            + "─" * 13
            + "┼"
            + "─" * 11
            + "┼"
            + "─" * 11
            + "┼"
            + "─" * 11
            + "┤"
        )
        top_border = (
            "┌"
            + "─" * 6
            + "┬"
            + "─" * 16
            + "┬"
            + "─" * 16
            + "┬"
            + "─" * 11
            + "┬"
            + "─" * 10
            + "┬"
            + "─" * 12
            + "┬"
            + "─" * 12
            + "┬"
            + "─" * 8
            + "┬"
            + "─" * 13
            + "┬"
            + "─" * 11
            + "┬"
            + "─" * 11
            + "┬"
            + "─" * 11
            + "┐"
        )
        bottom_border = (
            "└"
            + "─" * 6
            + "┴"
            + "─" * 16
            + "┴"
            + "─" * 16
            + "┴"
            + "─" * 11
            + "┴"
            + "─" * 10
            + "┴"
            + "─" * 12
            + "┴"
            + "─" * 12
            + "┴"
            + "─" * 8
            + "┴"
            + "─" * 13
            + "┴"
            + "─" * 11
            + "┴"
            + "─" * 11
            + "┴"
            + "─" * 11
            + "┘"
        )

        log_data(top_border)
        log_data(header)
        log_data(separator)

    display_count = min(len(pairs_df), max_display)
    for idx in range(display_count):
        row = pairs_df.iloc[idx]
        rank = idx + 1
        formatted_row = _format_pair_row(row, rank, use_border=True)
        if log_data:
            log_data(formatted_row)

    if log_data:
        log_data(bottom_border)

    if log_analysis:
        log_analysis("")

    if len(pairs_df) > max_display:
        if log_info:
            log_info(f"Showing top {max_display} of {len(pairs_df)} opportunities. Use --max-pairs to see more.")
