"""
Formatters Module

Provides formatting utilities for displaying prices, percentages,
timestamps, and PnL values in the GUI.
"""

from datetime import datetime


def format_price(price: float) -> str:
    """
    Format value as USD currency (2 decimals).

    Args:
        price: Value in USD

    Returns:
        Formatted price string (e.g., "$42,000.00")
    """
    return f"${price:,.2f}"


def format_asset_price(price: float, decimals: int = 5) -> str:
    """
    Format asset unit price without currency symbol, preserving more precision.

    Used for Entry / TP / SL / BE where we want to see the exact futures
    price (e.g. 0.00663) instead of a rounded $0.01.

    Args:
        price: Price value
        decimals: Maximum number of decimal places to show

    Returns:
        Formatted price string without currency symbol (e.g. "0.00663")
    """
    fmt = f"{{:.{decimals}f}}"
    s = fmt.format(price)
    # Trim trailing zeros and dot for cleaner display
    s = s.rstrip("0").rstrip(".")
    return s or "0"


def format_pnl(pnl: float) -> str:
    """
    Format profit/loss with sign.

    Args:
        pnl: PnL value

    Returns:
        Formatted PnL string with + or - sign (e.g., "+$123.45" or "-$56.78")
    """
    sign = "+" if pnl >= 0 else ""
    return f"{sign}${pnl:.2f}"


def format_percent(value: float) -> str:
    """
    Format percentage with sign.

    Args:
        value: Percentage value

    Returns:
        Formatted percentage string with + or - sign (e.g., "+5.23%" or "-2.15%")
    """
    sign = "+" if value >= 0 else ""
    return f"{sign}{value:.2f}%"


def format_timestamp(timestamp: str) -> str:
    """
    Format timestamp as relative time or absolute date.

    Args:
        timestamp: ISO format timestamp string or stringified timestamp

    Returns:
        Human-readable time string (e.g., "just now", "5m ago", "2024-01-15 10:30")
    """
    try:
        # Check if it's a numeric string (timestamp)
        if timestamp.replace(".", "", 1).isdigit():
            dt = datetime.fromtimestamp(float(timestamp))
        else:
            dt = datetime.fromisoformat(timestamp)

        now = datetime.now()
        diff = now - dt
        seconds = int(diff.total_seconds())

        if seconds < 60:
            return "just now"
        elif seconds < 3600:
            return f"{seconds // 60}m ago"
        elif seconds < 86400:
            return f"{seconds // 3600}h ago"
        else:
            return dt.strftime("%Y-%m-%d %H:%M")
    except (ValueError, AttributeError, TypeError, OverflowError):
        return timestamp
