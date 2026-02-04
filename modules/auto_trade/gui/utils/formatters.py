"""
Formatters Module

Provides formatting utilities for displaying prices, percentages,
timestamps, and PnL values in the GUI.
"""

from datetime import datetime


def format_price(price: float) -> str:
    """
    Format price as USD currency.

    Args:
        price: Price value

    Returns:
        Formatted price string (e.g., "$42,000.00")
    """
    return f"${price:,.2f}"


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
        timestamp: ISO format timestamp string

    Returns:
        Human-readable time string (e.g., "just now", "5m ago", "2024-01-15 10:30")
    """
    try:
        dt = datetime.fromisoformat(timestamp)
        now = datetime.now()
        diff = now - dt
        if diff.seconds < 60:
            return "just now"
        elif diff.seconds < 3600:
            return f"{diff.seconds // 60}m ago"
        elif diff.seconds < 86400:
            return f"{diff.seconds // 3600}h ago"
        else:
            return dt.strftime("%Y-%m-%d %H:%M")
    except (ValueError, AttributeError):
        return timestamp
