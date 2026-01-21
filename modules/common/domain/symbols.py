"""
Symbol normalization utilities for trading.

This module provides functions to normalize and validate trading symbols.
"""

from config import DEFAULT_QUOTE


def normalize_symbol(user_input: str, quote: str = DEFAULT_QUOTE) -> str:
    """
    Converts user input like 'xmr' into 'XMR/USDT'. Keeps existing slash pairs.

    Args:
        user_input: User input symbol (e.g., 'btc', 'BTC/USDT', 'btcusdt')
        quote: Quote currency (default: DEFAULT_QUOTE)

    Returns:
        Normalized symbol in format 'BASE/QUOTE' (e.g., 'BTC/USDT')
    """
    if not user_input:
        return f"BTC/{quote}"

    norm = user_input.strip().upper()
    if "/" in norm:
        return norm

    if norm.endswith(quote):
        return f"{norm[: -len(quote)]}/{quote}"

    return f"{norm}/{quote}"


def normalize_symbol_key(symbol: str) -> str:
    """
    Generates a compare-friendly key by uppercasing and stripping separators.

    Args:
        symbol: Symbol string (e.g., 'BTC/USDT', 'ETH-USDT')

    Returns:
        Normalized key string (e.g., 'BTCUSDT', 'ETHUSDT')
    """
    if not symbol:
        return ""
    return "".join(ch for ch in symbol.upper() if ch.isalnum())


__all__ = ["normalize_symbol", "normalize_symbol_key"]
