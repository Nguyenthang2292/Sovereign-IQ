"""
Domain-specific utilities for trading (symbols, timeframes).
"""

import re
from config import DEFAULT_QUOTE


# --- Timeframe Utilities ---

def timeframe_to_minutes(timeframe: str) -> int:
    """
    Converts a timeframe string like '30m', '1h', '1d' into minutes.

    Args:
        timeframe: Timeframe string (e.g., '30m', '1h', '1d', '1w')

    Returns:
        Number of minutes
    """
    match = re.match(r"^\s*(\d+)\s*([mhdw])\s*$", timeframe.lower())
    if not match:
        return 60  # default 1h

    value, unit = match.groups()
    value = int(value)

    if unit == "m":
        return value
    if unit == "h":
        return value * 60
    if unit == "d":
        return value * 60 * 24
    if unit == "w":
        return value * 60 * 24 * 7
    return 60


# --- Symbol Normalization Utilities ---

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
        return f"{norm[:-len(quote)]}/{quote}"

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


def days_to_candles(days: int, timeframe: str) -> int:
    """
    Convert number of days to number of candles based on timeframe.
    
    Args:
        days: Number of days
        timeframe: Timeframe string (e.g., '1h', '4h', '1d')
        
    Returns:
        Number of candles needed to cover the specified days
        
    Examples:
        days_to_candles(90, '1h') -> 2160  (90 days * 24 hours/day)
        days_to_candles(90, '4h') -> 540   (90 days * 6 candles/day)
        days_to_candles(90, '1d') -> 90    (90 days * 1 candle/day)
    """
    minutes_per_candle = timeframe_to_minutes(timeframe)
    minutes_per_day = 24 * 60  # 1440 minutes per day
    candles_per_day = minutes_per_day / minutes_per_candle
    return int(days * candles_per_day)
