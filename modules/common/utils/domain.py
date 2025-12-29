"""
Domain-specific utilities for trading (symbols, timeframes).
"""

import math
import re
from config import DEFAULT_QUOTE


# --- Timeframe Utilities ---

# Regex patterns for timeframe parsing (input is already stripped)
# Pattern 1: number + optional space + unit (e.g., '15m', '15 m')
TIMEFRAME_RE = re.compile(r"^(\d+)\s*([mhdw])$")
# Pattern 2: unit + optional space + number (e.g., 'm15', 'm 15', 'h1')
TIMEFRAME_REVERSE_RE = re.compile(r"^([mhdw])\s*(\d+)$")
# Pattern for normalized format (no spaces): number + unit
TIMEFRAME_NORMALIZED_RE = re.compile(r"^(\d+)([mhdw])$")


def normalize_timeframe(timeframe: str) -> str:
    """
    Normalizes timeframe string to standard format (number + unit).
    Accepts both formats: '15m' or 'm15', '1h' or 'h1', etc.
    
    Args:
        timeframe: Timeframe string in any format (e.g., '15m', 'm15', '1h', 'h1', '1d', 'd1')
        
    Returns:
        Normalized timeframe string in format 'number+unit' (e.g., '15m', '1h', '1d')
        
    Examples:
        normalize_timeframe('15m') -> '15m'
        normalize_timeframe('m15') -> '15m'
        normalize_timeframe('1h') -> '1h'
        normalize_timeframe('h1') -> '1h'
        normalize_timeframe('4h') -> '4h'
        normalize_timeframe('h4') -> '4h'
    """
    def _validate_timeframe_value(value: int, timeframe: str) -> None:
        """Validates that timeframe value is positive."""
        if value <= 0:
            raise ValueError(f"Invalid timeframe: value must be > 0, got '{timeframe}'")
    
    if not timeframe:
        return '1h'  # default
    
    timeframe = timeframe.strip().lower()
    
    # Pattern 1: number + optional space + unit (e.g., '15m', '15 m', '1h')
    match1 = TIMEFRAME_RE.match(timeframe)
    if match1:
        value, unit = match1.groups()
        value = int(value)
        _validate_timeframe_value(value, timeframe)
        return f"{value}{unit}"
    
    # Pattern 2: unit + optional space + number (e.g., 'm15', 'm 15', 'h1')
    match2 = TIMEFRAME_REVERSE_RE.match(timeframe)
    if match2:
        unit, value = match2.groups()
        value = int(value)
        _validate_timeframe_value(value, timeframe)
        return f"{value}{unit}"
    
    # If no match, return as-is (might be invalid, but let other functions handle it)
    return timeframe


def timeframe_to_minutes(timeframe: str) -> int:
    """
    Converts a timeframe string like '30m', '1h', '1d' into minutes.
    Accepts both formats: '15m' or 'm15', '1h' or 'h1', etc.

    Args:
        timeframe: Timeframe string (e.g., '30m', 'm30', '1h', 'h1', '1d', '1w')

    Returns:
        Number of minutes
    """
    # Normalize input to handle both '15m' and 'm15' formats
    normalized = normalize_timeframe(timeframe)
    
    # Input is already normalized, so use simpler pattern without whitespace matching
    match = TIMEFRAME_NORMALIZED_RE.match(normalized.lower())
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
    # All units handled above; regex ensures unit is in [mhdw]
    raise AssertionError(f"Unexpected unit '{unit}' - should be unreachable")


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
    return math.ceil(days * candles_per_day)
