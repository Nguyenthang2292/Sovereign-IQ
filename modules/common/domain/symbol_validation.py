"""
Symbol format validation for exchange symbols (e.g. Binance futures).

Validates symbol strings before use in fetching or processing to avoid
path/command injection and malformed input from external sources.
"""

from __future__ import annotations

import re
from typing import List

# Typical exchange symbols: alphanumeric, 2–30 chars (e.g. BTCUSDT, 1000PEPEUSDT).
# Reject separators and metacharacters that could be used for path/shell injection.
SYMBOL_PATTERN = re.compile(r"^[A-Za-z0-9]{2,30}$")


def validate_symbol(symbol: str) -> bool:
    """Return True if symbol has a safe, exchange-like format.

    Args:
        symbol: Symbol string (e.g. from exchange API or user input).

    Returns:
        True if format is valid (alphanumeric, length 2–30).
    """
    if not isinstance(symbol, str):
        return False
    s = symbol.strip()
    return bool(s and SYMBOL_PATTERN.fullmatch(s))


def require_valid_symbol(symbol: str) -> None:
    """Raise ValueError if symbol format is invalid.

    Use at entry points (e.g. process_symbol, scan) before using symbol
    in fetchers or file paths.

    Args:
        symbol: Symbol string to validate.

    Raises:
        ValueError: If symbol is not a string or does not match allowed format.
    """
    if not isinstance(symbol, str):
        raise ValueError(f"symbol must be a string, got {type(symbol).__name__}")
    s = symbol.strip()
    if not s:
        raise ValueError("symbol must be non-empty")
    if not SYMBOL_PATTERN.fullmatch(s):
        raise ValueError(
            f"symbol must be 2–30 alphanumeric characters (e.g. BTCUSDT), got {symbol!r}"
        )


def filter_valid_symbols(symbols: List[str]) -> List[str]:
    """Return only symbols that pass format validation.

    Use when processing lists from external sources (e.g. exchange API)
    to skip malformed entries without failing the whole batch.

    Args:
        symbols: List of symbol strings.

    Returns:
        List of symbols that pass validate_symbol().
    """
    if not symbols:
        return []
    return [s for s in symbols if validate_symbol(s)]


__all__ = ["validate_symbol", "require_valid_symbol", "filter_valid_symbols", "SYMBOL_PATTERN"]
