"""
Symbol normalization utilities for trading.

This module provides functions to normalize and validate trading symbols.

.. deprecated::
    Use :class:`modules.common.domain.symbol_codec.SymbolCodec` instead.
"""

import warnings

from config import DEFAULT_QUOTE


def _deprecated_to_ccxt(user_input: str, quote: str = DEFAULT_QUOTE) -> str:
    """
    Converts user input like 'xmr' into 'XMR/USDT'. Keeps existing slash pairs.

    .. deprecated::
        Use :class:`modules.common.domain.symbol_codec.SymbolCodec` instead.

    Args:
        user_input: User input symbol (e.g., 'btc', 'BTC/USDT', 'btcusdt')
        quote: Quote currency (default: DEFAULT_QUOTE)

    Returns:
        Normalized symbol in format 'BASE/QUOTE' (e.g., 'BTC/USDT')
    """
    warnings.warn(
        "normalize_symbol is deprecated. Use SymbolCodec.to_ccxt instead.",
        DeprecationWarning,
        stacklevel=2,
    )
    quote_norm = quote.strip().upper()
    if not quote_norm:
        raise ValueError("quote must be non-empty")

    if not user_input:
        return f"BTC/{quote_norm}"

    norm = user_input.strip().upper()
    if "/" in norm:
        return norm

    if norm == quote_norm:
        raise ValueError(f"invalid symbol '{user_input}': missing base currency")

    if norm.endswith(quote_norm):
        base = norm[: -len(quote_norm)]
        if not base:
            raise ValueError(f"invalid symbol '{user_input}': missing base currency")
        return f"{base}/{quote_norm}"

    return f"{norm}/{quote_norm}"


def _deprecated_to_db(symbol: str) -> str:
    """
    Generates a compare-friendly key by uppercasing and stripping separators.

    .. deprecated::
        Use :class:`modules.common.domain.symbol_codec.SymbolCodec` instead.

    Args:
        symbol: Symbol string (e.g., 'BTC/USDT', 'ETH-USDT')

    Returns:
        Normalized key string (e.g., 'BTCUSDT', 'ETHUSDT')
    """
    warnings.warn(
        "normalize_symbol_key is deprecated. Use SymbolCodec.to_db instead.",
        DeprecationWarning,
        stacklevel=2,
    )
    if not symbol:
        return ""
    return "".join(ch for ch in symbol.upper() if ch.isalnum())


normalize_symbol = _deprecated_to_ccxt
normalize_symbol_key = _deprecated_to_db


__all__ = ["normalize_symbol", "normalize_symbol_key"]
