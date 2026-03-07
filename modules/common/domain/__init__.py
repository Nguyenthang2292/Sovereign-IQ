"""
Trading domain utilities.

This package provides domain-specific utilities for:
- Symbol normalization and validation (use SymbolCodec)
- Order type resolution (use BinanceOrderType)
- Timeframe conversion and normalization

Deprecated (use SymbolCodec instead):
- normalize_symbol -> SymbolCodec.to_ccxt
- normalize_symbol_key -> SymbolCodec.to_db
"""

import warnings

from .order_type_codec import BinanceOrderType
from .symbol_codec import SymbolCodec
from .symbol_types import CcxtSymbol, DbSymbol, FuturesSymbol
from .symbol_validation import (
    SYMBOL_PATTERN,
    filter_valid_symbols,
    require_valid_symbol,
    validate_symbol,
)
from .symbols import normalize_symbol, normalize_symbol_key
from .timeframes import days_to_candles, normalize_timeframe, timeframe_to_minutes


def __getattr__(name: str):
    """Deprecation wrapper for legacy symbol functions."""
    if name == "normalize_symbol":
        warnings.warn(
            "normalize_symbol is deprecated. Use SymbolCodec.to_ccxt instead.",
            DeprecationWarning,
            stacklevel=2,
        )
        return normalize_symbol
    if name == "normalize_symbol_key":
        warnings.warn(
            "normalize_symbol_key is deprecated. Use SymbolCodec.to_db instead.",
            DeprecationWarning,
            stacklevel=2,
        )
        return normalize_symbol_key
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


__all__ = [
    # SymbolCodec (new API)
    "SymbolCodec",
    "BinanceOrderType",
    # Symbol types
    "DbSymbol",
    "CcxtSymbol",
    "FuturesSymbol",
    # Symbol validation (still valid)
    "validate_symbol",
    "require_valid_symbol",
    "filter_valid_symbols",
    "SYMBOL_PATTERN",
    # Timeframe utilities
    "normalize_timeframe",
    "timeframe_to_minutes",
    "days_to_candles",
    # Deprecated (kept for backward compatibility, use SymbolCodec instead)
    # "normalize_symbol",  # Use SymbolCodec.to_ccxt()
    # "normalize_symbol_key",  # Use SymbolCodec.to_db()
]
