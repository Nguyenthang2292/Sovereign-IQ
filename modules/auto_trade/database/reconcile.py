"""Compatibility shim for legacy reconcile imports.

Legacy tests import:
    from modules.auto_trade.database.reconcile import _symbol_to_ccxt, reconcile_orders_with_binance
"""

from __future__ import annotations

from modules.auto_trade.database import reconcile_orders_with_binance
from modules.common.domain.symbol_codec import SymbolCodec
from modules.common.domain.symbol_types import CcxtSymbol


_SYMBOL_CODEC = SymbolCodec()


def _symbol_to_ccxt(symbol: str | None) -> CcxtSymbol:
    """Normalize Binance-style symbols to CCXT spot-like format.

    Examples:
      BTCUSDT -> BTC/USDT
      BTCUSDT_PERP -> BTC/USDT
      BTC/USDT -> BTC/USDT
      BTC -> BTC/USDT
    """
    if not symbol:
        return CcxtSymbol("")

    s = symbol.strip().upper()
    if not s:
        return CcxtSymbol("")

    s = s.replace("-PERP", "").replace("_PERP", "")
    ccxt_symbol = str(_SYMBOL_CODEC.to_ccxt(s))

    if "/" not in ccxt_symbol:
        return CcxtSymbol(f"{ccxt_symbol}/USDT")

    return CcxtSymbol(ccxt_symbol)


_normalize_symbol = _symbol_to_ccxt

__all__ = ["_normalize_symbol", "_symbol_to_ccxt", "reconcile_orders_with_binance"]
