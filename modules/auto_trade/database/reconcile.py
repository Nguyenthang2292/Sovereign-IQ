"""Compatibility shim for legacy reconcile imports.

Legacy tests import:
    from modules.auto_trade.database.reconcile import _normalize_symbol, reconcile_orders_with_binance
"""

from __future__ import annotations

from modules.auto_trade.database import reconcile_orders_with_binance


def _normalize_symbol(symbol: str | None) -> str:
    """Normalize Binance-style symbols to CCXT spot-like format.

    Examples:
      BTCUSDT -> BTC/USDT
      BTCUSDT_PERP -> BTC/USDT
      BTC/USDT -> BTC/USDT
      BTC -> BTC/USDT
    """
    if not symbol:
        return ""

    s = symbol.strip().upper()
    if not s:
        return ""

    s = s.replace("-PERP", "").replace("_PERP", "")
    if "/" in s:
        return s

    if s.endswith("USDT") and len(s) > 4:
        return f"{s[:-4]}/USDT"

    return f"{s}/USDT"


__all__ = ["_normalize_symbol", "reconcile_orders_with_binance"]
