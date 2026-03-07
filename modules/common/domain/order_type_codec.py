from __future__ import annotations

from typing import Literal

OrderKind = Literal["tp", "sl", "market", "limit", "unknown"]

_CONDITIONAL_TYPES = frozenset(
    {
        "STOP_MARKET",
        "TAKE_PROFIT_MARKET",
        "STOP",
        "TAKE_PROFIT",
        "STOP_LOSS",
        "STOP_LOSS_LIMIT",
    }
)


class BinanceOrderType:
    """
    Resolve the TRUE Binance order type from a CCXT order dict.

    CCXT normalizes STOP_MARKET → 'market' at top level.
    The real type lives in order['info']['type'] or order['info']['origType'].
    """

    @staticmethod
    def resolve(order: dict) -> str:
        """Return the authoritative Binance order type string (e.g. 'STOP_MARKET')."""
        info = order.get("info") or {}
        if not isinstance(info, dict):
            info = {}
        return (info.get("type") or info.get("origType") or order.get("type") or "UNKNOWN").upper()

    @staticmethod
    def is_conditional(order: dict) -> bool:
        """True if this order must be cancelled via params={'stop': True}."""
        raw = BinanceOrderType.resolve(order)
        has_stop_price = bool(
            order.get("stopPrice") or order.get("triggerPrice") or (order.get("info") or {}).get("stopPrice")
        )
        return any(ctype in raw for ctype in _CONDITIONAL_TYPES) or has_stop_price

    @staticmethod
    def classify(
        order: dict,
        entry_price: float = 0.0,
        side: str = "",
    ) -> OrderKind:
        """Classify order as 'tp', 'sl', 'market', 'limit', or 'unknown'.

        Priority:
          1. Explicit Binance type string (TAKE_PROFIT_MARKET → 'tp')
          2. Price-vs-entry fallback when CCXT normalizes to generic 'market'
        """
        raw = BinanceOrderType.resolve(order)

        if "TAKE_PROFIT" in raw:
            return "tp"
        if ("STOP" in raw or "LOSS" in raw) and "TAKE_PROFIT" not in raw:
            return "sl"

        info = order.get("info") or {}
        sp_raw = (
            order.get("stopPrice") or order.get("triggerPrice") or info.get("stopPrice") or info.get("triggerPrice")
        )
        if sp_raw and entry_price > 0 and side:
            try:
                sp = float(sp_raw)
                s = side.lower()
                if s == "long":
                    return "tp" if sp > entry_price else "sl"
                if s == "short":
                    return "tp" if sp < entry_price else "sl"
            except (TypeError, ValueError):
                pass

        return "unknown"

    @staticmethod
    def cancel_params(order: dict) -> dict:
        """Return the correct params dict for exchange.cancel_order().

        Usage:
            exchange.cancel_order(order_id, symbol,
                                  params=BinanceOrderType.cancel_params(order))
        """
        return {"stop": True} if BinanceOrderType.is_conditional(order) else {}
