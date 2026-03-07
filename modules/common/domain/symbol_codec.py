from __future__ import annotations

import re
from typing import Optional

import ccxt

from .symbol_types import CcxtSymbol, DbSymbol, FuturesSymbol

QUOTE_CURRENCIES = ("USDT", "BUSD", "USDC", "FDUSD", "BTC", "ETH", "BNB")


class SymbolCodec:
    """
    Single source of truth for all symbol format conversions.

    Usage:
        codec = SymbolCodec(exchange)          # exchange-aware (best)
        codec = SymbolCodec()                  # exchange-free (heuristic fallback)

        db  = codec.to_db("BTC/USDT:USDT")    # → DbSymbol("BTCUSDT")
        fut = codec.to_futures("BTCUSDT")      # → FuturesSymbol("BTC/USDT:USDT")
        ccx = codec.to_ccxt("BTCUSDT")         # → CcxtSymbol("BTC/USDT")
    """

    def __init__(self, exchange: Optional[ccxt.Exchange] = None):
        self._exchange = exchange

    def to_db(self, symbol: str) -> DbSymbol:
        """Any format → DB canonical format (e.g. 'BTCUSDT')."""
        root = symbol.split(":", 1)[0]
        normalized = re.sub(r"[^A-Za-z0-9]", "", root).upper()
        return DbSymbol(normalized)

    def to_ccxt(self, symbol: str) -> CcxtSymbol:
        """Any format → CCXT spot format (e.g. 'BTC/USDT')."""
        db = self.to_db(symbol)
        for quote in QUOTE_CURRENCIES:
            if db.endswith(quote) and len(db) > len(quote):
                return CcxtSymbol(f"{db[:-len(quote)]}/{quote}")
        return CcxtSymbol(db)

    def to_futures(self, symbol: str) -> FuturesSymbol:
        """Any format → CCXT futures format (e.g. 'BTC/USDT:USDT')."""
        if self._exchange:
            return self._to_futures_via_exchange(symbol)
        return self._to_futures_heuristic(symbol)

    def _to_futures_via_exchange(self, symbol: str) -> FuturesSymbol:
        assert self._exchange is not None
        try:
            if not getattr(self._exchange, "markets", None):
                self._exchange.load_markets()
            market = self._exchange.market(symbol)
            if market and market.get("symbol"):
                return FuturesSymbol(str(market["symbol"]))
        except Exception:
            pass

        raw_id = self.to_db(symbol)
        try:
            by_id = getattr(self._exchange, "markets_by_id", {}) or {}
            candidates = by_id.get(raw_id)
            if isinstance(candidates, list) and candidates:
                candidates = candidates[0]
            if isinstance(candidates, dict) and candidates.get("symbol"):
                return FuturesSymbol(str(candidates["symbol"]))
        except Exception:
            pass

        return self._to_futures_heuristic(symbol)

    def _to_futures_heuristic(self, symbol: str) -> FuturesSymbol:
        ccxt_sym = self.to_ccxt(symbol)
        if "/" in ccxt_sym and ":" not in ccxt_sym:
            quote = ccxt_sym.split("/")[1]
            return FuturesSymbol(f"{ccxt_sym}:{quote}")
        return FuturesSymbol(ccxt_sym)

    def equal(self, a: str, b: str) -> bool:
        """True if two symbols refer to the same asset, regardless of format."""
        return self.to_db(a) == self.to_db(b)

    @staticmethod
    def sanitize_for_filename(symbol: str) -> str:
        """Return a filesystem-safe token from a symbol string."""
        return (symbol or "").replace("/", "_").replace(":", "_")
