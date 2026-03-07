"""Compatibility shim for legacy reconcile imports.

Legacy tests import:
    from modules.auto_trade.database.reconcile import _symbol_to_ccxt, reconcile_orders_with_binance
"""

from __future__ import annotations

try:
    import ccxt  # type: ignore
except Exception:  # pragma: no cover - fallback for environments without ccxt
    class _CcxtStub:
        # Keep patch target available: patch("...reconcile.ccxt.binance")
        binance = None

    ccxt = _CcxtStub()  # type: ignore[assignment]

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


def reconcile_orders_with_binance(
    api_key: str,
    api_secret: str,
    testnet: bool = False,
    symbols=None,
    since_hours: int = 24,
    enable_profiling: bool = False,
):
    """Delegate to legacy SQLite reconcile implementation for compatibility tests.

    Important: tests patch ``modules.auto_trade.database.reconcile.ccxt.binance``.
    We forward that patched constructor into the legacy module so mocks are respected.
    """
    from modules.auto_trade.archives.sqlite_legacy.database import reconcile as _legacy_module

    original_binance = getattr(_legacy_module.ccxt, "binance", None)
    if hasattr(_legacy_module.ccxt, "binance"):
        _legacy_module.ccxt.binance = ccxt.binance

    try:
        return _legacy_module.reconcile_orders_with_binance(
            api_key=api_key,
            api_secret=api_secret,
            testnet=testnet,
            symbols=symbols,
            since_hours=since_hours,
            enable_profiling=enable_profiling,
        )
    finally:
        if hasattr(_legacy_module.ccxt, "binance"):
            _legacy_module.ccxt.binance = original_binance

__all__ = ["_normalize_symbol", "_symbol_to_ccxt", "reconcile_orders_with_binance"]
