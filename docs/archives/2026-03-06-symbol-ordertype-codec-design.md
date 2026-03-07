# Symbol & OrderType Codec Design

**Date:** 2026-03-06  
**Status:** Approved for implementation  
**Context:** Resolve symbol format mismatch (CCXT vs Binance vs DB) and Binance conditional
order type normalization issues throughout the `auto_trade` module.

---

## Problem Statement

### Symbol Mismatch

Six different conversion approaches exist in the codebase with no single source of truth:

| Format | Example | Used at |
|--------|---------|---------|
| **DB format** | `BTCUSDT` | DynamoDB, `order_manager.py:297`, `position_sync_service.py` |
| **Binance raw** | `BTCUSDT` | `info.symbol` in CCXT responses |
| **CCXT slash** | `BTC/USDT` | `normalize_symbol()`, scanner, signal pipeline |
| **CCXT futures** | `BTC/USDT:USDT` | `_ccxt_futures_symbol()`, `fetch_open_orders`, `fetch_positions` |
| **`normalize_symbol_key`** | `BTCUSDT` | Comparison key — **BUG**: `SKL/USDT:USDT` → `SKLUSDTUSDT` doubles USDT |

**Scattered converters (6 different implementations):**

```
normalize_symbol()          → BTC/USDT       (common/domain/symbols.py)
normalize_symbol_key()      → BTCUSDT        (common/domain/symbols.py) ← BUG: SKL/USDT:USDT → SKLUSDTUSDT
_ccxt_futures_symbol()      → BTC/USDT:USDT  (binance/order_management.py)
_normalize_symbol()         → BTC/USDT       (database/reconcile.py)
_symbol_id()                → BTCUSDT        (tp_sl_sync.py)
_normalize_symbol_for_db()  → BTCUSDT        (tp_sl_sync.py)
inline: .replace("/","")    → BTCUSDT        (order_manager.py:297, position_sync.py:131...)
inline: .split(":")[0]...   → BTCUSDT        (position_sync.py:53, tp_sl_sync.py:22...)
```

### OrderType Mismatch

CCXT normalizes Binance conditional order types at the top-level `type` field:

| Binance API type | `info.type` / `info.origType` | CCXT top-level `type` | Cancel needs |
|---|---|---|---|
| `STOP_MARKET` | `STOP_MARKET` | `market` ❌ | `params={'stop': True}` |
| `TAKE_PROFIT_MARKET` | `TAKE_PROFIT_MARKET` | `market` ❌ | `params={'stop': True}` |
| `STOP` | `STOP` | `limit` ❌ | `params={'stop': True}` |
| `TAKE_PROFIT` | `TAKE_PROFIT` | `limit` ❌ | `params={'stop': True}` |
| `MARKET` | `MARKET` | `market` ✅ | not needed |
| `LIMIT` | `LIMIT` | `limit` ✅ | not needed |

This caused conditional TP/SL orders to **not be cancelled** when a position was closed,
leaving orphaned orders open on Binance.

---

## Solution: A + C Combined

Combine **Centralized Symbol Codec** (A) with **Type-safe Symbol Types** (C):

- Zero runtime overhead via Python `NewType`
- Single codec class for all conversions
- Type checker (mypy/pyright) catches format mismatches at development time
- Backward-compatible gradual migration

---

## Design Section 1: Type-Safe Symbol Types

Use Python `NewType` — zero runtime overhead, only exists at type-checking layer.
No wrap/unwrap needed; just cast once at the entry point.

```python
# modules/common/domain/symbol_types.py

from typing import NewType

# Three distinct symbol "kinds" — can no longer be confused by the type checker
DbSymbol      = NewType("DbSymbol",      str)  # "BTCUSDT"       ← DynamoDB
CcxtSymbol    = NewType("CcxtSymbol",    str)  # "BTC/USDT"      ← CCXT spot/scanner
FuturesSymbol = NewType("FuturesSymbol", str)  # "BTC/USDT:USDT" ← CCXT futures API
```

**Benefits:**

- mypy/pyright will error if you pass `DbSymbol` where `FuturesSymbol` is expected
- IDEs show which symbol format is in use at a glance
- `NewType` is an identity function at runtime — no performance cost

---

## Design Section 2: SymbolCodec — Single Source of Truth

One codec class with exchange-aware conversion:

```python
# modules/common/domain/symbol_codec.py

from __future__ import annotations
from typing import Optional
import ccxt

from .symbol_types import DbSymbol, CcxtSymbol, FuturesSymbol


QUOTE_CURRENCIES = ("USDT", "BUSD", "USDC", "FDUSD", "BTC", "ETH", "BNB")


class SymbolCodec:
    """
    Single source of truth for all symbol format conversions.

    Entry point: any string → typed symbol.
    All downstream code uses typed symbols to avoid silent format errors.

    Usage:
        codec = SymbolCodec(exchange)          # exchange-aware (best)
        codec = SymbolCodec()                  # exchange-free (regex fallback)

        db  = codec.to_db("BTC/USDT:USDT")    # → DbSymbol("BTCUSDT")
        fut = codec.to_futures("BTCUSDT")      # → FuturesSymbol("BTC/USDT:USDT")
        ccx = codec.to_ccxt("BTCUSDT")         # → CcxtSymbol("BTC/USDT")
    """

    def __init__(self, exchange: Optional[ccxt.Exchange] = None):
        self._exchange = exchange

    # ── Canonical conversions ─────────────────────────────────────────────

    def to_db(self, symbol: str) -> DbSymbol:
        """Any format → DB canonical format (e.g. 'BTCUSDT')."""
        return DbSymbol(symbol.replace("/", "").split(":")[0].upper())

    def to_ccxt(self, symbol: str) -> CcxtSymbol:
        """Any format → CCXT spot format (e.g. 'BTC/USDT')."""
        db = self.to_db(symbol)  # BTCUSDT
        for quote in QUOTE_CURRENCIES:
            if db.endswith(quote) and len(db) > len(quote):
                return CcxtSymbol(f"{db[:-len(quote)]}/{quote}")
        return CcxtSymbol(db)  # fallback: untouched

    def to_futures(self, symbol: str) -> FuturesSymbol:
        """Any format → CCXT futures format (e.g. 'BTC/USDT:USDT').

        Uses exchange.market() if exchange is available (most accurate).
        Falls back to regex heuristic.
        """
        if self._exchange:
            return self._to_futures_via_exchange(symbol)
        return self._to_futures_heuristic(symbol)

    def _to_futures_via_exchange(self, symbol: str) -> FuturesSymbol:
        """Use exchange market registry — most reliable path."""
        assert self._exchange is not None
        try:
            if not getattr(self._exchange, "markets", None):
                self._exchange.load_markets()
            market = self._exchange.market(symbol)
            if market and market.get("symbol"):
                return FuturesSymbol(str(market["symbol"]))
        except Exception:
            pass

        # Try by Binance raw id (e.g. "BTCUSDT")
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
        """Regex fallback when no exchange available."""
        ccxt_sym = self.to_ccxt(symbol)
        if "/" in ccxt_sym and ":" not in ccxt_sym:
            quote = ccxt_sym.split("/")[1]
            return FuturesSymbol(f"{ccxt_sym}:{quote}")
        return FuturesSymbol(ccxt_sym)

    # ── Comparison helper ─────────────────────────────────────────────────

    def equal(self, a: str, b: str) -> bool:
        """True if two symbols refer to the same asset, regardless of format."""
        return self.to_db(a) == self.to_db(b)
```

---

## Design Section 3: BinanceOrderTypeCodec

Resolves CCXT order type normalization once and for all:

```python
# modules/common/domain/order_type_codec.py

from __future__ import annotations
from typing import Literal

# Type-safe order kinds
OrderKind = Literal["tp", "sl", "market", "limit", "unknown"]

# Binance conditional order types that need params={'stop': True} for cancel
_CONDITIONAL_TYPES = frozenset({
    "STOP_MARKET", "TAKE_PROFIT_MARKET",
    "STOP", "TAKE_PROFIT",
    "STOP_LOSS", "STOP_LOSS_LIMIT",
})


class BinanceOrderType:
    """
    Resolve the TRUE Binance order type from a CCXT order dict.

    CCXT normalizes STOP_MARKET → 'market' at top level.
    The real type lives in order['info']['type'] or order['info']['origType'].

    This is the single place in the codebase that knows about this CCXT
    normalization quirk. All other code should delegate to this class.
    """

    @staticmethod
    def resolve(order: dict) -> str:
        """Return the authoritative Binance order type string (e.g. 'STOP_MARKET')."""
        info = order.get("info") or {}
        if not isinstance(info, dict):
            info = {}
        return (
            info.get("type")
            or info.get("origType")
            or order.get("type")
            or "UNKNOWN"
        ).upper()

    @staticmethod
    def is_conditional(order: dict) -> bool:
        """True if this order must be cancelled via params={'stop': True}.

        Conditional orders on Binance Futures live on a separate endpoint
        from basic orders. CCXT requires the {'stop': True} hint to route
        the cancel request to the correct endpoint.
        """
        raw = BinanceOrderType.resolve(order)
        has_stop_price = bool(
            order.get("stopPrice") or order.get("triggerPrice")
            or (order.get("info") or {}).get("stopPrice")
        )
        return (
            any(ctype in raw for ctype in _CONDITIONAL_TYPES)
            or has_stop_price
        )

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

        # Fallback: classify via stopPrice vs entry price
        info = order.get("info") or {}
        sp_raw = (
            order.get("stopPrice") or order.get("triggerPrice")
            or info.get("stopPrice") or info.get("triggerPrice")
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
```

**Key API:** `cancel_params(order)` is the ONLY thing callers need — they don't need to
know whether the order is conditional or not.

---

## Design Section 4: Migration Plan

### Backward-compatible gradual migration — no big-bang rewrite

**File structure (new files only):**

```
modules/common/domain/
├── symbols.py              (existing — keep normalize_symbol, normalize_symbol_key)
├── symbol_types.py         (NEW — NewType definitions)
├── symbol_codec.py         (NEW — SymbolCodec class)
└── order_type_codec.py     (NEW — BinanceOrderType class)
```

### Phase 1 — Foundation (immediate)

- [ ] Create `symbol_types.py` with `DbSymbol`, `CcxtSymbol`, `FuturesSymbol`
- [ ] Create `symbol_codec.py` with `SymbolCodec` class
- [ ] Create `order_type_codec.py` with `BinanceOrderType` class
- [ ] Add unit tests for all three modules

### Phase 2 — Fix critical hotspots first

Replace the most bug-prone usages before anything else:

| File | Change |
|------|--------|
| `binance/order_management.py` | `cancel_open_orders()` → `BinanceOrderType.cancel_params(order)` |
| `binance/order_management.py` | `_classify_order_kind()` → delegate to `BinanceOrderType.classify()` |
| `ensure_tp_sl_job.py` | `_count_conditional_orders()` → `BinanceOrderType.classify()` |
| `position_sync_service.py` | inline `replace("/","").split(":")[0]` → `SymbolCodec().to_db()` |
| `tp_sl_sync.py` | `_symbol_id()`, `_normalize_symbol_for_db()` → `SymbolCodec().to_db()` |
| `position_sync_service.py` | `_ccxt_futures_symbol()` call → `SymbolCodec(exchange).to_futures()` |

### Phase 3 — Annotate function signatures gradually

```python
# Before
def cancel_open_orders(self, symbol: str) -> Optional[dict]: ...
def fetch_tp_sl(client, symbol: str) -> tuple: ...

# After
def cancel_open_orders(self, symbol: DbSymbol) -> Optional[dict]: ...
def fetch_tp_sl(client, symbol: FuturesSymbol) -> tuple: ...
```

### Phase 4 — Deprecate old converters (optional)

Once all hotspots are migrated:

- Add `DeprecationWarning` to `_ccxt_futures_symbol()`, `_symbol_id()`, `_normalize_symbol_for_db()`
- Run mypy in strict mode on the `auto_trade` module
- Remove deprecated functions after all callers are updated

### Known Issues Fixed by This Design

| Bug | Root cause | Fixed by |
|-----|-----------|---------|
| Conditional TP/SL not cancelled after position close | `cancel_order()` missing `params={'stop': True}` | `BinanceOrderType.cancel_params()` |
| `normalize_symbol_key("SKL/USDT:USDT")` → `SKLUSDTUSDT` (double USDT) | Naive `strip separators` logic | `SymbolCodec.to_db()` uses `split(":")[0]` first |
| `get_position()` misses positions with double-USDT key | Upstream `normalize_symbol_key` bug | Fixed by `SymbolCodec.to_db()` |
| `cancel_open_orders(db_symbol)` fails silently for conditional orders | CCXT type normalization + wrong endpoint | `BinanceOrderType.is_conditional()` + `cancel_params()` |
| 6 different symbol normalization functions with subtle differences | No single source of truth | `SymbolCodec` replaces all |

---

## Testing Strategy

```python
# tests/common/domain/test_symbol_codec.py

import pytest
from modules.common.domain.symbol_codec import SymbolCodec

codec = SymbolCodec()  # no exchange — heuristic mode

@pytest.mark.parametrize("input_sym, expected_db", [
    ("BTCUSDT",        "BTCUSDT"),
    ("BTC/USDT",       "BTCUSDT"),
    ("BTC/USDT:USDT",  "BTCUSDT"),   # ← the double-USDT bug case
    ("SKL/USDT:USDT",  "SKLUSDT"),   # ← normalize_symbol_key would give SKLUSDTUSDT
    ("eth/usdt",       "ETHUSDT"),
])
def test_to_db(input_sym, expected_db):
    assert codec.to_db(input_sym) == expected_db

@pytest.mark.parametrize("a, b", [
    ("BTCUSDT",       "BTC/USDT"),
    ("BTC/USDT:USDT", "BTC/USDT"),
    ("SKL/USDT:USDT", "SKLUSDT"),
])
def test_equal(a, b):
    assert codec.equal(a, b)


# tests/common/domain/test_order_type_codec.py

from modules.common.domain.order_type_codec import BinanceOrderType

def make_order(ccxt_type, info_type=None, stop_price=None):
    o = {"type": ccxt_type, "info": {}}
    if info_type:
        o["info"]["type"] = info_type
    if stop_price:
        o["info"]["stopPrice"] = stop_price
        o["stopPrice"] = stop_price
    return o

@pytest.mark.parametrize("order, expected_conditional", [
    (make_order("market", "STOP_MARKET", 100.0),        True),
    (make_order("market", "TAKE_PROFIT_MARKET", 200.0), True),
    (make_order("market"),                              False),
    (make_order("limit"),                               False),
    (make_order("market", stop_price=150.0),            True),  # stopPrice fallback
])
def test_is_conditional(order, expected_conditional):
    assert BinanceOrderType.is_conditional(order) == expected_conditional

@pytest.mark.parametrize("order, entry, side, expected_kind", [
    (make_order("market", "TAKE_PROFIT_MARKET", 110.0), 100.0, "long",  "tp"),
    (make_order("market", "STOP_MARKET",         90.0), 100.0, "long",  "sl"),
    (make_order("market", stop_price=110.0),            100.0, "long",  "tp"),  # fallback
    (make_order("market", stop_price=90.0),             100.0, "long",  "sl"),  # fallback
])
def test_classify(order, entry, side, expected_kind):
    assert BinanceOrderType.classify(order, entry, side) == expected_kind
```
