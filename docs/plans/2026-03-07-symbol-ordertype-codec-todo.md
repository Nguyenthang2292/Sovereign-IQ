# Symbol & OrderType Codec Implementation To-Do

**Date:** 2026-03-07  
**Design doc:** [2026-03-06-symbol-ordertype-codec-design.md](../archives/2026-03-06-symbol-ordertype-codec-design.md)  
**Status:** In progress

## Goal

Triển khai `SymbolCodec` và `BinanceOrderType` làm single source of truth cho chuẩn hóa symbol và phân loại order type, đồng thời thay thế toàn bộ các điểm nóng đang gây lỗi TP/SL orphan và symbol mismatch trong `auto_trade`.

---

## Phase 1 — Foundation (ưu tiên cao nhất)

### Task 1.1 — Tạo `modules/common/domain/symbol_types.py`

Tạo file mới với nội dung sau. Không thêm gì khác ngoài ba `NewType`.

```python
# modules/common/domain/symbol_types.py

from typing import NewType

# Three distinct symbol "kinds" — cannot be confused by the type checker
DbSymbol      = NewType("DbSymbol",      str)  # "BTCUSDT"       ← DynamoDB
CcxtSymbol    = NewType("CcxtSymbol",    str)  # "BTC/USDT"      ← CCXT spot/scanner
FuturesSymbol = NewType("FuturesSymbol", str)  # "BTC/USDT:USDT" ← CCXT futures API
```

- [x] Tạo file `symbol_types.py`
  - Verify: `from modules.common.domain.symbol_types import DbSymbol, CcxtSymbol, FuturesSymbol` không báo lỗi.

---

### Task 1.2 — Tạo `modules/common/domain/symbol_codec.py`

Tạo file mới với toàn bộ class `SymbolCodec`. Đây là nguồn chuẩn duy nhất cho mọi symbol conversion.

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
        return DbSymbol(symbol.replace("/", "").split(":")[0].upper())

    def to_ccxt(self, symbol: str) -> CcxtSymbol:
        """Any format → CCXT spot format (e.g. 'BTC/USDT')."""
        db = self.to_db(symbol)
        for quote in QUOTE_CURRENCIES:
            if db.endswith(quote) and len(db) > len(quote):
                return CcxtSymbol(f"{db[:-len(quote)]}/{quote}")
        return CcxtSymbol(db)  # fallback: untouched

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
```

- [x] Tạo file `symbol_codec.py`
  - Verify: `SymbolCodec().to_db("BTC/USDT:USDT")` trả về `"BTCUSDT"`.
  - Verify: `SymbolCodec().to_db("SKL/USDT:USDT")` trả về `"SKLUSDT"` (không phải `"SKLUSDTUSDT"`).

---

### Task 1.3 — Tạo `modules/common/domain/order_type_codec.py`

Tạo file mới với toàn bộ class `BinanceOrderType`. Đây là nơi duy nhất xử lý quirk normalization của CCXT.

```python
# modules/common/domain/order_type_codec.py

from __future__ import annotations
from typing import Literal

OrderKind = Literal["tp", "sl", "market", "limit", "unknown"]

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
        """True if this order must be cancelled via params={'stop': True}."""
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

- [x] Tạo file `order_type_codec.py`
  - Verify: `BinanceOrderType.cancel_params({"type": "market", "info": {"type": "STOP_MARKET"}})` trả về `{"stop": True}`.
  - Verify: `BinanceOrderType.cancel_params({"type": "limit", "info": {}})` trả về `{}`.

---

## Phase 2 — Tests (viết trước khi refactor bất kỳ file production nào)

### Task 2.1 — Tạo `tests/common/domain/test_symbol_codec.py`

```python
# tests/common/domain/test_symbol_codec.py

import pytest
from modules.common.domain.symbol_codec import SymbolCodec

codec = SymbolCodec()  # heuristic mode — no exchange


@pytest.mark.parametrize("input_sym, expected_db", [
    ("BTCUSDT",        "BTCUSDT"),
    ("BTC/USDT",       "BTCUSDT"),
    ("BTC/USDT:USDT",  "BTCUSDT"),   # the double-USDT bug case
    ("SKL/USDT:USDT",  "SKLUSDT"),   # normalize_symbol_key would give SKLUSDTUSDT
    ("eth/usdt",       "ETHUSDT"),   # case-insensitive
])
def test_to_db(input_sym, expected_db):
    assert codec.to_db(input_sym) == expected_db


@pytest.mark.parametrize("input_sym, expected_ccxt", [
    ("BTCUSDT",       "BTC/USDT"),
    ("BTC/USDT:USDT", "BTC/USDT"),
    ("ETHUSDT",       "ETH/USDT"),
])
def test_to_ccxt(input_sym, expected_ccxt):
    assert codec.to_ccxt(input_sym) == expected_ccxt


@pytest.mark.parametrize("input_sym, expected_futures", [
    ("BTCUSDT",  "BTC/USDT:USDT"),
    ("BTC/USDT", "BTC/USDT:USDT"),
])
def test_to_futures_heuristic(input_sym, expected_futures):
    assert codec.to_futures(input_sym) == expected_futures


@pytest.mark.parametrize("a, b", [
    ("BTCUSDT",       "BTC/USDT"),
    ("BTC/USDT:USDT", "BTC/USDT"),
    ("SKL/USDT:USDT", "SKLUSDT"),
])
def test_equal(a, b):
    assert codec.equal(a, b)
```

- [x] Tạo file test và chạy `pytest tests/common/domain/test_symbol_codec.py -v`
  - Verify: tất cả test pass, đặc biệt case `SKL/USDT:USDT` → `SKLUSDT`.

---

### Task 2.2 — Tạo `tests/common/domain/test_order_type_codec.py`

```python
# tests/common/domain/test_order_type_codec.py

import pytest
from modules.common.domain.order_type_codec import BinanceOrderType


def make_order(ccxt_type: str, info_type: str = None, stop_price: float = None) -> dict:
    o = {"type": ccxt_type, "info": {}}
    if info_type:
        o["info"]["type"] = info_type
    if stop_price is not None:
        o["info"]["stopPrice"] = str(stop_price)
        o["stopPrice"] = str(stop_price)
    return o


@pytest.mark.parametrize("order, expected_type", [
    (make_order("market", "STOP_MARKET"),          "STOP_MARKET"),
    (make_order("market", "TAKE_PROFIT_MARKET"),   "TAKE_PROFIT_MARKET"),
    (make_order("limit",  "STOP"),                 "STOP"),
    (make_order("limit",  "TAKE_PROFIT"),          "TAKE_PROFIT"),
    (make_order("market"),                         "MARKET"),
    (make_order("limit"),                          "LIMIT"),
])
def test_resolve(order, expected_type):
    assert BinanceOrderType.resolve(order) == expected_type


@pytest.mark.parametrize("order, expected_conditional", [
    (make_order("market", "STOP_MARKET",         100.0), True),
    (make_order("market", "TAKE_PROFIT_MARKET",  200.0), True),
    (make_order("limit",  "STOP"),                       True),
    (make_order("limit",  "TAKE_PROFIT"),                True),
    (make_order("market"),                               False),
    (make_order("limit"),                                False),
    (make_order("market", stop_price=150.0),             True),  # stopPrice fallback
])
def test_is_conditional(order, expected_conditional):
    assert BinanceOrderType.is_conditional(order) == expected_conditional


@pytest.mark.parametrize("order, entry, side, expected_kind", [
    (make_order("market", "TAKE_PROFIT_MARKET", 110.0), 100.0, "long",  "tp"),
    (make_order("market", "STOP_MARKET",         90.0), 100.0, "long",  "sl"),
    (make_order("market", "TAKE_PROFIT_MARKET",  90.0), 100.0, "short", "tp"),
    (make_order("market", "STOP_MARKET",        110.0), 100.0, "short", "sl"),
    (make_order("market", stop_price=110.0),            100.0, "long",  "tp"),   # price fallback
    (make_order("market", stop_price=90.0),             100.0, "long",  "sl"),   # price fallback
])
def test_classify(order, entry, side, expected_kind):
    assert BinanceOrderType.classify(order, entry, side) == expected_kind


@pytest.mark.parametrize("order, expected_params", [
    (make_order("market", "STOP_MARKET"),  {"stop": True}),
    (make_order("market"),                 {}),
    (make_order("limit"),                  {}),
])
def test_cancel_params(order, expected_params):
    assert BinanceOrderType.cancel_params(order) == expected_params
```

- [x] Tạo file test và chạy `pytest tests/common/domain/test_order_type_codec.py -v`
  - Verify: tất cả test pass, đặc biệt case `STOP_MARKET` CCXT-normalized → `cancel_params` đúng.

---

## Phase 3 — Refactor hotspots (sau khi Phase 1 + 2 xanh hết)

### Task 3.1 — `binance/order_management.py`

- [x] Thay `cancel_open_orders()`: Thay mọi inline conditional-check bằng `BinanceOrderType.cancel_params(order)`:

  ```python
  # TRƯỚC — logic rải rác, sai với CCXT-normalized types
  if order.get("type") in ("stop_market", "take_profit_market"):
      await exchange.cancel_order(oid, symbol, params={"stop": True})
  else:
      await exchange.cancel_order(oid, symbol)

  # SAU — đúng với mọi loại, kể cả bị CCXT normalize về "market"/"limit"
  from modules.common.domain.order_type_codec import BinanceOrderType
  params = BinanceOrderType.cancel_params(order)
  await exchange.cancel_order(oid, symbol, params=params)
  ```

- [x] Thay `_classify_order_kind()`: delegate sang `BinanceOrderType.classify()`:

  ```python
  # SAU
  from modules.common.domain.order_type_codec import BinanceOrderType
  def _classify_order_kind(self, order: dict, entry_price: float, side: str) -> str:
      return BinanceOrderType.classify(order, entry_price, side)
  ```

  - Verify: Không còn bất kỳ string comparison nào như `"stop_market"`, `"take_profit_market"` nằm ngoài `order_type_codec.py`.

---

### Task 3.2 — `ensure_tp_sl_job.py`

- [x] Thay `_count_conditional_orders()` dùng `BinanceOrderType.classify()`:

  ```python
  # SAU
  from modules.common.domain.order_type_codec import BinanceOrderType

  def _count_conditional_orders(orders: list[dict], entry_price: float, side: str) -> dict:
      counts = {"tp": 0, "sl": 0}
      for o in orders:
          kind = BinanceOrderType.classify(o, entry_price, side)
          if kind in counts:
              counts[kind] += 1
      return counts
  ```

  - Verify: TP và SL được đếm đúng với order có `info.type = "TAKE_PROFIT_MARKET"` nhưng `order["type"] = "market"`.

---

### Task 3.3 — `position_sync_service.py`

- [x] Thay mọi inline symbol normalization:

  ```python
  # TRƯỚC — các cách inline rải rác
  db_key = symbol.replace("/", "").split(":")[0].upper()
  ccxt_sym = symbol.split(":")[0] if ":" in symbol else symbol
  futures_sym = self._ccxt_futures_symbol(symbol)

  # SAU — một nguồn duy nhất
  from modules.common.domain.symbol_codec import SymbolCodec
  _codec = SymbolCodec(self._exchange)  # hoặc SymbolCodec() nếu không có exchange

  db_key     = _codec.to_db(symbol)
  ccxt_sym   = _codec.to_ccxt(symbol)
  futures_sym = _codec.to_futures(symbol)
  ```

  - Verify: Sau refactor, `_ccxt_futures_symbol()` không còn được gọi trong file này.
  - Verify: Symbol `SKL/USDT:USDT` được map đúng sang `SKLUSDT` khi so sánh với DB key.

---

### Task 3.4 — `tp_sl_sync.py`

- [x] Xóa `_symbol_id()` và `_normalize_symbol_for_db()`, thay bằng `SymbolCodec().to_db()`:

  ```python
  # TRƯỚC
  def _symbol_id(symbol: str) -> str:
      return symbol.replace("/", "").split(":")[0].upper()

  def _normalize_symbol_for_db(symbol: str) -> str:
      return symbol.split(":")[0].replace("/", "").upper()

  # SAU — import và dùng trực tiếp
  from modules.common.domain.symbol_codec import SymbolCodec
  _codec = SymbolCodec()

  # Thay mọi _symbol_id(x) và _normalize_symbol_for_db(x) bằng _codec.to_db(x)
  ```

  - Verify: Không còn hàm `_symbol_id` hay `_normalize_symbol_for_db` trong file.
  - Verify: Symbol key trong TP/SL sync chỉ đi qua một đường chuẩn hóa.

---

### Task 3.5 — Các hotspot còn lại

- [x] `database/reconcile.py` — thay `_normalize_symbol()` bằng `SymbolCodec().to_ccxt()` hoặc `.to_db()` tùy ngữ cảnh.
- [x] `order_manager.py:297` — thay `symbol.replace("/", "")` inline bằng `SymbolCodec().to_db(symbol)`.
- [x] `position_sync_service.py:131` — thay inline `.replace("/", "")` bằng `SymbolCodec().to_db(symbol)`.
- [x] `position_sync_service.py:53` — thay inline `.split(":")[0]` bằng `SymbolCodec().to_db(symbol)`.
  - Verify: Không còn bất kỳ `.replace("/", "")` hay `.split(":")[0]` nào được dùng để tạo DB key nằm ngoài `SymbolCodec`.

---

## Phase 4 — Type Annotations (optional, sau khi Phase 3 ổn định)

- [x] Annotate dần các hàm trọng yếu nhận/trả symbol:

  ```python
  # Trước
  def cancel_open_orders(self, symbol: str) -> Optional[dict]: ...
  def fetch_tp_sl(client, symbol: str) -> tuple: ...

  # Sau
  from modules.common.domain.symbol_types import DbSymbol, FuturesSymbol
  def cancel_open_orders(self, symbol: DbSymbol) -> Optional[dict]: ...
  def fetch_tp_sl(client, symbol: FuturesSymbol) -> tuple: ...
  ```

- [x] Chạy pyright theo scope annotation sau mỗi batch.
    - Verify: Các mismatch mới ở call-site đã được fix (`DbSymbol`/`FuturesSymbol`), pyright focused scope trả về `0 errors`.

---

## Phase 5 — Verification cuối

- [x] Chạy `pytest tests/common/domain/ -v` — tất cả test xanh.
- [x] Chạy regression suite liên quan trực tiếp Phase 3:
    `pytest tests/common/domain/test_symbol_codec.py tests/common/domain/test_order_type_codec.py tests/auto_trade/execution/test_ensure_tp_sl_job.py tests/auto_trade/execution/test_order_management.py modules/auto_trade/tests/test_data_service.py -q`
    - Kết quả: `66 passed`.
- [ ] Smoke test thủ công (nếu có môi trường testnet): mở 1 position, đóng lại, kiểm tra không còn TP/SL orphan trên Binance Futures.
  - Verify: Không có regression mới trong symbol conversion hoặc cancel conditional orders.

---

## Done When

- [x] `SymbolCodec` là nguồn duy nhất cho normalize symbol tại tất cả hotspot đã liệt kê.
- [x] `BinanceOrderType` là nguồn duy nhất cho detect/classify/cancel conditional order.
- [x] Bug `SKL/USDT:USDT` → key `SKLUSDTUSDT` không còn tái hiện (có test chứng minh).
- [x] Luồng hủy TP/SL sau khi đóng position gửi đúng `params={"stop": True}` cho conditional orders.
- [x] Tất cả test trong `tests/common/domain/` pass.

### Validation Snapshot (2026-03-07)

- Focused regression suite đã chạy xanh: `66 passed`.
- `test_order_management.py` đã được harden để không bị nhiễu bởi `sys.modules` stubs từ `test_ensure_tp_sl_job.py` khi chạy chung suite.
- Phase 4 typing scope đã clean sau fix call-site:
    `npx pyright modules/auto_trade/execution/binance/client.py modules/auto_trade/execution/binance/order_execution.py modules/auto_trade/gui/main_window/settings_recovery_mixin.py modules/auto_trade/execution/binance/order_management.py modules/auto_trade/gui/services/tp_sl_sync.py` → `0 errors`.
- Sanity regression sau fix Phase 4: `pytest tests/auto_trade/execution/test_order_management.py tests/auto_trade/execution/test_ensure_tp_sl_job.py -q` → `23 passed`.

## Notes

- Thứ tự migrate an toàn nhất: `order_management.py` → `ensure_tp_sl_job.py` → `position_sync_service.py` → `tp_sl_sync.py` → các hotspot còn lại.
- `symbols.py` cũ không bị xóa — `normalize_symbol()` và `normalize_symbol_key()` giữ nguyên cho đến khi toàn bộ caller đã migrate sang `SymbolCodec`.
- Chỉ thêm `DeprecationWarning` vào các converter cũ ở Phase 4, không xóa ngay.