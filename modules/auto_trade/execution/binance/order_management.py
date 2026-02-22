"""
Order Management Module

Handles TP/SL modification and order cancellation.
"""

from typing import Any, Optional, cast

import ccxt

from modules.common.ui.logging import log_error, log_info, log_warn


def _log_sl_error(symbol: str, stop_loss_price: Optional[float], e: Exception) -> None:
    """Log SL modification failure; treat Binance -2021 with a clear message."""
    msg = str(e).strip()
    if "-2021" in msg or "immediately trigger" in msg.lower():
        log_warn(
            f"SL not set for {symbol}: order would trigger immediately. "
            f"Ensure SL is below mark for LONG and above mark for SHORT (Binance -2021)."
        )
    else:
        log_error(f"Failed to modify SL: {e}")


def _classify_order_kind(order: dict, entry_price: float = 0.0, side: str = "") -> str:
    """
    Classify a conditional order as 'tp', 'sl', or 'unknown'.

    Strategy:
    1. Try explicit type string first (TAKE_PROFIT_MARKET, STOP_MARKET, etc.).
    2. When CCXT normalises the type to 'market' / '', fall back to price comparison:
       - LONG:  stopPrice > entry → TP,  stopPrice <= entry → SL
       - SHORT: stopPrice < entry → TP,  stopPrice >= entry → SL

    Args:
        order:       CCXT order dict.
        entry_price: Position entry price (for fallback classification).
        side:        'long' or 'short' (for fallback classification).

    Returns:
        'tp' | 'sl' | 'unknown'
    """
    info = order.get("info") or {}
    if not isinstance(info, dict):
        info = {}

    otype = (info.get("type") or info.get("origType") or order.get("type") or "").upper()

    if "TAKE_PROFIT" in otype:
        return "tp"
    if ("STOP" in otype or "LOSS" in otype) and "TAKE_PROFIT" not in otype:
        return "sl"

    # ── Fallback: type is ambiguous (CCXT normalises to 'MARKET') ───────────
    sp_raw = order.get("stopPrice") or order.get("triggerPrice") or info.get("stopPrice") or info.get("triggerPrice")
    if not sp_raw:
        return "unknown"  # no stopPrice → not a conditional order at all

    try:
        sp = float(sp_raw)
    except (TypeError, ValueError):
        return "unknown"

    if entry_price > 0 and side:
        s = side.lower()
        if s == "long":
            return "tp" if sp > entry_price else "sl"
        if s == "short":
            return "tp" if sp < entry_price else "sl"

    # No context — can't distinguish safely; flag as 'unknown' so caller skips
    return "unknown"


def _get_mark_price_from_exchange(exchange: ccxt.binance, symbol: str) -> Optional[float]:
    """Fetch current mark price for symbol (futures). Returns None if unavailable."""
    try:
        ticker = cast(dict, exchange.fetch_ticker(symbol))
        info = ticker.get("info")
        if isinstance(info, dict) and info.get("markPrice") is not None:
            return float(info["markPrice"])
        last = ticker.get("last")
        if last is not None:
            return float(last)
        return None
    except Exception:
        return None


def _ccxt_futures_symbol(exchange: ccxt.binance, symbol: str) -> str:
    """
    Return the symbol format CCXT/Binance futures uses so fetch_open_orders finds orders.
    With defaultType 'future', markets are often BASE/QUOTE:USDT (e.g. SKL/USDT:USDT).
    If we pass only SKL/USDT, fetch_open_orders may return [] and we'd never cancel existing SL.
    """
    if not symbol:
        return symbol

    try:
        if not getattr(exchange, "markets", None):
            exchange.load_markets()
    except Exception:
        pass

    try:
        market = exchange.market(symbol)
        if market and market.get("symbol"):
            return str(market["symbol"])
    except Exception:
        pass

    normalized_id = symbol.replace("/", "").split(":")[0].upper()
    try:
        by_id = getattr(exchange, "markets_by_id", {}) or {}
        market_by_id = by_id.get(normalized_id)
        if isinstance(market_by_id, list) and market_by_id:
            first_market = market_by_id[0]
            if isinstance(first_market, dict) and first_market.get("symbol"):
                return str(first_market["symbol"])
        if isinstance(market_by_id, dict) and market_by_id.get("symbol"):
            return str(market_by_id["symbol"])
    except Exception:
        pass

    if ":" not in symbol and "/" in symbol:
        return f"{symbol}:USDT"

    if ":" not in symbol and "/" not in symbol:
        for quote in ("USDT", "BUSD", "USDC", "FDUSD", "BTC", "ETH", "BNB"):
            if normalized_id.endswith(quote) and len(normalized_id) > len(quote):
                base = normalized_id[: -len(quote)]
                return f"{base}/{quote}:{quote}"

    return symbol


def _fetch_all_open_orders(exchange: ccxt.binance, symbol: str) -> list:
    """Fetch BOTH Basic AND Conditional (stop) open orders for a symbol.

    Binance Futures separates orders into two categories:
      - Basic orders:       regular limit/market orders
      - Conditional orders: STOP_MARKET, TAKE_PROFIT_MARKET, STOP_LOSS, etc.

    CCXT's ``fetch_open_orders()`` by default only returns Basic orders.
    To get Conditional orders, we must pass ``params={'stop': True}``.

    Failure to fetch both means TP/SL detection fails, causing the
    EnsureTPSL job to flood Binance with duplicate conditional orders.

    Args:
        exchange: CCXT Binance exchange instance.
        symbol: Trading symbol in CCXT format (e.g. 'BAND/USDT:USDT').

    Returns:
        Combined list of all open orders (basic + conditional), deduplicated by id.
    """
    all_orders: list = []
    seen_ids: set = set()

    # 1. Basic orders (default)
    try:
        basic_orders = exchange.fetch_open_orders(symbol)
        for o in basic_orders:
            oid = o.get("id")
            if oid and oid not in seen_ids:
                seen_ids.add(oid)
                all_orders.append(o)
    except Exception as e:
        log_warn(f"_fetch_all_open_orders: basic query failed for {symbol}: {e}")

    # 2. Conditional (stop) orders - the critical missing piece
    try:
        stop_orders = exchange.fetch_open_orders(symbol, params={"stop": True})
        for o in stop_orders:
            oid = o.get("id")
            if oid and oid not in seen_ids:
                seen_ids.add(oid)
                all_orders.append(o)
    except Exception as e:
        # Some CCXT versions may not support the 'stop' param - log but don't crash
        log_warn(f"_fetch_all_open_orders: conditional query failed for {symbol}: {e}")

    return all_orders


class OrderManagement:
    """
    Handles TP/SL and order management operations.
    """

    def __init__(self, exchange: ccxt.binance, max_retries: int = 3, retry_delay: float = 1.0, dry_run: bool = False):
        """
        Initialize OrderManagement.

        Args:
            exchange: CCXT exchange instance
            max_retries: Maximum retry attempts
            retry_delay: Initial delay between retries
            dry_run: Simulate operations without executing
        """
        self.exchange = exchange
        self.max_retries = max_retries
        self.retry_delay = retry_delay
        self.dry_run = dry_run

    def _get_mark_price(self, symbol: str) -> Optional[float]:
        """Current mark price for symbol (futures). Returns None if unavailable."""
        return _get_mark_price_from_exchange(self.exchange, symbol)

    def modify_take_profit(
        self, symbol: str, position_id: Optional[str], take_profit_price: Optional[float] = None
    ) -> Optional[dict]:
        """
        Modify take profit order for a position.

        Args:
            symbol: Trading symbol
            position_id: Position ID (if available)
            take_profit_price: New TP price (None to cancel existing TP)

        Returns:
            Order result dict or None if failed
        """
        if self.dry_run:
            if take_profit_price:
                log_info(f"[DRY RUN] Would modify TP for {symbol} to ${take_profit_price:,.2f}")
            else:
                log_info(f"[DRY RUN] Would cancel TP for {symbol}")
            return {"dry_run": True, "symbol": symbol}

        try:
            # 1. Get current position
            from modules.auto_trade.execution.binance.position_management import PositionManagement

            position_mgr = PositionManagement(self.exchange, self.dry_run)
            position: Optional[dict] = position_mgr.get_position(symbol)
            if not position:
                log_error(f"No open position found for {symbol}")
                return None

            amount: float = abs(float(position.get("contracts", 0) or position.get("info", {}).get("positionAmt", 0)))
            if amount == 0:
                log_warn(f"Position size is 0 for {symbol}, cannot modify TP")
                return None

            side: str = str(position.get("side") or "")
            if not side:
                # Fallback
                amt: float = float(position.get("info", {}).get("positionAmt", 0))
                side = "long" if amt > 0 else "short"

            entry_price_tp: float = 0.0
            try:
                entry_price_tp = float(position.get("entryPrice") or position.get("info", {}).get("entryPrice", 0) or 0)
            except (TypeError, ValueError):
                pass

            tp_side: str = "sell" if side == "long" else "buy"

            ccxt_symbol_tp: str = _ccxt_futures_symbol(self.exchange, symbol)

            # 2. Cancel existing TP orders only (NOT SL orders).
            # Use _classify_order_kind so CCXT-normalised 'market' typed orders
            # are classified correctly via entry_price+side comparison.
            open_orders: list = _fetch_all_open_orders(self.exchange, ccxt_symbol_tp)
            cancelled_count: int = 0
            log_info(
                f"Cancelling existing TP orders for {symbol} (side={side}, entry={entry_price_tp}, "
                f"{len(open_orders)} total open orders)"
            )

            for order in open_orders:
                kind = _classify_order_kind(order, entry_price_tp, side)
                if kind == "tp":
                    try:
                        self.exchange.cancel_order(order["id"], ccxt_symbol_tp)
                        cancelled_count += 1
                        log_info(f"Cancelled existing TP order: {order['id']}")
                    except Exception as e:
                        log_warn(f"Failed to cancel TP order {order['id']}: {e}")

            # 3. Place new TP order if price provided
            if take_profit_price:
                log_info(f"Setting new TP for {symbol} at ${take_profit_price:,.2f}")

                tp_order = cast(
                    dict,
                    self.exchange.create_order(
                        symbol=ccxt_symbol_tp,
                        type=cast(Any, "TAKE_PROFIT_MARKET"),
                        side=tp_side,
                        amount=amount,
                        params={
                            "stopPrice": take_profit_price,
                            "reduceOnly": True,
                            "workingType": "MARK_PRICE",
                        },
                    ),
                )
                log_info(f"✅ Take Profit order updated at ${take_profit_price:,.2f}")
                return tp_order
            else:
                log_info(f"TP cancelled for {symbol}")
                return {"symbol": symbol, "cancelled_tp_count": cancelled_count}

        except Exception as e:
            log_error(f"Failed to modify TP: {e}")
            return None

    def modify_stop_loss(
        self, symbol: str, position_id: Optional[str], stop_loss_price: Optional[float] = None
    ) -> Optional[dict]:
        """
        Modify stop loss order for a position.

        Args:
            symbol: Trading symbol
            position_id: Position ID (if available)
            stop_loss_price: New SL price (None to cancel existing SL)

        Returns:
            Order result dict or None if failed
        """
        if self.dry_run:
            if stop_loss_price:
                log_info(f"[DRY RUN] Would modify SL for {symbol} to ${stop_loss_price:,.2f}")
            else:
                log_info(f"[DRY RUN] Would cancel SL for {symbol}")
            return {"dry_run": True, "symbol": symbol}

        try:
            # 1. Get current position
            from modules.auto_trade.execution.binance.position_management import PositionManagement

            position_mgr = PositionManagement(self.exchange, self.dry_run)
            position: Optional[dict] = position_mgr.get_position(symbol)
            if not position:
                log_error(f"No open position found for {symbol}")
                return None

            amount: float = abs(float(position.get("contracts", 0) or position.get("info", {}).get("positionAmt", 0)))
            if amount == 0:
                log_warn(f"Position size is 0 for {symbol}, cannot modify SL")
                return None

            side: str = str(position.get("side") or "")
            if not side:
                # Fallback
                amt: float = float(position.get("info", {}).get("positionAmt", 0))
                side = "long" if amt > 0 else "short"

            entry_price_sl: float = 0.0
            try:
                entry_price_sl = float(position.get("entryPrice") or position.get("info", {}).get("entryPrice", 0) or 0)
            except (TypeError, ValueError):
                pass

            sl_side: str = "sell" if side == "long" else "buy"

            # Use CCXT futures symbol so we find and cancel existing conditional orders (avoid duplicates)
            ccxt_symbol: str = _ccxt_futures_symbol(self.exchange, symbol)

            # 2. Cancel existing SL orders only (NOT TP orders).
            # Use _classify_order_kind so CCXT-normalised 'market' typed orders
            # are classified correctly via entry_price+side comparison.
            open_orders: list = _fetch_all_open_orders(self.exchange, ccxt_symbol)
            cancelled_count: int = 0
            log_info(
                f"Cancelling existing SL orders for {symbol} (side={side}, entry={entry_price_sl}, "
                f"{len(open_orders)} total open orders)"
            )

            for order in open_orders:
                kind = _classify_order_kind(order, entry_price_sl, side)
                if kind == "sl":
                    try:
                        self.exchange.cancel_order(order["id"], ccxt_symbol)
                        cancelled_count += 1
                        log_info(f"Cancelled existing SL order: {order['id']}")
                    except Exception as e:
                        log_warn(f"Failed to cancel SL order {order['id']}: {e}")

            # 3. Place new SL order if price provided (validate to avoid -2021 "Order would immediately trigger")
            if stop_loss_price:
                mark_price: Optional[float] = self._get_mark_price(symbol)
                if mark_price is not None and mark_price > 0:
                    would_trigger = (side == "long" and stop_loss_price >= mark_price) or (
                        side == "short" and stop_loss_price <= mark_price
                    )
                    if would_trigger:
                        log_warn(
                            f"Skipping SL at ${stop_loss_price:,.2f} for {symbol} {side}: "
                            f"would trigger immediately (mark=${mark_price:,.2f}). "
                            "For LONG, SL must be below mark; for SHORT, above mark."
                        )
                        return None

                log_info(f"Setting new SL for {symbol} at ${stop_loss_price:,.2f}")

                try:
                    sl_order = cast(
                        dict,
                        self.exchange.create_order(
                            symbol=ccxt_symbol,
                            type=cast(Any, "STOP_MARKET"),
                            side=sl_side,
                            amount=amount,
                            params={
                                "stopPrice": stop_loss_price,
                                "reduceOnly": True,
                                "workingType": "MARK_PRICE",
                            },
                        ),
                    )
                    log_info(f"✅ Stop Loss order updated at ${stop_loss_price:,.2f}")
                    return sl_order
                except Exception as create_err:  # noqa: BLE001
                    _log_sl_error(symbol, stop_loss_price, create_err)
                    return None
            else:
                log_info(f"SL cancelled for {symbol}")
                return {"symbol": symbol, "cancelled_sl_count": cancelled_count}

        except Exception as e:
            _log_sl_error(symbol, stop_loss_price, e)
            return None

    def modify_tp_sl(
        self,
        symbol: str,
        position_id: Optional[str] = None,
        take_profit_price: Optional[float] = None,
        stop_loss_price: Optional[float] = None,
    ) -> Optional[dict]:
        """
        Modify both take profit and stop loss for a position.

        Args:
            symbol: Trading symbol
            position_id: Position ID (if available)
            take_profit_price: New TP price (None to keep current)
            stop_loss_price: New SL price (None to keep current)

        Returns:
            Combined result dict or None if failed
        """
        results: dict = {}

        if take_profit_price is not None:
            tp_result: Optional[dict] = self.modify_take_profit(symbol, position_id, take_profit_price)
            results["tp_result"] = tp_result

        if stop_loss_price is not None:
            sl_result: Optional[dict] = self.modify_stop_loss(symbol, position_id, stop_loss_price)
            results["sl_result"] = sl_result

        return results if results else None

    def cancel_open_orders(self, symbol: str) -> Optional[dict]:
        """
        Cancel all open orders for a symbol (TP, SL, limit orders).

        Args:
            symbol: Trading symbol

        Returns:
            Cancel result dict with success count
        """
        if self.dry_run:
            log_info(f"[DRY RUN] Would cancel all open orders for {symbol}")
            return {"dry_run": True, "symbol": symbol, "cancelled_count": 0}

        try:
            log_info(f"Cancelling all open orders for {symbol}")
            ccxt_sym: str = _ccxt_futures_symbol(self.exchange, symbol)
            open_orders: list = _fetch_all_open_orders(self.exchange, ccxt_sym)

            cancelled_count: int = 0
            for order in open_orders:
                try:
                    self.exchange.cancel_order(order["id"], ccxt_sym)
                    cancelled_count += 1
                    log_info(f"  Cancelled order: {order['id']} ({order.get('type', 'N/A')})")
                except Exception as e:
                    log_warn(f"Failed to cancel order {order['id']}: {e}")

            log_info(f"✅ Cancelled {cancelled_count} open orders for {symbol}")
            return {"symbol": symbol, "cancelled_count": cancelled_count, "success": True}

        except Exception as e:
            log_error(f"Failed to cancel open orders: {e}")
            return {"symbol": symbol, "cancelled_count": 0, "success": False}
