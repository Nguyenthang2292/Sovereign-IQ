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
        market = exchange.market(symbol)
        if market and market.get("symbol"):
            return str(market["symbol"])
    except Exception:
        pass
    if ":" not in symbol and "/" in symbol:
        return f"{symbol}:USDT"
    return symbol


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

            tp_side: str = "sell" if side == "long" else "buy"

            ccxt_symbol_tp: str = _ccxt_futures_symbol(self.exchange, symbol)

            # 2. Cancel existing TP orders
            open_orders: list = self.exchange.fetch_open_orders(ccxt_symbol_tp)
            cancelled_count: int = 0

            for order in open_orders:
                info_tp = order.get("info") or {}
                order_type_tp = (
                    (info_tp.get("type") or info_tp.get("origType") or order.get("type") or "")
                ).lower()
                if "take_profit" in order_type_tp:
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
                        type=cast(Any, "take_profit_market"),
                        side=tp_side,
                        amount=amount,
                        params={
                            "stopPrice": take_profit_price,
                            "reduceOnly": True,
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

            sl_side: str = "sell" if side == "long" else "buy"

            # Use CCXT futures symbol so we find and cancel existing conditional orders (avoid duplicates)
            ccxt_symbol: str = _ccxt_futures_symbol(self.exchange, symbol)

            # 2. Cancel existing SL orders
            open_orders: list = self.exchange.fetch_open_orders(ccxt_symbol)
            cancelled_count: int = 0

            for order in open_orders:
                info_sl = order.get("info") or {}
                order_type_sl = (
                    (info_sl.get("type") or info_sl.get("origType") or order.get("type") or "")
                ).lower()
                if ("stop" in order_type_sl or "loss" in order_type_sl) and "take_profit" not in order_type_sl:
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
                            type=cast(Any, "stop_market"),
                            side=sl_side,
                            amount=amount,
                            params={
                                "stopPrice": stop_loss_price,
                                "reduceOnly": True,
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
            open_orders: list = self.exchange.fetch_open_orders(ccxt_sym)

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
