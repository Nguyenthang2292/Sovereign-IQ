"""TP/SL Synchronization Service.

Bidirectional sync between Binance Open Orders API and Database:
- Fetch TP/SL from Binance → Update DB
- Keep DB in sync with live exchange state
"""

from datetime import datetime, timezone
from typing import Any, Dict, Optional, Tuple, Union

from modules.common.domain.symbol_codec import SymbolCodec
from modules.common.domain.symbol_types import CcxtSymbol, DbSymbol, FuturesSymbol
from modules.common.domain.order_type_codec import BinanceOrderType
from modules.common.ui.logging import log_debug, log_error, log_info, log_warn


class TPSLSyncService:
    """Service to sync TP/SL between Binance and Database."""

    _codec = SymbolCodec()

    @staticmethod
    def _filter_orders_for_symbol(open_orders: list, symbol: Union[str, DbSymbol, FuturesSymbol]) -> list:
        """Filter open orders by symbol using both CCXT symbol and Binance raw symbol id."""
        target_id = TPSLSyncService._codec.to_db(symbol)
        matched_orders = []
        for order in open_orders:
            if not isinstance(order, dict):
                continue
            info = order.get("info") or {}
            order_symbol = str(order.get("symbol") or "")
            info_symbol = str(info.get("symbol") or "") if isinstance(info, dict) else ""
            if TPSLSyncService._codec.to_db(order_symbol) == target_id or info_symbol.upper() == target_id:
                matched_orders.append(order)
        return matched_orders

    @staticmethod
    def fetch_tp_sl_from_binance(
        client, symbol: Union[str, DbSymbol, CcxtSymbol, FuturesSymbol]
    ) -> Tuple[Optional[float], Optional[float], Optional[float]]:
        """
        Fetch TP/SL/BE from Binance Open Orders API.

        Args:
            client: BinanceClient instance
            symbol: Trading symbol in any format (DbSymbol: "BTCUSDT", CcxtSymbol: "BTC/USDT", etc.)

        Returns:
            Tuple of (take_profit, stop_loss, break_even)
        """
        take_profit = None
        stop_loss = None
        break_even = None

        try:
            # Use same symbol format as exchange (e.g. SKL/USDT:USDT) so we see conditional orders
            from modules.auto_trade.execution.binance.order_management import _ccxt_futures_symbol

            ccxt_symbol = _ccxt_futures_symbol(client.exchange, symbol)

            # Use _fetch_all_open_orders to get BOTH Basic AND Conditional orders.
            # Binance separates these into different API endpoints:
            #   - Basic: regular limit/market orders
            #   - Conditional: STOP_MARKET, TAKE_PROFIT_MARKET (our TP/SL!)
            # Without this, conditional orders are invisible → 'No TP/SL detected' → duplicates.
            from modules.auto_trade.execution.binance.order_management import _fetch_all_open_orders

            open_orders = _fetch_all_open_orders(client.exchange, ccxt_symbol)

            # Fallback: if symbol-scoped query misses conditional orders due symbol-format mismatch,
            # fetch all open orders and filter by symbol id.
            if not open_orders:
                try:
                    all_open_orders = client.exchange.fetch_open_orders()
                    open_orders = TPSLSyncService._filter_orders_for_symbol(all_open_orders, ccxt_symbol)
                    log_info(
                        f"[TPSLSync] Fallback query used for {symbol}: "
                        f"{len(open_orders)} matched of {len(all_open_orders)} total open orders"
                    )
                except Exception as fallback_error:
                    log_debug(f"[TPSLSync] Fallback fetch_open_orders() failed for {symbol}: {fallback_error}")

            log_info(f"[TPSLSync] Found {len(open_orders)} open orders for {symbol}")

            for order in open_orders:
                # Use BinanceOrderType.resolve for consistent order type detection
                order_type = BinanceOrderType.resolve(order)
                info = order.get("info") or {}
                if not isinstance(info, dict):
                    info = {}

                stop_price = order.get("stopPrice") or order.get("triggerPrice") or order.get("price", 0)
                if not stop_price and isinstance(info, dict):
                    stop_price = info.get("stopPrice") or info.get("triggerPrice") or info.get("price")

                log_debug(f"[TPSLSync] Order id={order.get('id')} type={order_type} stopPrice={stop_price}")

                # Classify the order using BinanceOrderType.classify for consistency
                kind = BinanceOrderType.classify(order)
                is_tp = kind == "tp"
                is_sl = kind == "sl"

                if not is_tp and not is_sl and stop_price:
                    # CCXT normalized type='MARKET' for a conditional order → classify by price
                    try:
                        sp_f = float(stop_price)
                        # Without position side context here, use a simple positional heuristic:
                        # first ambiguous stop order → assume TP, second → assume SL.
                        # EnsureTPSLJob._classify_conditional_order does a proper price-vs-entry
                        # comparison and is the authoritative guard against duplicate order creation.
                        if take_profit is None and stop_loss is None:
                            # First ambiguous order – assume TP
                            take_profit = sp_f
                            log_debug(f"[TPSLSync] Ambiguous order assumed TP id={order.get('id')} price={sp_f}")
                        elif take_profit is not None and stop_loss is None:
                            # Second ambiguous order – assume SL
                            stop_loss = sp_f
                            log_debug(f"[TPSLSync] Ambiguous order assumed SL id={order.get('id')} price={sp_f}")
                    except (TypeError, ValueError):
                        pass

                elif is_tp:
                    # Take Profit: Binance TAKE_PROFIT, TAKE_PROFIT_MARKET, TAKE_PROFIT_LIMIT
                    take_profit = float(stop_price) if stop_price else None
                    log_info(f"[TPSLSync] ✅ Detected TP for {symbol}: ${take_profit}")

                elif is_sl:
                    # Stop Loss: Binance STOP, STOP_MARKET, STOP_LOSS, STOP_LOSS_MARKET (all appear in Conditional)
                    # Require STOP or LOSS in type and exclude TAKE_PROFIT so we don't misclassify.
                    stop_loss = float(stop_price) if stop_price else None
                    log_info(f"[TPSLSync] ✅ Detected SL for {symbol}: ${stop_loss}")

            if take_profit is None and stop_loss is None:
                log_warn(f"[TPSLSync] ⚠️ No TP/SL orders detected for {symbol}")

            return take_profit, stop_loss, break_even

        except Exception as e:
            log_error(f"[TPSLSync] Error fetching from Binance for {symbol}: {e}")
            return None, None, None

    @staticmethod
    def detect_break_even(entry_price: float, stop_loss: Optional[float], side: str) -> Optional[float]:
        """
        Auto-detect if Break Even has been moved.

        Args:
            entry_price: Position entry price
            stop_loss: Current stop loss price
            side: Position side (LONG/SHORT)

        Returns:
            Break even price if moved, None otherwise
        """
        if stop_loss is None:
            return None

        try:
            if side.upper() == "LONG" and stop_loss >= entry_price:
                return stop_loss
            elif side.upper() == "SHORT" and stop_loss <= entry_price:
                return stop_loss
        except Exception as e:
            log_error(f"[TPSLSync] Error detecting BE: {e}")

        return None

    # Buffer below/above mark so SL is valid and does not trigger immediately (Binance -2021)
    _SL_MARK_BUFFER_PCT = 0.005  # 0.5%

    @staticmethod
    def _get_mark_price(client, symbol: FuturesSymbol) -> Optional[float]:
        """Get current mark price for symbol (futures). Returns None if unavailable."""
        try:
            exchange = getattr(client, "exchange", None)
            if exchange is None:
                return None
            ticker = exchange.fetch_ticker(symbol)
            info = ticker.get("info") if isinstance(ticker, dict) else None
            if isinstance(info, dict) and info.get("markPrice") is not None:
                return float(info["markPrice"])
            last = ticker.get("last") if isinstance(ticker, dict) else None
            return float(last) if last is not None else None
        except Exception:
            return None

    @staticmethod
    def sync_to_database(
        repo_context,
        symbol: Union[str, DbSymbol, CcxtSymbol, FuturesSymbol],
        take_profit: Optional[float],
        stop_loss: Optional[float],
    ) -> bool:
        """
        Sync TP/SL values to database order.

        Args:
            repo_context: RepositoryContext
            symbol: Trading symbol (any format: DbSymbol, CcxtSymbol, etc.)
            take_profit: TP price from Binance
            stop_loss: SL price from Binance

        Returns:
            True if updated, False otherwise
        """
        try:
            # Normalize symbol for DB query (handle SKL/USDT, SKL/USDT:USDT, SKLUSDT)
            symbol_normalized = TPSLSyncService._codec.to_db(symbol)

            # Try to find order
            orders = repo_context.orders.get_open_positions(symbol=symbol_normalized)

            if not orders:
                log_warn(f"[TPSLSync] No open order found in DB for {symbol} (tried: {symbol}, {symbol_normalized})")
                return False

            order = orders[0]
            log_debug(f"[TPSLSync] Found order in DB: ID={order.get('order_id')}, symbol={order.get('symbol')}")

            # Check if values changed
            changed = False

            update_data: Dict[str, Any] = {}

            if take_profit is not None and order.get("take_profit") != take_profit:
                old_tp = order.get("take_profit")
                update_data["take_profit"] = take_profit
                changed = True
                log_info(f"[TPSLSync] Updated TP for {symbol}: ${old_tp} → ${take_profit}")

            if stop_loss is not None and order.get("stop_loss") != stop_loss:
                old_sl = order.get("stop_loss")
                update_data["stop_loss"] = stop_loss
                changed = True
                log_info(f"[TPSLSync] Updated SL for {symbol}: ${old_sl} → ${stop_loss}")

                # Detect if BE was moved
                if order.get("entry_price"):
                    be_detected = TPSLSyncService.detect_break_even(
                        order.get("entry_price"), stop_loss, order.get("side")
                    )
                    if be_detected and not order.get("be_moved"):
                        update_data["be_moved"] = True
                        update_data["be_moved_at"] = datetime.now(timezone.utc).isoformat()
                        update_data["original_stop_loss"] = old_sl
                        log_info(f"[TPSLSync] BE detected for {symbol}! Moved to ${be_detected}")
                        changed = True

            if changed:
                order_id = order.get("order_id")
                if not order_id:
                    log_warn(f"[TPSLSync] Cannot update DB for {symbol}: order_id missing")
                    return False

                # Use update() not create_order() — create_order has ConditionExpression=pk.not_exists()
                # which always raises ConditionalCheckFailedException for existing orders.
                updated = repo_context.orders.update(order_id, update_data)
                if updated:
                    log_info(f"[TPSLSync] ✅ DB updated for {symbol}")
                    return True
                else:
                    log_warn(f"[TPSLSync] DB update returned False for {symbol} (order may not exist)")
                    return False
            else:
                log_debug(f"[TPSLSync] No changes needed for {symbol}")
                return False

        except Exception as e:
            log_error(f"[TPSLSync] Error syncing to DB for {symbol}: {e}")
            return False

    @staticmethod
    def sync_position_tp_sl(
        client, repo_context, symbol: DbSymbol, side: str, entry_price: float
    ) -> Dict[str, Optional[float]]:
        """
        Complete sync: Fetch from Binance → Update DB → Return values.

        Args:
            client: BinanceClient instance
            repo_context: RepositoryContext instance
            symbol: Trading symbol
            side: Position side (LONG/SHORT)
            entry_price: Position entry price

        Returns:
            Dict with take_profit, stop_loss, break_even
        """
        # Fetch from Binance
        take_profit, stop_loss, _ = TPSLSyncService.fetch_tp_sl_from_binance(client, symbol)

        # Sync to database
        if take_profit or stop_loss:
            TPSLSyncService.sync_to_database(repo_context, symbol, take_profit, stop_loss)

        # Detect break even
        break_even = TPSLSyncService.detect_break_even(entry_price, stop_loss, side)

        return {
            "take_profit": take_profit,
            "stop_loss": stop_loss,
            "break_even": break_even,
        }

    @staticmethod
    def ensure_tp_sl_on_binance(
        client,
        symbol: DbSymbol,
        side: str,
        entry_price: float,
        default_tp_pct: float,
        default_sl_pct: float,
    ) -> Dict[str, Optional[float]]:
        """
        If position is missing TP or SL on Binance, place them using config percentages.
        Call this after sync_position_tp_sl when take_profit or stop_loss is None.

        default_tp_pct / default_sl_pct are treated as ROI% on capital.
        The function fetches the actual leverage from the live Binance position and
        converts:  price_move% = roi% / leverage.

        Args:
            client: BinanceClient instance (must not be dry_run if you want real orders)
            symbol: Trading symbol (e.g. "SKL/USDT")
            side: LONG or SHORT
            entry_price: Position entry price
            default_tp_pct: Take profit ROI% (e.g. 5.0 = 5% on capital)
            default_sl_pct: Stop loss ROI% (e.g. 2.5 = 2.5% on capital)

        Returns:
            Dict with take_profit, stop_loss (prices now on exchange, or unchanged if failed)
        """
        take_profit, stop_loss, _ = TPSLSyncService.fetch_tp_sl_from_binance(client, symbol)

        # Fetch actual leverage from live Binance position
        actual_leverage: float = 1.0
        try:
            from modules.auto_trade.execution.binance.position_management import PositionManagement

            pos_mgr = PositionManagement(client.exchange)
            pos = pos_mgr.get_position(symbol)
            if pos:
                lev_raw = pos.get("leverage") or (pos.get("info") or {}).get("leverage")
                if lev_raw is not None:
                    try:
                        actual_leverage = float(lev_raw)
                    except (TypeError, ValueError):
                        pass
        except Exception as lev_err:
            log_debug(f"[TPSLSync] Could not fetch leverage for {symbol}: {lev_err}")

        if actual_leverage < 1.0:
            actual_leverage = 1.0

        log_info(f"[TPSLSync] ensure_tp_sl_on_binance: {symbol} leverage={actual_leverage}x ")

        # Secondary verification: check conditional orders directly using correct symbol format
        has_tp_conditional = take_profit is not None
        has_sl_conditional = stop_loss is not None

        if not has_tp_conditional or not has_sl_conditional:
            try:
                from modules.auto_trade.execution.binance.order_management import _ccxt_futures_symbol

                ccxt_sym = _ccxt_futures_symbol(client.exchange, symbol)
                from modules.auto_trade.execution.binance.order_management import _fetch_all_open_orders

                open_orders = _fetch_all_open_orders(client.exchange, ccxt_sym)
                for order_item in open_orders:
                    kind = BinanceOrderType.classify(order_item)
                    if kind == "tp":
                        has_tp_conditional = True
                        if take_profit is None:
                            info_item = order_item.get("info") or {}
                            sp = order_item.get("stopPrice") or (
                                info_item.get("stopPrice") if isinstance(info_item, dict) else None
                            )
                            take_profit = float(sp) if sp else None
                    elif kind == "sl":
                        has_sl_conditional = True
                        if stop_loss is None:
                            info_item = order_item.get("info") or {}
                            sp = order_item.get("stopPrice") or (
                                info_item.get("stopPrice") if isinstance(info_item, dict) else None
                            )
                            stop_loss = float(sp) if sp else None
            except Exception as e:
                log_debug(f"[TPSLSync] Secondary check failed for {symbol}: {e}")

        need_tp = not has_tp_conditional
        need_sl = not has_sl_conditional
        if not need_tp and not need_sl:
            return {"take_profit": take_profit, "stop_loss": stop_loss}

        if entry_price <= 0 or default_tp_pct <= 0 or default_sl_pct <= 0:
            return {"take_profit": take_profit, "stop_loss": stop_loss}

        # ROI% → price-move% by dividing by actual leverage
        tp_price_pct = default_tp_pct / actual_leverage
        sl_price_pct = default_sl_pct / actual_leverage

        side_upper = str(side).upper()
        if side_upper == "LONG":
            tp_price = entry_price * (1.0 + tp_price_pct / 100.0)
            sl_price = entry_price * (1.0 - sl_price_pct / 100.0)
        else:
            tp_price = entry_price * (1.0 - tp_price_pct / 100.0)
            sl_price = entry_price * (1.0 + sl_price_pct / 100.0)

        # Clamp SL so it is valid vs mark price (avoid Binance -2021 "Order would immediately trigger")
        if need_sl:
            mark_price = TPSLSyncService._get_mark_price(client, FuturesSymbol(ccxt_sym))
            if mark_price is not None and mark_price > 0:
                if side_upper == "LONG" and sl_price >= mark_price:
                    sl_price = mark_price * (1.0 - TPSLSyncService._SL_MARK_BUFFER_PCT)
                    log_info(f"[TPSLSync] Adjusted SL below mark for {symbol} LONG: ${sl_price:.6f}")
                elif side_upper == "SHORT" and sl_price <= mark_price:
                    sl_price = mark_price * (1.0 + TPSLSyncService._SL_MARK_BUFFER_PCT)
                    log_info(f"[TPSLSync] Adjusted SL above mark for {symbol} SHORT: ${sl_price:.6f}")

        if need_tp:
            try:
                res = client.modify_take_profit(symbol, None, tp_price)
                if res and (res.get("id") or res.get("dry_run")):
                    take_profit = tp_price
                    log_info(f"[TPSLSync] ✅ Placed TP for {symbol} at ${take_profit}")
            except Exception as e:
                log_error(f"[TPSLSync] Failed to place TP for {symbol}: {e}")
        if need_sl:
            try:
                res = client.modify_stop_loss(symbol, None, sl_price)
                if res and (res.get("id") or res.get("dry_run")):
                    stop_loss = sl_price
                    log_info(f"[TPSLSync] ✅ Placed SL for {symbol} at ${stop_loss}")
            except Exception as e:
                log_error(f"[TPSLSync] Failed to place SL for {symbol}: {e}")

        return {"take_profit": take_profit, "stop_loss": stop_loss}


__all__ = ["TPSLSyncService"]
