"""TP/SL Synchronization Service.

Bidirectional sync between Binance Open Orders API and Database:
- Fetch TP/SL from Binance → Update DB
- Keep DB in sync with live exchange state
"""

import logging
from datetime import datetime, timezone
from typing import Dict, Optional, Tuple

logger = logging.getLogger(__name__)


class TPSLSyncService:
    """Service to sync TP/SL between Binance and Database."""

    @staticmethod
    def _normalize_symbol_for_db(symbol: str) -> str:
        """
        Normalize symbol format for database query.
        
        Binance API returns various formats:
        - "SKL/USDT" → "SKLUSDT"
        - "SKL/USDT:USDT" → "SKLUSDT"
        - "SKLUSDT" → "SKLUSDT"
        
        Database might store as:
        - "SKLUSDT" or "SKLUSDT:USDT"
        
        Args:
            symbol: Symbol in any format
            
        Returns:
            Base symbol without separators (e.g., "SKLUSDT")
        """
        # Remove slash and colon suffixes
        normalized = symbol.replace("/", "").split(":")[0]
        return normalized

    @staticmethod
    def fetch_tp_sl_from_binance(client, symbol: str) -> Tuple[Optional[float], Optional[float], Optional[float]]:
        """
        Fetch TP/SL/BE from Binance Open Orders API.
        
        Args:
            client: BinanceClient instance
            symbol: Trading symbol (e.g., "BTC/USDT")
            
        Returns:
            Tuple of (take_profit, stop_loss, break_even)
        """
        take_profit = None
        stop_loss = None
        break_even = None

        try:
            # Fetch open orders for this symbol
            open_orders = client.exchange.fetch_open_orders(symbol)
            logger.info(f"[TPSLSync] Found {len(open_orders)} open orders for {symbol}")

            for order in open_orders:
                # IMPORTANT: Use info['type'] instead of top-level type
                # ccxt normalizes type to lowercase generic (e.g., 'market')
                # but info['type'] preserves Binance's exact type (e.g., 'TAKE_PROFIT_MARKET')
                order_type_main = order.get("type", "").upper()
                order_type_info = order.get("info", {}).get("type", "").upper()

                # Prefer info type (more specific)
                order_type = order_type_info if order_type_info else order_type_main

                # Log order details for debugging
                logger.debug(f"[TPSLSync] Order: main_type={order_type_main}, info_type={order_type_info}, stopPrice={order.get('stopPrice')}")

                # Get stop price from either stopPrice or triggerPrice field
                stop_price = order.get("stopPrice") or order.get("triggerPrice") or order.get("price", 0)

                # Detect Take Profit orders
                # Binance uses: TAKE_PROFIT, TAKE_PROFIT_MARKET, TAKE_PROFIT_LIMIT
                if "TAKE_PROFIT" in order_type:
                    take_profit = float(stop_price) if stop_price else None
                    logger.info(f"[TPSLSync] ✅ Detected TP for {symbol}: ${take_profit}")

                # Detect Stop Loss orders
                # Binance uses: STOP, STOP_MARKET, STOP_LOSS, STOP_LOSS_MARKET
                elif "STOP" in order_type and ("MARKET" in order_type or "LOSS" in order_type):
                    stop_loss = float(stop_price) if stop_price else None
                    logger.info(f"[TPSLSync] ✅ Detected SL for {symbol}: ${stop_loss}")

            if take_profit is None and stop_loss is None:
                logger.warning(f"[TPSLSync] ⚠️ No TP/SL orders detected for {symbol}")

            return take_profit, stop_loss, break_even

        except Exception as e:
            logger.error(f"[TPSLSync] Error fetching from Binance for {symbol}: {e}")
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
            logger.error(f"[TPSLSync] Error detecting BE: {e}")

        return None

    @staticmethod
    def sync_to_database(session, symbol: str, take_profit: Optional[float], stop_loss: Optional[float]) -> bool:
        """
        Sync TP/SL values to database order.
        
        Args:
            session: Database session
            symbol: Trading symbol
            take_profit: TP price from Binance
            stop_loss: SL price from Binance
            
        Returns:
            True if updated, False otherwise
        """
        try:
            from modules.auto_trade.database.models import Order

            # Normalize symbol for DB query (handle SKL/USDT, SKL/USDT:USDT, SKLUSDT)
            symbol_normalized = TPSLSyncService._normalize_symbol_for_db(symbol)

            # Try to find order with multiple symbol formats
            order = session.query(Order).filter(
                Order.status == "OPEN"
            ).filter(
                (Order.symbol == symbol) |
                (Order.symbol == symbol_normalized) |
                (Order.symbol.like(f"{symbol_normalized}%"))  # Match SKLUSDT:USDT
            ).order_by(Order.created_at.desc()).first()

            if not order:
                logger.warning(f"[TPSLSync] No open order found in DB for {symbol} (tried: {symbol}, {symbol_normalized})")
                return False

            logger.debug(f"[TPSLSync] Found order in DB: ID={order.id}, symbol={order.symbol}")

            # Check if values changed
            changed = False

            if take_profit is not None and order.take_profit != take_profit:
                old_tp = order.take_profit
                order.take_profit = take_profit
                changed = True
                logger.info(f"[TPSLSync] Updated TP for {symbol}: ${old_tp} → ${take_profit}")

            if stop_loss is not None and order.stop_loss != stop_loss:
                old_sl = order.stop_loss
                order.stop_loss = stop_loss
                changed = True
                logger.info(f"[TPSLSync] Updated SL for {symbol}: ${old_sl} → ${stop_loss}")

                # Detect if BE was moved
                if order.entry_price:
                    be_detected = TPSLSyncService.detect_break_even(
                        order.entry_price,
                        stop_loss,
                        order.side
                    )
                    if be_detected and not order.be_moved:
                        order.be_moved = True
                        order.be_moved_at = datetime.now(timezone.utc)
                        order.original_stop_loss = old_sl
                        logger.info(f"[TPSLSync] BE detected for {symbol}! Moved to ${be_detected}")
                        changed = True

            if changed:
                order.updated_at = datetime.now(timezone.utc)
                session.commit()
                logger.info(f"[TPSLSync] ✅ DB updated for {symbol}")
                return True
            else:
                logger.debug(f"[TPSLSync] No changes needed for {symbol}")
                return False

        except Exception as e:
            logger.error(f"[TPSLSync] Error syncing to DB for {symbol}: {e}")
            session.rollback()
            return False

    @staticmethod
    def sync_position_tp_sl(client, session, symbol: str, side: str, entry_price: float) -> Dict[str, Optional[float]]:
        """
        Complete sync: Fetch from Binance → Update DB → Return values.

        Args:
            client: BinanceClient instance
            session: Database session
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
            TPSLSyncService.sync_to_database(session, symbol, take_profit, stop_loss)

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
        symbol: str,
        side: str,
        entry_price: float,
        default_tp_pct: float,
        default_sl_pct: float,
    ) -> Dict[str, Optional[float]]:
        """
        If position is missing TP or SL on Binance, place them using config percentages.
        Call this after sync_position_tp_sl when take_profit or stop_loss is None.

        Args:
            client: BinanceClient instance (must not be dry_run if you want real orders)
            symbol: Trading symbol (e.g. "SKL/USDT")
            side: LONG or SHORT
            entry_price: Position entry price
            default_tp_pct: Take profit % (e.g. 5.0 for 5%)
            default_sl_pct: Stop loss % (e.g. 2.5 for 2.5%)

        Returns:
            Dict with take_profit, stop_loss (prices now on exchange, or unchanged if failed)
        """
        take_profit, stop_loss, _ = TPSLSyncService.fetch_tp_sl_from_binance(client, symbol)
        need_tp = take_profit is None
        need_sl = stop_loss is None
        if not need_tp and not need_sl:
            return {"take_profit": take_profit, "stop_loss": stop_loss}

        if entry_price <= 0 or default_tp_pct <= 0 or default_sl_pct <= 0:
            return {"take_profit": take_profit, "stop_loss": stop_loss}

        side_upper = str(side).upper()
        if side_upper == "LONG":
            tp_price = entry_price * (1.0 + default_tp_pct / 100.0)
            sl_price = entry_price * (1.0 - default_sl_pct / 100.0)
        else:
            tp_price = entry_price * (1.0 - default_tp_pct / 100.0)
            sl_price = entry_price * (1.0 + default_sl_pct / 100.0)

        if need_tp:
            try:
                res = client.modify_take_profit(symbol, None, tp_price)
                if res and (res.get("id") or res.get("dry_run")):
                    take_profit = tp_price
                    logger.info(f"[TPSLSync] ✅ Placed TP for {symbol} at ${take_profit}")
            except Exception as e:
                logger.error(f"[TPSLSync] Failed to place TP for {symbol}: {e}")
        if need_sl:
            try:
                res = client.modify_stop_loss(symbol, None, sl_price)
                if res and (res.get("id") or res.get("dry_run")):
                    stop_loss = sl_price
                    logger.info(f"[TPSLSync] ✅ Placed SL for {symbol} at ${stop_loss}")
            except Exception as e:
                logger.error(f"[TPSLSync] Failed to place SL for {symbol}: {e}")

        return {"take_profit": take_profit, "stop_loss": stop_loss}


__all__ = ["TPSLSyncService"]
