"""TP/SL Synchronization Service.

Bidirectional sync between Binance Open Orders API and Database:
- Fetch TP/SL from Binance → Update DB
- Keep DB in sync with live exchange state
"""

import logging
from typing import Optional, Dict, Tuple
from datetime import datetime, timezone

logger = logging.getLogger(__name__)


class TPSLSyncService:
    """Service to sync TP/SL between Binance and Database."""
    
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
                order_type = order.get("type", "").upper()
                stop_price = order.get("stopPrice", 0) or order.get("price", 0)
                
                # Detect Take Profit order
                if "TAKE_PROFIT" in order_type:
                    take_profit = float(stop_price)
                    logger.info(f"[TPSLSync] Found TP for {symbol}: ${take_profit}")
                
                # Detect Stop Loss order
                elif "STOP" in order_type and "MARKET" in order_type:
                    stop_loss = float(stop_price)
                    logger.info(f"[TPSLSync] Found SL for {symbol}: ${stop_loss}")
            
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
            
            # Find open order for this symbol
            order = session.query(Order).filter(
                Order.symbol == symbol,
                Order.status == "OPEN"
            ).order_by(Order.created_at.desc()).first()
            
            if not order:
                logger.warning(f"[TPSLSync] No open order found in DB for {symbol}")
                return False
            
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


__all__ = ["TPSLSyncService"]
