"""
Position Sync Service

Syncs existing Binance positions into the local database.
Useful when:
- Positions were opened manually on Binance
- Database was cleared/reset
- Miss-sync occurred due to errors
"""

import logging
from datetime import datetime, timezone
from typing import Dict, List, Optional

from modules.auto_trade.database.models import Order

logger = logging.getLogger(__name__)


class PositionSyncService:
    """Service for syncing Binance positions to database."""

    @staticmethod
    def fetch_binance_positions(client) -> List[Dict]:
        """
        Fetch all open positions from Binance.

        Args:
            client: BinanceClient instance

        Returns:
            List of position dictionaries
        """
        try:
            # Fetch all positions (including zero-sized)
            positions = client.exchange.fetch_positions()

            logger.info(f"[PositionSync] Raw API returned {len(positions) if positions else 0} positions")

            # Debug: Log first position structure if available
            if positions and len(positions) > 0:
                sample = positions[0]
                logger.debug(f"[PositionSync] Sample position structure: {list(sample.keys())}")

            # Filter only positions with non-zero size
            open_positions = []
            for pos in positions:
                try:
                    contracts = float(pos.get("contracts", 0) or 0)
                    if contracts == 0:
                        continue

                    symbol = pos.get("symbol", "")
                    if not symbol:
                        logger.warning(f"[PositionSync] Skipping position with no symbol: {pos}")
                        continue

                    side = pos.get("side", "").upper()
                    entry_price = float(pos.get("entryPrice", 0) or 0)
                    notional = float(pos.get("notional", 0) or 0)

                    # Safe int conversion with None handling
                    leverage_raw = pos.get("leverage")
                    leverage = int(leverage_raw) if leverage_raw is not None else 1

                    # Fetch TP/SL from open orders
                    tp_price, sl_price = PositionSyncService._fetch_tp_sl_orders(
                        client, symbol
                    )

                    open_positions.append({
                        "symbol": symbol,
                        "side": side,
                        "contracts": abs(contracts),
                        "entry_price": entry_price,
                        "notional": abs(notional),
                        "leverage": leverage,
                        "take_profit": tp_price,
                        "stop_loss": sl_price,
                    })

                except (ValueError, TypeError) as e:
                    logger.warning(f"[PositionSync] Error parsing position {pos.get('symbol')}: {e}")
                    continue

            logger.info(f"[PositionSync] Fetched {len(open_positions)} open positions from Binance")
            return open_positions

        except Exception as e:
            logger.error(f"[PositionSync] Error fetching positions: {e}")
            return []

    @staticmethod
    def _fetch_tp_sl_orders(client, symbol: str) -> tuple[Optional[float], Optional[float]]:
        """
        Fetch TP/SL prices from open orders.

        Args:
            client: BinanceClient instance
            symbol: Trading symbol

        Returns:
            Tuple of (take_profit, stop_loss)
        """
        try:
            open_orders = client.exchange.fetch_open_orders(symbol)

            logger.info(f"[PositionSync] Found {len(open_orders)} open orders for {symbol}")

            tp_price = None
            sl_price = None

            for order in open_orders:
                # IMPORTANT: Use info['type'] instead of top-level type
                # ccxt normalizes type to lowercase generic (e.g., 'market')
                # but info['type'] preserves Binance's exact type (e.g., 'TAKE_PROFIT_MARKET')
                order_type_main = order.get("type", "").upper()
                order_type_info = order.get("info", {}).get("type", "").upper()
                
                # Prefer info type (more specific)
                order_type = order_type_info if order_type_info else order_type_main
                
                # Log order details for debugging
                logger.debug(f"[PositionSync] Order: main_type={order_type_main}, info_type={order_type_info}, stopPrice={order.get('stopPrice')}")
                
                # Get stop price from either stopPrice or triggerPrice field
                stop_price = order.get("stopPrice") or order.get("triggerPrice") or order.get("price", 0)
                
                # Detect Take Profit orders
                # Binance uses: TAKE_PROFIT, TAKE_PROFIT_MARKET, TAKE_PROFIT_LIMIT
                if "TAKE_PROFIT" in order_type:
                    tp_price = float(stop_price) if stop_price else None
                    logger.info(f"[PositionSync] ✅ Detected TP order: type={order_type}, price=${tp_price}")
                
                # Detect Stop Loss orders
                # Binance uses: STOP, STOP_MARKET, STOP_LOSS, STOP_LOSS_MARKET
                elif "STOP" in order_type and ("MARKET" in order_type or "LOSS" in order_type):
                    sl_price = float(stop_price) if stop_price else None
                    logger.info(f"[PositionSync] ✅ Detected SL order: type={order_type}, price=${sl_price}")

            if tp_price is None and sl_price is None:
                logger.warning(f"[PositionSync] ⚠️ No TP/SL orders detected for {symbol}")

            return tp_price, sl_price

        except Exception as e:
            logger.error(f"[PositionSync] Error fetching orders for {symbol}: {e}")
            return None, None

    @staticmethod
    def sync_position_to_db(session, position: Dict, order_source: str = "MANUAL") -> tuple[Optional[Order], bool]:
        """
        Sync a single position to database.

        Args:
            session: Database session
            position: Position dictionary from Binance
            order_source: Order source tag (default: MANUAL)

        Returns:
            Tuple of (Order object or None, is_new: bool)
        """
        try:
            symbol = position["symbol"]
            side = position["side"]

            # Check if position already exists in DB
            existing = session.query(Order).filter(
                Order.symbol == symbol.replace("/", ""),
                Order.status == "OPEN"
            ).first()

            if existing:
                logger.info(f"[PositionSync] Position {symbol} already exists in DB (ID={existing.id})")
                return existing, False

            # Create new order record
            now = datetime.now(timezone.utc)

            new_order = Order(
                order_id=f"SYNC_{int(now.timestamp())}",  # Synthetic order ID
                client_order_id=f"SYNC_{symbol}_{int(now.timestamp())}",
                symbol=symbol.replace("/", ""),  # Remove slash for consistency
                side=side,
                order_type="MARKET",
                order_source=order_source,
                execution_mode="MANUAL",
                entry_price=position["entry_price"],
                amount=position["contracts"],
                leverage=position["leverage"],
                stop_loss=position.get("stop_loss"),
                take_profit=position.get("take_profit"),
                status="OPEN",
                pnl=0.0,
                pnl_percentage=0.0,
                be_moved=False,
                trailing_step_index=0,
                martingale_step=0,
                created_at=now,
                opened_at=now,
            )

            session.add(new_order)
            session.commit()

            logger.info(f"[PositionSync] ✅ Synced {symbol} to DB (ID={new_order.id})")
            return new_order, True

        except Exception as e:
            logger.error(f"[PositionSync] Error syncing position {position.get('symbol')}: {e}")
            session.rollback()
            return None, False

    @staticmethod
    def sync_all_positions(client, db_manager) -> Dict[str, int]:
        """
        Sync all open Binance positions to database.

        Args:
            client: BinanceClient instance
            db_manager: DatabaseManager instance

        Returns:
            Dict with sync statistics: {
                "fetched": int,
                "synced": int,
                "existing": int,
                "failed": int
            }
        """
        stats = {
            "fetched": 0,
            "synced": 0,
            "existing": 0,
            "failed": 0,
        }

        try:
            # Fetch positions from Binance
            positions = PositionSyncService.fetch_binance_positions(client)
            stats["fetched"] = len(positions)

            if not positions:
                logger.info("[PositionSync] No open positions found on Binance")
                return stats

            # Sync each position to DB
            with db_manager.session_scope() as session:
                for pos in positions:
                    order, is_new = PositionSyncService.sync_position_to_db(session, pos)

                    if order:
                        if is_new:
                            stats["synced"] += 1
                        else:
                            stats["existing"] += 1
                    else:
                        stats["failed"] += 1

            logger.info(
                f"[PositionSync] Sync completed: "
                f"{stats['synced']} synced, "
                f"{stats['existing']} existing, "
                f"{stats['failed']} failed"
            )

        except Exception as e:
            logger.error(f"[PositionSync] Fatal error during sync: {e}")

        return stats


__all__ = ["PositionSyncService"]
