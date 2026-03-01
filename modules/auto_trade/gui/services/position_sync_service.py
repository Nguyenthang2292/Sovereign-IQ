"""
Position Sync Service

Syncs existing Binance positions into the local database.
Useful when:
- Positions were opened manually on Binance
- Database was cleared/reset
- Miss-sync occurred due to errors
"""

from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

from modules.common.ui.logging import log_error, log_info, log_warn


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

            log_info(f"[PositionSync] Raw API returned {len(positions) if positions else 0} positions")

            # Filter only positions with non-zero size
            open_positions = []
            for pos in positions:
                try:
                    contracts = float(pos.get("contracts", 0) or 0)
                    if contracts == 0:
                        continue

                    symbol_raw = pos.get("symbol", "")
                    if not symbol_raw:
                        log_warn(f"[PositionSync] Skipping position with no symbol: {pos}")
                        continue

                    # CCXT futures symbols look like "DOGE/USDT:USDT" or "BTC/USDT:USDT".
                    # Normalise to plain Binance format ("DOGEUSDT") so the symbol is
                    # consistent with what the rest of the system stores in DynamoDB.
                    symbol = symbol_raw.split(":")[0].replace("/", "")  # DOGE/USDT:USDT → DOGEUSDT

                    side = pos.get("side", "").upper()
                    entry_price = float(pos.get("entryPrice", 0) or pos.get("info", {}).get("entryPrice", 0) or 0)
                    notional = float(pos.get("notional", 0) or 0)

                    # Safe int conversion with None handling
                    leverage_raw = pos.get("leverage")
                    leverage = int(leverage_raw) if leverage_raw is not None else 1

                    # Fetch TP/SL from open orders (use raw CCXT symbol for the API call)
                    tp_price, sl_price = PositionSyncService._fetch_tp_sl_orders(client, symbol_raw)

                    open_positions.append(
                        {
                            "symbol": symbol,  # normalised: DOGEUSDT
                            "symbol_ccxt": symbol_raw,  # CCXT form kept for debugging
                            "side": side,
                            "contracts": abs(contracts),
                            "entry_price": entry_price,
                            "notional": abs(notional),
                            "leverage": leverage,
                            "take_profit": tp_price,
                            "stop_loss": sl_price,
                        }
                    )

                except (ValueError, TypeError) as e:
                    log_warn(f"[PositionSync] Error parsing position {pos.get('symbol')}: {e}")
                    continue

            log_info(f"[PositionSync] Fetched {len(open_positions)} open positions from Binance")
            return open_positions

        except Exception as e:
            log_error(f"[PositionSync] Error fetching positions: {e}")
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
            from modules.auto_trade.gui.services.tp_sl_sync import TPSLSyncService

            tp_price, sl_price, _ = TPSLSyncService.fetch_tp_sl_from_binance(client, symbol)
            if tp_price is None and sl_price is None:
                log_warn(f"[PositionSync] ⚠️ No TP/SL orders detected for {symbol}")
            return tp_price, sl_price

        except Exception as e:
            log_error(f"[PositionSync] Error fetching orders for {symbol}: {e}")
            return None, None

    @staticmethod
    def sync_position_to_db(ctx, position: Dict, order_source: str = "PROGRAMMATIC") -> tuple[bool, bool]:
        """
        Sync a single position to database using RepositoryContext.

        Args:
            ctx: RepositoryContext
            position: Position dictionary from Binance
            order_source: Order source tag (default: PROGRAMMATIC so it appears in all queries)

        Returns:
            Tuple of (success: bool, is_new: bool)
        """
        try:
            symbol = position["symbol"]
            side = position["side"]
            symbol_normalized = symbol.replace("/", "")

            # Check if position already exists in DB.
            # get_open_positions only returns PROGRAMMATIC orders; that covers both
            # auto-placed and previously-synced positions.
            existing_list = ctx.orders.get_open_positions(symbol=symbol_normalized)
            if existing_list:
                existing = existing_list[0]
                log_info(f"[PositionSync] Position {symbol} already exists in DB (order_id={existing.get('order_id')})")
                return True, False

            # Create new order record — tagged PROGRAMMATIC so it is visible
            # to all downstream queries (trailing stop, breakeven, GUI, etc.).
            now = datetime.now(timezone.utc)

            order_data: Dict[str, Any] = {
                "order_id": f"SYNC_{int(now.timestamp())}",
                "client_order_id": f"SYNC_{symbol_normalized}_{int(now.timestamp())}",
                "symbol": symbol_normalized,
                "side": side,
                "order_type": "MARKET",
                "order_source": order_source,  # PROGRAMMATIC → shows in all queries
                "execution_mode": "SYNCED",  # distinct from AUTO/MANUAL for auditing
                "entry_price": float(position["entry_price"]),
                "amount": float(position["contracts"]),
                "leverage": int(position["leverage"]),
                "stop_loss": float(position["stop_loss"]) if position.get("stop_loss") else None,
                "take_profit": float(position["take_profit"]) if position.get("take_profit") else None,
                "status": "OPEN",
                "pnl": 0.0,
                "pnl_percentage": 0.0,
                "be_moved": False,
                "trailing_step_index": 0,
                "martingale_step": 0,
                "created_at": now.isoformat(),
                "opened_at": now.isoformat(),
            }

            ctx.orders.create_order(order_data)
            log_info(f"[PositionSync] ✅ Synced {symbol} to DB (order_id={order_data.get('order_id')})")
            return True, True

        except Exception as e:
            log_error(f"[PositionSync] Error syncing position {position.get('symbol')}: {e}")
            return False, False

    @staticmethod
    def sync_all_positions(client) -> Dict[str, int]:
        """
        Sync all open Binance positions to database using DynamoDB.

        This performs TWO operations:
        1. INSERT: positions open on Binance but missing from DB → create
        2. CLOSE:  positions in DB marked OPEN but no longer on Binance → mark CLOSED

        Args:
            client: BinanceClient instance

        Returns:
            Dict with sync statistics
        """
        stats = {
            "fetched": 0,
            "synced": 0,
            "existing": 0,
            "failed": 0,
            "closed": 0,
        }

        try:
            from modules.auto_trade.database.repository.context import RepositoryContext

            binance_positions = PositionSyncService.fetch_binance_positions(client)
            stats["fetched"] = len(binance_positions)

            ctx = RepositoryContext.from_env()

            # ── Phase 1: Insert missing positions into DB ──────────────────
            binance_open_symbols: set[str] = set()
            for pos in binance_positions:
                symbol = pos.get("symbol", "").replace("/", "")
                if symbol:
                    binance_open_symbols.add(symbol)

                success, is_new = PositionSyncService.sync_position_to_db(ctx, pos)
                if success:
                    if is_new:
                        stats["synced"] += 1
                    else:
                        stats["existing"] += 1
                else:
                    stats["failed"] += 1

            # ── Phase 2: Close stale DB positions ──────────────────────────
            # Query all OPEN orders in DB and close any whose symbol is no
            # longer open on Binance (TP/SL triggered, liquidated, manual close).
            try:
                db_open_orders = ctx.orders.get_open_positions()  # GSI3: PROGRAMMATIC#OPEN
                for order in db_open_orders:
                    db_symbol = (order.get("symbol") or "").replace("/", "")
                    if db_symbol and db_symbol not in binance_open_symbols:
                        order_id = order.get("order_id")
                        if order_id:
                            ok = ctx.orders.update_order_status(order_id, "CLOSED")
                            if ok:
                                stats["closed"] += 1
                                log_info(f"[PositionSync] 🔴 Closed stale DB order: {db_symbol} (order_id={order_id})")
                                # Cancel any orphaned conditional orders on Binance
                                try:
                                    cancel_res = client.cancel_open_orders(db_symbol)
                                    log_info(f"[PositionSync] Cancelled orphaned conditional orders for {db_symbol}: {cancel_res}")
                                except Exception as exc:
                                    log_warn(f"[PositionSync] Error cancelling orphaned conditional orders for {db_symbol}: {exc}")
                            else:
                                log_warn(f"[PositionSync] Could not close stale order {order_id} for {db_symbol}")
            except Exception as close_err:
                log_error(f"[PositionSync] Error closing stale positions: {close_err}")

            log_info(
                f"[PositionSync] Sync completed: "
                f"{stats['synced']} synced, "
                f"{stats['existing']} existing, "
                f"{stats['closed']} closed, "
                f"{stats['failed']} failed"
            )

        except Exception as e:
            log_error(f"[PositionSync] Fatal error during sync: {e}")

        return stats


__all__ = ["PositionSyncService"]
