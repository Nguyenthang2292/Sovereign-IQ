"""WebSocket callback handlers for real-time updates."""

from modules.auto_trade.monitoring.account_monitor import BalanceSnapshot, OrderSnapshot
from modules.auto_trade.monitoring.position_monitor import PositionSnapshot


class WebSocketHandler:
    """Handles WebSocket callbacks and updates UI components."""

    def __init__(self, parent):
        self.parent = parent
        self._ws_trailing_handler = None
        self._ws_negative_be_handler = None

    def register_callbacks(self):
        """Register callbacks for WebSocket real-time updates."""
        from modules.auto_trade.gui.utils.modes import TradingMode

        if self.parent.mode == TradingMode.DRY_RUN:
            return

        self.parent.ws_data_service.on_position_update(self._on_position_update)
        self.parent.ws_data_service.on_balance_update(self._on_balance_update)
        self.parent.ws_data_service.on_order_update(self._on_order_update)

        # WebSocket-driven trailing stop: same logic as timer job, triggered on each position update (debounced)
        from modules.auto_trade.execution.trailing_stop_ws_handler import create_websocket_trailing_stop_handler

        self._ws_trailing_handler = create_websocket_trailing_stop_handler(
            settings_manager=self.parent.settings_manager,
            binance_client=None,  # Optional: pass from data_service when available
            debounce_seconds=2.0,
        )
        self.parent.ws_data_service.on_position_update(self._ws_trailing_handler.on_position_update)

        # WebSocket-driven negative breakeven: same logic as timer job, triggered on each position update (debounced)
        from modules.auto_trade.execution.negative_breakeven_ws_handler import (
            create_websocket_negative_breakeven_handler,
        )

        self._ws_negative_be_handler = create_websocket_negative_breakeven_handler(
            settings_manager=self.parent.settings_manager,
            binance_client=None,  # Optional: pass from data_service when available
            debounce_seconds=2.0,
        )
        self.parent.ws_data_service.on_position_update(self._ws_negative_be_handler.on_position_update)

        print("✅ WebSocket callbacks registered")

    def _on_position_update(self, position: PositionSnapshot):
        """Handle position update from WebSocket (called from background thread)."""
        self.parent.after(0, lambda: self._update_position_display(position))

    def _update_position_display(self, position: PositionSnapshot):
        """Update position display in GUI (runs in main thread)."""
        positions_list = self._convert_positions_to_dicts(self.parent.ws_data_service.get_positions())

        print(f"[WebSocket] Position update: {len(positions_list)} positions to display")
        for p in positions_list:
            print(f"  - {p['symbol']} {p['side']}: size={p['size']}, entry={p['entry_price']}, pnl={p['pnl']:.2f}")

        if hasattr(self.parent, "positions_frame"):
            self.parent.positions_frame.update_positions(positions_list)

        self.parent._update_timestamp()
        if hasattr(self.parent, "status_bar") and hasattr(self.parent, "ws_data_service"):
            self.parent.status_bar.set_connection_status(self.parent.ws_data_service.is_connected)

    def _on_balance_update(self, balance: BalanceSnapshot):
        """Handle balance update from WebSocket."""
        self.parent.after(0, lambda: self._update_balance_display(balance))

    def _update_balance_display(self, balance: BalanceSnapshot):
        """Update balance display in GUI (keys must match AccountFrame.update_data)."""
        # Calculate total unrealized P&L from all positions
        unrealized_pnl = 0.0
        try:
            positions = self.parent.ws_data_service.get_positions()
            for pos in positions:
                unrealized_pnl += pos.unrealized_pnl
        except Exception as e:
            print(f"[WebSocket] Could not calculate unrealized P&L: {e}")

        account_data = {
            "balance": balance.total,
            "available": balance.free,
            "margin_used": balance.used,
            "unrealized_pnl": unrealized_pnl,
            "daily_pnl": 0.0,
            "daily_pnl_percent": 0.0,
        }

        if hasattr(self.parent, "account_frame"):
            self.parent.account_frame.update_data(account_data)

        self.parent._update_timestamp()
        if hasattr(self.parent, "status_bar") and hasattr(self.parent, "ws_data_service"):
            self.parent.status_bar.set_connection_status(self.parent.ws_data_service.is_connected)

    def _on_order_update(self, order: OrderSnapshot):
        """Handle order update from WebSocket."""
        self.parent.after(0, lambda: self._update_order_display(order))

    def _update_order_display(self, order: OrderSnapshot):
        """Update order display and show notifications."""
        if order.status == "closed":
            print(f"✅ Order filled: {order.symbol} {order.side.upper()} {order.filled}/{order.amount}")
        elif order.status == "canceled":
            print(f"❌ Order canceled: {order.symbol}")
        elif order.status == "rejected":
            print(f"⛔ Order rejected: {order.symbol}")

        if order.status == "closed":
            positions_list = self._convert_positions_to_dicts(self.parent.ws_data_service.get_positions())
            if hasattr(self.parent, "positions_frame"):
                self.parent.positions_frame.update_positions(positions_list)

        self.parent._update_timestamp()

    def _convert_positions_to_dicts(self, positions) -> list:
        """Convert PositionSnapshot objects to dict format for UI."""
        result = []
        for p in positions:
            # Fetch TP/SL/BE from Binance and sync to DB
            take_profit = None
            stop_loss = None
            break_even = None

            print(f"[WebSocket] Syncing TP/SL/BE for {p.symbol}...")

            # Use TPSLSyncService for bidirectional sync
            if hasattr(self.parent, 'data_service'):
                try:
                    from modules.auto_trade.execution.binance_client import BinanceClient
                    from modules.auto_trade.gui.utils.tp_sl_sync import TPSLSyncService

                    # Create client
                    client = BinanceClient(
                        api_key=self.parent.data_service.api_key,
                        api_secret=self.parent.data_service.api_secret,
                        testnet=self.parent.data_service.testnet,
                        dry_run=False,
                    )

                    # Fetch from Binance and sync to DB in one call
                    if self.parent.data_service.database_manager:
                        with self.parent.data_service.database_manager.session_scope() as session:
                            sync_result = TPSLSyncService.sync_position_tp_sl(
                                client=client,
                                session=session,
                                symbol=p.symbol,
                                side=p.side,
                                entry_price=p.entry_price
                            )

                            take_profit = sync_result.get("take_profit")
                            stop_loss = sync_result.get("stop_loss")
                            break_even = sync_result.get("break_even")

                            print(f"[WebSocket] ✅ Synced {p.symbol}: TP=${take_profit}, SL=${stop_loss}, BE=${break_even}")
                    else:
                        # No DB, fetch from Binance only
                        tp, sl, _ = TPSLSyncService.fetch_tp_sl_from_binance(client, p.symbol)
                        take_profit = tp
                        stop_loss = sl
                        break_even = TPSLSyncService.detect_break_even(p.entry_price, sl, p.side)
                        print(f"[WebSocket] Fetched {p.symbol}: TP=${take_profit}, SL=${stop_loss} (no DB sync)")

                except Exception as e:
                    print(f"[WebSocket] Sync failed for {p.symbol}: {e}")

                    # Fallback to DB-only if everything fails
                    if hasattr(self.parent, 'data_service') and self.parent.data_service.database_manager:
                        try:
                            from modules.auto_trade.database.models import Order
                            with self.parent.data_service.database_manager.session_scope() as session:
                                db_orders = session.query(Order).filter(
                                    Order.symbol == p.symbol,
                                    Order.status == "OPEN"
                                ).order_by(Order.created_at.desc()).all()

                                if db_orders:
                                    order = db_orders[0]
                                    take_profit = order.take_profit
                                    stop_loss = order.stop_loss
                                    be_moved_flag = getattr(order, 'be_moved', False)
                                    if be_moved_flag is True and stop_loss is not None:
                                        break_even = stop_loss
                                    print(f"[WebSocket]   Fallback to DB-only: TP={take_profit}, SL={stop_loss}")
                        except Exception as db_err:
                            print(f"[WebSocket]   DB fallback failed: {db_err}")

            # For GUI we want Size in quote currency (USD), not contracts.
            notional = getattr(p, "notional", 0.0) or 0.0
            if not notional and p.entry_price:
                notional = abs(p.position_amt * p.entry_price)

            # Margin used: from snapshot or approximate (notional / leverage)
            margin_used = getattr(p, "margin_used", 0.0) or 0.0
            if margin_used <= 0 and notional and p.leverage:
                margin_used = notional / p.leverage

            result.append(
                {
                    "symbol": p.symbol,
                    "side": p.side.upper(),
                    # Size shown in USD
                    "size": notional,
                    # Preserve raw contracts separately for advanced views if needed
                    "contracts": abs(p.position_amt),
                    "entry_price": p.entry_price,
                    "current_price": p.mark_price,  # PositionsFrame expects "current_price"
                    "pnl": p.unrealized_pnl,
                    "pnl_percent": p.unrealized_pnl_percent,
                    "leverage": p.leverage,
                    "margin_used": margin_used,
                    "liquidation_price": p.liquidation_price,
                    "take_profit": take_profit,
                    "stop_loss": stop_loss,
                    "break_even": break_even,
                }
            )

        return result
