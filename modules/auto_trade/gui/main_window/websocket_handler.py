"""WebSocket callback handlers for real-time updates."""

from modules.common.ui.logging import log_debug, log_error, log_info
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

        binance_client = self._get_binance_client()

        # WebSocket-driven trailing stop: same logic as timer job, triggered on each position update (debounced)
        from modules.auto_trade.execution.trailing_stop_ws_handler import create_websocket_trailing_stop_handler

        self._ws_trailing_handler = create_websocket_trailing_stop_handler(
            settings_manager=self.parent.settings_manager,
            binance_client=binance_client,
            debounce_seconds=2.0,
        )
        self.parent.ws_data_service.on_position_update(self._ws_trailing_handler.on_position_update)

        # WebSocket-driven negative breakeven: same logic as timer job, triggered on each position update (debounced)
        from modules.auto_trade.execution.negative_breakeven_ws_handler import (
            create_websocket_negative_breakeven_handler,
        )

        self._ws_negative_be_handler = create_websocket_negative_breakeven_handler(
            settings_manager=self.parent.settings_manager,
            binance_client=binance_client,
            debounce_seconds=2.0,
        )
        self.parent.ws_data_service.on_position_update(self._ws_negative_be_handler.on_position_update)

        log_info("[WebSocket] Callbacks registered")

    def _get_binance_client(self):
        """Build BinanceClient from current dashboard credentials when available."""
        try:
            if hasattr(self.parent, "data_service") and hasattr(self.parent.data_service, "_get_or_create_client"):
                return self.parent.data_service._get_or_create_client()
        except Exception:
            pass
        return None

    def _on_position_update(self, position: PositionSnapshot):
        """Handle position update from WebSocket (called from background thread)."""
        self.parent.after(0, lambda: self._update_position_display(position))

    def _update_position_display(self, position: PositionSnapshot):
        """Update position display in GUI (runs in main thread)."""
        positions_list = self._convert_positions_to_dicts(self.parent.ws_data_service.get_positions())

        log_debug(f"[WebSocket] Position update: {len(positions_list)} positions to display")
        for p in positions_list:
            log_debug(
                f"[WebSocket]   - {p['symbol']} {p['side']}: size={p['size']}, "
                f"entry={p['entry_price']}, pnl={p['pnl']:.2f}"
            )

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
            log_error(f"[WebSocket] Could not calculate unrealized P&L: {e}")

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
            log_info(f"[WebSocket] Order filled: {order.symbol} {order.side.upper()} {order.filled}/{order.amount}")
        elif order.status == "canceled":
            log_info(f"[WebSocket] Order canceled: {order.symbol}")
        elif order.status == "rejected":
            log_error(f"[WebSocket] Order rejected: {order.symbol}")

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

            log_debug(f"[WebSocket] Syncing TP/SL/BE for {p.symbol}...")

            # Use data_service cached explicit TP/SL
            if hasattr(self.parent, "data_service") and hasattr(self.parent.data_service, "get_cached_tpsl"):
                try:
                    tpsl = self.parent.data_service.get_cached_tpsl(p.symbol)
                    take_profit = tpsl.get("take_profit")
                    stop_loss = tpsl.get("stop_loss")
                    break_even = tpsl.get("break_even")
                    log_debug(f"[WebSocket] Fetched {p.symbol}: TP=${take_profit}, SL=${stop_loss}, BE=${break_even}")
                except Exception as e:
                    log_error(f"[WebSocket] Cache fetch failed for {p.symbol}: {e}")

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
