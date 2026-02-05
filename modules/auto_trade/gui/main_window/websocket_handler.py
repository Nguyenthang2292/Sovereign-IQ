"""WebSocket callback handlers for real-time updates."""

from modules.auto_trade.monitoring.account_monitor import BalanceSnapshot, OrderSnapshot
from modules.auto_trade.monitoring.position_monitor import PositionSnapshot


class WebSocketHandler:
    """Handles WebSocket callbacks and updates UI components."""

    def __init__(self, parent):
        self.parent = parent

    def register_callbacks(self):
        """Register callbacks for WebSocket real-time updates."""
        from gui.utils.modes import TradingMode

        if self.parent.mode == TradingMode.DRY_RUN:
            return

        self.parent.ws_data_service.on_position_update(self._on_position_update)
        self.parent.ws_data_service.on_balance_update(self._on_balance_update)
        self.parent.ws_data_service.on_order_update(self._on_order_update)
        print("✅ WebSocket callbacks registered")

    def _on_position_update(self, position: PositionSnapshot):
        """Handle position update from WebSocket (called from background thread)."""
        self.parent.after(0, lambda: self._update_position_display(position))

    def _update_position_display(self, position: PositionSnapshot):
        """Update position display in GUI (runs in main thread)."""
        positions_list = self._convert_positions_to_dicts(self.parent.ws_data_service.get_positions())

        if hasattr(self.parent, "positions_frame"):
            self.parent.positions_frame.update_positions(positions_list)

        self.parent._update_timestamp()

    def _on_balance_update(self, balance: BalanceSnapshot):
        """Handle balance update from WebSocket."""
        self.parent.after(0, lambda: self._update_balance_display(balance))

    def _update_balance_display(self, balance: BalanceSnapshot):
        """Update balance display in GUI."""
        account_data = {
            "balance": balance.total,
            "free_balance": balance.free,
            "used_balance": balance.used,
            "equity": balance.total,
        }

        if hasattr(self.parent, "account_frame"):
            self.parent.account_frame.update_data(account_data)

        self.parent._update_timestamp()

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
        return [
            {
                "symbol": p.symbol,
                "side": p.side.upper(),
                "size": p.position_amt,
                "entry_price": p.entry_price,
                "mark_price": p.mark_price,
                "pnl": p.unrealized_pnl,
                "pnl_percent": p.unrealized_pnl_percent,
                "leverage": p.leverage,
            }
            for p in positions
        ]
