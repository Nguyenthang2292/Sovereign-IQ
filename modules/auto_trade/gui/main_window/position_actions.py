"""Position action handlers for trading operations."""

from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from .main_window import AutoTradeDashboard


class PositionActionHandler:
    """Handles position-related actions from GUI."""

    def __init__(self, parent: "AutoTradeDashboard"):
        self.parent = parent

    def handle_action(self, action_data: dict) -> dict[str, Any]:
        """Handle position actions from GUI."""
        print(f"Position action received: {action_data}")

        if not self.parent.data_service.exchange_manager:
            print("Error: Exchange manager not initialized")
            return {"success": False, "error": "Exchange manager unavailable"}

        mgr = self.parent.data_service.exchange_manager
        client = getattr(mgr, "client", None)
        if client is None or not hasattr(client, "close_position"):
            return {"success": False, "error": "Position actions not available (no trading client)"}
        target = client

        action = action_data.get("action")
        symbol = action_data.get("symbol")

        try:
            if action == "close_position":
                return self._close_position(target, action_data)
            elif action == "partial_close":
                return self._partial_close(target, action_data)
            elif action == "modify_tp_sl":
                return self._modify_tp_sl(target, action_data)
            elif action == "add_margin":
                return self._add_margin(target, action_data)
            elif action == "cancel_orders":
                return target.cancel_open_orders(symbol)

        except AttributeError:
            print(f"Error: Target {target} does not support action {action}")
            return {"success": False, "error": f"Method not supported: {action}"}
        except Exception as e:
            print(f"Error executing {action}: {e}")
            return {"success": False, "error": str(e)}

        return {"success": False, "error": "Unknown action"}

    def _close_position(self, target, action_data: dict) -> dict[str, Any]:
        """Execute close position action."""
        side = action_data.get("side")
        size = action_data.get("size")
        close_type = action_data.get("type", "market")
        limit_price = action_data.get("limit_price")
        return target.close_position(action_data.get("symbol"), side, size, close_type, limit_price)

    def _partial_close(self, target, action_data: dict) -> dict[str, Any]:
        """Execute partial close action."""
        symbol = action_data.get("symbol")
        side = action_data.get("side")
        size = action_data.get("size")
        return target.close_position(symbol, side, size, "market")

    def _modify_tp_sl(self, target, action_data: dict) -> dict[str, Any]:
        """Execute modify TP/SL action."""
        symbol = action_data.get("symbol")
        position_id = action_data.get("position_id")
        tp = action_data.get("take_profit")
        sl = action_data.get("stop_loss")
        return target.modify_tp_sl(symbol, position_id, tp, sl)

    def _add_margin(self, target, action_data: dict) -> dict[str, Any]:
        """Execute add margin action."""
        symbol = action_data.get("symbol")
        amount = action_data.get("amount")
        return target.modify_margin(symbol, amount, type=1)
