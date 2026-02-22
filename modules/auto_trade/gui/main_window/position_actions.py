"""Position action handlers for trading operations."""

from typing import TYPE_CHECKING, Any

from modules.common.ui.logging import log_error, log_info

if TYPE_CHECKING:
    from .main_window import AutoTradeDashboard


class PositionActionHandler:
    """Handles position-related actions from GUI."""

    def __init__(self, parent: "AutoTradeDashboard"):
        self.parent = parent

    def handle_action(self, action_data: dict) -> dict[str, Any]:
        """Handle position actions from GUI."""
        log_info(f"Position action received: {action_data}")

        if not self.parent.data_service.exchange_manager:
            log_error("Exchange manager not initialized")
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
            elif action == "sync_positions":
                return self.sync_positions_from_binance()

        except AttributeError:
            log_error(f"Target {target} does not support action {action}")
            return {"success": False, "error": f"Method not supported: {action}"}
        except Exception as e:
            log_error(f"Error executing {action}: {e}")
            return {"success": False, "error": str(e)}

        return {"success": False, "error": "Unknown action"}

    def sync_positions_from_binance(self) -> dict[str, Any]:
        """
        Sync existing Binance positions into database.

        Returns:
            Result dictionary with success status and statistics
        """
        try:
            from modules.auto_trade.execution.binance_client import BinanceClient
            from modules.auto_trade.gui.utils.credential_manager import CredentialManager
            from modules.auto_trade.gui.utils.position_sync_service import PositionSyncService

            log_info("[PositionSync] Starting manual position sync...")

            # Get credentials
            credential_manager = CredentialManager()
            api_config = credential_manager.load_credentials("binance")

            # Get testnet setting
            api_settings = self.parent.settings_manager.get("api", {})
            testnet = api_settings.get("mode", "").upper() == "TESTNET"

            # Create Binance client
            client = BinanceClient(
                api_key=api_config.get("api_key") or "",
                api_secret=api_config.get("api_secret") or "",
                testnet=testnet,
                dry_run=False,
            )

            # Perform sync (RepositoryContext is created internally)
            stats = PositionSyncService.sync_all_positions(client)

            log_info(f"[PositionSync] Sync completed: {stats}")

            return {
                "success": True,
                "stats": stats,
                "message": f"Synced {stats['synced']} positions, {stats['existing']} already existed",
            }

        except Exception as e:
            log_error(f"[PositionSync] Fatal error: {e}", exc_info=True)
            return {"success": False, "error": str(e), "message": f"Sync failed: {str(e)}"}

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
