"""
Dry Run Executor Module

Simulates order execution for testing and development without
executing real trades on an exchange.
"""

from typing import Any, Dict, Optional

# Local imports
from modules.auto_trade.gui.utils.dry_run_db import DryRunDB
from modules.auto_trade.gui.utils.mock_price_feed import MockPriceFeed


class DryRunExecutor:
    """
    Executes simulated trades for dry-run mode.

    Handles order placement, position closing, and TP/SL modification
    without connecting to a real exchange.
    """

    def __init__(self) -> None:
        """Initialize dry run executor with mock price feed and database."""
        self.price_feed = MockPriceFeed()
        self.db = DryRunDB()

    def place_order(
        self,
        symbol: str,
        side: str,
        amount: float,
        leverage: int,
        tp: Optional[float] = None,
        sl: Optional[float] = None,
    ) -> Dict[str, Any]:
        """
        Place a simulated order.

        Args:
            symbol: Trading symbol (e.g., "BTC/USDT")
            side: Order side ("LONG" or "SHORT")
            amount: Order size in base currency
            leverage: Leverage multiplier
            tp: Take profit price (optional)
            sl: Stop loss price (optional)

        Returns:
            Dictionary with order result containing success status and details
        """
        try:
            current_price = self.price_feed.get_current_price(symbol)

            entry_price = current_price
            position_id = self.db.insert_position(
                symbol=symbol,
                side=side,
                entry_price=entry_price,
                current_price=current_price,
                size=amount,
                leverage=leverage,
                take_profit=tp,
                stop_loss=sl,
            )

            return {
                "success": True,
                "order_id": position_id,
                "symbol": symbol,
                "side": side,
                "entry_price": entry_price,
                "size": amount,
                "message": "Order placed successfully in DRY_RUN mode",
            }
        except Exception as e:
            return {"success": False, "error": str(e), "message": f"Failed to place order: {e}"}

    def close_position(self, symbol: str, side: str, size: float) -> Dict[str, Any]:
        """
        Close a simulated position.

        Args:
            symbol: Trading symbol
            side: Position side ("LONG" or "SHORT")
            size: Size to close

        Returns:
            Dictionary with close result containing success status and details
        """
        try:
            current_price = self.price_feed.get_current_price(symbol)
            positions = self.db.get_open_positions_by_symbol(symbol, side)

            if not positions:
                return {"success": False, "error": "No open positions found", "message": "No positions to close"}

            total_size = sum(pos.get("size", 0) for pos in positions)
            close_size = min(size, total_size)

            for pos in positions:
                pos_size = pos.get("size", 0)
                if close_size <= 0:
                    break

                close_this = min(pos_size, close_size)
                entry_price = float(pos.get("entry_price", 0))

                if side == "LONG":
                    pnl = (current_price - entry_price) * close_this
                else:
                    pnl = (entry_price - current_price) * close_this

                pos_id = pos.get("id")
                if pos_id is not None:
                    self.db.update_position(position_id=int(pos_id), current_price=current_price, unrealized_pnl=pnl)

                close_size -= close_this

            return {
                "success": True,
                "symbol": symbol,
                "side": side,
                "size": size,
                "current_price": current_price,
                "message": "Position closed successfully in DRY_RUN mode",
            }
        except Exception as e:
            return {"success": False, "error": str(e), "message": f"Failed to close position: {e}"}

    def modify_tp_sl(self, symbol: str, tp_price: Optional[float], sl_price: Optional[float]) -> Dict[str, Any]:
        """
        Modify take profit and stop loss for simulated positions.

        Args:
            symbol: Trading symbol
            tp_price: New take profit price (optional)
            sl_price: New stop loss price (optional)

        Returns:
            Dictionary with modification result
        """
        try:
            positions = self.db.get_open_positions_by_symbol(symbol)

            if not positions:
                return {"success": False, "error": "No open positions found", "message": "No positions to modify"}

            for pos in positions:
                pos_id = pos.get("id")
                if pos_id is not None:
                    self.db.update_position(position_id=int(pos_id), take_profit=tp_price, stop_loss=sl_price)

            return {
                "success": True,
                "symbol": symbol,
                "take_profit": tp_price,
                "stop_loss": sl_price,
                "message": "TP/SL modified successfully in DRY_RUN mode",
            }
        except Exception as e:
            return {"success": False, "error": str(e), "message": f"Failed to modify TP/SL: {e}"}
