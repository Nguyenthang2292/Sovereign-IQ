from typing import Dict, Optional
from gui.utils.mock_price_feed import MockPriceFeed
from gui.utils.dry_run_db import DryRunDB


class DryRunExecutor:
    def __init__(self):
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
    ) -> Dict:
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
                "message": f"Order placed successfully in DRY_RUN mode",
            }
        except Exception as e:
            return {"success": False, "error": str(e), "message": f"Failed to place order: {e}"}

    def close_position(self, symbol: str, side: str, size: float) -> Dict:
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

                self.db.update_position(position_id=pos.get("id"), current_price=current_price, unrealized_pnl=pnl)

                close_size -= close_this

            return {
                "success": True,
                "symbol": symbol,
                "side": side,
                "size": size,
                "current_price": current_price,
                "message": f"Position closed successfully in DRY_RUN mode",
            }
        except Exception as e:
            return {"success": False, "error": str(e), "message": f"Failed to close position: {e}"}

    def modify_tp_sl(self, symbol: str, tp_price: Optional[float], sl_price: Optional[float]) -> Dict:
        try:
            positions = self.db.get_open_positions_by_symbol(symbol)

            if not positions:
                return {"success": False, "error": "No open positions found", "message": "No positions to modify"}

            for pos in positions:
                self.db.update_position(position_id=pos.get("id"), take_profit=tp_price, stop_loss=sl_price)

            return {
                "success": True,
                "symbol": symbol,
                "take_profit": tp_price,
                "stop_loss": sl_price,
                "message": f"TP/SL modified successfully in DRY_RUN mode",
            }
        except Exception as e:
            return {"success": False, "error": str(e), "message": f"Failed to modify TP/SL: {e}"}
