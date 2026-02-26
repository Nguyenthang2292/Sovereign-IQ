"""
WebSocket Negative Breakeven Handler
=====================================

Integrates negative breakeven with WebSocket position updates.
Called from PositionMonitor when position updates arrive.

Created: 2026-02-06
"""

import time
from typing import Any, Dict, List, Optional

from modules.auto_trade.database import get_open_positions, mark_be_moved
from modules.auto_trade.execution.binance_client import BinanceClient
from modules.auto_trade.execution.negative_breakeven import NegativeBreakevenLogic
from modules.common.ui.logging import log_error, log_info, log_warn


def _symbol_for_ccxt(symbol: str) -> str:
    """Convert DB symbol (e.g. SKLUSDT) to CCXT format (SKL/USDT) for API calls."""
    s = (symbol or "").strip()
    if "/" in s:
        return s
    if s.endswith("USDT"):
        return s[:-4] + "/USDT"
    return s + "/USDT" if s else s


class WebSocketNegativeBreakevenHandler:
    """
    Handles negative breakeven updates triggered by WebSocket position updates.

    Debounces updates per symbol to avoid excessive API calls.
    """

    def __init__(
        self,
        settings_manager,
        binance_client: Optional[BinanceClient] = None,
        debounce_seconds: float = 2.0,
    ):
        """
        Initialize WebSocket negative breakeven handler.

        Args:
            settings_manager: Settings manager to get TP/SL settings
            binance_client: Binance client for modifying TP orders
            debounce_seconds: Minimum time between updates per symbol
        """
        self.settings_manager = settings_manager
        self.binance_client = binance_client
        self.debounce_seconds = debounce_seconds
        self._last_update_times: Dict[str, float] = {}  # symbol -> timestamp

    def on_position_update(self, position_snapshot: Any) -> None:
        """
        Handle position update from WebSocket.

        Called by PositionMonitor when position updates arrive.

        Args:
            position_snapshot: PositionSnapshot object with mark price
        """
        try:
            # Check debounce
            symbol: str = str(position_snapshot.symbol)
            now: float = time.time()
            last_update: float = self._last_update_times.get(symbol, 0.0)

            if now - last_update < self.debounce_seconds:
                return  # Skip if too soon

            # Get TP/SL settings
            tp_sl_settings: dict = self.settings_manager.get("tp_sl", {})
            negative_be_enabled: bool = bool(tp_sl_settings.get("negative_be_enabled", False))
            negative_be_threshold_pct: float = float(tp_sl_settings.get("negative_be_threshold_pct", 2.0))

            if not negative_be_enabled:
                return

            if negative_be_threshold_pct <= 0:
                return

            # Get mark price from position snapshot
            mark_price: float = float(position_snapshot.mark_price)
            if not mark_price or mark_price <= 0:
                log_warn(f"Invalid mark price for {symbol}: {mark_price}")
                return

            # Find open orders for this symbol
            open_orders: Optional[List[Any]] = get_open_positions(symbol=symbol.replace("/", ""))

            if not open_orders:
                return

            for order in open_orders:
                try:
                    # Skip if breakeven already moved
                    if order.get("be_moved", False):
                        continue

                    # Check if we should trigger negative breakeven
                    should_trigger: bool = NegativeBreakevenLogic.should_trigger(
                        entry_price=float(order.get("entry_price", 0.0)),
                        mark_price=mark_price,
                        stop_loss=float(order.get("stop_loss", 0.0)),
                        side=str(order.get("side", "")),
                        threshold_pct=negative_be_threshold_pct,
                        be_moved=bool(order.get("be_moved", False)),
                    )

                    if not should_trigger:
                        continue

                    # Should trigger - calculate new TP (entry price)
                    new_tp: float = NegativeBreakevenLogic.get_new_take_profit(float(order.get("entry_price", 0.0)))

                    if self.binance_client:
                        try:
                            order_symbol: str = str(order.get("symbol", ""))
                            ccxt_symbol: str = _symbol_for_ccxt(order_symbol)
                            # Modify take profit on exchange
                            modify_result: Optional[dict] = self.binance_client.modify_take_profit(
                                symbol=ccxt_symbol,
                                position_id=None,
                                take_profit_price=new_tp,
                            )
                            success: bool = modify_result is not None and (
                                bool(modify_result.get("success"))
                                or bool(modify_result.get("id"))
                                or bool(modify_result.get("dry_run"))
                            )
                            if success:
                                old_tp: Optional[float] = order.get("take_profit")

                                # Use mark_be_moved to update DB
                                mark_be_moved(
                                    order_id=str(order.get("order_id", "")),
                                    new_take_profit=new_tp,
                                    verify_programmatic=True,
                                )

                                self._last_update_times[symbol] = now
                                log_info(
                                    f"[WS] Negative breakeven triggered for {order_symbol} "
                                    f"{order.get('side', '')}: TP {old_tp} → {new_tp} (entry price)"
                                )
                            else:
                                error_msg: str = str((modify_result or {}).get("error", "Unknown error"))
                                log_error(f"[WS] Failed to modify TP for {order.get('order_id', '')}: {error_msg}")
                        except Exception as modify_exc:
                            log_error(f"[WS] Error modifying TP for {order.get('order_id', '')}: {modify_exc}")
                    else:
                        # No Binance client / dry-run mode - just log
                        log_info(
                            f"[WS] Negative breakeven would trigger for {order.get('symbol', '')}: "
                            f"TP {order.get('take_profit')} → {new_tp}"
                        )
                        self._last_update_times[symbol] = now

                except Exception as order_exc:
                    log_error(f"[WS] Error processing order {order.get('order_id', '')}: {order_exc}")

        except Exception as e:
            log_error(f"[WS] Error in negative breakeven handler: {e}")


def create_websocket_negative_breakeven_handler(
    settings_manager,
    binance_client: Optional[BinanceClient] = None,
    debounce_seconds: float = 2.0,
) -> WebSocketNegativeBreakevenHandler:
    """
    Factory function to create a WebSocketNegativeBreakevenHandler.

    Args:
        settings_manager: Settings manager
        binance_client: Optional Binance client
        debounce_seconds: Debounce time between updates

    Returns:
        WebSocketNegativeBreakevenHandler instance
    """
    return WebSocketNegativeBreakevenHandler(
        settings_manager=settings_manager,
        binance_client=binance_client,
        debounce_seconds=debounce_seconds,
    )
