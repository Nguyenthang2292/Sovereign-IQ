"""
WebSocket Trailing Stop Handler
================================

Integrates trailing stop with WebSocket position updates.
Called from PositionMonitor when position updates arrive.

Created: 2026-02-06
"""

import logging
import time
from typing import Dict, Optional

from database import get_open_positions, session_scope
from execution.binance_client import BinanceClient
from execution.trailing_stop import calculate_trailing_stop

logger = logging.getLogger(__name__)


class WebSocketTrailingStopHandler:
    """
    Handles trailing stop updates triggered by WebSocket position updates.

    Debounces updates per symbol to avoid excessive API calls.
    """

    def __init__(
        self,
        settings_manager,
        binance_client: Optional[BinanceClient] = None,
        debounce_seconds: float = 2.0,
    ):
        """
        Initialize WebSocket trailing stop handler.

        Args:
            settings_manager: Settings manager to get TP/SL settings
            binance_client: Binance client for modifying SL orders
            debounce_seconds: Minimum time between updates per symbol
        """
        self.settings_manager = settings_manager
        self.binance_client = binance_client
        self.debounce_seconds = debounce_seconds
        self._last_update_times: Dict[str, float] = {}  # symbol -> timestamp

    def on_position_update(self, position_snapshot):
        """
        Handle position update from WebSocket.

        Called by PositionMonitor when position updates arrive.

        Args:
            position_snapshot: PositionSnapshot object with mark price
        """
        try:
            # Check debounce
            symbol = position_snapshot.symbol
            now = time.time()
            last_update = self._last_update_times.get(symbol, 0)

            if now - last_update < self.debounce_seconds:
                return  # Skip if too soon

            # Get TP/SL settings
            tp_sl_settings = self.settings_manager.get("tp_sl", {})
            trailing_stop_enabled = tp_sl_settings.get("trailing_stop", False)
            trailing_step_pct = tp_sl_settings.get("trailing_step_pct", 2.0)
            trailing_limit_steps = tp_sl_settings.get("trailing_limit_steps", False)
            trailing_max_steps = tp_sl_settings.get("trailing_max_steps", 5)

            if not trailing_stop_enabled:
                return

            if trailing_step_pct <= 0:
                return

            # Get mark price from position snapshot
            mark_price = position_snapshot.mark_price
            if not mark_price or mark_price <= 0:
                logger.warning(f"Invalid mark price for {symbol}: {mark_price}")
                return

            # Find open orders for this symbol
            with session_scope() as session:
                open_orders = get_open_positions(session, symbol=symbol.replace("/", ""))

                if not open_orders:
                    return

                for order in open_orders:
                    try:
                        # Calculate trailing stop
                        trailing_result = calculate_trailing_stop(
                            entry_price=order.entry_price,
                            current_price=mark_price,
                            side=order.side,
                            step_index=order.trailing_step_index,
                            step_pct=trailing_step_pct,
                            current_sl=order.stop_loss,
                            limit_steps=trailing_limit_steps,
                            max_steps=trailing_max_steps,
                        )

                        if not trailing_result.should_step:
                            continue

                        # We should step - update SL
                        new_sl = trailing_result.new_sl_price

                        if self.binance_client and new_sl:
                            try:
                                # Modify stop loss on exchange
                                modify_result = self.binance_client.modify_stop_loss(
                                    symbol=str(order.symbol),
                                    position_id=None,
                                    stop_loss_price=new_sl,
                                )
                                success = modify_result is not None and (
                                    modify_result.get("success")
                                    or modify_result.get("id")
                                    or modify_result.get("dry_run")
                                )
                                if success:
                                    # Update order in database
                                    old_sl = order.stop_loss
                                    order.stop_loss = new_sl
                                    order.trailing_step_index = trailing_result.next_step_index

                                    self._last_update_times[symbol] = now

                                    logger.info(
                                        f"[WS] Trailing stop updated for {order.symbol} {order.side}: "
                                        f"SL {old_sl} → {new_sl} (step {order.trailing_step_index})"
                                    )
                                else:
                                    error_msg = (modify_result or {}).get("error", "Unknown error")
                                    logger.error(f"[WS] Failed to modify SL for {order.order_id}: {error_msg}")

                            except Exception as e:
                                logger.error(f"[WS] Error modifying SL for {order.order_id}: {e}")
                        else:
                            # Dry run - just log
                            logger.info(
                                f"[WS] Trailing stop would update for {order.symbol}: SL {order.stop_loss} → {new_sl}"
                            )

                            # Still update database in dry run mode
                            order.stop_loss = new_sl
                            order.trailing_step_index = trailing_result.next_step_index
                            self._last_update_times[symbol] = now

                    except Exception as e:
                        logger.error(f"[WS] Error processing order {order.order_id}: {e}")

                # Commit all changes
                session.commit()

        except Exception as e:
            logger.error(f"[WS] Error in trailing stop handler: {e}")


def create_websocket_trailing_stop_handler(
    settings_manager,
    binance_client: Optional[BinanceClient] = None,
    debounce_seconds: float = 2.0,
) -> WebSocketTrailingStopHandler:
    """
    Factory function to create a WebSocketTrailingStopHandler.

    Args:
        settings_manager: Settings manager
        binance_client: Optional Binance client
        debounce_seconds: Debounce time between updates

    Returns:
        WebSocketTrailingStopHandler instance
    """
    return WebSocketTrailingStopHandler(
        settings_manager=settings_manager,
        binance_client=binance_client,
        debounce_seconds=debounce_seconds,
    )
