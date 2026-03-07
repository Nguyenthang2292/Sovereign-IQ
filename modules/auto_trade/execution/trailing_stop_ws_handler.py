"""
WebSocket Trailing Stop Handler
================================

Integrates trailing stop with WebSocket position updates.
Called from PositionMonitor when position updates arrive.

Created: 2026-02-06
"""

import time
from typing import Any, Dict, List, Optional

from modules.auto_trade.database import get_open_positions
from modules.auto_trade.execution.binance_client import BinanceClient
from modules.auto_trade.execution.trailing_stop import TrailingStopResult, calculate_trailing_stop
from modules.common.domain.symbol_codec import SymbolCodec
from modules.common.ui.logging import log_error, log_info, log_warn

_SYMBOL_CODEC = SymbolCodec()


def _symbol_for_ccxt(symbol: str) -> str:
    """Convert any symbol format to CCXT spot format for API calls."""
    return str(_SYMBOL_CODEC.to_ccxt(symbol))


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
            trailing_stop_enabled: bool = bool(tp_sl_settings.get("trailing_stop", False))
            trailing_step_pct: float = float(tp_sl_settings.get("trailing_step_pct", 2.0))
            trailing_limit_steps: bool = bool(tp_sl_settings.get("trailing_limit_steps", False))
            trailing_max_steps: int = int(tp_sl_settings.get("trailing_max_steps", 5))

            if not trailing_stop_enabled:
                return

            if trailing_step_pct <= 0:
                return

            # Get mark price from position snapshot
            mark_price: float = float(position_snapshot.mark_price)
            if not mark_price or mark_price <= 0:
                log_warn(f"Invalid mark price for {symbol}: {mark_price}")
                return

            # Find open orders for this symbol
            open_orders: Optional[List[Any]] = get_open_positions(symbol=_SYMBOL_CODEC.to_db(symbol))

            if not open_orders:
                return

            for order in open_orders:
                try:
                    # Calculate trailing stop
                    trailing_result: TrailingStopResult = calculate_trailing_stop(
                        entry_price=float(order.get("entry_price", 0.0)),
                        current_price=mark_price,
                        side=str(order.get("side", "")),
                        step_index=int(order.get("trailing_step_index", 0)),
                        step_pct=trailing_step_pct,
                        current_sl=(float(order.get("stop_loss", 0.0)) if order.get("stop_loss") is not None else None),
                        limit_steps=trailing_limit_steps,
                        max_steps=trailing_max_steps,
                    )

                    if not trailing_result.should_step:
                        continue

                    # We should step - update SL
                    new_sl: Optional[float] = trailing_result.new_sl_price

                    if self.binance_client and new_sl:
                        try:
                            order_symbol: str = str(order.get("symbol", ""))
                            ccxt_symbol: str = _symbol_for_ccxt(order_symbol)
                            # Modify stop loss on exchange
                            modify_result: Optional[dict] = self.binance_client.modify_stop_loss(
                                symbol=ccxt_symbol,
                                position_id=None,
                                stop_loss_price=new_sl,
                            )
                            success: bool = modify_result is not None and (
                                bool(modify_result.get("success"))
                                or bool(modify_result.get("id"))
                                or bool(modify_result.get("dry_run"))
                            )
                            if success:
                                # Update order in DynamoDB via RepositoryContext
                                old_sl: Optional[float] = order.get("stop_loss")
                                from modules.auto_trade.database import RepositoryContext

                                ctx = RepositoryContext.from_env()
                                ctx.orders.update(
                                    order.get("order_id"),
                                    {
                                        "stop_loss": new_sl,
                                        "trailing_step_index": trailing_result.next_step_index,
                                    },
                                )
                                self._last_update_times[symbol] = now
                                log_info(
                                    f"[WS] Trailing stop updated for {order_symbol} "
                                    f"{order.get('side', '')}: "
                                    f"SL {old_sl} → {new_sl} (step {order.get('trailing_step_index', 0)})"
                                )
                            else:
                                error_msg: str = str((modify_result or {}).get("error", "Unknown error"))
                                log_error(f"[WS] Failed to modify SL for {order.get('order_id', '')}: {error_msg}")
                        except Exception as modify_exc:
                            log_error(f"[WS] Error modifying SL for {order.get('order_id', '')}: {modify_exc}")
                    else:
                        # No Binance client / dry-run mode - just log
                        log_info(
                            f"[WS] Trailing stop would update for {order.get('symbol', '')}: "
                            f"SL {order.get('stop_loss')} → {new_sl}"
                        )
                        self._last_update_times[symbol] = now

                except Exception as order_exc:
                    log_error(f"[WS] Error processing order {order.get('order_id', '')}: {order_exc}")

        except Exception as e:
            log_error(f"[WS] Error in trailing stop handler: {e}")


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
