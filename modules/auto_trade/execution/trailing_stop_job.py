"""
Trailing Stop Timer Job
========================

Polling-based trailing stop job that runs every 30 seconds.
Checks all open PROGRAMMATIC orders and updates SL when profit reaches step thresholds.

Created: 2026-02-06
"""

import logging
from typing import Dict, List, Optional, Tuple

from sqlalchemy.orm import Session

from database.models import Order
from database.queries import get_open_positions
from execution.binance_client import BinanceClient
from execution.trailing_stop import calculate_trailing_stop

logger = logging.getLogger(__name__)


class TrailingStopJob:
    """
    Timer-based trailing stop job.

    Polls open orders every 30 seconds and updates stop loss
    when profit reaches step thresholds (BE → +step% → +2*step% …).
    """

    def __init__(
        self,
        settings_manager,
        db_session_scope,
        binance_client: Optional[BinanceClient] = None,
    ):
        """
        Initialize trailing stop job.

        Args:
            settings_manager: Settings manager to get TP/SL settings
            db_session_scope: Database session scope (context manager)
            binance_client: Binance client for modifying SL orders (optional)
        """
        self.settings_manager = settings_manager
        self.db_session_scope = db_session_scope
        self.binance_client = binance_client
        self._last_update_times: Dict[str, float] = {}  # Symbol -> last update timestamp

    def run(self) -> Dict[str, any]:
        """
        Execute trailing stop check for all open orders.

        Returns:
            Dictionary with results summary
        """
        results = {
            "orders_checked": 0,
            "orders_updated": 0,
            "errors": [],
            "updates": [],
        }

        try:
            # Get TP/SL settings
            tp_sl_settings = self.settings_manager.get("tp_sl", {})
            trailing_stop_enabled = tp_sl_settings.get("trailing_stop", False)
            trailing_step_pct = tp_sl_settings.get("trailing_step_pct", 2.0)
            trailing_limit_steps = tp_sl_settings.get("trailing_limit_steps", False)
            trailing_max_steps = tp_sl_settings.get("trailing_max_steps", 5)

            if not trailing_stop_enabled:
                logger.debug("Trailing stop is disabled, skipping")
                return results

            if trailing_step_pct <= 0:
                logger.warning(f"Invalid trailing step percentage: {trailing_step_pct}")
                return results

            # Get all open programmatic orders
            with self.db_session_scope() as session:
                open_orders = get_open_positions(session)

                if not open_orders:
                    logger.debug("No open orders to check")
                    return results

                # Group orders by symbol for efficient price fetching
                orders_by_symbol: Dict[str, List[Order]] = {}
                for order in open_orders:
                    if order.symbol not in orders_by_symbol:
                        orders_by_symbol[order.symbol] = []
                    orders_by_symbol[order.symbol].append(order)

                # Process each symbol
                for symbol, orders in orders_by_symbol.items():
                    try:
                        # Fetch current mark price
                        mark_price = self._get_mark_price(symbol)
                        if mark_price is None:
                            logger.warning(f"Could not get mark price for {symbol}")
                            continue

                        # Process each order for this symbol
                        for order in orders:
                            results["orders_checked"] += 1

                            try:
                                update_result = self._process_order(
                                    session,
                                    order,
                                    mark_price,
                                    trailing_step_pct,
                                    trailing_limit_steps,
                                    trailing_max_steps,
                                )

                                if update_result["updated"]:
                                    results["orders_updated"] += 1
                                    results["updates"].append(update_result)

                            except Exception as e:
                                error_msg = f"Error processing order {order.order_id}: {e}"
                                logger.error(error_msg)
                                results["errors"].append(error_msg)

                    except Exception as e:
                        error_msg = f"Error processing symbol {symbol}: {e}"
                        logger.error(error_msg)
                        results["errors"].append(error_msg)

                # Commit all changes
                session.commit()

        except Exception as e:
            error_msg = f"Error in trailing stop job: {e}"
            logger.error(error_msg)
            results["errors"].append(error_msg)

        return results

    def _get_mark_price(self, symbol: str) -> Optional[float]:
        """
        Get current mark price for a symbol.

        Args:
            symbol: Trading symbol (e.g., 'BTCUSDT')

        Returns:
            Current mark price or None if unavailable
        """
        try:
            if self.binance_client:
                # Use Binance client to fetch ticker
                ticker = self.binance_client.fetch_ticker(symbol)
                if ticker and "last" in ticker:
                    return float(ticker["last"])
            else:
                logger.warning("No Binance client available for fetching mark price")
                return None
        except Exception as e:
            logger.error(f"Error fetching mark price for {symbol}: {e}")
            return None

    def _process_order(
        self,
        session: Session,
        order: Order,
        mark_price: float,
        step_pct: float,
        limit_steps: bool,
        max_steps: int,
    ) -> Dict:
        """
        Process a single order for trailing stop.

        Args:
            session: Database session
            order: Order object
            mark_price: Current mark price
            step_pct: Step percentage
            limit_steps: Whether to limit steps
            max_steps: Maximum steps allowed

        Returns:
            Dictionary with update result
        """
        result = {
            "order_id": order.order_id,
            "symbol": order.symbol,
            "updated": False,
            "message": "",
            "old_sl": order.stop_loss,
            "new_sl": None,
            "step_index": order.trailing_step_index,
        }

        # Check if we should step the trailing stop
        trailing_result = calculate_trailing_stop(
            entry_price=order.entry_price,
            current_price=mark_price,
            side=order.side,
            step_index=order.trailing_step_index,
            step_pct=step_pct,
            current_sl=order.stop_loss,
            limit_steps=limit_steps,
            max_steps=max_steps,
        )

        if not trailing_result.should_step:
            result["message"] = trailing_result.message
            return result

        # We should step - update SL on exchange first
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
                    # Update order in database (setattr for SQLAlchemy Column type checker)
                    old_sl = order.stop_loss
                    setattr(order, "stop_loss", new_sl)
                    setattr(order, "trailing_step_index", trailing_result.next_step_index)

                    result["updated"] = True
                    result["new_sl"] = new_sl
                    result["step_index"] = trailing_result.next_step_index
                    result["message"] = (
                        f"Trailing stop stepped from {old_sl} to {new_sl} "
                        f"(step {order.trailing_step_index - 1} → {order.trailing_step_index})"
                    )

                    logger.info(
                        f"Trailing stop updated for {order.symbol} {order.side}: "
                        f"SL {old_sl} → {new_sl} (step {order.trailing_step_index})"
                    )
                else:
                    error_msg = (modify_result or {}).get("error", "Unknown error")
                    result["message"] = f"Failed to modify SL on exchange: {error_msg}"
                    logger.error(f"Failed to modify SL for {order.order_id}: {error_msg}")

            except Exception as e:
                result["message"] = f"Error modifying SL: {e}"
                logger.error(f"Error modifying SL for {order.order_id}: {e}")
        else:
            # Dry run or no client - just log what would happen
            result["message"] = f"Would step SL to {new_sl} (dry run or no client)"
            logger.info(f"[DRY RUN] Trailing stop would update for {order.symbol}: SL {order.stop_loss} → {new_sl}")

            # Still update database in dry run mode for tracking
            if not self.binance_client:
                setattr(order, "stop_loss", new_sl)
                setattr(order, "trailing_step_index", trailing_result.next_step_index)
                result["updated"] = True
                result["new_sl"] = new_sl
                result["step_index"] = trailing_result.next_step_index

        return result


def create_trailing_stop_job(
    settings_manager,
    db_session_scope,
    binance_client: Optional[BinanceClient] = None,
) -> TrailingStopJob:
    """
    Factory function to create a TrailingStopJob instance.

    Args:
        settings_manager: Settings manager
        db_session_scope: Database session scope
        binance_client: Optional Binance client

    Returns:
        TrailingStopJob instance
    """
    return TrailingStopJob(
        settings_manager=settings_manager,
        db_session_scope=db_session_scope,
        binance_client=binance_client,
    )
