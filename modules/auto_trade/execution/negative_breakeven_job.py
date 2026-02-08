"""
Negative Breakeven Timer Job
=============================

Polling-based negative breakeven job that runs every 30 seconds.
Checks all open PROGRAMMATIC orders and moves TP to entry when position
is losing by threshold percentage but hasn't hit stop loss yet.

Created: 2026-02-06
"""

import logging
from typing import Any, Dict, List, Optional

from database.models import Order
from database.queries import get_open_positions, mark_be_moved
from sqlalchemy.orm import Session

from execution.binance_client import BinanceClient
from execution.negative_breakeven import NegativeBreakevenLogic

logger = logging.getLogger(__name__)


class NegativeBreakevenJob:
    """
    Timer-based negative breakeven job.

    Polls open orders every 30 seconds and moves take profit to entry price
    when position loss reaches threshold and hasn't hit stop loss yet.
    """

    def __init__(
        self,
        settings_manager,
        db_session_scope,
        binance_client: Optional[BinanceClient] = None,
    ):
        """
        Initialize negative breakeven job.

        Args:
            settings_manager: Settings manager to get TP/SL settings
            db_session_scope: Database session scope (context manager)
            binance_client: Binance client for modifying TP orders (optional)
        """
        self.settings_manager = settings_manager
        self.db_session_scope = db_session_scope
        self.binance_client = binance_client

    def run(self) -> Dict[str, Any]:
        """
        Execute negative breakeven check for all open orders.

        Returns:
            Dictionary with results summary
        """
        results: Dict[str, Any] = {
            "orders_checked": 0,
            "orders_updated": 0,
            "errors": [],
            "updates": [],
        }

        try:
            # Get TP/SL settings
            tp_sl_settings: dict = self.settings_manager.get("tp_sl", {})
            negative_be_enabled: bool = bool(tp_sl_settings.get("negative_be_enabled", False))
            negative_be_threshold_pct: float = float(tp_sl_settings.get("negative_be_threshold_pct", 2.0))

            if not negative_be_enabled:
                logger.debug("Negative breakeven is disabled, skipping")
                return results

            if negative_be_threshold_pct <= 0:
                logger.warning(f"Invalid negative BE threshold: {negative_be_threshold_pct}")
                return results

            # Get all open programmatic orders
            with self.db_session_scope() as session:
                open_orders: Optional[List[Order]] = get_open_positions(session)

                if not open_orders:
                    logger.debug("No open orders to check")
                    return results

                # Group orders by symbol for efficient price fetching
                orders_by_symbol: Dict[str, List[Order]] = {}
                for order in open_orders:
                    sym: str = str(getattr(order, "symbol", ""))
                    if sym not in orders_by_symbol:
                        orders_by_symbol[sym] = []
                    orders_by_symbol[sym].append(order)

                # Process each symbol
                for symbol, orders in orders_by_symbol.items():
                    try:
                        # Fetch current mark price
                        mark_price: Optional[float] = self._get_mark_price(symbol)
                        if mark_price is None:
                            logger.warning(f"Could not get mark price for {symbol}")
                            continue

                        # Process each order for this symbol
                        for order in orders:
                            results["orders_checked"] += 1

                            try:
                                update_result: Dict[str, Any] = self._process_order(
                                    session,
                                    order,
                                    mark_price,
                                    negative_be_threshold_pct,
                                )

                                if update_result["updated"]:
                                    results["orders_updated"] += 1
                                    results["updates"].append(update_result)

                            except Exception as e:
                                error_msg_order: str = f"Error processing order {getattr(order, 'order_id', '')}: {e}"
                                logger.error(error_msg_order)
                                results["errors"].append(error_msg_order)

                    except Exception as e:
                        error_msg_sym: str = f"Error processing symbol {symbol}: {e}"
                        logger.error(error_msg_sym)
                        results["errors"].append(error_msg_sym)

                # Commit all changes
                session.commit()

        except Exception as e:
            error_msg: str = f"Error in negative breakeven job: {e}"
            logger.error(error_msg)
            results["errors"].append(error_msg)
            return results

        return results

    def _get_mark_price(self, symbol: str) -> Optional[float]:
        """
        Get current mark price for a symbol.

        Args:
            symbol: Trading symbol (e.g., 'BTCUSDT')

        Returns:
            Current mark price or None if unavailable
        """
        if not self.binance_client:
            logger.warning("No Binance client available for fetching mark price")
            return None
        try:
            ticker = self.binance_client.fetch_ticker(symbol)
            if ticker and "last" in ticker:
                return float(ticker["last"])
        except Exception as e:
            logger.error(f"Error fetching mark price for {symbol}: {e}")
        return None

    def _process_order(
        self,
        session: Session,
        order: Order,
        mark_price: float,
        threshold_pct: float,
    ) -> Dict[str, Any]:
        """
        Process a single order for negative breakeven.

        Args:
            session: Database session
            order: Order object
            mark_price: Current mark price
            threshold_pct: Negative breakeven threshold percentage

        Returns:
            Dictionary with update result
        """
        entry_price: float = float(getattr(order, "entry_price", 0.0))
        side: str = str(getattr(order, "side", ""))
        stop_loss: float = float(getattr(order, "stop_loss", 0.0))
        be_moved: bool = bool(getattr(order, "be_moved", False))
        order_id_str: str = str(getattr(order, "order_id", ""))

        result: Dict[str, Any] = {
            "order_id": order_id_str,
            "symbol": str(getattr(order, "symbol", "")),
            "updated": False,
            "message": "",
            "old_tp": getattr(order, "take_profit", None),
            "new_tp": None,
        }

        # Skip if breakeven already moved
        if be_moved:
            result["message"] = "Breakeven already moved, skipping"
            return result

        # Check if we should trigger negative breakeven
        should_trigger: bool = NegativeBreakevenLogic.should_trigger(
            entry_price=entry_price,
            mark_price=mark_price,
            stop_loss=stop_loss,
            side=side,
            threshold_pct=threshold_pct,
            be_moved=be_moved,
        )

        if not should_trigger:
            profit_pct: float = NegativeBreakevenLogic.calculate_profit_pct(
                entry_price=entry_price,
                mark_price=mark_price,
                side=side,
            )
            result["message"] = f"Conditions not met (profit: {profit_pct:.2f}%)"
            return result

        # Should trigger - calculate new TP (entry price)
        new_tp: float = NegativeBreakevenLogic.get_new_take_profit(entry_price)

        # Update TP on exchange first
        if self.binance_client:
            try:
                modify_result: Optional[dict] = self.binance_client.modify_take_profit(
                    symbol=str(getattr(order, "symbol", "")),
                    position_id=None,
                    take_profit_price=new_tp,
                )
                success: bool = modify_result is not None and (
                    bool(modify_result.get("success"))
                    or bool(modify_result.get("id"))
                    or bool(modify_result.get("dry_run"))
                )
                if success:
                    # Update order in database
                    old_tp: Optional[float] = getattr(order, "take_profit", None)

                    # Use mark_be_moved to update DB
                    mark_be_moved(
                        session=session,
                        order_id=order_id_str,
                        new_take_profit=new_tp,
                        verify_programmatic=True,
                    )

                    result["updated"] = True
                    result["new_tp"] = new_tp
                    result["message"] = (
                        f"Negative breakeven triggered: TP moved from {old_tp} to {new_tp} (entry price)"
                    )

                    sym_str: str = str(getattr(order, "symbol", ""))
                    logger.info(f"Negative breakeven triggered for {sym_str} {side}: TP {old_tp} → {new_tp}")
                else:
                    error_msg: str = str((modify_result or {}).get("error", "Unknown error"))
                    result["message"] = f"Failed to modify TP on exchange: {error_msg}"
                    logger.error(f"Failed to modify TP for {order_id_str}: {error_msg}")

            except Exception as e:
                result["message"] = f"Error modifying TP: {e}"
                logger.error(f"Error modifying TP for {order_id_str}: {e}")
        else:
            # Dry run or no client - just log what would happen
            old_tp_dry: Optional[float] = getattr(order, "take_profit", None)
            result["message"] = f"Would move TP to {new_tp} (dry run or no client)"
            sym_str_dry: str = str(getattr(order, "symbol", ""))
            logger.info(f"[DRY RUN] Negative breakeven would trigger for {sym_str_dry}: TP {old_tp_dry} → {new_tp}")

            # Still update database in dry run mode for tracking
            if not self.binance_client:
                mark_be_moved(
                    session=session,
                    order_id=order_id_str,
                    new_take_profit=new_tp,
                    verify_programmatic=True,
                )
                result["updated"] = True
                result["new_tp"] = new_tp

        return result


def create_negative_breakeven_job(
    settings_manager,
    db_session_scope,
    binance_client: Optional[BinanceClient] = None,
) -> NegativeBreakevenJob:
    """
    Factory function to create a NegativeBreakevenJob instance.

    Args:
        settings_manager: Settings manager
        db_session_scope: Database session scope
        binance_client: Optional Binance client

    Returns:
        NegativeBreakevenJob instance
    """
    return NegativeBreakevenJob(
        settings_manager=settings_manager,
        db_session_scope=db_session_scope,
        binance_client=binance_client,
    )
