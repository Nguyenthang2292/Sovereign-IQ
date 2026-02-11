"""
Ensure TP/SL Job
================

Runs periodically when auto-trade is on. For each open PROGRAMMATIC position,
checks if TP and/or SL exist on Binance; if missing, places them from config
(default_tp, default_sl percentages).
"""

import logging
from typing import Any, Dict, List, Optional, Tuple

from database.models import Order
from database.queries import get_open_positions
from sqlalchemy.orm import Session

from execution.binance_client import BinanceClient

logger = logging.getLogger(__name__)


def _tp_sl_prices_from_pct(
    entry_price: float,
    side: str,
    tp_pct: float,
    sl_pct: float,
) -> Tuple[float, float]:
    """
    Compute TP and SL prices from entry and percentages.

    LONG: TP above entry, SL below. SHORT: TP below entry, SL above.

    Args:
        entry_price: Entry price
        side: 'LONG' or 'SHORT'
        tp_pct: Take profit percentage (e.g. 5.0 for 5%)
        sl_pct: Stop loss percentage (e.g. 2.5 for 2.5%)

    Returns:
        (tp_price, sl_price)
    """
    if not entry_price or entry_price <= 0:
        return 0.0, 0.0
    mult_tp = 1.0 + (tp_pct / 100.0)
    mult_sl = 1.0 - (sl_pct / 100.0)
    if str(side).upper() == "SHORT":
        mult_tp = 1.0 - (tp_pct / 100.0)
        mult_sl = 1.0 + (sl_pct / 100.0)
    return entry_price * mult_tp, entry_price * mult_sl


class EnsureTPSLJob:
    """
    Job that ensures every open programmatic position has TP and SL on Binance.
    Only adds missing TP/SL; does not change existing ones.
    """

    def __init__(
        self,
        settings_manager,
        db_session_scope,
        binance_client: Optional[BinanceClient] = None,
    ):
        self.settings_manager = settings_manager
        self.db_session_scope = db_session_scope
        self.binance_client = binance_client

    def run(self) -> Dict[str, Any]:
        """
        For each open PROGRAMMATIC order, fetch TP/SL from Binance.
        If TP or SL is missing, place it from config (default_tp, default_sl %).
        """
        results: Dict[str, Any] = {
            "orders_checked": 0,
            "tp_added": 0,
            "sl_added": 0,
            "errors": [],
            "updates": [],
        }

        try:
            tp_sl_settings: dict = self.settings_manager.get("tp_sl", {}) or {}
            default_tp: float = float(tp_sl_settings.get("default_tp", 5.0))
            default_sl: float = float(tp_sl_settings.get("default_sl", 2.5))

            if default_tp <= 0 or default_sl <= 0:
                logger.warning("Ensure TP/SL: invalid default_tp or default_sl, skipping")
                return results

            with self.db_session_scope() as session:
                open_orders: List[Order] = get_open_positions(session)
                if not open_orders:
                    return results

                for order in open_orders:
                    results["orders_checked"] += 1
                    symbol: str = str(getattr(order, "symbol", ""))
                    if not symbol:
                        continue
                    try:
                        self._process_order(
                            session,
                            order,
                            symbol,
                            default_tp,
                            default_sl,
                            results,
                        )
                    except Exception as e:
                        msg = f"Ensure TP/SL order {getattr(order, 'order_id', '')}: {e}"
                        logger.exception(msg)
                        results["errors"].append(msg)

                session.commit()

        except Exception as e:
            logger.exception(f"Ensure TP/SL job: {e}")
            results["errors"].append(str(e))

        return results

    def _process_order(
        self,
        session: Session,
        order: Order,
        symbol: str,
        default_tp: float,
        default_sl: float,
        results: Dict[str, Any],
    ) -> None:
        entry_price = float(getattr(order, "entry_price", 0.0))
        side = str(getattr(order, "side", "LONG")).upper()
        if not entry_price or entry_price <= 0:
            return

        # Fetch current TP/SL from Binance
        current_tp: Optional[float] = None
        current_sl: Optional[float] = None
        if self.binance_client:
            try:
                from modules.auto_trade.gui.utils.tp_sl_sync import TPSLSyncService

                current_tp, current_sl, _ = TPSLSyncService.fetch_tp_sl_from_binance(
                    self.binance_client, symbol
                )
            except Exception as e:
                logger.warning(f"Ensure TP/SL: could not fetch Binance orders for {symbol}: {e}")
                return

        need_tp = current_tp is None
        need_sl = current_sl is None
        if not need_tp and not need_sl:
            return

        tp_price, sl_price = _tp_sl_prices_from_pct(entry_price, side, default_tp, default_sl)

        if need_tp and self.binance_client and tp_price:
            res = self.binance_client.modify_take_profit(symbol, None, tp_price)
            success = res is not None and (
                res.get("id") or res.get("dry_run") or res.get("success")
            )
            if success:
                order.take_profit = tp_price
                results["tp_added"] += 1
                results["updates"].append(
                    {"symbol": symbol, "action": "tp_added", "price": tp_price}
                )
                logger.info(f"Ensure TP/SL: added TP for {symbol} at {tp_price}")
            else:
                results["errors"].append(f"Failed to place TP for {symbol}")

        if need_sl and self.binance_client and sl_price:
            res = self.binance_client.modify_stop_loss(symbol, None, sl_price)
            success = res is not None and (
                res.get("id") or res.get("dry_run") or res.get("success")
            )
            if success:
                order.stop_loss = sl_price
                results["sl_added"] += 1
                results["updates"].append(
                    {"symbol": symbol, "action": "sl_added", "price": sl_price}
                )
                logger.info(f"Ensure TP/SL: added SL for {symbol} at {sl_price}")
            else:
                results["errors"].append(f"Failed to place SL for {symbol}")


def create_ensure_tp_sl_job(
    settings_manager,
    db_session_scope,
    binance_client: Optional[BinanceClient] = None,
) -> EnsureTPSLJob:
    """Factory to create EnsureTPSLJob."""
    return EnsureTPSLJob(
        settings_manager=settings_manager,
        db_session_scope=db_session_scope,
        binance_client=binance_client,
    )
