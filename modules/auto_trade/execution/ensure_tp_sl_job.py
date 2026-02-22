"""
Ensure TP/SL Job
================

Runs periodically when auto-trade is on. For each open PROGRAMMATIC position,
checks if TP and/or SL exist on Binance; if missing, places them from config
(default_tp, default_sl percentages).

Uses BinanceClient.modify_take_profit() / modify_stop_loss() which follow the
idempotent cancel-then-create pattern — preventing duplicate conditional orders.

Created: 2026-02-06
Refactored: 2026-02-20 (DynamoDB only)
Fixed: 2026-02-21 (duplicate conditional order bug — 200 order flood)
"""

from typing import Any, Dict, List, Optional, Tuple

from modules.auto_trade.database import RepositoryContext, get_open_positions
from modules.auto_trade.execution.binance.order_management import (
    _ccxt_futures_symbol,
    _classify_order_kind,
    _fetch_all_open_orders,
)
from modules.auto_trade.execution.binance_client import BinanceClient
from modules.common.ui.logging import log_debug, log_error, log_info, log_warn

# Maximum conditional orders allowed per symbol before we refuse to create more.
# Valid state is exactly 1 TP + 1 SL = 2. Allow a tiny buffer (e.g. 4) for
# race conditions, but reject anything much higher to prevent order floods.
_MAX_CONDITIONAL_ORDERS_PER_SYMBOL = 4


def _tp_sl_prices_from_pct(
    entry_price: float,
    side: str,
    tp_pct: float,
    sl_pct: float,
    leverage: float = 1.0,
    roi_mode: bool = False,
) -> Tuple[float, float]:
    """
    Compute TP and SL prices from entry and percentages.

    LONG: TP above entry, SL below. SHORT: TP below entry, SL above.

    Args:
        entry_price: Entry price.
        side: 'LONG' or 'SHORT'.
        tp_pct: Percentage value (meaning depends on roi_mode).
        sl_pct: Percentage value (meaning depends on roi_mode).
        leverage: Actual position leverage fetched from Binance (default=1 = no scaling).
        roi_mode: If True, tp_pct/sl_pct are ROI percentages on capital, so divide by
                  leverage to convert to price-move percentages.
                  If False (default), tp_pct/sl_pct are already price-move percentages.

    Returns:
        (tp_price, sl_price)
    """
    if not entry_price or entry_price <= 0:
        return 0.0, 0.0

    if roi_mode and leverage > 0:
        # Convert ROI% → price-move%:  price_pct = roi_pct / leverage
        tp_price_pct = tp_pct / leverage
        sl_price_pct = sl_pct / leverage
    else:
        tp_price_pct = tp_pct
        sl_price_pct = sl_pct

    mult_tp = 1.0 + (tp_price_pct / 100.0)
    mult_sl = 1.0 - (sl_price_pct / 100.0)
    if str(side).upper() == "SHORT":
        mult_tp = 1.0 - (tp_price_pct / 100.0)
        mult_sl = 1.0 + (sl_price_pct / 100.0)
    return entry_price * mult_tp, entry_price * mult_sl


def _count_conditional_orders(open_orders_list: list, entry_price: float = 0.0, side: str = "") -> dict:
    """Count TP and SL conditional orders.

    Uses _classify_order_kind (from order_management) so that CCXT-normalized
    'market' typed conditional orders are counted correctly via their stopPrice.

    Returns:
        Dict with 'tp' and 'sl' counts.
    """
    counts: Dict[str, int] = {"tp": 0, "sl": 0}
    for order_item in open_orders_list:
        kind = _classify_order_kind(order_item, entry_price, side)
        if kind == "tp":
            counts["tp"] += 1
        elif kind == "sl":
            counts["sl"] += 1
    return counts


class EnsureTPSLJob:
    """
    Job that ensures every open programmatic position has TP and SL on Binance.
    Only adds missing TP/SL; does not change existing ones.

    Uses idempotent modify_take_profit / modify_stop_loss which cancel existing
    conditional orders before creating new ones, preventing duplicate orders.
    """

    def __init__(
        self,
        settings_manager,
        repo_context: Optional[RepositoryContext] = None,
        binance_client: Optional[BinanceClient] = None,
    ):
        self.settings_manager = settings_manager
        self.repo_context = repo_context
        self.binance_client = binance_client

    def _get_repo_context(self) -> RepositoryContext:
        """Get or create RepositoryContext."""
        if self.repo_context is None:
            self.repo_context = RepositoryContext.from_env()
        return self.repo_context

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
                log_warn("Ensure TP/SL: invalid default_tp or default_sl, skipping")
                return results

            open_orders: List[Dict[str, Any]] = get_open_positions()
            if not open_orders:
                return results

            for order in open_orders:
                results["orders_checked"] += 1
                symbol: str = str(order.get("symbol", ""))
                if not symbol:
                    continue
                try:
                    self._process_order(
                        order,
                        symbol,
                        default_tp,
                        default_sl,
                        results,
                    )
                except Exception as e:
                    msg = f"Ensure TP/SL order {order.get('order_id', '')}: {e}"
                    log_error(msg, exc_info=True)
                    results["errors"].append(msg)

        except Exception as e:
            log_error(f"Ensure TP/SL job: {e}", exc_info=True)
            results["errors"].append(str(e))

        return results

    def _fetch_existing_tp_sl(
        self,
        symbol: str,
        entry_price: float = 0.0,
        side: str = "",
    ) -> Tuple[Optional[float], Optional[float], bool, bool, bool]:
        """
        Fetch existing TP/SL from Binance using a single fetch_open_orders call.

        Args:
            symbol: Trading symbol
            entry_price: Position entry price (used for TP/SL classification when
                         CCXT normalizes the order type to 'MARKET').
            side: Position side ('LONG' or 'SHORT') — same purpose.

        Returns:
            (current_tp, current_sl, has_tp, has_sl, detection_succeeded)
            detection_succeeded is False when fetch_open_orders fails entirely,
            signalling that we must NOT create orders (fail-closed).
        """
        current_tp: Optional[float] = None
        current_sl: Optional[float] = None
        has_tp: bool = False
        has_sl: bool = False

        if not self.binance_client:
            return None, None, False, False, True

        try:
            ccxt_sym = _ccxt_futures_symbol(self.binance_client.exchange, symbol)
            # Use _fetch_all_open_orders to get BOTH Basic AND Conditional orders.
            # This is critical: Binance separates these into different endpoints.
            # Without this, conditional orders (STOP_MARKET, TAKE_PROFIT_MARKET)
            # would be invisible, causing duplicate order creation.
            open_orders_list = _fetch_all_open_orders(self.binance_client.exchange, ccxt_sym)

            # Fallback: if combined query returned nothing, try broad symbol filter
            if not open_orders_list:
                try:
                    all_orders = self.binance_client.exchange.fetch_open_orders()
                    # Filter by symbol id
                    target_id = symbol.replace("/", "").split(":")[0].upper()
                    open_orders_list = []
                    for o in all_orders:
                        info = o.get("info") or {}
                        o_sym = str(o.get("symbol") or "").replace("/", "").split(":")[0].upper()
                        i_sym = (str(info.get("symbol") or "").upper()) if isinstance(info, dict) else ""
                        if o_sym == target_id or i_sym == target_id:
                            open_orders_list.append(o)
                    if open_orders_list:
                        log_info(f"Ensure TP/SL: Fallback found {len(open_orders_list)} orders for {symbol}")
                except Exception as fb_err:
                    log_debug(f"Ensure TP/SL: fallback fetch failed for {symbol}: {fb_err}")

            for order_item in open_orders_list:
                kind = _classify_order_kind(order_item, entry_price, side)
                stop_price_raw = (
                    order_item.get("stopPrice")
                    or order_item.get("triggerPrice")
                    or (order_item.get("info") or {}).get("stopPrice")
                )
                stop_price = float(stop_price_raw) if stop_price_raw else None

                if kind == "tp":
                    has_tp = True
                    if current_tp is None and stop_price:
                        current_tp = stop_price
                elif kind == "sl":
                    has_sl = True
                    if current_sl is None and stop_price:
                        current_sl = stop_price

            # Guard: refuse to create more orders if too many conditional orders exist
            counts = _count_conditional_orders(open_orders_list, entry_price, side)
            if counts["tp"] + counts["sl"] >= _MAX_CONDITIONAL_ORDERS_PER_SYMBOL:
                log_warn(
                    f"Ensure TP/SL: {symbol} has {counts['tp']} TP + {counts['sl']} SL "
                    f"conditional orders (>= {_MAX_CONDITIONAL_ORDERS_PER_SYMBOL}). "
                    f"Skipping to avoid flooding. Clean up duplicates first."
                )
                # Treat as "both exist" so no new orders are created
                return current_tp, current_sl, True, True, True

            return current_tp, current_sl, has_tp, has_sl, True

        except Exception as e:
            log_warn(f"Ensure TP/SL: could not fetch orders for {symbol}: {e}")
            # FAIL-CLOSED: detection failed, do NOT create orders
            return None, None, False, False, False

    def _process_order(
        self,
        order: Dict[str, Any],
        symbol: str,
        default_tp: float,
        default_sl: float,
        results: Dict[str, Any],
    ) -> None:
        entry_price = float(order.get("entry_price", 0.0))
        side = str(order.get("side", "LONG")).upper()
        if not entry_price or entry_price <= 0:
            return

        # ── Fetch actual leverage from Binance position ───────────────────────
        # The GUI 'trading.default_leverage' may differ from what was used when
        # the position was opened. Always read the real leverage from the live
        # position object so TP/SL prices are scaled correctly.
        actual_leverage: float = float(order.get("leverage") or 1.0)
        if self.binance_client and actual_leverage <= 1.0:
            try:
                from modules.auto_trade.execution.binance.position_management import PositionManagement

                pos_mgr = PositionManagement(self.binance_client.exchange)
                pos = pos_mgr.get_position(symbol)
                if pos:
                    # CCXT top-level 'leverage', or info.leverage
                    lev_raw = pos.get("leverage") or (pos.get("info") or {}).get("leverage")
                    if lev_raw is not None:
                        try:
                            actual_leverage = float(lev_raw)
                        except (TypeError, ValueError):
                            pass
            except Exception as lev_err:
                log_debug(f"Ensure TP/SL: could not fetch leverage for {symbol}: {lev_err}")

        if actual_leverage < 1.0:
            actual_leverage = 1.0

        # Always use ROI mode: default_tp/sl are % ROI on capital.
        # price_move% = roi% / actual_leverage
        roi_mode = True

        log_info(
            f"Ensure TP/SL: {symbol} entry={entry_price} side={side} "
            f"leverage={actual_leverage}x roi_mode=ROI "
            f"default_tp={default_tp}% default_sl={default_sl}%"
        )

        # Fetch existing TP/SL with fail-closed approach:
        # if detection fails, we do NOT create new orders.
        current_tp, current_sl, has_tp, has_sl, detection_ok = self._fetch_existing_tp_sl(
            symbol, entry_price=entry_price, side=side
        )

        if not detection_ok:
            log_info(f"Ensure TP/SL: detection failed for {symbol}, skipping order creation to avoid duplicates")
            return

        tp_missing = current_tp is None and not has_tp
        sl_missing = current_sl is None and not has_sl

        if not tp_missing and not sl_missing:
            log_debug(f"Ensure TP/SL: {symbol} already has TP and SL")
            return

        tp_price, sl_price = _tp_sl_prices_from_pct(
            entry_price,
            side,
            default_tp,
            default_sl,
            leverage=actual_leverage,
            roi_mode=roi_mode,
        )

        # Convert DB symbol to CCXT format for BinanceClient methods
        ccxt_sym = _ccxt_futures_symbol(self.binance_client.exchange, symbol) if self.binance_client else symbol

        if tp_missing and tp_price > 0:
            if self.binance_client:
                try:
                    # Use modify_take_profit which: cancels existing TP → places new
                    # TAKE_PROFIT_MARKET order → validates price. Idempotent.
                    tp_result = self.binance_client.modify_take_profit(
                        symbol=ccxt_sym,
                        position_id=None,
                        take_profit_price=tp_price,
                    )
                    if tp_result and (tp_result.get("id") or tp_result.get("dry_run")):
                        results["tp_added"] += 1
                        results["updates"].append({"symbol": symbol, "type": "TP", "price": tp_price})
                        log_info(f"Ensure TP/SL: Added TP for {symbol} at {tp_price}")
                    else:
                        log_warn(f"Ensure TP/SL: modify_take_profit returned no id for {symbol}")
                except Exception as e:
                    log_error(f"Ensure TP/SL: Failed to add TP for {symbol}: {e}")
                    results["errors"].append(f"TP add failed for {symbol}: {e}")
            else:
                results["tp_added"] += 1

        if sl_missing and sl_price > 0:
            if self.binance_client:
                try:
                    # Use modify_stop_loss which: cancels existing SL → places new
                    # STOP_MARKET order → validates against mark price. Idempotent.
                    sl_result = self.binance_client.modify_stop_loss(
                        symbol=ccxt_sym,
                        position_id=None,
                        stop_loss_price=sl_price,
                    )
                    if sl_result and (sl_result.get("id") or sl_result.get("dry_run")):
                        results["sl_added"] += 1
                        results["updates"].append({"symbol": symbol, "type": "SL", "price": sl_price})
                        log_info(f"Ensure TP/SL: Added SL for {symbol} at {sl_price}")
                    else:
                        log_warn(f"Ensure TP/SL: modify_stop_loss returned no id for {symbol}")
                except Exception as e:
                    log_error(f"Ensure TP/SL: Failed to add SL for {symbol}: {e}")
                    results["errors"].append(f"SL add failed for {symbol}: {e}")
            else:
                results["sl_added"] += 1


def create_ensure_tp_sl_job(
    settings_manager,
    repo_context: Optional[RepositoryContext] = None,
    binance_client: Optional[BinanceClient] = None,
) -> EnsureTPSLJob:
    """
    Factory function to create an EnsureTPSLJob instance.

    Args:
        settings_manager: Settings manager
        repo_context: RepositoryContext for DynamoDB (optional)
        binance_client: Optional Binance client

    Returns:
        EnsureTPSLJob instance
    """
    return EnsureTPSLJob(
        settings_manager=settings_manager,
        repo_context=repo_context,
        binance_client=binance_client,
    )
