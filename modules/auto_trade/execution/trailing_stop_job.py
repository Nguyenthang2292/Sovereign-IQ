"""
Trailing Stop Timer Job
========================

Polling-based trailing stop job that runs every 30 seconds.
Checks all open PROGRAMMATIC orders and updates SL when profit reaches step thresholds.

Created: 2026-02-06
Refactored: 2026-02-20 (DynamoDB only)
"""

from modules.common.ui.logging import log_info, log_error, log_warn, log_debug, log_success, log_system
from typing import Any, Dict, List, Optional

from modules.auto_trade.database import RepositoryContext, get_open_positions
from modules.auto_trade.execution.binance_client import BinanceClient
from modules.auto_trade.execution.trailing_stop import TrailingStopResult, calculate_trailing_stop



def _symbol_for_ccxt(symbol: str) -> str:
    """Convert DB symbol (e.g. SKLUSDT) to CCXT format (SKL/USDT) for API calls."""
    s = (symbol or "").strip()
    if "/" in s:
        return s
    if s.endswith("USDT"):
        return s[:-4] + "/USDT"
    return s + "/USDT" if s else s


class TrailingStopJob:
    """
    Timer-based trailing stop job.

    Polls open orders every 30 seconds and updates stop loss
    when profit reaches step thresholds (BE → +step% → +2*step% …).
    """

    def __init__(
        self,
        settings_manager,
        repo_context: Optional[RepositoryContext] = None,
        binance_client: Optional[BinanceClient] = None,
    ):
        """
        Initialize trailing stop job.

        Args:
            settings_manager: Settings manager to get TP/SL settings
            repo_context: RepositoryContext for DynamoDB access (optional, creates if not provided)
            binance_client: Binance client for modifying SL orders (optional)
        """
        self.settings_manager = settings_manager
        self.repo_context = repo_context
        self.binance_client = binance_client
        self._last_update_times: Dict[str, float] = {}

    def _get_repo_context(self) -> RepositoryContext:
        """Get or create RepositoryContext."""
        if self.repo_context is None:
            self.repo_context = RepositoryContext.from_env()
        return self.repo_context

    def run(self) -> Dict[str, Any]:
        """
        Execute trailing stop check for all open orders.

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
            tp_sl_settings: dict = self.settings_manager.get("tp_sl", {})
            trailing_stop_enabled: bool = bool(tp_sl_settings.get("trailing_stop", False))
            trailing_step_pct: float = float(tp_sl_settings.get("trailing_step_pct", 2.0))
            trailing_limit_steps: bool = bool(tp_sl_settings.get("trailing_limit_steps", False))
            trailing_max_steps: int = int(tp_sl_settings.get("trailing_max_steps", 5))

            if not trailing_stop_enabled:
                log_debug("Trailing stop is disabled, skipping")
                return results

            if trailing_step_pct <= 0:
                log_warn(f"Invalid trailing step percentage: {trailing_step_pct}")
                return results

            open_orders: List[Dict[str, Any]] = get_open_positions()

            if not open_orders:
                log_debug("No open orders to check")
                return results

            orders_by_symbol: Dict[str, List[Dict[str, Any]]] = {}
            for order in open_orders:
                sym: str = str(order.get("symbol", ""))
                if sym not in orders_by_symbol:
                    orders_by_symbol[sym] = []
                orders_by_symbol[sym].append(order)

            for symbol, orders in orders_by_symbol.items():
                try:
                    mark_price: Optional[float] = self._get_mark_price(symbol)
                    if mark_price is None:
                        log_warn(f"Could not get mark price for {symbol}")
                        continue

                    for order in orders:
                        results["orders_checked"] += 1

                        try:
                            update_result: Dict[str, Any] = self._process_order(
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
                            error_msg_order: str = f"Error processing order {order.get('order_id', '')}: {e}"
                            log_error(error_msg_order)
                            results["errors"].append(error_msg_order)

                except Exception as e:
                    error_msg_sym: str = f"Error processing symbol {symbol}: {e}"
                    log_error(error_msg_sym)
                    results["errors"].append(error_msg_sym)

        except Exception as e:
            error_msg: str = f"Error in trailing stop job: {e}"
            log_error(error_msg)
            results["errors"].append(error_msg)
            return results

        return results

    def _get_mark_price(self, symbol: str) -> Optional[float]:
        """Get current mark price for a symbol."""
        try:
            if not self.binance_client:
                log_warn("No Binance client available for fetching mark price")
                return None
            ccxt_symbol = _symbol_for_ccxt(symbol)
            ticker = self.binance_client.fetch_ticker(ccxt_symbol)
            if not ticker:
                return None
            info = ticker.get("info") or {}
            mark = info.get("markPrice") if isinstance(info, dict) else None
            if mark is not None:
                try:
                    return float(mark)
                except (TypeError, ValueError):
                    pass
            if "last" in ticker:
                return float(ticker["last"])
            return None
        except Exception as e:
            log_error(f"Error fetching mark price for {symbol}: {e}")
            return None

    def _process_order(
        self,
        order: Dict[str, Any],
        mark_price: float,
        step_pct: float,
        limit_steps: bool,
        max_steps: int,
    ) -> Dict[str, Any]:
        """Process a single order for trailing stop."""
        order_id_raw: Any = order.get("order_id")
        order_id: str = str(order_id_raw) if order_id_raw else ""
        if not order_id:
            return {
                "order_id": "",
                "symbol": str(order.get("symbol", "")),
                "updated": False,
                "message": "Missing order_id",
                "old_sl": None,
                "new_sl": None,
                "step_index": 0,
            }

        entry_price: float = float(order.get("entry_price", 0.0))
        side: str = str(order.get("side", ""))
        step_index: int = int(order.get("trailing_step_index", 0))
        stop_loss: Optional[float] = order.get("stop_loss")
        current_sl: Optional[float] = float(stop_loss) if stop_loss is not None else None

        result: Dict[str, Any] = {
            "order_id": order_id,
            "symbol": str(order.get("symbol", "")),
            "updated": False,
            "message": "",
            "old_sl": stop_loss,
            "new_sl": None,
            "step_index": step_index,
        }

        trailing_result: TrailingStopResult = calculate_trailing_stop(
            entry_price=entry_price,
            current_price=mark_price,
            side=side,
            step_index=step_index,
            step_pct=step_pct,
            current_sl=current_sl,
            limit_steps=limit_steps,
            max_steps=max_steps,
        )

        if not trailing_result.should_step:
            result["message"] = trailing_result.message
            return result

        new_sl: Optional[float] = trailing_result.new_sl_price

        if self.binance_client and new_sl:
            try:
                order_symbol: str = str(order.get("symbol", ""))
                ccxt_symbol: str = _symbol_for_ccxt(order_symbol)
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
                if success and order_id:
                    ctx = self._get_repo_context()
                    ctx.orders.update(
                        order_id,
                        {
                            "stop_loss": new_sl,
                            "trailing_step_index": trailing_result.next_step_index,
                        },
                    )

                    result["updated"] = True
                    result["new_sl"] = new_sl
                    result["step_index"] = trailing_result.next_step_index
                    result["message"] = (
                        f"Trailing stop stepped from {stop_loss} to {new_sl} "
                        f"(step {step_index} → {trailing_result.next_step_index})"
                    )

                    log_info(
                        f"Trailing stop updated for {order_symbol} {side}: "
                        f"SL {stop_loss} → {new_sl} (step {trailing_result.next_step_index})"
                    )
                else:
                    error_msg: str = str((modify_result or {}).get("error", "Unknown error"))
                    result["message"] = f"Failed to modify SL on exchange: {error_msg}"
                    log_error(f"Failed to modify SL for {order.get('order_id', '')}: {error_msg}")

            except Exception as e:
                result["message"] = f"Error modifying SL: {e}"
                log_error(f"Error modifying SL for {order.get('order_id', '')}: {e}")
        else:
            result["message"] = f"Would step SL to {new_sl} (dry run or no client)"
            sym_str: str = str(order.get("symbol", ""))
            log_info(f"[DRY RUN] Trailing stop would update for {sym_str}: SL {stop_loss} → {new_sl}")

            if not self.binance_client and order_id:
                ctx = self._get_repo_context()
                ctx.orders.update(
                    order_id,
                    {
                        "stop_loss": new_sl,
                        "trailing_step_index": trailing_result.next_step_index,
                    },
                )
                result["updated"] = True
                result["new_sl"] = new_sl
                result["step_index"] = trailing_result.next_step_index

        return result


def create_trailing_stop_job(
    settings_manager,
    repo_context: Optional[RepositoryContext] = None,
    binance_client: Optional[BinanceClient] = None,
) -> TrailingStopJob:
    """
    Factory function to create a TrailingStopJob instance.

    Args:
        settings_manager: Settings manager
        repo_context: RepositoryContext for DynamoDB (optional)
        binance_client: Optional Binance client

    Returns:
        TrailingStopJob instance
    """
    return TrailingStopJob(
        settings_manager=settings_manager,
        repo_context=repo_context,
        binance_client=binance_client,
    )
