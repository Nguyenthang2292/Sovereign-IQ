"""
Negative Breakeven Timer Job
===========================

Polling-based negative breakeven job that runs every 30 seconds.
Checks all open PROGRAMMATIC orders and moves TP to entry when position
is losing by threshold percentage but hasn't hit stop loss yet.

Created: 2026-02-06
Refactored: 2026-02-20 (DynamoDB only)
"""

from typing import Any, Dict, List, Optional

from modules.auto_trade.database import RepositoryContext, get_open_positions
from modules.auto_trade.execution.binance_client import BinanceClient
from modules.auto_trade.execution.negative_breakeven import NegativeBreakevenLogic
from modules.common.domain.symbol_codec import SymbolCodec
from modules.common.ui.logging import log_debug, log_error, log_info, log_warn

_SYMBOL_CODEC = SymbolCodec()


def _symbol_for_ccxt(symbol: str) -> str:
    """Convert any symbol format to CCXT spot format for API calls."""
    return str(_SYMBOL_CODEC.to_ccxt(symbol))


class NegativeBreakevenJob:
    """
    Timer-based negative breakeven job.

    Polls open orders every 30 seconds and moves take profit to entry price
    when position loss reaches threshold and hasn't hit stop loss yet.
    """

    def __init__(
        self,
        settings_manager,
        repo_context: Optional[RepositoryContext] = None,
        binance_client: Optional[BinanceClient] = None,
    ):
        """
        Initialize negative breakeven job.

        Args:
            settings_manager: Settings manager to get TP/SL settings
            repo_context: RepositoryContext for DynamoDB access (optional)
            binance_client: Binance client for modifying TP orders (optional)
        """
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
            tp_sl_settings: dict = self.settings_manager.get("tp_sl", {})
            negative_be_enabled: bool = bool(tp_sl_settings.get("negative_be_enabled", False))
            negative_be_threshold_pct: float = float(tp_sl_settings.get("negative_be_threshold_pct", 2.0))

            if not negative_be_enabled:
                log_debug("Negative breakeven is disabled, skipping")
                return results

            if negative_be_threshold_pct <= 0:
                log_warn(f"Invalid negative BE threshold: {negative_be_threshold_pct}")
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
                                negative_be_threshold_pct,
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
            error_msg: str = f"Error in negative breakeven job: {e}"
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
        threshold_pct: float,
    ) -> Dict[str, Any]:
        """Process a single order for negative breakeven."""
        entry_price: float = float(order.get("entry_price", 0.0))
        side: str = str(order.get("side", "")).upper()
        stop_loss: Optional[float] = order.get("stop_loss")
        take_profit: Optional[float] = order.get("take_profit")
        be_moved: bool = bool(order.get("be_moved", False))

        result: Dict[str, Any] = {
            "order_id": str(order.get("order_id", "")),
            "symbol": str(order.get("symbol", "")),
            "updated": False,
            "message": "",
            "old_tp": take_profit,
            "new_tp": None,
        }

        if be_moved:
            result["message"] = "Already moved to breakeven"
            return result

        if not entry_price or entry_price <= 0:
            result["message"] = "No valid entry price"
            return result

        # Use the canonical logic module — handles LONG/SHORT correctly
        profit_pct: float = NegativeBreakevenLogic.calculate_profit_pct(entry_price, mark_price, side)
        sl_val: float = float(stop_loss) if stop_loss else 0.0

        if not NegativeBreakevenLogic.should_trigger(
            entry_price=entry_price,
            mark_price=mark_price,
            stop_loss=sl_val,
            side=side,
            threshold_pct=threshold_pct,
            be_moved=be_moved,
        ):
            result["message"] = f"No trigger: profit={profit_pct:.2f}%, threshold={threshold_pct:.2f}%"
            return result

        if take_profit and abs(take_profit - entry_price) < entry_price * 0.001:
            result["message"] = "TP already at entry"
            return result

        new_tp: float = entry_price

        if self.binance_client:
            try:
                order_symbol: str = str(order.get("symbol", ""))
                ccxt_symbol: str = _symbol_for_ccxt(order_symbol)
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
                    ctx = self._get_repo_context()
                    ctx.orders.update(
                        str(order.get("order_id", "")),
                        {"take_profit": new_tp, "be_moved": True},
                    )

                    result["updated"] = True
                    result["new_tp"] = new_tp
                    result["message"] = f"TP moved to entry ({take_profit} → {new_tp})"

                    log_info(
                        f"Negative BE for {order_symbol} {side}: TP {take_profit} → {new_tp} (profit: {profit_pct:.2f}%)"
                    )
                else:
                    error_msg: str = str((modify_result or {}).get("error", "Unknown error"))
                    result["message"] = f"Failed to modify TP on exchange: {error_msg}"
                    log_error(f"Failed to modify TP for {order.get('order_id', '')}: {error_msg}")

            except Exception as e:
                result["message"] = f"Error modifying TP: {e}"
                log_error(f"Error modifying TP for {order.get('order_id', '')}: {e}")
        else:
            result["message"] = "Would move TP to entry (dry run)"
            log_info(
                f"[DRY RUN] Negative BE would update for {order.get('symbol', '')}: TP {take_profit} → {new_tp}"
            )

            if not self.binance_client:
                ctx = self._get_repo_context()
                ctx.orders.update(
                    str(order.get("order_id", "")),
                    {"take_profit": new_tp, "be_moved": True},
                )
                result["updated"] = True
                result["new_tp"] = new_tp

        return result


def create_negative_breakeven_job(
    settings_manager,
    repo_context: Optional[RepositoryContext] = None,
    binance_client: Optional[BinanceClient] = None,
) -> NegativeBreakevenJob:
    """
    Factory function to create a NegativeBreakevenJob instance.

    Args:
        settings_manager: Settings manager
        repo_context: RepositoryContext for DynamoDB (optional)
        binance_client: Optional Binance client

    Returns:
        NegativeBreakevenJob instance
    """
    return NegativeBreakevenJob(
        settings_manager=settings_manager,
        repo_context=repo_context,
        binance_client=binance_client,
    )
