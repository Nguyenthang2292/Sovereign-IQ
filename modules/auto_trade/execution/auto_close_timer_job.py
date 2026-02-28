"""
Auto Close Timer Job
====================

Polling job for scheduled exits.
Checks all open PROGRAMMATIC orders and triggers a quasi-market close when:
- max duration timeout is reached, or
- daily close cutoff is reached.
"""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

from modules.auto_trade.database import RepositoryContext, get_open_positions
from modules.auto_trade.execution.auto_close_timer import (
    AutoCloseDecision,
    execute_auto_close,
    evaluate_order_for_auto_close,
    get_order_id,
    to_utc_iso,
)
from modules.auto_trade.execution.binance_client import BinanceClient
from modules.common.ui.logging import log_debug, log_error, log_info, log_warn


class AutoCloseTimerJob:
    """Timer-based scheduled-exit job."""

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
        if self.repo_context is None:
            self.repo_context = RepositoryContext.from_env()
        return self.repo_context

    def run(self) -> Dict[str, Any]:
        """Run one polling cycle for auto-close timers."""
        results: Dict[str, Any] = {
            "orders_checked": 0,
            "orders_triggered": 0,
            "errors": [],
            "updates": [],
        }

        try:
            cfg: Dict[str, Any] = self.settings_manager.get("auto_close", {}) or {}
            enabled = bool(cfg.get("enabled", False))
            if not enabled:
                log_debug("Auto-close timer is disabled, skipping")
                return results

            tp_offset_pct = float(cfg.get("tp_offset_pct", 0.05) or 0.05)
            now_utc = datetime.now(timezone.utc)

            open_orders: List[Dict[str, Any]] = get_open_positions()
            if not open_orders:
                log_debug("Auto-close timer: no open orders")
                return results

            for order in open_orders:
                results["orders_checked"] += 1

                try:
                    decision: AutoCloseDecision = evaluate_order_for_auto_close(
                        order=order,
                        now_utc=now_utc,
                        auto_close_cfg=cfg,
                    )

                    if not decision.should_close:
                        continue

                    order_id = get_order_id(order)
                    if not order_id:
                        msg = "Auto-close skipped order with missing order_id"
                        log_warn(msg)
                        results["errors"].append(msg)
                        continue

                    execution_result = execute_auto_close(
                        order=order,
                        reason=decision.reason,
                        binance_client=self.binance_client,
                        tp_offset_pct=tp_offset_pct,
                    )

                    if not execution_result.success:
                        error_msg = (
                            f"Auto-close failed for {order.get('symbol', '')} ({order_id}): "
                            f"{execution_result.message}"
                        )
                        log_error(error_msg)
                        results["errors"].append(error_msg)
                        continue

                    updates = {
                        "auto_close_triggered": True,
                        "auto_close_reason": decision.reason,
                        "auto_close_triggered_at": to_utc_iso(execution_result.trigger_time_utc),
                    }
                    if execution_result.target_tp is not None:
                        updates["auto_close_target_tp"] = float(execution_result.target_tp)

                    if decision.reason == "daily_close":
                        updates["auto_close_last_daily_date"] = now_utc.date().isoformat()

                    ctx = self._get_repo_context()
                    update_ok = ctx.orders.update(order_id, updates)
                    if not update_ok:
                        msg = f"Auto-close exchange trigger succeeded but DB update failed for {order_id}"
                        log_warn(msg)
                        results["errors"].append(msg)

                    results["orders_triggered"] += 1
                    update_item: Dict[str, Any] = {
                        "order_id": order_id,
                        "symbol": str(order.get("symbol", "")),
                        "reason": decision.reason,
                        "trigger": decision.trigger_label,
                        "deadline_utc": to_utc_iso(decision.deadline_utc) if decision.deadline_utc else "",
                        "target_tp": execution_result.target_tp,
                    }
                    results["updates"].append(update_item)

                except Exception as order_exc:
                    err = f"Auto-close order processing error ({order.get('order_id', '')}): {order_exc}"
                    log_error(err)
                    results["errors"].append(err)

            if results["orders_triggered"] > 0:
                log_info(
                    f"Auto-close timer: checked={results['orders_checked']}, triggered={results['orders_triggered']}"
                )

        except Exception as exc:
            err = f"Error in auto-close timer job: {exc}"
            log_error(err)
            results["errors"].append(err)

        return results


def create_auto_close_timer_job(
    settings_manager,
    repo_context: Optional[RepositoryContext] = None,
    binance_client: Optional[BinanceClient] = None,
) -> AutoCloseTimerJob:
    """Factory for AutoCloseTimerJob."""
    return AutoCloseTimerJob(
        settings_manager=settings_manager,
        repo_context=repo_context,
        binance_client=binance_client,
    )
