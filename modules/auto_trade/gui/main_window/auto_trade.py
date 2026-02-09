"""Auto-trading logic and signal processing."""

import time
from typing import TYPE_CHECKING, Optional

if TYPE_CHECKING:
    from .main_window import AutoTradeDashboard
    from .updaters import UpdaterManager


class AutoTradeManager:
    """Manages auto-trading execution and signal processing."""

    def __init__(self, parent: "AutoTradeDashboard"):
        self.parent = parent
        self.updater: Optional["UpdaterManager"] = None

    def _get_binance_client(self):
        """
        Build a BinanceClient from the dashboard's data_service when credentials are available.
        Used by trailing stop and negative breakeven jobs for mark price and order updates.
        """
        from modules.auto_trade.execution.binance_client import BinanceClient
        from modules.auto_trade.gui.utils.modes import TradingMode

        ds = self.parent.data_service
        api_key = getattr(ds, "api_key", "") or ""
        api_secret = getattr(ds, "api_secret", "") or ""
        if not api_key or not api_secret:
            return None
        testnet = getattr(ds, "testnet", False)
        mode = getattr(self.parent, "mode", None) or getattr(
            ds, "mode", None
        ) or self.parent.settings_manager.get("api.mode", TradingMode.DRY_RUN)
        dry_run = mode == TradingMode.DRY_RUN
        try:
            return BinanceClient(
                api_key=api_key,
                api_secret=api_secret,
                testnet=testnet,
                dry_run=dry_run,
            )
        except Exception:
            return None

    def start(self):
        """Start auto-trading loop, periodic Binance reconcile, and updaters."""
        from .updaters import UpdaterManager

        updater = UpdaterManager(self.parent)
        self.updater = updater
        updater.create_auto_trade_updater(self._auto_trade_cycle, interval=60)
        updater.create_reconcile_updater(self._reconcile_cycle, interval=3600)
        updater.create_trailing_stop_updater(self._trailing_stop_cycle, interval=30)
        updater.create_negative_breakeven_updater(self._negative_breakeven_cycle, interval=30)
        print("Auto-trading started (with trailing stop and negative breakeven)")

    def stop(self):
        """Stop auto-trading loop and all updaters."""
        if self.updater:
            for updater_name in ["auto_trade", "reconcile", "trailing_stop", "negative_breakeven"]:
                if updater_name in self.updater.updaters:
                    self.updater.updaters[updater_name].stop()
            print("Auto-trading stopped (including trailing stop and negative breakeven)")

    def _trailing_stop_cycle(self):
        """Run trailing stop check every 30 seconds."""
        try:
            from modules.auto_trade.database import session_scope
            from modules.auto_trade.execution.trailing_stop_job import create_trailing_stop_job

            # Create and run trailing stop job (pass client for mark price and SL updates)
            binance_client = self._get_binance_client()
            job = create_trailing_stop_job(
                settings_manager=self.parent.settings_manager,
                db_session_scope=session_scope,
                binance_client=binance_client,
            )

            result = job.run()

            # Log results if there were updates
            if result["orders_updated"] > 0:
                msg = (
                    f"Trailing stop: checked={result['orders_checked']}, "
                    f"updated={result['orders_updated']}"
                )
                print(msg)
                for update in result["updates"]:
                    print(
                        f"  - {update['symbol']}: SL {update['old_sl']} → {update['new_sl']} "
                        f"(step {update['step_index']})"
                    )

            if result["errors"]:
                for error in result["errors"][:3]:  # Log first 3 errors only
                    print(f"Trailing stop error: {error}")

        except Exception as e:
            print(f"Trailing stop cycle error: {e}")

    def _negative_breakeven_cycle(self):
        """Run negative breakeven check every 30 seconds."""
        try:
            from modules.auto_trade.database import session_scope
            from modules.auto_trade.execution.negative_breakeven_job import create_negative_breakeven_job

            # Create and run negative breakeven job (pass client for mark price and TP updates)
            binance_client = self._get_binance_client()
            job = create_negative_breakeven_job(
                settings_manager=self.parent.settings_manager,
                db_session_scope=session_scope,
                binance_client=binance_client,
            )

            result = job.run()

            # Log results if there were updates
            if result["orders_updated"] > 0:
                print(f"Negative breakeven: checked={result['orders_checked']}, updated={result['orders_updated']}")
                for update in result["updates"]:
                    print(f"  - {update['symbol']}: TP {update['old_tp']} → {update['new_tp']} (moved to entry)")

            if result["errors"]:
                for error in result["errors"][:3]:  # Log first 3 errors only
                    print(f"Negative breakeven error: {error}")

        except Exception as e:
            print(f"Negative breakeven cycle error: {e}")

    def _reconcile_cycle(self):
        """Reconcile AT_* orders from Binance into DB (run periodically when auto-trade is on)."""
        try:
            from modules.auto_trade.database import reconcile_orders_with_binance

            ds = self.parent.data_service
            api_key = getattr(ds, "api_key", "") or ""
            api_secret = getattr(ds, "api_secret", "") or ""
            if not api_key or not api_secret:
                return
            testnet = getattr(ds, "testnet", False)
            symbols = self.parent.settings_manager.get("filters.symbol_whitelist") or None
            result = reconcile_orders_with_binance(
                api_key=api_key,
                api_secret=api_secret,
                testnet=testnet,
                symbols=symbols,
                since_hours=24,
            )
            if result.get("inserted", 0) or result.get("errors"):
                print(f"Reconcile: inserted={result.get('inserted', 0)}, errors={len(result.get('errors', []))}")
        except Exception as e:
            print(f"Reconcile cycle error: {e}")

    def _auto_trade_cycle(self):
        """
        Auto-trading cycle:
        1. Check for new qualifying signals
        2. Validate against risk rules
        3. Execute trade if conditions met
        """
        try:
            from modules.auto_trade.execution.order_executor import OrderExecutor

            # Signals are "fresh" if created within this many seconds (5 minutes)
            FRESH_SIGNAL_MAX_AGE_SECONDS = 300
            signals = self.parent.data_service.get_signals(min_score=0.7)
            now = time.time()
            fresh_signals = [
                s
                for s in signals
                if isinstance(s.get("created_at_ts"), (int, float))
                and (now - float(s["created_at_ts"])) < FRESH_SIGNAL_MAX_AGE_SECONDS
            ]
            if not fresh_signals:
                print("No fresh signals for auto-trade (<5 minutes)")
                return
            fresh_signals.sort(key=lambda s: float(s.get("score", 0.0)), reverse=True)
            best = fresh_signals[0]
            print(f"Auto-trade selected: {best.get('symbol')} (score={float(best.get('score', 0)):.2f})")

            default_position_size = self.parent.settings_manager.get("trading.default_position_size", 100.0)
            default_leverage = self.parent.settings_manager.get("trading.default_leverage", 2)

            from .risk_manager import RiskManager

            risk_manager = RiskManager(self.parent)
            if not risk_manager.check_limits(
                symbol=str(best.get("symbol") or ""),
                position_size=default_position_size,
                leverage=default_leverage,
            ):
                print("Risk limits exceeded, skipping trade")
                return

            sig_dict = {
                "symbol": best.get("symbol"),
                "signal": best.get("signal"),
                "score": best.get("score", 0.0),
                "created_at_ts": best.get("created_at_ts", 0.0),
            }
            tp_sl = self.parent.settings_manager.get("tp_sl", {}) or {}
            executor = OrderExecutor()
            result = executor.execute_from_signal(sig_dict, tp_sl_settings=tp_sl)

            if result and result.get("success"):
                print(f"Auto-trade executed: {sig_dict['symbol']}")
                self.parent.after(0, self.parent.refresh_positions)
                self.parent.after(0, self.parent.refresh_account)

        except Exception as e:
            print(f"Error in auto-trade cycle: {e}")

    def _convert_signals(self, signals: list) -> list:
        """Convert signal dicts to SignalResult objects."""
        from modules.auto_trade.core.atc_scanner import SignalResult

        xb_signals = []
        for s in signals:
            sym = (s.get("symbol") or "").strip()
            if sym and "/" not in sym:
                sym = f"{sym.replace('USDT', '')}/USDT"
            if not sym:
                continue
            xb_signals.append(
                SignalResult(
                    symbol=sym,
                    score=float(s.get("score", 0)),
                    signal_type=(s.get("signal") or "LONG").upper(),
                    details={"time": s.get("time", "")},
                    strengths={},
                )
            )
        return xb_signals
