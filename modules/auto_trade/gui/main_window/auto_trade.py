"""Auto-trading logic and signal processing."""

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from .main_window import AutoTradeDashboard


class AutoTradeManager:
    """Manages auto-trading execution and signal processing."""

    def __init__(self, parent: "AutoTradeDashboard"):
        self.parent = parent
        self.updater = None

    def start(self):
        """Start auto-trading loop."""
        from .updaters import UpdaterManager

        self.updater = UpdaterManager(self.parent)
        self.updater.create_auto_trade_updater(self._auto_trade_cycle, interval=60)
        print("Auto-trading started")

    def stop(self):
        """Stop auto-trading loop."""
        if self.updater and "auto_trade" in self.updater.updaters:
            self.updater.updaters["auto_trade"].stop()
            print("Auto-trading stopped")

    def _auto_trade_cycle(self):
        """
        Auto-trading cycle:
        1. Check for new qualifying signals
        2. Validate against risk rules
        3. Execute trade if conditions met
        """
        try:
            from modules.auto_trade.core.atc_scanner import SignalResult
            from modules.auto_trade.core.signal_selector import SignalSelector
            from modules.auto_trade.execution.order_executor import OrderExecutor

            signals = self.parent.data_service.get_signals(min_score=0.7)
            xb_signals = self._convert_signals(signals)

            selector = SignalSelector()
            selected_signal = selector.select_best_signal(xb_signals, gemini_signals={})

            if not selected_signal:
                print("No qualifying signals for auto-trade")
                return

            default_position_size = self.parent.settings_manager.get("trading.default_position_size", 100.0)
            default_leverage = self.parent.settings_manager.get("trading.default_leverage", 2)

            from .risk_manager import RiskManager

            risk_manager = RiskManager(self.parent)
            if not risk_manager.check_limits(
                symbol=selected_signal.symbol,
                position_size=default_position_size,
                leverage=default_leverage,
            ):
                print("Risk limits exceeded, skipping trade")
                return

            sig_dict = {
                "symbol": selected_signal.symbol,
                "signal": selected_signal.signal_type,
                "score": selected_signal.final_score
                if hasattr(selected_signal, "final_score")
                else selected_signal.score,
            }
            executor = OrderExecutor()
            result = executor.execute_from_signal(sig_dict)

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
