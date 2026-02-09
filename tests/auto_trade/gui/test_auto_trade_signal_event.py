from types import SimpleNamespace
from unittest.mock import MagicMock, patch

from modules.auto_trade.monitoring.event_system import EventSystem, EventType


class DummyUpdaterManager:
    def __init__(self, parent):
        self.parent = parent
        self.updaters = {}

    def _register(self, name):
        self.updaters[name] = MagicMock()

    def create_auto_trade_updater(self, callback, interval=60):
        self._register("auto_trade")

    def create_reconcile_updater(self, callback, interval=3600):
        self._register("reconcile")

    def create_trailing_stop_updater(self, callback, interval=30):
        self._register("trailing_stop")

    def create_negative_breakeven_updater(self, callback, interval=30):
        self._register("negative_breakeven")


class ImmediateThread:
    def __init__(self, target, daemon=True, name=None):
        self._target = target

    def start(self):
        self._target()


def _make_parent():
    parent = SimpleNamespace()
    parent.event_bus = EventSystem()
    parent.settings_manager = MagicMock()
    parent.data_service = MagicMock()
    parent.after = MagicMock()
    return parent


def test_signal_event_triggers_auto_trade_cycle():
    from modules.auto_trade.gui.main_window.auto_trade import AutoTradeManager

    parent = _make_parent()
    manager = AutoTradeManager(parent)  # type: ignore[arg-type]
    called = {"value": False}

    def fake_cycle():
        called["value"] = True

    manager._auto_trade_cycle = fake_cycle

    with patch("modules.auto_trade.gui.main_window.updaters.UpdaterManager", DummyUpdaterManager):
        with patch("modules.auto_trade.gui.main_window.auto_trade.threading.Thread", ImmediateThread):
            manager.start()
            parent.event_bus.publish(EventType.SIGNAL_GENERATED, {"symbol": "BTC/USDT"})

    assert called["value"] is True


def test_signal_event_skips_when_cycle_running():
    from modules.auto_trade.gui.main_window.auto_trade import AutoTradeManager

    parent = _make_parent()
    manager = AutoTradeManager(parent)  # type: ignore[arg-type]
    manager._trading_running = True

    with patch("modules.auto_trade.gui.main_window.auto_trade.threading.Thread") as mock_thread:
        manager._on_signal_event(SimpleNamespace(data={"symbol": "ETH/USDT"}))

    assert not mock_thread.called
