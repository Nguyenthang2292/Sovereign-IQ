import time
from types import SimpleNamespace
from unittest.mock import MagicMock, patch


def _make_parent(*, signals, tp_sl=None):
    tp_sl = tp_sl or {"default_tp": 7.0, "default_sl": 3.0}

    class _Settings:
        def get(self, key, default=None):
            if key == "tp_sl":
                return tp_sl
            if key == "trading.default_position_size":
                return 100.0
            if key == "trading.default_leverage":
                return 2
            if key == "filters.symbol_whitelist":
                return None
            return default

    parent = SimpleNamespace()
    parent.data_service = SimpleNamespace(get_signals=MagicMock(return_value=signals))
    parent.settings_manager = _Settings()
    parent.after = MagicMock()
    parent.refresh_positions = MagicMock()
    parent.refresh_account = MagicMock()
    return parent


def test_auto_trade_picks_best_fresh_signal_by_score_and_passes_tp_sl():
    from modules.auto_trade.gui.main_window.auto_trade import AutoTradeManager

    now = 1000.0
    signals = [
        {"symbol": "AAAUSDT", "signal": "LONG", "score": 0.80, "created_at_ts": now - 1},   # fresh
        {"symbol": "BBBUSDT", "signal": "SHORT", "score": 0.90, "created_at_ts": now - 50}, # fresh (best)
        {"symbol": "CCCUSDT", "signal": "LONG", "score": 0.99, "created_at_ts": now - 200}, # stale
    ]
    parent = _make_parent(signals=signals, tp_sl={"default_tp": 9.0, "default_sl": 4.0})

    # Risk always ok
    with patch("modules.auto_trade.gui.main_window.risk_manager.RiskManager") as MockRisk:
        MockRisk.return_value.check_limits.return_value = True

        # Capture executor call
        with patch("modules.auto_trade.execution.order_executor.OrderExecutor") as MockExecutor:
            MockExecutor.return_value.execute_from_signal.return_value = {"success": False}

            with patch.object(time, "time", return_value=now):
                AutoTradeManager(parent)._auto_trade_cycle()

            MockExecutor.return_value.execute_from_signal.assert_called_once()
            args, kwargs = MockExecutor.return_value.execute_from_signal.call_args
            assert args[0]["symbol"] == "BBBUSDT"
            assert args[0]["signal"] == "SHORT"
            assert float(args[0]["score"]) == 0.90
            assert kwargs["tp_sl_settings"] == {"default_tp": 9.0, "default_sl": 4.0}


def test_auto_trade_skips_when_no_fresh_signals():
    from modules.auto_trade.gui.main_window.auto_trade import AutoTradeManager

    now = 1000.0
    signals = [
        {"symbol": "AAAUSDT", "signal": "LONG", "score": 0.80, "created_at_ts": now - 120},
        {"symbol": "BBBUSDT", "signal": "SHORT", "score": 0.90, "created_at_ts": now - 300},
    ]
    parent = _make_parent(signals=signals)

    with patch("modules.auto_trade.gui.main_window.risk_manager.RiskManager") as MockRisk:
        MockRisk.return_value.check_limits.return_value = True
        with patch("modules.auto_trade.execution.order_executor.OrderExecutor") as MockExecutor:
            with patch.object(time, "time", return_value=now):
                AutoTradeManager(parent)._auto_trade_cycle()
            MockExecutor.return_value.execute_from_signal.assert_not_called()


def test_get_binance_client_returns_none_when_no_credentials():
    """_get_binance_client returns None when data_service has no api_key or api_secret."""
    from modules.auto_trade.gui.main_window.auto_trade import AutoTradeManager

    parent = SimpleNamespace()
    parent.data_service = SimpleNamespace(api_key="", api_secret="", testnet=False)
    parent.settings_manager = MagicMock()
    parent.settings_manager.get.return_value = "DRY_RUN"
    parent.mode = "DRY_RUN"

    manager = AutoTradeManager(parent)
    assert manager._get_binance_client() is None


def test_get_binance_client_returns_client_when_credentials_present():
    """_get_binance_client returns a BinanceClient when credentials are set."""
    from modules.auto_trade.gui.main_window.auto_trade import AutoTradeManager

    parent = SimpleNamespace()
    parent.data_service = SimpleNamespace(
        api_key="test_key",
        api_secret="test_secret",
        testnet=True,
    )
    parent.settings_manager = MagicMock()
    parent.settings_manager.get.return_value = "DEMO"
    parent.mode = "DEMO"

    with patch("modules.auto_trade.execution.binance_client.BinanceClient") as MockClient:
        MockClient.return_value = MagicMock()
        manager = AutoTradeManager(parent)
        client = manager._get_binance_client()
        assert client is not None
        MockClient.assert_called_once_with(
            api_key="test_key",
            api_secret="test_secret",
            testnet=True,
            dry_run=False,
        )

