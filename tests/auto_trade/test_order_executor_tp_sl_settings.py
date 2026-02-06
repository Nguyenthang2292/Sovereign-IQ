from unittest.mock import Mock, patch

import pytest


class _DummyClient:
    def __init__(self, *args, **kwargs):
        self.exchange = Mock()
        self.exchange.fetch_ticker = Mock(return_value={"last": 100.0})


class _CapturingOrderManager:
    def __init__(self, *args, **kwargs):
        self.last_signal = None

    def execute_signal(self, final_signal):
        self.last_signal = final_signal
        return {"order_id": "dummy"}


@pytest.mark.parametrize(
    "signal_type,tp_pct,sl_pct,expected_tp,expected_sl",
    [
        ("LONG", 10.0, 4.0, 110.0, 96.0),
        ("SHORT", 10.0, 4.0, 90.0, 104.0),
    ],
)
def test_execute_from_signal_uses_tp_sl_settings(signal_type, tp_pct, sl_pct, expected_tp, expected_sl):
    from modules.auto_trade.execution.order_executor import OrderExecutor

    manager = _CapturingOrderManager()
    with patch("modules.auto_trade.execution.order_executor.ExchangeManager"), patch(
        "modules.auto_trade.execution.order_executor.DataFetcher"
    ), patch("modules.auto_trade.execution.order_executor.BinanceClient", _DummyClient), patch(
        "modules.auto_trade.execution.order_executor.OrderManager", return_value=manager
    ):
        executor = OrderExecutor(api_key="k", api_secret="s", testnet=True, dry_run=True)
        result = executor.execute_from_signal(
            {"symbol": "BTCUSDT", "signal": signal_type, "score": 1.0},
            tp_sl_settings={"default_tp": tp_pct, "default_sl": sl_pct},
        )

        assert result["success"] is True
        assert manager.last_signal is not None
        assert manager.last_signal.entry_price == 100.0
        assert manager.last_signal.take_profit == pytest.approx(expected_tp)
        assert manager.last_signal.stop_loss == pytest.approx(expected_sl)

