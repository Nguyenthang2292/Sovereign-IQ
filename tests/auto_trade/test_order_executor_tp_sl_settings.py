from unittest.mock import Mock, patch

import pytest

from modules.order_book.models import OBIDecision


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


def _snapshot_for_obi(*, bid_qty: float, ask_qty: float):
    from modules.order_book.models import OrderBookSnapshot

    return OrderBookSnapshot(
        symbol="BTC/USDT",
        bids=[(50000.0, bid_qty)],
        asks=[(50001.0, ask_qty)],
        timestamp=1234567890.0,
    )


def _trades_for_obi(*, buy_qty: float, sell_qty: float):
    from modules.order_book.models import AggTrade

    return [
        AggTrade(price=50000.0, quantity=buy_qty, timestamp=1.0, is_buyer_maker=False),
        AggTrade(price=50001.0, quantity=sell_qty, timestamp=2.0, is_buyer_maker=True),
    ]


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


def test_order_executor_init_without_order_book_config_keeps_gate_none():
    from modules.auto_trade.execution.order_executor import OrderExecutor

    with patch("modules.auto_trade.execution.order_executor.BinanceClient", _DummyClient):
        executor = OrderExecutor(api_key="k", api_secret="s", testnet=True, dry_run=True)

    assert executor._order_book_imbalance_gate is None


def test_execute_from_signal_returns_skipped_when_order_book_gate_conflicts():
    from modules.auto_trade.execution.order_executor import OrderExecutor

    manager = _CapturingOrderManager()
    gate_instance = Mock()
    gate_instance.check.return_value = (OBIDecision.SKIP, None)

    with patch("modules.auto_trade.execution.order_executor.ExchangeManager"), patch(
        "modules.auto_trade.execution.order_executor.DataFetcher"
    ), patch("modules.auto_trade.execution.order_executor.BinanceClient", _DummyClient), patch(
        "modules.auto_trade.execution.order_executor.OrderManager", return_value=manager
    ), patch("modules.order_book.order_book_imbalance_gate.OrderBookImbalanceGate", return_value=gate_instance):
        executor = OrderExecutor(
            api_key="k",
            api_secret="s",
            testnet=True,
            dry_run=True,
            order_book_imbalance_config={"enabled": True},
        )
        result = executor.execute_from_signal(
            {"symbol": "BTCUSDT", "signal": "LONG", "score": 1.0},
            tp_sl_settings={"default_tp": 10.0, "default_sl": 4.0},
        )

    assert result == {
        "success": False,
        "skipped": True,
        "reason": "ORDER_BOOK_IMBALANCE_CONFLICT",
    }
    assert manager.last_signal is None


def test_execute_from_signal_continues_when_order_book_gate_passes():
    from modules.auto_trade.execution.order_executor import OrderExecutor

    manager = _CapturingOrderManager()
    gate_instance = Mock()
    gate_instance.check.return_value = (OBIDecision.PASS, None)

    with patch("modules.auto_trade.execution.order_executor.ExchangeManager"), patch(
        "modules.auto_trade.execution.order_executor.DataFetcher"
    ), patch("modules.auto_trade.execution.order_executor.BinanceClient", _DummyClient), patch(
        "modules.auto_trade.execution.order_executor.OrderManager", return_value=manager
    ), patch("modules.order_book.order_book_imbalance_gate.OrderBookImbalanceGate", return_value=gate_instance):
        executor = OrderExecutor(
            api_key="k",
            api_secret="s",
            testnet=True,
            dry_run=True,
            order_book_imbalance_config={"enabled": True},
        )
        result = executor.execute_from_signal(
            {"symbol": "BTCUSDT", "signal": "LONG", "score": 1.0},
            tp_sl_settings={"default_tp": 10.0, "default_sl": 4.0},
        )

    assert result["success"] is True
    assert manager.last_signal is not None


def test_order_executor_maps_ob_depth_to_depth_limit_for_gate():
    from modules.auto_trade.execution.order_executor import OrderExecutor

    with patch("modules.auto_trade.execution.order_executor.BinanceClient", _DummyClient), patch(
        "modules.order_book.order_book_imbalance_gate.OrderBookImbalanceGate"
    ) as mock_gate:
        OrderExecutor(
            api_key="k",
            api_secret="s",
            testnet=True,
            dry_run=True,
            order_book_imbalance_config={"enabled": True, "ob_depth": 55},
        )

    gate_kwargs = mock_gate.call_args.kwargs
    assert gate_kwargs["depth_limit"] == 55
    assert "ob_depth" not in gate_kwargs


def test_execute_from_signal_dry_run_with_real_gate_passes_on_positive_score():
    from modules.auto_trade.execution.order_executor import OrderExecutor

    manager = _CapturingOrderManager()
    snapshot = _snapshot_for_obi(bid_qty=100.0, ask_qty=10.0)
    trades = _trades_for_obi(buy_qty=10.0, sell_qty=1.0)

    with patch("modules.auto_trade.execution.order_executor.ExchangeManager"), patch(
        "modules.auto_trade.execution.order_executor.DataFetcher"
    ), patch("modules.auto_trade.execution.order_executor.BinanceClient", _DummyClient), patch(
        "modules.auto_trade.execution.order_executor.OrderManager", return_value=manager
    ), patch("modules.order_book.order_book_imbalance_gate.fetch_depth", return_value=snapshot), patch(
        "modules.order_book.order_book_imbalance_gate.fetch_agg_trades", return_value=trades
    ):
        executor = OrderExecutor(
            api_key="k",
            api_secret="s",
            testnet=True,
            dry_run=True,
            order_book_imbalance_config={
                "enabled": True,
                "threshold": 0.15,
                "retry_wait_seconds": 0,
                "max_retries": 1,
            },
        )
        result = executor.execute_from_signal(
            {"symbol": "BTCUSDT", "signal": "LONG", "score": 1.0},
            tp_sl_settings={"default_tp": 10.0, "default_sl": 4.0},
        )

    assert result["success"] is True
    assert manager.last_signal is not None


def test_execute_from_signal_dry_run_with_real_gate_skips_after_retry():
    from modules.auto_trade.execution.order_executor import OrderExecutor

    manager = _CapturingOrderManager()
    snapshot = _snapshot_for_obi(bid_qty=10.0, ask_qty=100.0)
    trades = _trades_for_obi(buy_qty=1.0, sell_qty=10.0)

    with patch("modules.auto_trade.execution.order_executor.ExchangeManager"), patch(
        "modules.auto_trade.execution.order_executor.DataFetcher"
    ), patch("modules.auto_trade.execution.order_executor.BinanceClient", _DummyClient), patch(
        "modules.auto_trade.execution.order_executor.OrderManager", return_value=manager
    ), patch("modules.order_book.order_book_imbalance_gate.fetch_depth", return_value=snapshot), patch(
        "modules.order_book.order_book_imbalance_gate.fetch_agg_trades", return_value=trades
    ):
        executor = OrderExecutor(
            api_key="k",
            api_secret="s",
            testnet=True,
            dry_run=True,
            order_book_imbalance_config={
                "enabled": True,
                "threshold": 0.15,
                "retry_wait_seconds": 0,
                "max_retries": 1,
            },
        )
        result = executor.execute_from_signal(
            {"symbol": "BTCUSDT", "signal": "LONG", "score": 1.0},
            tp_sl_settings={"default_tp": 10.0, "default_sl": 4.0},
        )

    assert result == {
        "success": False,
        "skipped": True,
        "reason": "ORDER_BOOK_IMBALANCE_CONFLICT",
    }
    assert manager.last_signal is None

