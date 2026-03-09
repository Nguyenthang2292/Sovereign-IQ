from datetime import datetime, timezone
from unittest.mock import MagicMock, Mock, patch

from modules.auto_trade.core.signal_selector import FinalSignal
from modules.auto_trade.execution.adaptive_close_calculator import AdaptiveCloseResult
from modules.auto_trade.execution.order_builder import OrderTicket
from modules.auto_trade.execution.order_manager import OrderManager


class _DummyManagerClient:
    def __init__(self, *args, **kwargs):
        self.exchange = Mock()
        self.exchange.fetch_ticker = Mock(return_value={"last": 100.0})

    def create_market_order(self, order):
        return {
            "market_order": {"id": "ignored", "timestamp": 1700000000000},
            "entry_price": 100.0,
            "order_ticket": order.to_dict(),
        }


class _FakeRepoContext:
    def __init__(self):
        self.orders = Mock()


def _build_order_manager(fake_repo, adaptive_calculator):
    data_fetcher = Mock()
    data_fetcher.fetch_binance_futures_positions = Mock(return_value=[])

    with patch("modules.auto_trade.execution.order_manager.BinanceClient", _DummyManagerClient), patch.object(
        OrderManager, "_init_adaptive_close_calculator", return_value=adaptive_calculator
    ), patch("modules.auto_trade.database.repository.context.RepositoryContext.from_env", return_value=fake_repo), patch(
        "modules.auto_trade.database.queries.get_system_state", return_value=False
    ):
        manager = OrderManager(
            data_fetcher=data_fetcher,
            api_key="k",
            api_secret="s",
            testnet=True,
            dry_run=False,
        )

    manager.risk_manager.calculate_position_size = Mock(return_value=100.0)
    manager._fetch_account_balance = Mock(return_value=1000.0)
    manager.order_validator.validate_pre_order = Mock(return_value=True)
    manager.order_validator.validate_post_order = Mock(return_value=True)
    manager._create_market_order = Mock(
        return_value={
            "market_order": {"id": "ord_meta_1", "timestamp": 1700000000000, "clientOrderId": "cid_meta_1"},
            "entry_price": 101.0,
            "order_ticket": OrderTicket(
                symbol="BTC/USDT",
                side="BUY",
                amount=100.0,
                leverage=2,
                take_profit_price=110.0,
                stop_loss_price=95.0,
            ).to_dict(),
        }
    )
    return manager


def test_order_flow_metadata_uses_adaptive_calculator_and_persists_new_fields():
    fake_repo = _FakeRepoContext()
    adaptive_calculator = MagicMock()
    adaptive_calculator.compute_adaptive_deadline_with_meta.return_value = AdaptiveCloseResult(
        deadline_utc=datetime(2026, 3, 9, 12, 30, 45, tzinfo=timezone.utc),
        source="adaptive",
        duration_hours=5.5,
        pelt_hours=4.2,
        hmm_hours=6.1,
    )

    manager = _build_order_manager(fake_repo=fake_repo, adaptive_calculator=adaptive_calculator)

    signal = FinalSignal(
        symbol="BTC/USDT",
        signal_type="LONG",
        entry_price=100.0,
        stop_loss=95.0,
        take_profit=110.0,
        leverage=2,
    )
    with patch("modules.auto_trade.database.repository.context.RepositoryContext.from_env", return_value=fake_repo):
        result = manager.execute_signal(signal)

    assert result is not None
    adaptive_calculator.compute_adaptive_deadline_with_meta.assert_called_once()
    assert fake_repo.orders.create_order.call_count == 1

    order_payload = fake_repo.orders.create_order.call_args.args[0]
    assert order_payload["auto_close_deadline_utc"] == "2026-03-09T12:30:45Z"
    assert order_payload["auto_close_deadline_source"] == "adaptive"
    assert order_payload["adaptive_close_duration_hours"] == 5.5
    assert order_payload["adaptive_close_pelt_hours"] == 4.2
    assert order_payload["adaptive_close_hmm_hours"] == 6.1


def test_order_flow_metadata_omits_none_pelt_hmm_values():
    fake_repo = _FakeRepoContext()
    adaptive_calculator = MagicMock()
    adaptive_calculator.compute_adaptive_deadline_with_meta.return_value = AdaptiveCloseResult(
        deadline_utc=datetime(2026, 3, 9, 12, 30, 45, tzinfo=timezone.utc),
        source="static",
        duration_hours=4.0,
        pelt_hours=None,
        hmm_hours=None,
    )

    manager = _build_order_manager(fake_repo=fake_repo, adaptive_calculator=adaptive_calculator)

    signal = FinalSignal(
        symbol="BTC/USDT",
        signal_type="LONG",
        entry_price=100.0,
        stop_loss=95.0,
        take_profit=110.0,
        leverage=2,
    )
    with patch("modules.auto_trade.database.repository.context.RepositoryContext.from_env", return_value=fake_repo):
        result = manager.execute_signal(signal)

    assert result is not None
    assert fake_repo.orders.create_order.call_count == 1
    order_payload = fake_repo.orders.create_order.call_args.args[0]

    assert order_payload["auto_close_deadline_source"] == "static"
    assert order_payload["adaptive_close_duration_hours"] == 4.0
    assert "adaptive_close_pelt_hours" not in order_payload
    assert "adaptive_close_hmm_hours" not in order_payload
