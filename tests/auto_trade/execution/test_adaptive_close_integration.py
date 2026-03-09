from datetime import timedelta
from unittest.mock import Mock, patch

from modules.auto_trade.core.signal_selector import FinalSignal
from modules.auto_trade.execution.adaptive_close_calculator import AdaptiveCloseResult
from modules.auto_trade.execution.order_builder import OrderTicket
from modules.auto_trade.execution.order_executor import OrderExecutor
from modules.auto_trade.execution.order_manager import OrderManager


class _DummyExecutorClient:
    def __init__(self, *args, **kwargs):
        self.exchange = Mock()
        self.exchange.fetch_ticker = Mock(return_value={"last": 100.0})


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


class _FakeAdaptiveCalculator:
    def __init__(self, offset_hours: float):
        self.offset_hours = offset_hours

    def compute_adaptive_deadline(self, symbol, opened_at, ohlcv_df=None):
        return opened_at + timedelta(hours=self.offset_hours)

    def compute_adaptive_deadline_with_meta(self, symbol, opened_at, ohlcv_df=None):
        deadline = opened_at + timedelta(hours=self.offset_hours)
        return AdaptiveCloseResult(
            deadline_utc=deadline,
            source="adaptive",
            duration_hours=self.offset_hours,
            pelt_hours=self.offset_hours - 0.5,
            hmm_hours=self.offset_hours + 0.5,
        )


class _FakeRepoContext:
    def __init__(self):
        self.orders = Mock()


def test_order_executor_place_order_persists_adaptive_deadline():
    fake_repo = _FakeRepoContext()
    fake_adaptive = _FakeAdaptiveCalculator(offset_hours=3.0)

    with patch("modules.auto_trade.execution.order_executor.BinanceClient", _DummyExecutorClient), patch.object(
        OrderExecutor, "_init_adaptive_close_calculator", return_value=fake_adaptive
    ), patch("modules.auto_trade.database.repository.context.RepositoryContext.from_env", return_value=fake_repo):
        executor = OrderExecutor(api_key="k", api_secret="s", testnet=True, dry_run=False)
        executor._create_market_order = Mock(
            return_value={
                "market_order": {"id": "ord_1", "timestamp": 1700000000000},
                "entry_price": 100.0,
            }
        )

        result = executor.place_order(
            symbol="BTC/USDT",
            side="long",
            amount=50.0,
            leverage=2,
            take_profit=110.0,
            stop_loss=95.0,
        )

    assert result["success"] is True
    assert fake_repo.orders.create_order.call_count == 1
    order_payload = fake_repo.orders.create_order.call_args.args[0]
    assert order_payload["execution_mode"] == "MANUAL"
    assert "auto_close_deadline_utc" in order_payload
    assert isinstance(order_payload["auto_close_deadline_utc"], str)
    assert order_payload["auto_close_deadline_source"] == "adaptive"
    assert order_payload["adaptive_close_duration_hours"] == 3.0
    assert order_payload["adaptive_close_pelt_hours"] == 2.5
    assert order_payload["adaptive_close_hmm_hours"] == 3.5


def test_order_manager_execute_signal_persists_adaptive_deadline():
    fake_repo = _FakeRepoContext()
    fake_adaptive = _FakeAdaptiveCalculator(offset_hours=2.5)

    data_fetcher = Mock()
    data_fetcher.fetch_binance_futures_positions = Mock(return_value=[])

    with patch("modules.auto_trade.execution.order_manager.BinanceClient", _DummyManagerClient), patch.object(
        OrderManager, "_init_adaptive_close_calculator", return_value=fake_adaptive
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
                "market_order": {"id": "ord_2", "timestamp": 1700000000000, "clientOrderId": "cid_1"},
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

        signal = FinalSignal(
            symbol="BTC/USDT",
            signal_type="LONG",
            entry_price=100.0,
            stop_loss=95.0,
            take_profit=110.0,
            leverage=2,
        )
        result = manager.execute_signal(signal)

    assert result is not None
    assert fake_repo.orders.create_order.call_count == 1
    order_payload = fake_repo.orders.create_order.call_args.args[0]
    assert order_payload["execution_mode"] == "AUTO"
    assert "auto_close_deadline_utc" in order_payload
    assert isinstance(order_payload["auto_close_deadline_utc"], str)
    assert order_payload["auto_close_deadline_source"] == "adaptive"
    assert order_payload["adaptive_close_duration_hours"] == 2.5
    assert order_payload["adaptive_close_pelt_hours"] == 2.0
    assert order_payload["adaptive_close_hmm_hours"] == 3.0
