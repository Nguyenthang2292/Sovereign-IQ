import json
from pathlib import Path
from unittest.mock import MagicMock

import pytest

from modules.auto_trade.core.signal_selector import FinalSignal
from modules.auto_trade.execution.order_executor import OrderExecutor
from modules.auto_trade.execution.order_manager import OrderManager
from modules.auto_trade.execution.order_builder import OrderTicket
from modules.auto_trade.websocket.client import BinanceWebSocketClient


def _build_valid_signal() -> FinalSignal:
    return FinalSignal(
        symbol="BTC/USDT",
        signal_type="LONG",
        entry_price=10000.0,
        stop_loss=9900.0,
        take_profit=10100.0,
        leverage=2,
        score=80.0,
    )


def test_order_manager_retry_fetch_ticker_recovers_second_attempt(monkeypatch):
    monkeypatch.setattr("tenacity.nap.sleep", lambda _seconds: None)
    monkeypatch.setattr("modules.auto_trade.database.queries.get_system_state", lambda _key: False)
    monkeypatch.setattr("modules.auto_trade.database.queries.set_system_state", lambda _key, _value: None)

    data_fetcher = MagicMock()
    manager = OrderManager(data_fetcher=data_fetcher, api_key="k", api_secret="s", dry_run=True)

    manager.binance_client.exchange.fetch_ticker = MagicMock(side_effect=[Exception("network"), {"last": 100.0}])

    ticker = manager._fetch_ticker("BTC/USDT")

    assert ticker["last"] == 100.0
    assert manager.binance_client.exchange.fetch_ticker.call_count == 2


def test_order_executor_retry_fetch_ticker_recovers_second_attempt(monkeypatch):
    monkeypatch.setattr("tenacity.nap.sleep", lambda _seconds: None)

    executor = object.__new__(OrderExecutor)
    executor._client = MagicMock()
    executor._client.exchange.fetch_ticker = MagicMock(side_effect=[Exception("network"), {"last": 200.0}])

    ticker = OrderExecutor._fetch_ticker(executor, "BTC/USDT")

    assert ticker["last"] == 200.0
    assert executor._client.exchange.fetch_ticker.call_count == 2


def test_order_manager_db_failure_writes_fallback_jsonl(tmp_path, monkeypatch):
    monkeypatch.setattr("modules.auto_trade.database.queries.get_system_state", lambda _key: False)
    monkeypatch.setattr("modules.auto_trade.database.queries.set_system_state", lambda _key, _value: None)
    monkeypatch.setattr("modules.auto_trade.execution.order_manager.Path.home", classmethod(lambda _cls: tmp_path))

    class _RepoContext:
        @staticmethod
        def from_env():
            raise RuntimeError("db down")

    monkeypatch.setattr("modules.auto_trade.database.repository.context.RepositoryContext", _RepoContext)

    data_fetcher = MagicMock()
    manager = OrderManager(data_fetcher=data_fetcher, api_key="k", api_secret="s", dry_run=False)

    manager.check_open_positions = MagicMock(return_value=None)
    manager.risk_manager.calculate_position_size = MagicMock(return_value=100.0)
    manager._fetch_ticker = MagicMock(return_value={"last": 10000.0})
    manager._fetch_account_balance = MagicMock(return_value=1000.0)
    manager.order_validator.validate_pre_order = MagicMock(return_value=True)
    manager.order_validator.validate_post_order = MagicMock(return_value=True)
    manager._create_market_order = MagicMock(return_value={"market_order": {"id": "1"}, "entry_price": 10000.0})

    result = manager.execute_signal(_build_valid_signal())

    assert result is not None
    fallback_file = Path(tmp_path) / ".auto_trade" / "fallback_orders.jsonl"
    assert fallback_file.exists()
    line = fallback_file.read_text(encoding="utf-8").strip().splitlines()[0]
    payload = json.loads(line)
    assert payload["order_id"] == "1"
    assert payload["symbol"] == "BTCUSDT"


@pytest.mark.asyncio
async def test_websocket_staleness_logs_warning(monkeypatch):
    client = object.__new__(BinanceWebSocketClient)
    client.running = True
    client._last_msg_time = 0.0
    client.staleness_timeout = 300

    warnings: list[str] = []

    async def _fake_sleep(_seconds: float):
        client.running = False

    monkeypatch.setattr("modules.auto_trade.websocket.client.asyncio.sleep", _fake_sleep)
    monkeypatch.setattr("modules.auto_trade.websocket.client.time.time", lambda: 301.0)
    monkeypatch.setattr("modules.auto_trade.websocket.client.log_warn", lambda message: warnings.append(message))
    monkeypatch.setattr("modules.auto_trade.websocket.client.log_info", lambda _message: None)

    await BinanceWebSocketClient._monitor_staleness(client)

    assert warnings
    assert "stale" in warnings[0].lower()


def test_order_executor_retry_create_order_recovers_second_attempt(monkeypatch):
    monkeypatch.setattr("tenacity.nap.sleep", lambda _seconds: None)

    executor = object.__new__(OrderExecutor)
    executor._client = MagicMock()
    ticket = OrderTicket(symbol="BTC/USDT", side="BUY", amount=10.0, leverage=2)
    executor._client.create_market_order = MagicMock(side_effect=[Exception("network"), {"market_order": {"id": "2"}}])

    result = OrderExecutor._create_market_order(executor, ticket)

    assert result == {"market_order": {"id": "2"}}
    assert executor._client.create_market_order.call_count == 2
