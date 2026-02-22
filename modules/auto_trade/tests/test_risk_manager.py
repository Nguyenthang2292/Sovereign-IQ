from unittest.mock import MagicMock

from modules.auto_trade.execution.risk_manager import RiskManager


def test_risk_manager_sizing_and_gate(monkeypatch):
    # Mock database queries since RiskManager.__init__ tries to load state
    monkeypatch.setattr("modules.auto_trade.database.queries.get_system_state", lambda _key: False)
    monkeypatch.setattr("modules.auto_trade.database.queries.set_system_state", lambda _key, _value: None)

    fetcher_mock = MagicMock()
    # fetch_binance_account_balance returns the mocked value
    fetcher_mock.fetch_binance_account_balance.return_value = 1000.0

    rm = RiskManager(data_fetcher=fetcher_mock, balance_percentage=0.1, max_position_size=500.0)

    # 1000 * 0.1 = 100
    size = rm.calculate_position_size("mock_key", "mock_secret")
    assert size == 100.0

    # Trigger emergency stop
    rm.trigger_emergency_stop("Test")
    assert rm.is_emergency_stop_active is True

    # Sizing should fail if emergency stop is active
    size_after_stop = rm.calculate_position_size("mock_key", "mock_secret")
    assert size_after_stop is None


def test_emergency_stop_persists_across_reinstantiation(monkeypatch):
    state = {"emergency_stop": False}

    def _get_system_state(key: str):
        return state.get(key, False)

    def _set_system_state(key: str, value):
        state[key] = value

    monkeypatch.setattr("modules.auto_trade.database.queries.get_system_state", _get_system_state)
    monkeypatch.setattr("modules.auto_trade.database.queries.set_system_state", _set_system_state)

    fetcher_mock = MagicMock()
    rm = RiskManager(data_fetcher=fetcher_mock)
    assert rm.is_emergency_stop_active is False

    rm.trigger_emergency_stop("test persist")
    assert state["emergency_stop"] is True

    rm_fresh = RiskManager(data_fetcher=fetcher_mock)
    assert rm_fresh.is_emergency_stop_active is True
