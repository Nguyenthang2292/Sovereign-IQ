"""
Pytest fixtures for gui.utils tests.
"""

import os
import tempfile
from pathlib import Path
from unittest.mock import MagicMock, Mock

import pytest


@pytest.fixture
def temp_env_file(tmp_path):
    """Create a temporary .env file."""
    env_file = tmp_path / ".env"
    env_file.touch()
    return env_file


@pytest.fixture
def temp_db_file(tmp_path):
    """Create a temporary database file path."""
    db_file = tmp_path / "test_dry_run.db"
    return db_file


@pytest.fixture
def temp_settings_file(tmp_path):
    """Create a temporary settings file path."""
    settings_file = tmp_path / "test_settings.yaml"
    return settings_file


@pytest.fixture
def mock_env_vars(monkeypatch):
    """Mock environment variables."""
    test_env = {
        "BINANCE_API_KEY": "test_api_key",
        "BINANCE_API_SECRET": "test_api_secret",
        "BINANCE_TESTNET": "false",
    }

    for key, value in test_env.items():
        monkeypatch.setenv(key, value)

    return test_env


@pytest.fixture
def mock_exchange_manager():
    """Mock ExchangeManager."""
    mock = MagicMock()
    mock.api_key = "test_key"
    mock.api_secret = "test_secret"
    mock.testnet = False
    return mock


@pytest.fixture
def mock_data_fetcher():
    """Mock DataFetcher."""
    mock = MagicMock()
    mock.fetch_ticker.return_value = {"last": 42000.0}
    mock.fetch_binance_account_balance.return_value = 10000.0
    mock.fetch_binance_futures_positions.return_value = []
    return mock


@pytest.fixture
def mock_database_manager():
    """Mock DatabaseManager."""
    mock = MagicMock()
    mock.session_scope.return_value.__enter__ = Mock()
    mock.session_scope.return_value.__exit__ = Mock()
    return mock


@pytest.fixture
def sample_position_data():
    """Sample position data for testing."""
    return {
        "symbol": "BTC/USDT",
        "side": "LONG",
        "entry_price": 42000.0,
        "current_price": 42500.0,
        "size": 0.1,
        "leverage": 10,
        "take_profit": 44000.0,
        "stop_loss": 40000.0,
    }


@pytest.fixture
def sample_settings():
    """Sample settings data for testing."""
    return {
        "risk": {
            "max_position_size": 100.0,
            "max_open_positions": 3,
            "max_daily_loss": 50.0,
            "default_leverage": "10x",
        },
        "filters": {
            "min_signal_score": 0.7,
            "enable_xgboost": True,
            "symbol_whitelist": "BTC/USDT\nETH/USDT\nSOL/USDT",
            "min_volume": 50.0,
            "timeframe": "1h",
        },
        "api": {
            "exchange": "Demo",
            "mode": "DRY_RUN",
            "api_key": "",
            "api_secret": "",
        },
        "tp_sl": {
            "default_tp": 5.0,
            "default_sl": 2.5,
            "trailing_stop": False,
            "mode": "Percentage",
        },
        "scanner": {
            "scan_interval": 5,
            "timeframe": "1h",
            "symbol_list": "Top 20",
            "auto_start": True,
            "running": False,
        },
        "ui": {
            "theme": "dark",
            "font_size": 12,
            "window_size": {"width": 1200, "height": 800},
            "last_active_tab": "Dashboard",
            "column_visibility": {},
            "widget_order": {},
        },
    }
