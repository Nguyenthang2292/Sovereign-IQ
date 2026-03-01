import pytest
import time
from unittest.mock import patch, MagicMock

from modules.auto_trade.gui.services.data_service import DataService


def test_data_service_dry_run_mode():
    ds = DataService(mode="DRY_RUN")

    # Should use mock data
    account = ds.get_account_data()
    assert account["balance"] == 10000.0

    stats = ds.get_quick_stats()
    assert stats["mode"] == "DRY_RUN"


@patch("modules.auto_trade.execution.binance_client.BinanceClient")
def test_data_service_client_caching(mock_client_class):
    mock_client_class.side_effect = lambda *args, **kwargs: MagicMock()
    # Set up mock credentials
    with patch("os.getenv", return_value="fake"):
        ds = DataService(mode="PRODUCTION")

    ds.api_key = "test_key"
    ds.api_secret = "test_secret"

    # First call creates client
    client1 = ds._get_or_create_client()
    assert client1 is not None
    assert mock_client_class.call_count == 1

    # Second call returns cached client
    client2 = ds._get_or_create_client()
    assert client2 is client1
    assert mock_client_class.call_count == 1

    # Invalidate cache like SETTINGS_SAVED event would
    with patch("modules.auto_trade.gui.services.credential_manager.CredentialManager.load_credentials") as mock_load:
        mock_load.return_value = {"api_key": "new_key", "api_secret": "new_secret"}
        ds._on_settings_saved(None)
        ds._reload_credentials()

    # Third call creates new client
    client3 = ds._get_or_create_client()
    assert client3 is not client1
    assert mock_client_class.call_count == 2


@patch("modules.auto_trade.gui.services.tp_sl_sync.TPSLSyncService.sync_position_tp_sl")
@patch("modules.auto_trade.gui.services.data_service.DataService._get_or_create_client")
def test_data_service_tpsl_caching(mock_get_client, mock_sync):
    mock_get_client.return_value = MagicMock()
    mock_sync.return_value = {"take_profit": 1.0, "stop_loss": 0.5, "break_even": None}

    ds = DataService(mode="PRODUCTION")
    ds.repo_context = MagicMock()

    # First call, should call sync
    res1 = ds.get_cached_tpsl("BTCUSDT", ttl_seconds=10)
    assert res1["take_profit"] == 1.0
    assert mock_sync.call_count == 1

    # Second call, within TTL, should use cache
    res2 = ds.get_cached_tpsl("BTCUSDT", ttl_seconds=10)
    assert res2["take_profit"] == 1.0
    assert mock_sync.call_count == 1

    # Simulate time passing by modifying the cache time
    ds._tpsl_cache_time["BTCUSDT"] = time.monotonic() - 20

    # Third call, expired TTL, should call sync
    res3 = ds.get_cached_tpsl("BTCUSDT", ttl_seconds=10)
    assert mock_sync.call_count == 2
