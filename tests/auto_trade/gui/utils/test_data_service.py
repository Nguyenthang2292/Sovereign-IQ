"""
Comprehensive tests for DataService.

Tests cover:
- Initialization in different modes
- Price fetching
- Account data retrieval
- Signal and position management
- Mock data fallbacks
- Error handling
"""

from unittest.mock import MagicMock, Mock, patch

import pytest

from modules.auto_trade.gui.utils.data_service import DataService


class TestDataService:
    """Test DataService functionality."""

    def test_init_dry_run_mode(self):
        """Test initialization in DRY_RUN mode."""
        service = DataService(mode="DRY_RUN")

        assert service.mode == "DRY_RUN"
        assert service.mock_price_feed is not None
        assert service.data_fetcher is None  # Should not initialize in dry run
        assert service.exchange_manager is None

    def test_init_demo_mode(self, mock_env_vars):
        """Test initialization in DEMO mode."""
        # DataFetcher and ExchangeManager are lazy-loaded, so no need to patch at module level
        service = DataService(mode="DEMO")

        assert service.mode == "DEMO"
        assert service.mock_price_feed is not None  # Always available as fallback

    def test_init_production_mode(self, mock_env_vars):
        """Test initialization in PRODUCTION mode."""
        service = DataService(mode="PRODUCTION")

        assert service.mode == "PRODUCTION"
        assert service.mock_price_feed is not None

    def test_get_current_price_dry_run(self):
        """Test getting current price in DRY_RUN mode."""
        service = DataService(mode="DRY_RUN")

        price = service.get_current_price("BTC/USDT")

        assert isinstance(price, float)
        assert price > 0

    def test_get_current_price_from_exchange(self, mock_env_vars):
        """Test getting current price from exchange."""
        service = DataService(mode="DEMO")

        # Mock data fetcher to return a price
        service.data_fetcher = MagicMock()
        service.data_fetcher.fetch_ticker.return_value = {"last": 42000.0}

        price = service.get_current_price("BTC/USDT")

        assert price == 42000.0
        service.data_fetcher.fetch_ticker.assert_called_once_with("BTCUSDT")

    def test_get_current_price_fallback_on_error(self, mock_env_vars):
        """Test fallback to mock price on exchange error."""
        service = DataService(mode="DEMO")

        # Mock data fetcher to raise error
        service.data_fetcher = MagicMock()
        service.data_fetcher.fetch_ticker.side_effect = Exception("Network error")

        price = service.get_current_price("BTC/USDT")

        # Should fallback to mock price
        assert isinstance(price, float)
        assert price > 0

    def test_get_account_data_dry_run(self):
        """Test getting account data in DRY_RUN mode."""
        service = DataService(mode="DRY_RUN")

        account_data = service.get_account_data()

        assert account_data is not None
        assert "balance" in account_data
        assert "available" in account_data
        assert account_data["balance"] == 10000.0  # Default dry run balance

    def test_get_account_data_from_exchange(self, mock_env_vars):
        """Test getting account data from exchange."""
        service = DataService(mode="DEMO")

        # Mock data fetcher
        service.data_fetcher = MagicMock()
        service.data_fetcher.fetch_binance_account_balance.return_value = 5000.0
        service.data_fetcher.fetch_binance_futures_positions.return_value = []

        service.api_key = "test_key"
        service.api_secret = "test_secret"

        account_data = service.get_account_data()

        assert account_data is not None
        assert account_data["balance"] == 5000.0

    def test_get_quick_stats_dry_run(self):
        """Test getting quick stats in DRY_RUN mode."""
        service = DataService(mode="DRY_RUN")

        stats = service.get_quick_stats()

        assert stats is not None
        assert "open_positions" in stats
        assert "today_trades" in stats
        assert "win_rate" in stats
        assert "mode" in stats
        assert stats["mode"] == "DRY_RUN"

    def test_get_signals(self):
        """Test getting signals from database."""
        service = DataService(mode="DRY_RUN")

        # If no database, should return demo signals
        signals = service.get_signals(min_score=0.7)

        assert len(signals) > 0
        assert "created_at" in signals[0]
        assert "created_at_ts" in signals[0]
        assert isinstance(signals[0]["created_at_ts"], (int, float))

    def test_get_signals_fallback(self):
        """Test getting signals fallback to demo data."""
        service = DataService(mode="DRY_RUN")
        service.database_manager = None  # No database

        signals = service.get_signals()

        # Should return demo signals
        assert len(signals) > 0
        assert signals[0]["symbol"] == "BTCUSDT"
        assert "created_at" in signals[0]
        assert "created_at_ts" in signals[0]

    def test_get_positions_dry_run(self):
        """Test getting positions in DRY_RUN mode."""
        service = DataService(mode="DRY_RUN")

        # Will return empty list if no positions in DB
        positions = service.get_positions()

        assert isinstance(positions, list)

    def test_get_positions_from_exchange(self, mock_env_vars):
        """Test getting positions from exchange."""
        service = DataService(mode="DEMO")

        # Mock data fetcher
        service.data_fetcher = MagicMock()
        service.data_fetcher.fetch_binance_futures_positions.return_value = [
            {
                "symbol": "BTCUSDT",
                "direction": "LONG",
                "entry_price": 42000.0,
                "contracts": 0.1,
                "size_usdt": 4200.0,
            }
        ]

        service.api_key = "test_key"
        service.api_secret = "test_secret"

        positions = service.get_positions()

        assert len(positions) > 0

    def test_mock_price_feed_centralization(self):
        """Test that mock price feed is centralized."""
        service = DataService(mode="DRY_RUN")

        # Get price multiple times
        price1 = service.get_current_price("BTC/USDT")
        price2 = service.get_current_price("BTC/USDT")

        # Prices should come from same mock instance
        assert isinstance(price1, float)
        assert isinstance(price2, float)

    def test_error_handling_in_price_fetch(self):
        """Test error handling in price fetching."""
        service = DataService(mode="DRY_RUN")

        # Force an error by setting mock_price_feed to None then getting it
        service.mock_price_feed = None

        # Should not crash, should create new mock
        price = service.get_current_price("BTC/USDT")
        assert isinstance(price, float)

    def test_environment_variables_loaded(self, mock_env_vars):
        """Test that environment variables are loaded correctly."""
        service = DataService(mode="DEMO")

        assert service.api_key == "test_api_key"
        assert service.api_secret == "test_api_secret"
        assert service.testnet is False

    def test_demo_account_data_fallback(self):
        """Test demo account data fallback."""
        service = DataService(mode="DEMO")
        service.data_fetcher = None  # No data fetcher

        account_data = service.get_account_data()

        # Should return demo data
        assert account_data["balance"] == 1000.0  # type: ignore[index]
